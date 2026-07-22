# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of the XTTS-v2 GPT transformer core (Block 3) — prefill AND decode.

Mirrors the CPU reference (reference/xtts_gpt_ref.py, which holds both reference_forward and
reference_generate) by keeping both paths in one module, sharing a single `_gpt_layer` so the
transformer-block math exists once and prefill/decode stay numerically consistent by construction.

    run_prefill:  inputs_embeds [1, S, 1024] -> 30x block (causal mask) -> ln_f -> final_norm
                  = latents [1, S, 1024]     (the return_latent path that feeds the vocoder)

    run_generate: greedy KV-cache decode. Prompt prefill [prefix, start] captures per-layer K/V
                  caches (causal-masked). Each step embeds the previous code, appends its K/V to
                  the caches, and attends over the whole cache with NO mask (all cached positions
                  are past -> already causal). Token embed, mel_head, and argmax/stop run on host.

fp32 (HiFi3 + fp32 accumulation) to match the reference precision. HF Conv1D weights are [in,out],
so they feed ttnn.linear directly (NO transpose). Embeddings/positions/mel_head live on host.

Validate against goldens from reference/xtts_gpt_ref.py:
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/xtts_v2/tt/ttnn_xtts_gpt.py
"""

import os

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import (
    DEFAULT_CKPT,
    LN_EPS,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    START_AUDIO_TOKEN,
    STOP_AUDIO_TOKEN,
    load_gen_head,
    load_gpt_core_state,
    pcc,
)

HEAD_DIM = N_EMBD // N_HEAD  # 64
SCALE = 1.0 / (HEAD_DIM**0.5)
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt")
GEN_DIR = os.path.join(GOLDEN_DIR, "generate")

# fp32 to match the reference: fp32 tensors + HiFi3 math with fp32 accumulation (on Wormhole,
# HiFi4 + fp32-acc is worse than HiFi3 due to a documented HW bug).
DTYPE = ttnn.float32
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def _to_dev(t, device, dtype=DTYPE):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def load_ttnn_weights(device, ckpt_path=DEFAULT_CKPT):
    """Convert the GPT-core checkpoint tensors to on-device ttnn tensors (HF Conv1D [in,out] as-is)."""
    core = load_gpt_core_state(ckpt_path)  # keys: h.{i}.*, ln_f.*, final_norm.*
    f = lambda k: core[k].float()
    layers = []
    for i in range(N_LAYER):
        p = f"h.{i}."
        layers.append(
            {
                "ln_1_w": _to_dev(f(p + "ln_1.weight"), device),
                "ln_1_b": _to_dev(f(p + "ln_1.bias"), device),
                "attn_w": _to_dev(f(p + "attn.c_attn.weight"), device),  # [1024, 3072] (in,out)
                "attn_b": _to_dev(f(p + "attn.c_attn.bias"), device),
                "proj_w": _to_dev(f(p + "attn.c_proj.weight"), device),  # [1024, 1024]
                "proj_b": _to_dev(f(p + "attn.c_proj.bias"), device),
                "ln_2_w": _to_dev(f(p + "ln_2.weight"), device),
                "ln_2_b": _to_dev(f(p + "ln_2.bias"), device),
                "fc_w": _to_dev(f(p + "mlp.c_fc.weight"), device),  # [1024, 4096]
                "fc_b": _to_dev(f(p + "mlp.c_fc.bias"), device),
                "mproj_w": _to_dev(f(p + "mlp.c_proj.weight"), device),  # [4096, 1024]
                "mproj_b": _to_dev(f(p + "mlp.c_proj.bias"), device),
            }
        )
    tail = {
        "ln_f_w": _to_dev(f("ln_f.weight"), device),
        "ln_f_b": _to_dev(f("ln_f.bias"), device),
        "fn_w": _to_dev(f("final_norm.weight"), device),
        "fn_b": _to_dev(f("final_norm.bias"), device),
    }
    return layers, tail


# --------------------------------------------------------------------------------------
# Shared building blocks (used by both prefill and decode)
# --------------------------------------------------------------------------------------
def _split_heads(t, seq_len):  # [1, seq, 1024] -> [1, n_head, seq, head_dim]
    t = ttnn.reshape(t, [1, seq_len, N_HEAD, HEAD_DIM])
    return ttnn.permute(t, [0, 2, 1, 3])


def _qkv(h, w, seq_len):
    qkv = ttnn.linear(h, w["attn_w"], bias=w["attn_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)  # [1,seq,3072]
    q = ttnn.slice(qkv, [0, 0, 0], [1, seq_len, N_EMBD])
    k = ttnn.slice(qkv, [0, 0, N_EMBD], [1, seq_len, 2 * N_EMBD])
    v = ttnn.slice(qkv, [0, 0, 2 * N_EMBD], [1, seq_len, 3 * N_EMBD])
    return _split_heads(q, seq_len), _split_heads(k, seq_len), _split_heads(v, seq_len)


def _attn_out(q, k, v, mask, q_seq_len):
    """q [1,nh,q_seq,hd], k/v [1,nh,kv_seq,hd] -> [1, q_seq, 1024]. kv_seq may exceed q_seq (cache)."""
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    scores = ttnn.multiply(scores, SCALE)
    if mask is not None:
        scores = ttnn.add(scores, mask)
    attn = ttnn.softmax(scores, dim=-1)
    out = ttnn.matmul(attn, v, compute_kernel_config=COMPUTE_KERNEL_CONFIG)  # [1, nh, q_seq, hd]
    out = ttnn.permute(out, [0, 2, 1, 3])
    return ttnn.reshape(out, [1, q_seq_len, N_EMBD])


def _mlp(x, w):
    x = ttnn.linear(x, w["fc_w"], bias=w["fc_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)  # [1, S, 4096]
    x = ttnn.gelu(x)  # HF gelu_new (tanh approx)
    return ttnn.linear(x, w["mproj_w"], bias=w["mproj_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)  # [1,S,1024]


def _gpt_layer(x, w, seq_len, mask=None, k_cache=None, v_cache=None):
    """One GPT-2 block. Shared by prefill and decode:
      - prefill: mask=causal, no cache; returned (k, v) become the initial cache.
      - decode:  mask=None, k_cache/v_cache given; new K/V appended, attention over full cache.
    Returns (x, k, v) where k/v are the full (possibly cache-appended) K/V for this layer."""
    h = ttnn.layer_norm(x, weight=w["ln_1_w"], bias=w["ln_1_b"], epsilon=LN_EPS, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    q, k, v = _qkv(h, w, seq_len)
    if k_cache is not None:
        k = ttnn.concat([k_cache, k], dim=2)
        v = ttnn.concat([v_cache, v], dim=2)
    attn = _attn_out(q, k, v, mask, seq_len)
    x = ttnn.add(x, ttnn.linear(attn, w["proj_w"], bias=w["proj_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG))
    h = ttnn.layer_norm(x, weight=w["ln_2_w"], bias=w["ln_2_b"], epsilon=LN_EPS, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    x = ttnn.add(x, _mlp(h, w))
    return x, k, v


def _apply_tail(x, tail):
    """GPT2 ln_f followed by XTTS's extra final_norm."""
    x = ttnn.layer_norm(x, weight=tail["ln_f_w"], bias=tail["ln_f_b"], epsilon=LN_EPS, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
    return ttnn.layer_norm(x, weight=tail["fn_w"], bias=tail["fn_b"], epsilon=LN_EPS, compute_kernel_config=COMPUTE_KERNEL_CONFIG)


def _causal_mask(seq_len, device):
    m = torch.zeros(seq_len, seq_len).masked_fill(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), -1e9)
    return _to_dev(m.reshape(1, 1, seq_len, seq_len), device)


# --------------------------------------------------------------------------------------
# Prefill (return_latent path)
# --------------------------------------------------------------------------------------
def run_prefill(device, inputs_embeds, weights=None, ckpt_path=DEFAULT_CKPT):
    """inputs_embeds: torch [1, S, 1024] -> latents: torch [1, S, 1024]."""
    if weights is None:
        weights = load_ttnn_weights(device, ckpt_path)
    layers, tail = weights
    seq = inputs_embeds.shape[1]
    mask = _causal_mask(seq, device)
    x = _to_dev(inputs_embeds, device)
    for w in layers:
        x, _, _ = _gpt_layer(x, w, seq, mask=mask)
    return ttnn.to_torch(_apply_tail(x, tail)).float()


# --------------------------------------------------------------------------------------
# Decode (greedy, KV-cache)
# --------------------------------------------------------------------------------------
def run_generate(device, prefix_emb, heads, max_new=24, weights=None, ckpt_path=DEFAULT_CKPT):
    """prefix_emb: torch [1, P, 1024]; heads: dict from load_gen_head.
    Returns dict(codes [T], latents [1,T,1024], logits [1,T,1026])."""
    if weights is None:
        weights = load_ttnn_weights(device, ckpt_path)
    layers, tail = weights
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]

    def head(latent):  # host: [1,1,1024] -> [1,1,1026]
        return latent @ mh_w.t() + mh_b

    def last_latent(x):
        return ttnn.to_torch(_apply_tail(x, tail)).float()[:, -1:, :]

    # prompt prefill: [prefix, start] -> per-layer K/V caches + code_0
    start_emb = (mel_emb[START_AUDIO_TOKEN] + mel_pos[0]).view(1, 1, -1)
    inp = torch.cat([prefix_emb, start_emb], dim=1)
    seq = inp.shape[1]
    mask = _causal_mask(seq, device)
    x = _to_dev(inp, device)
    k_caches, v_caches = [], []
    for w in layers:
        x, k, v = _gpt_layer(x, w, seq, mask=mask)
        k_caches.append(k)
        v_caches.append(v)
    latent = last_latent(x)
    logits = head(latent)
    code = int(logits.argmax(-1))
    codes, lat_list, log_list = [code], [latent], [logits]

    # decode loop
    for m in range(1, max_new):
        if code == STOP_AUDIO_TOKEN:
            break
        emb = (mel_emb[code] + mel_pos[m]).view(1, 1, -1)
        x = _to_dev(emb, device)
        for li, w in enumerate(layers):
            x, k_caches[li], v_caches[li] = _gpt_layer(x, w, 1, mask=None, k_cache=k_caches[li], v_cache=v_caches[li])
        latent = last_latent(x)
        logits = head(latent)
        code = int(logits.argmax(-1))
        codes.append(code)
        lat_list.append(latent)
        log_list.append(logits)

    return {"codes": torch.tensor(codes), "latents": torch.cat(lat_list, dim=1), "logits": torch.cat(log_list, dim=1)}


def main():
    device = ttnn.open_device(device_id=0)
    try:
        weights = load_ttnn_weights(device, DEFAULT_CKPT)

        # prefill vs golden
        inputs_embeds = torch.load(os.path.join(GOLDEN_DIR, "inputs_embeds.pt"))
        golden = torch.load(os.path.join(GOLDEN_DIR, "latents.pt"))
        out = run_prefill(device, inputs_embeds, weights=weights)
        print(f"[ttnn] prefill latents {tuple(out.shape)}  PCC = {pcc(out, golden):.6f}")

        # decode vs golden
        prefix = torch.load(os.path.join(GEN_DIR, "prefix_emb.pt"))
        ref_codes = torch.load(os.path.join(GEN_DIR, "codes.pt"))
        ref_logits = torch.load(os.path.join(GEN_DIR, "logits.pt"))
        heads = load_gen_head(DEFAULT_CKPT)
        gen = run_generate(device, prefix, heads, max_new=ref_codes.numel(), weights=weights)
        k = ref_codes.numel()
        print(f"[ttnn] decode codes match: {bool(torch.equal(gen['codes'][:k], ref_codes[:k]))}")
        print(f"[ttnn] decode logits PCC = {pcc(gen['logits'][:, :k], ref_logits[:, :k]):.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
