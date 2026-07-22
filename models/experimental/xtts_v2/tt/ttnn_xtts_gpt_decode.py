# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of the XTTS-v2 GPT transformer core (Block 3), DECODE path.

Greedy autoregressive generation with a per-layer KV-cache, matching the CPU reference
(reference/xtts_gpt_ref.py: reference_generate):

    prompt prefill: [prefix_emb, mel_emb[start] + mel_pos[0]]  -> per-layer K/V caches
                    + last-position latent -> mel_head -> code_0
    step m:         mel_emb[code_{m-1}] + mel_pos[m]  ->  (append K/V to cache, attend over
                    the whole cache, no mask) -> latent -> mel_head -> code_m
    stop at stop_token or max_new.

Positions come only from mel_pos (GPT2's wpe is nulled). The token embedding, the mel_head
projection, and argmax/stop run on host (outside the transformer block), same boundary as the
reference. The transformer stack + KV-cache run on device (fp32 / HiFi3, from the prefill port).

Validate against reference goldens (golden/gpt/generate/):
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/xtts_v2/tt/ttnn_xtts_gpt_decode.py
"""

import os

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import (
    DEFAULT_CKPT,
    LN_EPS,
    START_AUDIO_TOKEN,
    STOP_AUDIO_TOKEN,
    load_gen_head,
    pcc,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import (
    COMPUTE_KERNEL_CONFIG as CKC,
    HEAD_DIM,
    N_EMBD,
    N_HEAD,
    _mlp,
    _to_dev,
    load_ttnn_weights,
)

GEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt", "generate")
SCALE = 1.0 / (HEAD_DIM**0.5)


def _split_heads(t, seq_len):  # [1, seq, 1024] -> [1, n_head, seq, head_dim]
    t = ttnn.reshape(t, [1, seq_len, N_HEAD, HEAD_DIM])
    return ttnn.permute(t, [0, 2, 1, 3])


def _qkv(h, w, seq_len):
    qkv = ttnn.linear(h, w["attn_w"], bias=w["attn_b"], compute_kernel_config=CKC)  # [1, seq, 3072]
    q = ttnn.slice(qkv, [0, 0, 0], [1, seq_len, N_EMBD])
    k = ttnn.slice(qkv, [0, 0, N_EMBD], [1, seq_len, 2 * N_EMBD])
    v = ttnn.slice(qkv, [0, 0, 2 * N_EMBD], [1, seq_len, 3 * N_EMBD])
    return _split_heads(q, seq_len), _split_heads(k, seq_len), _split_heads(v, seq_len)


def _attn_out(scores_q, kv_k, kv_v, mask, seq_len):
    scores = ttnn.matmul(scores_q, ttnn.transpose(kv_k, -2, -1), compute_kernel_config=CKC)
    scores = ttnn.multiply(scores, SCALE)
    if mask is not None:
        scores = ttnn.add(scores, mask)
    attn = ttnn.softmax(scores, dim=-1)
    out = ttnn.matmul(attn, kv_v, compute_kernel_config=CKC)  # [1, n_head, seq, head_dim]
    out = ttnn.permute(out, [0, 2, 1, 3])
    return ttnn.reshape(out, [1, seq_len, N_EMBD])


def _prefill_layer(x, w, seq_len, causal_mask):
    """Prompt prefill: returns (x, k_cache, v_cache) for this layer."""
    h = ttnn.layer_norm(x, weight=w["ln_1_w"], bias=w["ln_1_b"], epsilon=LN_EPS, compute_kernel_config=CKC)
    q, k, v = _qkv(h, w, seq_len)
    attn = _attn_out(q, k, v, causal_mask, seq_len)
    x = ttnn.add(x, ttnn.linear(attn, w["proj_w"], bias=w["proj_b"], compute_kernel_config=CKC))
    h = ttnn.layer_norm(x, weight=w["ln_2_w"], bias=w["ln_2_b"], epsilon=LN_EPS, compute_kernel_config=CKC)
    x = ttnn.add(x, _mlp(h, w))
    return x, k, v  # k, v: [1, n_head, seq, head_dim]


def _decode_layer(x, w, k_cache, v_cache):
    """One decode step (single token). Appends to the cache and attends over it (no mask)."""
    h = ttnn.layer_norm(x, weight=w["ln_1_w"], bias=w["ln_1_b"], epsilon=LN_EPS, compute_kernel_config=CKC)
    q, k, v = _qkv(h, w, 1)  # each [1, n_head, 1, head_dim]
    k_cache = ttnn.concat([k_cache, k], dim=2)  # [1, n_head, L+1, head_dim]
    v_cache = ttnn.concat([v_cache, v], dim=2)
    attn = _attn_out(q, k_cache, v_cache, None, 1)  # new token attends over full cache -> [1,1,1024]
    x = ttnn.add(x, ttnn.linear(attn, w["proj_w"], bias=w["proj_b"], compute_kernel_config=CKC))
    h = ttnn.layer_norm(x, weight=w["ln_2_w"], bias=w["ln_2_b"], epsilon=LN_EPS, compute_kernel_config=CKC)
    x = ttnn.add(x, _mlp(h, w))
    return x, k_cache, v_cache


def _last_latent(x, tail, seq_len):
    """Apply ln_f + final_norm; return the LAST position's latent as torch [1, 1, 1024]."""
    x = ttnn.layer_norm(x, weight=tail["ln_f_w"], bias=tail["ln_f_b"], epsilon=LN_EPS, compute_kernel_config=CKC)
    x = ttnn.layer_norm(x, weight=tail["fn_w"], bias=tail["fn_b"], epsilon=LN_EPS, compute_kernel_config=CKC)
    return ttnn.to_torch(x).float()[:, -1:, :]


def run_generate(device, prefix_emb, heads, max_new=24, ckpt_path=DEFAULT_CKPT, weights=None):
    """prefix_emb: torch [1, P, 1024]; heads: dict of host tensors from load_gen_head.
    Returns dict(codes [T], latents [1,T,1024], logits [1,T,1026])."""
    if weights is None:
        weights = load_ttnn_weights(device, ckpt_path)
    layers, tail = weights
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]

    def head(latent):  # host: [1,1,1024] -> [1,1,1026]
        return latent @ mh_w.t() + mh_b

    # ---- prompt prefill: [prefix, start] through all layers, capturing K/V caches ----
    start_emb = (mel_emb[START_AUDIO_TOKEN] + mel_pos[0]).view(1, 1, -1)
    inp = torch.cat([prefix_emb, start_emb], dim=1)  # [1, P+1, 1024]
    seq = inp.shape[1]
    mask_t = torch.zeros(seq, seq).masked_fill(torch.triu(torch.ones(seq, seq), diagonal=1).bool(), -1e9)
    x = _to_dev(inp, device)
    mask = _to_dev(mask_t.reshape(1, 1, seq, seq), device)

    k_caches, v_caches = [], []
    for w in layers:
        x, k, v = _prefill_layer(x, w, seq, mask)
        k_caches.append(k)
        v_caches.append(v)

    latent = _last_latent(x, tail, seq)
    logits = head(latent)
    code = int(logits.argmax(-1))
    codes, lat_list, log_list = [code], [latent], [logits]

    # ---- decode loop ----
    for m in range(1, max_new):
        if code == STOP_AUDIO_TOKEN:
            break
        emb = (mel_emb[code] + mel_pos[m]).view(1, 1, -1)
        x = _to_dev(emb, device)
        for li, w in enumerate(layers):
            x, k_caches[li], v_caches[li] = _decode_layer(x, w, k_caches[li], v_caches[li])
        latent = _last_latent(x, tail, 1)
        logits = head(latent)
        code = int(logits.argmax(-1))
        codes.append(code)
        lat_list.append(latent)
        log_list.append(logits)

    return {"codes": torch.tensor(codes), "latents": torch.cat(lat_list, dim=1), "logits": torch.cat(log_list, dim=1)}


def main():
    prefix = torch.load(os.path.join(GEN_DIR, "prefix_emb.pt"))
    ref_codes = torch.load(os.path.join(GEN_DIR, "codes.pt"))
    ref_logits = torch.load(os.path.join(GEN_DIR, "logits.pt"))
    ref_latents = torch.load(os.path.join(GEN_DIR, "latents.pt"))
    heads = load_gen_head(DEFAULT_CKPT)

    device = ttnn.open_device(device_id=0)
    try:
        out = run_generate(device, prefix, heads, max_new=ref_codes.numel())
    finally:
        ttnn.close_device(device)

    k = min(out["codes"].numel(), ref_codes.numel())
    print(f"[ttnn] ours codes : {out['codes'][:k].tolist()}")
    print(f"[ttnn] ref  codes : {ref_codes[:k].tolist()}")
    print(f"[ttnn] codes match (first {k}): {bool(torch.equal(out['codes'][:k], ref_codes[:k]))}")
    print(f"[ttnn] logits PCC  = {pcc(out['logits'][:, :k], ref_logits[:, :k]):.6f}")
    print(f"[ttnn] latents PCC = {pcc(out['latents'][:, :k], ref_latents[:, :k]):.6f}")


if __name__ == "__main__":
    main()
