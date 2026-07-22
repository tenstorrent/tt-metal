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


def load_ttnn_weights(device, ckpt_path=DEFAULT_CKPT, dtype=DTYPE, matmul_dtype=None):
    """Convert the GPT-core checkpoint tensors to on-device ttnn tensors (HF Conv1D [in,out] as-is).
    dtype=float32 (default) for the accuracy-first prefill path; bfloat16 for the fast decode path.
    matmul_dtype (optional) sets a *separate* precision for the four big weight matrices only
    (attn/proj/c_fc/c_proj) — bfloat8_b there halves the per-token weight DRAM traffic that dominates
    decode, while layer-norm/bias stay at `dtype`. Defaults to `dtype` (uniform precision)."""
    core = load_gpt_core_state(ckpt_path)  # keys: h.{i}.*, ln_f.*, final_norm.*
    to = lambda k: _to_dev(core[k].float(), device, dtype)
    to_mm = lambda k: _to_dev(core[k].float(), device, matmul_dtype or dtype)  # weight matrices
    layers = []
    for i in range(N_LAYER):
        p = f"h.{i}."
        layers.append(
            {
                "ln_1_w": to(p + "ln_1.weight"),
                "ln_1_b": to(p + "ln_1.bias"),
                "attn_w": to_mm(p + "attn.c_attn.weight"),  # [1024, 3072] (in,out)
                "attn_b": to(p + "attn.c_attn.bias"),
                "proj_w": to_mm(p + "attn.c_proj.weight"),  # [1024, 1024]
                "proj_b": to(p + "attn.c_proj.bias"),
                "ln_2_w": to(p + "ln_2.weight"),
                "ln_2_b": to(p + "ln_2.bias"),
                "fc_w": to_mm(p + "mlp.c_fc.weight"),  # [1024, 4096]
                "fc_b": to(p + "mlp.c_fc.bias"),
                "mproj_w": to_mm(p + "mlp.c_proj.weight"),  # [4096, 1024]
                "mproj_b": to(p + "mlp.c_proj.bias"),
            }
        )
    tail = {
        "ln_f_w": to("ln_f.weight"),
        "ln_f_b": to("ln_f.bias"),
        "fn_w": to("final_norm.weight"),
        "fn_b": to("final_norm.bias"),
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


# --------------------------------------------------------------------------------------
# Fast decode: bf16 flash-decode + paged KV-cache + a position "mailbox" tensor, captured
# into a device trace and replayed per token (one host dispatch/token instead of ~600).
# --------------------------------------------------------------------------------------
class TracedGPTDecoder:
    """One 30-layer decode step, trace-captured and replayed per token. The position lives in a
    device tensor (`_pos`) threaded into paged_update_cache (which cache line to write) and
    scaled_dot_product_attention_decode (how much of the cache to read), so a single captured
    graph serves every position. bf16 (flash-decode + paged cache are bf16-only). Needs the device
    opened with a trace_region_size. mel_head / argmax / embedding stay on host (see generate_traced)."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, max_seq=128):
        self.device = device
        # Faithful by default: bf16 everywhere (flash-decode/cache are bf16-only anyway).
        # bfloat8_b on the four matmul weights buys ~7% (12.4->11.5 ms/token) by halving per-token
        # weight DRAM traffic, with codes bit-identical and latents PCC 0.99931 — but it's a
        # precision trade we're deferring until we can A/B *real audio* end-to-end. Enable with
        # matmul_dtype=ttnn.bfloat8_b when that day comes.
        self.layers, self.tail = load_ttnn_weights(device, ckpt_path, dtype=ttnn.bfloat16)
        # sdpa_decode is wrong when the cache length is an odd number of 32-tiles -> round to a multiple of 64.
        self.max_seq = ((max_seq + 63) // 64) * 64
        zc = torch.zeros(1, N_HEAD, self.max_seq, HEAD_DIM)
        mk = lambda: ttnn.from_torch(zc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.k_cache = [mk() for _ in range(N_LAYER)]
        self.v_cache = [mk() for _ in range(N_LAYER)]
        self._zeros = mk()  # preallocated (before any trace) so reset() never allocates during a trace
        self._pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), device=device)  # the position mailbox
        self._in = ttnn.from_torch(torch.zeros(1, 1, N_EMBD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # paged_update_cache wants the new K/V height-sharded (1 core suffices for batch=1)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        self._shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (32, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Width-sharded LayerNorm for decode: over a single-token [1,1,1024], the 1024-dim (= 32-tile)
        # reduction runs effectively single-core in the interleaved op. Spreading it width-wise across a
        # 32-core grid (1 tile/core) parallelizes the reduction — ~62 LayerNorms/token, so it adds up.
        # Configs + expanded weights are built here (before trace capture) so the trace stays safe.
        SH = 32  # shard height = one tile; N_EMBD // SH = 32-wide hidden slice per core
        wsh = N_EMBD // SH
        cg = ttnn.CoreGrid(x=8, y=SH // 8)  # 8 x 4 = 32 cores
        self._ln_cfg = ttnn.create_sharded_memory_config(
            shape=(SH, wsh), core_grid=cg, strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )
        self._ln_prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[cg.x, cg.y],
            subblock_w=wsh // 32, block_h=SH // 32, block_w=wsh // 32, inplace=False,
        )
        # The sharded kernel wants gamma/beta as [1, SH, dim]; re-expand the loaded [dim] vectors.
        _sh = lambda t: _to_dev(ttnn.to_torch(t).float().view(1, 1, N_EMBD).expand(1, SH, N_EMBD).contiguous(), device, ttnn.bfloat16)
        for w in self.layers:
            w["ln_1_w"], w["ln_1_b"] = _sh(w["ln_1_w"]), _sh(w["ln_1_b"])
            w["ln_2_w"], w["ln_2_b"] = _sh(w["ln_2_w"]), _sh(w["ln_2_b"])
        for kk in ("ln_f_w", "ln_f_b", "fn_w", "fn_b"):
            self.tail[kk] = _sh(self.tail[kk])

        self.trace_id = None
        self._out = None

    def _ln(self, x, g, b):  # width-sharded LayerNorm across 32 cores, then back to interleaved
        xs = ttnn.interleaved_to_sharded(x, self._ln_cfg)
        y = ttnn.layer_norm(
            xs, weight=g, bias=b, epsilon=LN_EPS, program_config=self._ln_prog,
            memory_config=self._ln_cfg, compute_kernel_config=COMPUTE_KERNEL_CONFIG,
        )
        return ttnn.sharded_to_interleaved(y)

    def _decode_step(self, x):  # x [1,1,1024] -> latent [1,1,1024]
        for li in range(N_LAYER):
            w = self.layers[li]
            qkv = ttnn.linear(self._ln(x, w["ln_1_w"], w["ln_1_b"]), w["attn_w"], bias=w["attn_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)
            q = ttnn.reshape(ttnn.slice(qkv, [0, 0, 0], [1, 1, N_EMBD]), [1, 1, N_HEAD, HEAD_DIM])
            k = ttnn.reshape(ttnn.slice(qkv, [0, 0, N_EMBD], [1, 1, 2 * N_EMBD]), [1, 1, N_HEAD, HEAD_DIM])
            v = ttnn.reshape(ttnn.slice(qkv, [0, 0, 2 * N_EMBD], [1, 1, 3 * N_EMBD]), [1, 1, N_HEAD, HEAD_DIM])
            ttnn.experimental.paged_update_cache(self.k_cache[li], ttnn.interleaved_to_sharded(k, self._shard), update_idxs_tensor=self._pos, page_table=None)
            ttnn.experimental.paged_update_cache(self.v_cache[li], ttnn.interleaved_to_sharded(v, self._shard), update_idxs_tensor=self._pos, page_table=None)
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q, self.k_cache[li], self.v_cache[li], cur_pos_tensor=self._pos, scale=SCALE, compute_kernel_config=COMPUTE_KERNEL_CONFIG
            )
            attn = ttnn.reshape(attn, [1, 1, N_EMBD])
            x = ttnn.add(x, ttnn.linear(attn, w["proj_w"], bias=w["proj_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG))
            x = ttnn.add(x, _mlp(self._ln(x, w["ln_2_w"], w["ln_2_b"]), w))
        x = self._ln(x, self.tail["ln_f_w"], self.tail["ln_f_b"])  # sharded ln_f ...
        return self._ln(x, self.tail["fn_w"], self.tail["fn_b"])  # ... + final_norm

    def reset(self):
        for c in self.k_cache + self.v_cache:
            ttnn.copy(self._zeros, c)

    def _set_pos(self, p):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([p], dtype=torch.int32)), self._pos)

    def _set_input(self, emb):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._in)

    def capture(self):
        """Warm up (compile kernels; trace can't compile) then capture the step into a trace."""
        self.reset()
        self._set_pos(0)
        self._decode_step(self._in)
        ttnn.synchronize_device(self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._out = self._decode_step(self._in)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self.reset()

    def step(self, emb, pos):  # emb torch [1,1,1024]; pos int -> latent torch [1,1,1024]
        self._set_input(emb)
        self._set_pos(pos)
        if self.trace_id is not None:
            ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
            out = self._out
        else:
            out = self._decode_step(self._in)  # eager path (for correctness validation before capture)
        return ttnn.to_torch(out).to(torch.float32)


def generate_traced(device, prefix_emb, heads, max_new=24, max_seq=128, use_trace=True):
    """Greedy generation with the traced decoder. Feeds the prompt token-by-token to fill the cache,
    then decodes; mel_head + argmax + embedding run on host. Returns dict(codes [T], latents [1,T,1024])."""
    dec = TracedGPTDecoder(device, max_seq=max_seq)
    if use_trace:
        dec.capture()
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]
    head = lambda latent: latent @ mh_w.t() + mh_b

    dec.reset()
    pos = 0
    for t in range(prefix_emb.shape[1]):  # prompt prefill (fill cache, discard latents)
        dec.step(prefix_emb[:, t : t + 1, :].contiguous(), pos)
        pos += 1
    lat = dec.step((mel_emb[START_AUDIO_TOKEN] + mel_pos[0]).view(1, 1, -1), pos)  # start token
    pos += 1
    code = int(head(lat).argmax(-1))
    codes, lats = [code], [lat]
    for m in range(1, max_new):
        lat = dec.step((mel_emb[code] + mel_pos[m]).view(1, 1, -1), pos)
        pos += 1
        code = int(head(lat).argmax(-1))
        codes.append(code)
        lats.append(lat)
    return {"codes": torch.tensor(codes), "latents": torch.cat(lats, dim=1)}


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
