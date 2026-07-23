# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of the XTTS-v2 GPT transformer core (Block 3) — greedy autoregressive decode.

The block is `TracedGPTDecoder`: one 30-layer GPT-2 decode step (bf16 flash-decode + paged KV-cache),
captured into a device trace and replayed per token. The position lives in a device tensor threaded
through paged_fused_update_cache and scaled_dot_product_attention_decode, so a single captured graph
serves every position. Each step embeds the previous code, updates the cache, and attends over it.
The per-token latents (final_norm output) are what feed the vocoder; token embed / mel_head / argmax
run on host (see `generate_traced`).

HF Conv1D weights are [in,out], so they feed ttnn.linear directly (NO transpose). Width-sharded
LayerNorm + fused QKV-heads + fused KV-cache keep the step lean (~9.4 ms/token). bf16 throughout,
with fp32 accumulation (HiFi3 + fp32_dest_acc) to stay tight vs the fp32 reference.

Validate against the CPU reference (reference/xtts_gpt_ref.py: reference_generate):
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
)

HEAD_DIM = N_EMBD // N_HEAD  # 64
SCALE = 1.0 / (HEAD_DIM**0.5)
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt")
GEN_DIR = os.path.join(GOLDEN_DIR, "generate")

# bf16 tensors with fp32 accumulation (HiFi3 + fp32_dest_acc) to stay tight vs the fp32 reference;
# on Wormhole HiFi4 + fp32-acc is worse than HiFi3 due to a documented HW bug. (flash-decode +
# paged cache are bf16-only anyway.) DTYPE is the _to_dev default; the decoder also passes it explicitly.
DTYPE = ttnn.bfloat16
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def _to_dev(t, device, dtype=DTYPE):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def load_ttnn_weights(device, ckpt_path=DEFAULT_CKPT, dtype=DTYPE):
    """Convert the GPT-core checkpoint tensors to on-device ttnn tensors (HF Conv1D [in,out] as-is).
    dtype=float32 (default) for the accuracy-first prefill path; bfloat16 for the fast decode path."""
    core = load_gpt_core_state(ckpt_path)  # keys: h.{i}.*, ln_f.*, final_norm.*
    to = lambda k: _to_dev(core[k].float(), device, dtype)
    layers = []
    for i in range(N_LAYER):
        p = f"h.{i}."
        layers.append(
            {
                "ln_1_w": to(p + "ln_1.weight"),
                "ln_1_b": to(p + "ln_1.bias"),
                "attn_w": to(p + "attn.c_attn.weight"),  # [1024, 3072] (in,out)
                "attn_b": to(p + "attn.c_attn.bias"),
                "proj_w": to(p + "attn.c_proj.weight"),  # [1024, 1024]
                "proj_b": to(p + "attn.c_proj.bias"),
                "ln_2_w": to(p + "ln_2.weight"),
                "ln_2_b": to(p + "ln_2.bias"),
                "fc_w": to(p + "mlp.c_fc.weight"),  # [1024, 4096]
                "fc_b": to(p + "mlp.c_fc.bias"),
                "mproj_w": to(p + "mlp.c_proj.weight"),  # [4096, 1024]
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


def _mlp(x, w):
    x = ttnn.linear(x, w["fc_w"], bias=w["fc_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)  # [1, S, 4096]
    x = ttnn.gelu(x)  # HF gelu_new (tanh approx)
    return ttnn.linear(x, w["mproj_w"], bias=w["mproj_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)  # [1,S,1024]


# --------------------------------------------------------------------------------------
# The block: bf16 flash-decode + paged KV-cache + a position "mailbox" tensor, captured
# into a device trace and replayed per token (one host dispatch/token instead of ~600).
# --------------------------------------------------------------------------------------
class TracedGPTDecoder:
    """One 30-layer decode step, trace-captured and replayed per token. The position lives in a
    device tensor (`_pos`) threaded into paged_fused_update_cache (which cache line to write) and
    scaled_dot_product_attention_decode (how much of the cache to read), so a single captured
    graph serves every position. bf16 (flash-decode + paged cache are bf16-only). Needs the device
    opened with a trace_region_size. mel_head / argmax / embedding stay on host (see generate_traced)."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, max_seq=128):
        self.device = device
        # bf16 everywhere (flash-decode + paged cache are bf16-only anyway).
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
        # paged_fused_update_cache does K+V in one kernel but needs them on non-overlapping cores;
        # nlp_create_qkv_heads_decode puts both K and V on core (0,0), so move V to core (1,0) first.
        self._v_cfg1 = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                (32, HEAD_DIM),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
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
        # Keep the plain [dim] weights too (ln_*_wp/bp): the batched prefill pass runs over [1, P, 1024],
        # for which the single-token width-sharded config doesn't apply — it uses plain interleaved LN.
        _sh = lambda t: _to_dev(ttnn.to_torch(t).float().view(1, 1, N_EMBD).expand(1, SH, N_EMBD).contiguous(), device, ttnn.bfloat16)
        for w in self.layers:
            w["ln_1_wp"], w["ln_1_bp"] = w["ln_1_w"], w["ln_1_b"]
            w["ln_2_wp"], w["ln_2_bp"] = w["ln_2_w"], w["ln_2_b"]
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

    def prefill(self, prefix_emb):
        """Fill the KV-cache for prompt positions 0..P-1 in ONE batched pass over the P prompt tokens,
        instead of P single-token decode steps. Each layer's K/V weights are read once (not P times),
        so this is weight-bandwidth-amortized. Latents are discarded — only the caches seed decode.
        Eager (not traced); run after reset() and before the first decode step. prefix_emb: torch [1,P,1024]."""
        P = prefix_emb.shape[1]
        x = _to_dev(prefix_emb, self.device, ttnn.bfloat16)  # [1, P, 1024]
        for li in range(N_LAYER):
            w = self.layers[li]
            h = ttnn.layer_norm(x, weight=w["ln_1_wp"], bias=w["ln_1_bp"], epsilon=LN_EPS, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
            qkv = ttnn.linear(h, w["attn_w"], bias=w["attn_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)  # [1,P,3072]
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.reshape(qkv, [1, 1, P, 3 * N_EMBD]), num_heads=N_HEAD, num_kv_heads=N_HEAD,
                transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # each [1, N_HEAD, P, HEAD_DIM]
            ttnn.fill_cache(self.k_cache[li], k, 0)  # write positions 0..P-1 in one shot
            ttnn.fill_cache(self.v_cache[li], v, 0)
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=SCALE, compute_kernel_config=COMPUTE_KERNEL_CONFIG
            )  # [1, N_HEAD, P, HEAD_DIM]
            attn = ttnn.reshape(ttnn.experimental.nlp_concat_heads(attn), [1, P, N_EMBD])  # concat heads (fused; inverse of nlp_create_qkv_heads)
            x = ttnn.add(x, ttnn.linear(attn, w["proj_w"], bias=w["proj_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG))
            x = ttnn.add(x, _mlp(ttnn.layer_norm(x, weight=w["ln_2_wp"], bias=w["ln_2_bp"], epsilon=LN_EPS, compute_kernel_config=COMPUTE_KERNEL_CONFIG), w))

    def _decode_step(self, x):  # x [1,1,1024] -> latent [1,1,1024]
        for li in range(N_LAYER):
            w = self.layers[li]
            qkv = ttnn.linear(self._ln(x, w["ln_1_w"], w["ln_1_b"]), w["attn_w"], bias=w["attn_b"], compute_kernel_config=COMPUTE_KERNEL_CONFIG)
            # Fused per-head Q/K/V split (replaces 3 slice + 3 reshape + 2 interleaved_to_sharded).
            # Outputs are height-sharded in L1: K/V feed paged_update_cache directly, Q feeds sdpa_decode.
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(qkv, [1, 1, 1, 3 * N_EMBD]), num_heads=N_HEAD, num_kv_heads=N_HEAD
            )
            # Fused K+V cache update in one kernel (V moved to a separate core so the two don't overlap).
            ttnn.experimental.paged_fused_update_cache(
                self.k_cache[li], k, self.v_cache[li], ttnn.to_memory_config(v, self._v_cfg1),
                update_idxs_tensor=self._pos, page_table=None,
            )
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
        """Warm up (compile kernels; trace can't compile) then capture the step into a trace.
        Warms up at a scratch cache slot (max_seq-1, never a real decode position) and does NOT
        zero the cache, so it can run AFTER prefill without clobbering the prompt's K/V. Allocating
        buffers is only safe while no trace exists, so prefill must run before this."""
        self._set_pos(self.max_seq - 1)
        self._decode_step(self._in)
        ttnn.synchronize_device(self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._out = self._decode_step(self._in)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def step(self, emb, pos):  # emb torch [1,1,1024]; pos int -> latent torch [1,1,1024]
        self._set_input(emb)
        self._set_pos(pos)
        if self.trace_id is not None:
            ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
            out = self._out
        else:
            out = self._decode_step(self._in)  # eager path (for correctness validation before capture)
        return ttnn.to_torch(out).to(torch.float32)


def _select_token(logits, prev, temperature, top_k, top_p, repetition_penalty):
    """HF-style token sampling on the mel logits (host side): repetition penalty -> temperature ->
    top-k -> top-p (nucleus) -> multinomial. XTTS needs sampling; pure argmax collapses into a
    repeated-code loop. `prev` = audio codes generated so far (for the repetition penalty)."""
    logits = logits.clone().float().view(-1)
    if repetition_penalty and repetition_penalty != 1.0 and prev:
        idx = torch.tensor(sorted(set(prev)))
        s = logits[idx]
        logits[idx] = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)
    if temperature and temperature != 1.0:
        logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        remove = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sl[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, si, sl)
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))


def generate_traced(device, prefix_emb, heads, max_new=24, max_seq=128, use_trace=True, stop_token=None,
                    do_sample=False, temperature=0.75, top_k=50, top_p=0.85, repetition_penalty=10.0, seed=None):
    """Autoregressive generation with the traced decoder. Fills the cache with a single batched prefill
    pass over the prompt, then decodes token by token; mel_head + token selection + embedding run on host.
    do_sample=False (default) is greedy argmax — deterministic, for the PCC tests. do_sample=True uses
    HF-style sampling with the XTTS defaults (temp 0.75 / top_k 50 / top_p 0.85 / repetition_penalty 10),
    which the model REQUIRES for real speech (greedy collapses into a repeated code).
    stop_token (e.g. STOP_AUDIO_TOKEN) ends generation once emitted. Returns dict(codes [T], latents [1,T,1024])."""
    if seed is not None:
        torch.manual_seed(seed)
    dec = TracedGPTDecoder(device, max_seq=max_seq)
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]
    head = lambda latent: latent @ mh_w.t() + mh_b

    def pick(lat, prev):
        logits = head(lat)
        if do_sample:
            return _select_token(logits, prev, temperature, top_k, top_p, repetition_penalty)
        return int(logits.argmax(-1))

    dec.reset()
    dec.prefill(prefix_emb)  # fill cache positions 0..P-1 in one batched pass (before any trace exists)
    if use_trace:
        dec.capture()  # capture the single-token decode step; leaves the prefilled cache intact
    pos = prefix_emb.shape[1]
    lat = dec.step((mel_emb[START_AUDIO_TOKEN] + mel_pos[0]).view(1, 1, -1), pos)  # start token at pos P
    pos += 1
    code = pick(lat, [])
    codes, lats = [code], [lat]
    for m in range(1, max_new):
        lat = dec.step((mel_emb[code] + mel_pos[m]).view(1, 1, -1), pos)
        pos += 1
        code = pick(lat, codes)
        if stop_token is not None and code == stop_token:
            break
        codes.append(code)
        lats.append(lat)
    return {"codes": torch.tensor(codes), "latents": torch.cat(lats, dim=1)}


def main():
    device = ttnn.open_device(device_id=0, trace_region_size=50_000_000)
    try:
        prefix = torch.load(os.path.join(GEN_DIR, "prefix_emb.pt"))
        ref_codes = torch.load(os.path.join(GEN_DIR, "codes.pt"))
        heads = load_gen_head(DEFAULT_CKPT)
        k = ref_codes.numel()
        gen = generate_traced(device, prefix, heads, max_new=k, use_trace=True)
        print(f"[ttnn] traced decode codes match: {bool(torch.equal(gen['codes'][:k], ref_codes[:k]))}")
        print(f"[ttnn] traced decode latents {tuple(gen['latents'].shape)}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
