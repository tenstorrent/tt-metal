# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini attention core on sdpa_decode, both layer types. Drop-in for `attention.py`:

    from models.demos.inkling2_mini.tt.attention_fast import DecodePositionsFast as DecodePositions
    from models.demos.inkling2_mini.tt.attention_fast import Inkling2AttentionFast as Inkling2Attention

Same ctor `(device, weights, is_global)`, `new_state(max_ctx, cache_dtype)`, frozen
`decode(xq, state, positions, xk=None, xv=None)` and output as `Inkling2Attention`. The
projections / k-v conv / qgrs slicing are `Inkling2Attention`'s (E3), collected in `_project`.

Local layers: `scaled_dot_product_attention_decode` on the 1024-slot ring, non-causal. The
per-head `attn_mask [1,1,24,1024]` carries rel bias + ring validity, `attention_sink` the sinks.
The kernel adds both to raw q.k before `scale`, so q is pre-scaled and `scale=1`: the bf16 bias
and sink then enter without another rounding.

Global layers: logits `near ++ far ++ sink`, one softmax (HF), as two exact partials:
- near + sink: the ring with rel bias by matmul + softmax (fixed cost); output O_n, log-sum-exp L_n.
- far (pos - key >= 1024): `flash_multi_latent_attention_decode` with `cur_pos = pos - 1024`,
  so only the valid chunks are read, and `attention_sink = L_n` (the near mass joins its
  denominator). The output is the far share A of the full output; a constant-1 value column
  gives the far mass a. `O = (A r + O_n (1 - a)) / (a r + 1 - a)`, `r = exp(s_bf16 - L_n)`
  undoes the bf16 rounding of the sink. Far is zeroed for pos < 1024.
- far cache: one `kv_full [1, 8, max_ctx, 288] = [v(128) | 1, 0 x 31 | k(128)]` per (layer, pass).
  MLA mode uses the first 160 columns as V and reuses K from L1: 288 columns per key instead of
  256 (K + V), +12.5% memory, no separate V read.

`DecodePositionsFast` = `DecodePositions` without the O(max_ctx) far mask, plus `far`.
"""

import os

import torch

import ttnn
from models.demos.inkling2_mini.tt.attention import (
    D_REL,
    EPS,
    GROUPS,
    HEAD_DIM,
    KV0,
    N_HEADS,
    N_KV,
    NEG,
    QK_CLIP,
    REL_CLIP,
    ROWS,
    TAPS,
    WINDOW,
    AttentionState,
    DecodePositions,
    Inkling2Attention,
    _DramShardedLinear,
)
from models.demos.inkling2_mini.tt.fused.attn import AttnNear, AttnOut, AttnPrep, AttnWindow

V_WIDTH = 160  # v(128) + mass column block
KV_WIDTH = V_WIDTH + HEAD_DIM  # 288
SCALE = HEAD_DIM**-0.5
FAR_CHUNK = 128  # sdpa K chunk of the far cache (P1 sweep: 128 < 256 < 512 ms at 120K)
# local ring (1024 slots): 4 cores per kv head x 256-key chunks, 37.4 -> 31.7 us per call vs the default 13 cores x 128
# (each of 13 cores got < 1 chunk; the tree reduction dominated)
LOCAL_CHUNK = 256
LOCAL_CORES_PER_HEAD = 4
# k and v ring writes as one paged_fused_update_cache program; INKLING_KV_FUSED=0: two paged_update_cache calls
FUSED_KV_WRITE = os.environ.get("INKLING_KV_FUSED", "1") != "0"


class DecodePositionsFast(DecodePositions):
    """`DecodePositions` minus `kpos` / `far_mask`; adds `far = (far_pos, far_valid)`:
    far_pos = max(pos - 1024, 0) (int32 [1]), far_valid = (pos >= 1024) (fp32 [1,1,1,1])."""

    def __init__(self, device, max_ctx):
        self.device = device
        self.max_ctx = max_ctx
        self.inputs = {k: ttnn.to_device(v, device) for k, v in self._host(0).items()}

    def _host(self, pos):
        host = super()._host(pos)
        host["far_pos"] = ttnn.from_torch(torch.tensor([max(pos - WINDOW, 0)], dtype=torch.int32), dtype=ttnn.int32)
        valid = torch.tensor(float(pos >= WINDOW)).view(1, 1, 1, 1)
        host["far_valid"] = ttnn.from_torch(valid, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        return host

    def prepare(self):
        inp = self.inputs
        self.near_mask = ttnn.multiply(ttnn.gt(inp["dist"], inp["pos_f"]), NEG)
        self.tau = inp["tau"]
        self.dist_idx = inp["dist_idx"]
        self.pos = inp["pos"]
        self.ring = inp["ring"]
        self.far = (inp["far_pos"], inp["far_valid"])


class FastAttentionState(AttentionState):
    """Ring + conv histories as `AttentionState`; global far cache is one `kv_full` (module doc)."""

    def __init__(self, device, is_global, max_ctx, cache_dtype):
        super().__init__(device, False, max_ctx, cache_dtype)  # ring + conv histories
        self.kv_full = None
        if is_global:
            self.kv_full = ttnn.zeros(
                [1, N_KV, max_ctx, KV_WIDTH], dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )

    def tensors(self):
        out = super().tensors()
        return out if self.kv_full is None else out + [self.kv_full]

    def fill(self, keys, values):
        """Load post-norm keys / values [P, 8, 128] of positions 0..P-1 (tests). Conv histories untouched."""
        p = keys.shape[0]
        lo = max(0, p - WINDOW)
        slots = torch.arange(lo, p) % WINDOW
        dst = []
        for cache, src in ((self.k_ring, keys), (self.v_ring, values)):
            ring = torch.zeros(N_KV, WINDOW, HEAD_DIM)
            ring[:, slots] = src[lo:].transpose(0, 1).float()
            dst.append((cache, ring))
        if self.kv_full is not None:
            full = torch.zeros(N_KV, self.kv_full.shape[-2], KV_WIDTH)
            full[:, :p, :HEAD_DIM] = values.transpose(0, 1).float()
            full[:, :p, HEAD_DIM] = 1.0
            full[:, :p, V_WIDTH:] = keys.transpose(0, 1).float()
            dst.append((self.kv_full, full))
        for tensor, value in dst:
            host = ttnn.from_torch(value.reshape(list(tensor.shape)), dtype=tensor.dtype, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, tensor)


class Inkling2AttentionFast(Inkling2Attention):
    """`Inkling2Attention` with the scores / softmax on sdpa_decode (module doc).

    FUSED: the glue around the projections and the attention core runs as multi-core programs (tt/fused/attn.py:
    conv window, q/k/v prep with the cache writes, output gate with the global merge) instead of ~50 ttnn ops;
    False keeps the op-by-op path (reference / A-B timing)."""

    FUSED = True

    def __init__(self, device, weights, is_global, prefetcher=None):
        super().__init__(device, weights, is_global, prefetcher)
        dram = ttnn.DRAM_MEMORY_CONFIG
        self._init_fused(weights)

        def put(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

        grid = device.compute_with_storage_grid_size()

        def sdpa_cfg(chunk, cores_per_head=16):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=0,
                k_chunk_size=chunk,
                exp_approx_mode=False,
                max_cores_per_head_batch=cores_per_head,
            )

        self.sink_cols = put(torch.ones(1, 1, N_HEADS, 32), ttnn.float32)
        self.local_cfg = sdpa_cfg(LOCAL_CHUNK, LOCAL_CORES_PER_HEAD)
        # k and v ring writes in one paged_fused_update_cache program: k's row on core (0, 0), v's on (0, 1)
        self.kv_shard_v = ttnn.create_sharded_memory_config(
            shape=(32, HEAD_DIM),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if not is_global:
            # local core runs at scale=1: fold SCALE into q's pre-scale and clip threshold (t -> t * SCALE),
            # so q leaves _project already scaled with one bf16 rounding
            self.q_scale = ttnn.multiply(self.q_scale, SCALE)
            self.q_red = put(torch.full((1, 1, HEAD_DIM, 1), QK_CLIP**-2 / SCALE**2))
            return
        mass = torch.zeros(1, 1, N_KV, V_WIDTH - HEAD_DIM)
        mass[..., 0] = 1.0
        self.mass_col = put(mass)
        self.q_pad = put(torch.zeros(1, 1, N_HEADS, V_WIDTH))
        self.far_cfg = sdpa_cfg(FAR_CHUNK)
        self.kv_full_shard = ttnn.create_sharded_memory_config(
            shape=(32, KV_WIDTH),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _init_fused(self, weights):
        def torch_of(name):
            return ttnn.to_torch(weights[name]).float()

        width = N_KV * HEAD_DIM
        taps = {}
        for kv in ("k", "v"):
            w = torch_of(f"{kv}_conv").view(TAPS, width).clone()
            w[0] += 1.0
            taps[kv] = w * self.conv_scale
        rel_proj = torch_of("rel_proj").view(D_REL, WINDOW)
        # q leaves the fused prep pre-scaled by SCALE (both layer types), clip threshold scaled to match
        self.prep = AttnPrep(
            self.device,
            torch_of("q_clip_scale").view(N_HEADS, HEAD_DIM) * SCALE,
            torch_of("rel_clip_scale").view(N_HEADS, D_REL),
            rel_proj,
            taps,
            q_inv=QK_CLIP**-2 / SCALE**2,
            k_inv=QK_CLIP**-2,
            rel_inv=REL_CLIP**-2,
            eps=EPS,
            kv0=KV0,
            dyn=self.is_global,
            conv_scale=self.conv_scale,
        )
        self.out = AttnOut(self.device, self.w_o.act_mem)
        self.near = AttnNear(self.device) if self.is_global else None
        # the prefetcher's projection takes an interleaved input: keep the op-by-op window there
        self.window = (
            AttnWindow(self.device, self.w_all.act_mem) if isinstance(self.w_all, _DramShardedLinear) else None
        )

    def new_state(self, max_ctx, cache_dtype=ttnn.bfloat16):
        if self.is_global and max_ctx % FAR_CHUNK:
            raise ValueError(f"max_ctx {max_ctx} not a multiple of {FAR_CHUNK}")
        return FastAttentionState(self.device, self.is_global, max_ctx, cache_dtype)

    # ---------------------------------------------------------------- pieces
    def _rel_bias(self, rel, positions):
        """rel_states [1,1,24,16] @ proj[:, dist(slot)] -> [1,8,3,1024] bf16 (as attention.py)."""
        prof = ttnn.embedding(positions.dist_idx, self.rel_table, layout=ttnn.TILE_LAYOUT)  # [1, 1024, 16]
        prof = ttnn.reshape(prof, [1, 1, WINDOW, D_REL])
        bias = ttnn.matmul(rel, prof, transpose_b=True, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)
        return ttnn.reshape(bias, [1, N_KV, GROUPS, WINDOW])

    def _local(self, q, bias, sink, state, positions):
        # q pre-scaled and scale=1: mask and sink are then plain logits (bias, sink are bf16 already)
        mask = ttnn.add(bias, positions.near_mask, dtype=ttnn.bfloat16)
        sink_t = ttnn.multiply(ttnn.reshape(sink, [1, 1, N_HEADS, 1]), self.sink_cols, dtype=ttnn.bfloat16)
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            state.k_ring,
            state.v_ring,
            is_causal=False,
            attn_mask=ttnn.reshape(mask, [1, 1, N_HEADS, WINDOW]),
            attention_sink=ttnn.reshape(sink_t, [N_HEADS, 32]),
            scale=1.0,
            program_config=self.local_cfg,
            compute_kernel_config=self.hifi4,
        )  # [1,1,24,128] bf16

    def _global(self, q, bias, sink_blk, state, positions):
        """Op-by-op global core: near + sink partial, far partial, exact merge (module doc). sink_blk [1,8,3,32]
        fp32: col 0 = sink logit, others masked. The fused path runs the same math in AttnNear / AttnOut."""
        # near + sink partial: normalized output and log-sum-exp
        qg = ttnn.reshape(q, [1, N_KV, GROUPS, HEAD_DIM])
        near = ttnn.matmul(qg, state.k_ring, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
        near = ttnn.add(ttnn.add(ttnn.multiply(near, SCALE), bias), positions.near_mask)
        logits = ttnn.concat([near, sink_blk], dim=-1)
        # softmax and log-sum-exp from one exp pass
        m = ttnn.max(logits, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(logits, m))
        z = ttnn.sum(e, dim=-1, keepdim=True)  # [1,8,3,1]; a matmul with a ones column runs on one core (~140 us)
        e_near = ttnn.slice(e, [0, 0, 0, 0], [1, N_KV, GROUPS, WINDOW])
        p_near = ttnn.multiply(e_near, ttnn.reciprocal(z), dtype=ttnn.bfloat16)
        o_near = ttnn.matmul(p_near, state.v_ring, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
        lse = ttnn.add(m, ttnn.log(z))

        # far partial with the near mass as its sink: sdpa `attention_sink` [24, 32] bf16 in raw q.k units
        far_pos, valid = positions.far
        raw = ttnn.reshape(ttnn.multiply(lse, 1.0 / SCALE), [1, 1, N_HEADS, 1])
        sink_t = ttnn.multiply(raw, self.sink_cols, dtype=ttnn.bfloat16)
        s_hat = ttnn.typecast(ttnn.slice(sink_t, [0, 0, 0, 0], [1, 1, N_HEADS, 1]), ttnn.float32)
        s_hat = ttnn.reshape(ttnn.multiply(s_hat, SCALE), [1, N_KV, GROUPS, 1])
        r = ttnn.exp(ttnn.subtract(s_hat, lse))  # undoes the bf16 rounding of the sink
        out = ttnn.transformer.flash_multi_latent_attention_decode(
            ttnn.concat([self.q_pad, q], dim=-1),  # [1,1,24,288], zeros meet the v / mass columns
            state.kv_full,
            head_dim_v=V_WIDTH,
            is_causal=True,
            cur_pos_tensor=far_pos,
            attention_sink=ttnn.reshape(sink_t, [N_HEADS, 32]),
            scale=SCALE,
            program_config=self.far_cfg,
            compute_kernel_config=self.hifi4,
        )
        out = ttnn.multiply(ttnn.reshape(ttnn.typecast(out, ttnn.float32), [1, N_KV, GROUPS, V_WIDTH]), valid)
        A = ttnn.slice(out, [0, 0, 0, 0], [1, N_KV, GROUPS, HEAD_DIM])
        a = ttnn.slice(out, [0, 0, 0, HEAD_DIM], [1, N_KV, GROUPS, HEAD_DIM + 1])

        # exact merge
        near_w = ttnn.rsub(a, 1.0)
        num = ttnn.add(ttnn.multiply(A, r), ttnn.multiply(o_near, near_w))
        den = ttnn.add(ttnn.multiply(a, r), near_w)
        return ttnn.typecast(ttnn.multiply(num, ttnn.reciprocal(den)), ttnn.bfloat16)

    # ---------------------------------------------------------------- decode
    def decode(self, xq, state, positions, xk=None, xv=None):
        """Same contract as `Inkling2Attention.decode`; `positions` is a `DecodePositionsFast`."""
        xk = xq if xk is None else xk
        xv = xq if xv is None else xv
        if not self.FUSED:
            return self._decode_ops(xq, state, positions, xk, xv)
        if self.window is None:
            proj, mode = self._window(state, xq, xk, xv)
        else:
            mode = "single" if (xk is xq and xv is xq) else "triple"
            act = self.window(state.x_hist, [xq] if mode == "single" else [xq, xk, xv])
            proj = self.w_all(act, ttnn.bfloat16, self.mm_cfg)
        rows = ROWS[mode]
        q, mask, sink_t = self.prep(proj, positions, {"q": 0, "k": rows["k"], "v": rows["v"]}, state)
        if self.is_global:  # q pre-scaled; prep's mask (bias + near mask) and q_grp are in kv-group layout
            o_near, r, sink_far = self.near(self.prep.q_grp, mask, sink_t, state.k_ring, state.v_ring)
            far_pos, valid = positions.far
            far = ttnn.transformer.flash_multi_latent_attention_decode(
                ttnn.concat([self.q_pad, q], dim=-1),  # [1,1,24,288], zeros meet the v / mass columns
                state.kv_full,
                head_dim_v=V_WIDTH,
                is_causal=True,
                cur_pos_tensor=far_pos,
                attention_sink=sink_far,
                scale=1.0,
                program_config=self.far_cfg,
                compute_kernel_config=self.hifi4,
            )
            return self.w_o(self.out(far, proj, (o_near, r, valid)), ttnn.bfloat16, self.mm_cfg)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            state.k_ring,
            state.v_ring,
            is_causal=False,
            attn_mask=mask,
            attention_sink=sink_t,
            scale=1.0,
            program_config=self.local_cfg,
            compute_kernel_config=self.hifi4,
        )
        return self.w_o(self.out(attn, proj), ttnn.bfloat16, self.mm_cfg)

    def _decode_ops(self, xq, state, positions, xk, xv):
        q, rel, gate, sink, k, v = self._project(state, positions, xq, xk, xv)  # E3 seam
        if FUSED_KV_WRITE:
            ttnn.experimental.paged_fused_update_cache(
                state.k_ring,
                ttnn.to_memory_config(k, self.kv_shard),
                state.v_ring,
                ttnn.to_memory_config(v, self.kv_shard_v),
                update_idxs_tensor=positions.ring,
            )
        else:
            self._write(state.k_ring, k, positions.ring)
            self._write(state.v_ring, v, positions.ring)
        if self.is_global:
            row = ttnn.to_memory_config(ttnn.concat([v, self.mass_col, k], dim=-1), self.kv_full_shard)
            ttnn.experimental.paged_update_cache(state.kv_full, row, update_idxs_tensor=positions.pos)

        bias = self._rel_bias(rel, positions)
        if self.is_global:
            attn = self._global(q, bias, ttnn.add(self.sink_pad, ttnn.typecast(sink, ttnn.float32)), state, positions)
        else:
            attn = self._local(q, bias, sink, state, positions)
        attn = ttnn.reshape(attn, [1, 1, 1, N_HEADS * HEAD_DIM])
        gated = ttnn.multiply(ttnn.typecast(attn, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
        return self.w_o(ttnn.typecast(gated, ttnn.bfloat16), ttnn.bfloat16, self.mm_cfg)
