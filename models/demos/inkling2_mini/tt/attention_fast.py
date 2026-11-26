# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini attention core on sdpa_decode, both layer types. Drop-in for `attention.py`:

    from models.demos.inkling2_mini.tt.attention_fast import DecodePositionsFast as DecodePositions
    from models.demos.inkling2_mini.tt.attention_fast import Inkling2AttentionFast as Inkling2Attention

Same ctor `(device, weights, is_global)`, `new_state(max_ctx, cache_dtype)`, frozen
`decode(xq, state, positions, xk=None, xv=None)` and output as `Inkling2Attention`. The
projections / k-v conv / qgrs slicing are `Inkling2Attention`'s, collected in `_project`.

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
- far cache: one `kv_full [1, 8, max_ctx, 256] = [v(128) | k(128)]` per (layer, pass). The kernel sees
  rows `[v(128) | 1, 0 x 31 | k(128)]` (288): MLA mode uses the first 160 columns as V and reuses K
  from L1, no separate V read. The mass block `1, 0 x 31` is the same for every key, so the reader
  writes it itself (sdpa `unit_k_tile_col`); a build without that option stores it (288 columns).

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
    HIDDEN,
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
from models.demos.inkling2_mini.tt.fused import attn as FA
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attn import AttnNear, AttnOut, AttnPrep, AttnWindow, KvCommit, RowsOut
from models.demos.inkling2_mini.tt.fused.attn_wo import AttnWo

V_WIDTH = 160  # v(128) + mass column block
KV_WIDTH = V_WIDTH + HEAD_DIM  # 288: the far call's q and the cache rows as the kernel sees them
SCALE = HEAD_DIM**-0.5
# verify (decode_rows): the output gate / merge of all R tokens in one program (fused/attn.py AttnOut.rows) instead of
# one per token; INKLING_OUT_ROWS=0: one per token
OUT_ROWS = os.environ.get("INKLING_OUT_ROWS", "1") == "1"
FAR_CHUNK = 128  # sdpa K chunk of the far cache (measured at 120K: 128 beats 256 and 512)
# needs the sdpa_decode options of the custom C++ build; a stock build has neither field
FAR_OPTS = (
    dict(qk_skip_head_dim=V_WIDTH, const_k_tile_col=HEAD_DIM // 32)
    if hasattr(ttnn.SDPAProgramConfig, "qk_skip_head_dim")
    else {}
)
# The far cache does not store the mass block: 256 columns per key instead of 288, same output (the reader writes
# the block). Needs sdpa_decode `unit_k_tile_col`; INKLING_FAR_MASS=1 keeps the stored block (A/B on one build).
FAR_UNIT = hasattr(ttnn.SDPAProgramConfig, "unit_k_tile_col") and os.environ.get("INKLING_FAR_MASS") != "1"
if FAR_UNIT:
    FAR_OPTS["unit_k_tile_col"] = True
V_MEM = HEAD_DIM if FAR_UNIT else V_WIDTH  # columns of the cache before k
KV_MEM = V_MEM + HEAD_DIM  # columns of the cache: 256, or 288 with the stored mass block
# far call with the kv groups as the batch (q [1, 8, 3, 288], cache viewed as [8, 1, S, 288]): 16-row Q tiles, so
# the softmax runs on half the rows; exact exp keeps it bit-equal to the 24-head call (custom C++ build only)
FAR_GRP = hasattr(ttnn.SDPAProgramConfig, "exact_exp_half_tile")
# opt-in, changes the output: far cache dtype (None: the cache dtype; ttnn.bfloat4_b with bfp8 rings halves the far
# reads at long context). The decode writer converts the bfp8 row (round to nearest). Decode only: prefill
# and multi-token verify refuse it.
# INKLING_FAR_KV=bfp4 turns it on (fits 512K on one p100a: 10,368 B / token instead of 19,584).
FAR_DTYPE = ttnn.bfloat4_b if os.environ.get("INKLING_FAR_KV") == "bfp4" else None
FAR_CHUNK_BFP4 = 256  # bfp4 far cache: 256 beats 128 at 120K (615 vs 720 us per global use)
# opt-in, changes the output: INKLING_FAR_KV=k8v4 keeps the far keys and values in two caches, k in the cache dtype
# (bfp8) and v in bfp4: 13,312 B / token instead of 17,408. `kv_full` [1, 8, max_ctx, 128] then holds k only and
# `v_full` the values; the far call reads both (sdpa `unit_v_tile_col` writes the mass block), q is 128 wide.
# Decode only, one token per step, bfp8 rings, fused path (as the bfp4 cache above).
FAR_SPLIT = os.environ.get("INKLING_FAR_KV") == "k8v4"
assert not FAR_SPLIT or hasattr(
    ttnn.SDPAProgramConfig, "unit_v_tile_col"
), "INKLING_FAR_KV=k8v4: build lacks the option"
FAR_SPLIT_CHUNK = int(os.environ.get("INKLING_FAR_SPLIT_CHUNK", 128))
if FAR_SPLIT:
    FA.FAR_Q = (HEAD_DIM // 32, 0)  # the prep writes the far q as [1, 8, 3, 128]
# local ring (1024 slots): 4 cores per kv head x 256-key chunks, 37.4 -> 31.7 us per call vs the default 13 cores x 128
# (each of 13 cores got < 1 chunk; the tree reduction dominated)
LOCAL_CHUNK = 256
LOCAL_CORES_PER_HEAD = 4
# Local sdpa on an 8 x 4 grid (its 32 work cores, rows 0..3): bit-equal to the full grid (same cores per head and
# chunking) and leaves the other cores free for fill kernels riding inside the call (SDPA_HOST). None = full grid.
LOCAL_GRID = (8, 4)
# SDPA_HOST(cores) -> ttnn.ProgramDescriptor | None: extra kernels for the local sdpa program (tt/mk/overlap.py sets
# it while staging is installed); cores = the (x, y) set the sdpa call uses.
SDPA_HOST = None
W_ALL_OUT_MEM = ttnn.L1_MEMORY_CONFIG
_OUT = {}


def _shared_out(device, cols, dtype, mem):
    """A persistent [1, 1, 32, cols] output tensor shared by every layer with that width."""
    key = (id(device), cols, dtype, str(mem))  # the entry keeps `device` alive, so its id is not reused
    if key not in _OUT or not _OUT[key][1].is_allocated():
        shape = ttnn.Shape([1, 1, 32, cols])
        _OUT[key] = (device, ttnn.allocate_tensor_on_device(shape, dtype, ttnn.TILE_LAYOUT, device, mem))
    return _OUT[key][1]


class DecodePositionsFast(DecodePositions):
    """`DecodePositions` minus `kpos` / `far_mask`; adds `far = (far_pos, far_valid)` and `far_grp = (far_pos8,
    far_valid)`: far_pos = max(pos - 1024, 0) (int32 [1]; far_pos8 the same x 8, one per kv group), far_valid =
    (pos >= 1024) (fp32 [1,1,1,1])."""

    def __init__(self, device, max_ctx):
        self.device = device
        self.max_ctx = max_ctx
        self.inputs = {k: ttnn.to_device(v, device) for k, v in self._host(0).items()}

    def _host(self, pos):
        host = super()._host(pos)
        host["far_pos"] = ttnn.from_torch(torch.tensor([max(pos - WINDOW, 0)], dtype=torch.int32), dtype=ttnn.int32)
        host["far_pos8"] = ttnn.from_torch(
            torch.tensor([max(pos - WINDOW, 0)] * N_KV, dtype=torch.int32), dtype=ttnn.int32
        )
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
        self.far_grp = (inp["far_pos8"], inp["far_valid"])


class _NoFar:
    """RowsOut view for decode_rows' per-token far path: the prep writes token t's far q to rows.q_far_tok[t]."""

    def __init__(self, rows, t):
        self._rows = rows
        self.q_far = rows.q_far_tok[t]
        self.far_shared = False

    def __getattr__(self, name):
        return getattr(self._rows, name)


class FastAttentionState(AttentionState):
    """Ring + conv histories as `AttentionState`; global far cache is one `kv_full` (module doc)."""

    def __init__(self, device, is_global, max_ctx, cache_dtype):
        super().__init__(device, False, max_ctx, cache_dtype)  # ring + conv histories
        # the rings as [8, 1, 1024, 128] (same buffers): sdpa with the kv groups as the batch (multi-token verify)
        self.k_grp, self.v_grp = (ttnn.reshape(r, [N_KV, 1, WINDOW, HEAD_DIM]) for r in (self.k_ring, self.v_ring))
        assert self.k_grp.buffer_address() == self.k_ring.buffer_address()
        self.kv_full = self.kv_grp = self.v_full = self.v_far_grp = None  # v_grp is the ring view above
        if is_global and FAR_SPLIT:  # keys and values apart (kv_full = the keys), values in bfp4
            self.kv_full, self.v_full = (
                ttnn.zeros([1, N_KV, max_ctx, HEAD_DIM], dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
                for dt in (cache_dtype, ttnn.bfloat4_b)
            )
            self.kv_grp, self.v_far_grp = (
                ttnn.reshape(t, [N_KV, 1, max_ctx, HEAD_DIM]) for t in (self.kv_full, self.v_full)
            )
            assert self.v_far_grp.buffer_address() == self.v_full.buffer_address()
        elif is_global:
            self.kv_full = ttnn.zeros(
                [1, N_KV, max_ctx, KV_MEM], dtype=FAR_DTYPE or cache_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            # the same buffer as [8, 1, max_ctx, KV_MEM] (the far call with kv groups as the batch)
            self.kv_grp = ttnn.reshape(self.kv_full, [N_KV, 1, max_ctx, KV_MEM])
            assert self.kv_grp.buffer_address() == self.kv_full.buffer_address()

    def tensors(self):
        out = super().tensors()
        return out + [t for t in (self.kv_full, self.v_full) if t is not None]

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
        if self.v_full is not None:
            for cache, src in ((self.kv_full, keys), (self.v_full, values)):
                full = torch.zeros(N_KV, cache.shape[-2], HEAD_DIM)
                full[:, :p] = src.transpose(0, 1).float()
                dst.append((cache, full))
        elif self.kv_full is not None:
            full = torch.zeros(N_KV, self.kv_full.shape[-2], KV_MEM)
            full[:, :p, :HEAD_DIM] = values.transpose(0, 1).float()
            if not FAR_UNIT:
                full[:, :p, HEAD_DIM] = 1.0
            full[:, :p, V_MEM:] = keys.transpose(0, 1).float()
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

    def __init__(self, device, weights, is_global, prefetcher=None, kv_fused=True):
        """kv_fused: k and v ring writes as one paged_fused_update_cache program; False: two paged_update_cache
        calls (A/B runs)."""
        self.kv_fused = kv_fused
        super().__init__(device, weights, is_global, prefetcher)
        dram = ttnn.DRAM_MEMORY_CONFIG
        self._init_fused(weights)

        def put(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

        grid = device.compute_with_storage_grid_size()

        def sdpa_cfg(chunk, cores_per_head=16, g=None, **kw):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=g or grid,
                q_chunk_size=0,
                k_chunk_size=chunk,
                exp_approx_mode=False,
                max_cores_per_head_batch=cores_per_head,
                **kw,
            )

        self.sink_cols = put(torch.ones(1, 1, N_HEADS, 32), ttnn.float32)
        lg = ttnn.CoreCoord(*LOCAL_GRID) if LOCAL_GRID is not None else grid
        self.local_cfg = sdpa_cfg(LOCAL_CHUNK, LOCAL_CORES_PER_HEAD, g=lg)
        self.local_cores = {(x, y) for x in range(lg.x) for y in range(lg.y)}
        # multi-token verify (decode_rows): sdpa_decode with pending rows, (token, kv group) batches x 4 cores
        self.rows_cfg = sdpa_cfg(LOCAL_CHUNK, LOCAL_CORES_PER_HEAD, exact_exp_half_tile=True)
        # R = 4: the second sdpa call of the pass starts at token 2 (sdpa pending_tok0)
        self.rows_cfgs = {0: self.rows_cfg}
        if hasattr(ttnn.SDPAProgramConfig, "pending_tok0"):
            self.rows_cfgs[2] = sdpa_cfg(LOCAL_CHUNK, LOCAL_CORES_PER_HEAD, exact_exp_half_tile=True, pending_tok0=2)
        # shared verify far call: the R tokens' rows in one call, each token's own causal bound (rows_per_token)
        self.far_rows_cfg = (
            sdpa_cfg(FAR_CHUNK, **FAR_OPTS, exact_exp_half_tile=True, rows_per_token=GROUPS)
            if is_global and hasattr(ttnn.SDPAProgramConfig, "rows_per_token")
            else None
        )
        self.commit = KvCommit(self.device)
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
        # far q is zero on the v / mass columns: QK skips them; the mass column block [1, 0 x 31] is the same for
        # every written key (unwritten keys are causally masked), so it is read once per K buffer slot
        self.far_chunk = FAR_CHUNK_BFP4 if FAR_DTYPE == ttnn.bfloat4_b else FAR_CHUNK
        self.far_cfg = sdpa_cfg(self.far_chunk, **FAR_OPTS)
        self.far_grp_cfg = sdpa_cfg(self.far_chunk, **FAR_OPTS, exact_exp_half_tile=True) if FAR_GRP else None
        if FAR_SPLIT:
            self.far_grp_cfg = sdpa_cfg(FAR_SPLIT_CHUNK, unit_v_tile_col=True, exact_exp_half_tile=True)
        self.kv_full_shard = ttnn.create_sharded_memory_config(
            shape=(32, KV_MEM),
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
        # o-projection + post-attention conv in one program (decode(post=...))
        self.wo = AttnWo(self.w_o.bank, HIDDEN) if self.w_o.bank is not None else None
        self.near = AttnNear(self.device, sink_grp=FAR_GRP) if self.is_global else None
        # the prefetcher's projection takes an interleaved input: keep the op-by-op window there
        self.window = (
            AttnWindow(self.device, self.w_all.act_mem) if isinstance(self.w_all, _DramShardedLinear) else None
        )
        # The window's input is a zeroed [1, 1, 32, C] L1 tensor (_shared_l1) of which only rows 0..8 (ROWS) are ever
        # written: w_all reads only those rows of each activation tile (fused/kernels/x_rows.hpp), the same bytes in
        # its CB with ~1/4 of the NOC traffic on the 8 shard cores. (w_o's 1-row input read this way was slower.)
        if self.window is not None and self.w_all.bank is not None:
            self.w_all.bank.x_rows = 9
            # its output (550 KB of tiles, read by the prep and AttnOut programs right after) lives in L1: the prep's
            # row reads no longer go to DRAM (same bytes). One persistent buffer per shape for all layers (each app's
            # consumers are done before the next app's w_all writes it): no L1 allocation per call, which made the
            # eager path ~40 ms per step slower.
            b = self.w_all.bank
            b.out_mem = W_ALL_OUT_MEM
            b.out_buf = _shared_out(self.device, b.out_cols, b.out_dtype, W_ALL_OUT_MEM)
            # bank_linear.py EIGHT: each core's 8-row window of the 9 rows. Triple mode: v columns are read at rows 5..8
            # (window rows 1..8), q / gate / rel / sink at row 0 and k at rows 1..4 (rows 0..7); the global layers' dynamic
            # taps at the k / v lag-0 rows 1 and 5 (either window). Single mode: rows 0..3.
            q1, v0, v1 = KV0 // 32, (KV0 + N_KV * HEAD_DIM) // 32, (KV0 + 2 * N_KV * HEAD_DIM) // 32
            win = self.window  # the input mode of the window the projection reads: its last call (decode or not)

            def x_base(part):
                cols = [g0 + t for ncq, g0 in part for t in range(ncq)]
                if not getattr(win, "triple", False) or not any(v0 <= c < v1 for c in cols):
                    return 0
                assert min(cols) >= q1, "a core with q columns (row 0) and v columns (rows 5..8)"
                return 1

            b.x_base = x_base

    def new_state(self, max_ctx, cache_dtype=ttnn.bfloat16):
        if self.is_global and max_ctx % self.far_chunk:
            raise ValueError(f"max_ctx {max_ctx} not a multiple of {self.far_chunk}")
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
    def decode(self, xq, state, positions, xk=None, xv=None, post=None):
        """Same contract as `Inkling2Attention.decode`; `positions` is a `DecodePositionsFast`.
        post = (FusedShortConv, ring, phase, resid or None): returns conv_add(a, ring, phase, resid, fp32,
        row_out=False) instead of a; the fused path runs that conv inside the o-projection program."""
        xk = xq if xk is None else xk
        xv = xq if xv is None else xv
        if not self.FUSED:
            return self._post(self._decode_ops(xq, state, positions, xk, xv), post)
        F.RIDE = (
            self.prep.ride(positions) if FA.PREP_PRE and FA.PREP_SLIM else None
        )  # the prep's input fetch, in the w_all program
        if self.window is None:
            proj, mode = self._window(state, xq, xk, xv)
        else:
            mode = "single" if (xk is xq and xv is xq) else "triple"
            act = self.window(state.x_hist, [xq] if mode == "single" else [xq, xk, xv])
            proj = self.w_all(act, ttnn.bfloat16, self.mm_cfg)
        F.RIDE = None
        rows = ROWS[mode]
        q, mask, sink_t = self.prep(proj, positions, {"q": 0, "k": rows["k"], "v": rows["v"]}, state)
        if self.is_global:  # q pre-scaled; prep's mask (bias + near mask) and q_grp are in kv-group layout
            o_near, r, sink_far = self.near(self.prep.q_grp, mask, sink_t, state.k_ring, state.v_ring)
            if FAR_GRP:
                far_pos, valid = positions.far_grp
                far = ttnn.transformer.flash_multi_latent_attention_decode(
                    self.prep.q_far,  # [1, 8, 3, 288], written by the prep (FAR_SPLIT: [1, 8, 3, 128])
                    state.kv_grp,
                    *([state.v_far_grp] if FAR_SPLIT else []),
                    head_dim_v=V_WIDTH,
                    is_causal=True,
                    cur_pos_tensor=far_pos,
                    attention_sink=sink_far,  # [8 x 32, 32]
                    scale=1.0,
                    program_config=self.far_grp_cfg,
                    compute_kernel_config=self.hifi4,
                )
                return self._out_proj(self.out(far, proj, (o_near, r, valid)), post)
            assert not FAR_SPLIT, "INKLING_FAR_KV=k8v4 needs the far call with the kv groups as the batch"
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
            return self._out_proj(self.out(far, proj, (o_near, r, valid)), post)
        extra = SDPA_HOST(self.local_cores) if SDPA_HOST is not None else None
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
            **({"extra_kernels": extra} if extra is not None else {}),  # older builds lack the argument
        )
        return self._out_proj(self.out(attn, proj), post)

    def _rows_proj(self, xs, state, rows):
        """One conv window + projection for all tokens of a verify pass: token t's rows at 4 t .. 4 t + 3 (the window
        state is written by commit_rows). Returns (proj, row of token t)."""
        assert self.window is not None, "decode_rows needs the fused window (not the prefetcher projection)"
        act, rows.win_g = self.window.rows(state.x_hist, xs, rows.win)
        bank = getattr(self.w_all, "bank", None)
        bank = getattr(bank, "st", None) and bank.st.bl or bank  # decode staging proxy: its BankLinear
        x_rows = bank.x_rows if bank is not None else None
        if bank is not None:
            bank.x_rows = 0  # every row of the activation tiles
        try:
            proj = self.w_all(act, ttnn.bfloat16, self.mm_cfg)
        finally:
            if bank is not None:
                bank.x_rows = x_rows
        return proj

    @staticmethod
    def _rows_sel(rows, t):
        """Projection rows of token t (q, and k / v lag 0) in a verify pass's window (AttnWindow.rows)."""
        b = rows.win_g * t
        r = ROWS["single" if rows.win_g == 4 else "triple"]
        return {"q": b, "k": b + r["k"], "v": b + r["v"]}

    def rows_out(self, R, state):
        """Scratch of a multi-token verify pass (decode_rows) for R tokens, R <= 4."""
        rows = RowsOut(self.device, R, state.k_ring.dtype, is_global=self.is_global)
        rows.win = ttnn.zeros_like(state.x_hist)
        return rows

    def decode_rows(self, xs, state, positions, rows, p0=None):
        """Multi-token verify: R = len(xs) <= 4 tokens at consecutive positions (R <= 3 in the triple input mode);
        xs[t] the token's input row (single mode) or [xq, xk, xv] (triple), positions[t] its prepared DecodePositionsFast, rows a rows_out(R). Each token's window /
        projection and the prep run as one program each (its k / v rows go to rows.pending, the ring is not written), then one
        sdpa_decode over the ring with the pending rows, then each token's output gate into o-projection input row t.
        Returns that input (self.out.act, rows 0 .. R-1): row t is bit-equal to row 0 after plain decode of token t.
        Call commit_rows after the accept to write the accepted rows into the ring.
        Global layers run each token's prep / near / far / output merge in order (no shared attention call: the far
        call's core split is per token); the near part reads the pending rows the same way."""
        assert self.FUSED and len(xs) == rows.R
        if self.is_global:
            assert FAR_GRP
            R = rows.R
            # one far call for all tokens when their far bounds share a K chunk (exact, sdpa rows_per_token);
            # p0: token 0's position (host int). Otherwise one far call per token.
            shared = (
                self.far_rows_cfg is not None
                and R > 1
                and p0 is not None
                and p0 >= WINDOW
                and (p0 - WINDOW) // FAR_CHUNK == (p0 + R - 1 - WINDOW) // FAR_CHUNK
            )
            proj = self._rows_proj(xs, state, rows)
            sels = [self._rows_sel(rows, t) for t in range(R)]
            views = [rows if shared else _NoFar(rows, t) for t in range(R)]
            for t0, n in rows.chunks:  # one prep program per chunk of <= 3 tokens
                sl = slice(t0, t0 + n)
                self.prep.rows(proj, positions[sl], sels[sl], state, views[sl], t0=t0)
            # near + combine of all tokens in one program each (token t reads pending rows 0 .. t)
            sinks = [rows.sink_far if shared else rows.sink_tok[t] for t in range(R)]
            items = [
                (
                    rows.q_grp[t],
                    rows.mask_g[t],
                    rows.sink_blk[t],
                    rows.o_near[t],
                    rows.r[t],
                    sinks[t],
                    t if shared else 0,
                )
                for t in range(R)
            ]
            for t0, n in rows.chunks:
                self.near.rows(items[t0 : t0 + n], state.k_ring, state.v_ring, rows, positions[0].pos, t0=t0)
            if not shared:
                for t, pos in enumerate(positions):
                    far_pos, valid = pos.far_grp
                    far = ttnn.transformer.flash_multi_latent_attention_decode(
                        rows.q_far_tok[t],
                        state.kv_grp,
                        head_dim_v=V_WIDTH,
                        is_causal=True,
                        cur_pos_tensor=far_pos,
                        attention_sink=sinks[t],
                        scale=1.0,
                        program_config=self.far_grp_cfg,
                        compute_kernel_config=self.hifi4,
                    )
                    self.out(far, proj, (rows.o_near[t], rows.r[t], valid), row=t, proj_row=rows.win_g * t)
            if shared:
                far = ttnn.transformer.flash_multi_latent_attention_decode(
                    rows.q_far,
                    state.kv_grp,
                    head_dim_v=V_WIDTH,
                    is_causal=True,
                    cur_pos_tensor=positions[R - 1].far_grp[0],
                    attention_sink=rows.sink_far,
                    scale=1.0,
                    program_config=self.far_rows_cfg,
                    compute_kernel_config=self.hifi4,
                )
                if OUT_ROWS:  # every token's output merge in one program
                    items = [(far, (rows.o_near[t], rows.r[t], positions[t].far_grp[1]), GROUPS * t) for t in range(R)]
                    return self.out.rows(proj, items, rows.win_g)
                for t in range(R):
                    self.out(
                        far,
                        proj,
                        (rows.o_near[t], rows.r[t], positions[t].far_grp[1]),
                        row=t,
                        far_row=GROUPS * t,
                        proj_row=rows.win_g * t,
                    )
            return self.out.act
        proj = self._rows_proj(xs, state, rows)
        sels = [self._rows_sel(rows, t) for t in range(rows.R)]
        for t0, n in rows.chunks:
            self.prep.rows(proj, positions[t0 : t0 + n], sels[t0 : t0 + n], state, [rows] * n, t0=t0)
        outs = []
        for k, (t0, n) in enumerate(rows.chunks):  # one sdpa call per chunk (pending_tok0: its first token)
            outs.append(
                ttnn.transformer.scaled_dot_product_attention_decode(
                    rows.qc[k],
                    state.k_grp,
                    state.v_grp,
                    is_causal=False,
                    attn_mask=rows.maskc[k],
                    attention_sink=rows.sinkc[k],
                    scale=1.0,
                    program_config=self._rows_cfg(t0),
                    compute_kernel_config=self.hifi4,
                    pending_kv=rows.pending,
                    pending_pos=positions[0].pos,
                )
            )
        if OUT_ROWS:  # every token's output gate in one program
            return self.out.rows(proj, [(outs[rows.local(t)[0]], rows.local(t)[1]) for t in range(rows.R)], rows.win_g)
        for t in range(rows.R):
            k, lt = rows.local(t)
            self.out(outs[k], proj, row=t, proj_row=rows.win_g * t, batch=lt)
        return self.out.act

    def _rows_cfg(self, t0):
        assert t0 in self.rows_cfgs, "R = 4 needs sdpa_decode pending_tok0 (a newer build)"
        return self.rows_cfgs[t0]

    def commit_rows(self, state, rows, positions0, n):
        """After the accept of a decode_rows pass: tokens 0 .. n - 1's k / v rows into the ring, as plain decode would
        have written them. positions0: token 0's DecodePositionsFast."""
        self.commit(rows, state.k_ring, state.v_ring, positions0.pos, n)
        self.window.commit(state.x_hist, rows.win, n, rows.win_g)

    @staticmethod
    def _post(a, post):
        if post is None:
            return a
        conv, ring, phase, resid = post
        return conv.conv_add(a, ring, phase, resid, ttnn.float32, row_out=False)

    def _out_proj(self, gated, post):
        if post is not None and self.wo is not None:
            resid = post[3]
            if resid is None or (F.twin(resid) is not None and F.twin(resid)[1] == 1):
                return self.wo(gated, conv=post)
        return self._post(self.w_o(gated, ttnn.bfloat16, self.mm_cfg), post)

    def _decode_ops(self, xq, state, positions, xk, xv):
        q, rel, gate, sink, k, v = self._project(state, positions, xq, xk, xv)
        if self.kv_fused:
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
            row = ttnn.concat([v, k] if FAR_UNIT else [v, self.mass_col, k], dim=-1)
            row = ttnn.to_memory_config(row, self.kv_full_shard)
            ttnn.experimental.paged_update_cache(state.kv_full, row, update_idxs_tensor=positions.pos)

        bias = self._rel_bias(rel, positions)
        if self.is_global:
            attn = self._global(q, bias, ttnn.add(self.sink_pad, ttnn.typecast(sink, ttnn.float32)), state, positions)
        else:
            attn = self._local(q, bias, sink, state, positions)
        attn = ttnn.reshape(attn, [1, 1, 1, N_HEADS * HEAD_DIM])
        gated = ttnn.multiply(ttnn.typecast(attn, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
        return self.w_o(ttnn.typecast(gated, ttnn.bfloat16), ttnn.bfloat16, self.mm_cfg)
