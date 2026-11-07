# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 context-parallel long prefill on the Wormhole Galaxy 8x4 mesh (design B).

Layout per device (r, c), S = padded prefill length:
  - column c owns ring slices c and 7 - c (S/4 tokens). Its local order is [slice c ; slice 7 - c].
  - row r owns q heads 8r..8r+7, kv head r and MLP columns 3200r..3200r+3199 (the stored weight split over rows).
  - residual: local rows [S/32 * r, S/32 * (r + 1)) of column c, full hidden width: [1, 1, S/32, 5120] bf16.
  - per layer the stored prefill weights are all-gathered over the columns (axis 1) into the per-row matrices
    (the stored tensors are not touched; the gathered copy is freed after the layer).
  - attention / MLP inputs: the local norm output all-gathered over the rows (axis 0) -> [S/4, 5120] bf8; WO / FF2
    outputs are reduce-scattered over the rows back to [S/32, 5120].
  - K / V of the column's S/4 rows are gathered over the columns and reordered to the natural order with a
    permuting paged_fill_cache into a scratch buffer, which feeds the KV cache fill and ring SDPA. The paged KV
    cache is column-local (user u lives on column u // 8; block ids repeat across columns), so only the owner column
    writes it (the device page table is -1 on the other columns) and SDPA, which runs on every column, must not read
    K / V through it.
  - attention core: one model-local program per device (custom_ops/qwen_cp_sdpa) on the scratch K / V (identity page
    table) computing both owned slices, written in the concat-heads layout; QWEN_CP_LOCAL_SDPA=0 runs the two stock
    chunked SDPA calls instead (call j = slice slices(c)[j], per-device chunk_start_idx_tensor, chunked_sdpa_pair),
    bit-identical; QWEN_CP_CHUNKED_SDPA=0 keeps ring SDPA.

Guard (cp_prefill_applies): Qwen, Wormhole, prefetcher, batch 1, start_pos 0 (no prefix cache),
8192 < S <= QWEN_OPT_MAX_SEQ_LEN (tt/qwen_opt_limits.py), S % 1024 == 0, env QWEN_CP_PREFILL=1 (the default).
QWEN_CP_PREFILL=0 selects the keep-14 seq-parallel path.
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import (
    QWEN_OPT_MAX_SEQ_LEN,
    prefill_norm_eps,
    qwen_opt_seq_len_ok,
)

ROWS, COLS = 8, 4
KV_BLOCK = 64  # reorder scratch block (rows); S / 8 is a multiple of it for every S % 1024 == 0
HIDDEN = 5120
GATHER_DIMS = {"wqkv": 2, "wo": 3, "w1": 2, "w3": 2, "w2": 3}
WEIGHT_AG_KW = dict(num_workers_per_link=6)  # job139: all five 1.201 -> 1.156 ms


def cp_prefill_enabled() -> bool:
    return os.environ.get("QWEN_CP_PREFILL", "1") == "1"


def cp_prefill_applies(args, use_prefetcher, prefetcher_setup, mode, batch_size, start_pos, seq_len) -> bool:
    return (
        cp_prefill_enabled()
        and mode == "prefill"
        and getattr(args, "is_qwen", False)
        and use_prefetcher
        and not getattr(args, "is_blackhole", False)
        and prefetcher_setup is not None
        and tuple(args.cluster_shape) == (ROWS, COLS)
        and batch_size == 1
        and start_pos == 0
        and seq_len > 8192
        and qwen_opt_seq_len_ok(seq_len)  # S > QWEN_OPT_MAX_SEQ_LEN: main's per-layer path
        and seq_len % 1024 == 0
    )


# ---------------------------------------------------------------------------------------------------------------
# Layout (host math)
# ---------------------------------------------------------------------------------------------------------------
class CPLayout:
    """Index math of layout B for one padded length S."""

    def __init__(self, seq_len: int, rows: int = ROWS, cols: int = COLS):
        assert seq_len % (rows * cols * 32) == 0, seq_len
        self.S, self.rows, self.cols = seq_len, rows, cols
        self.sl = seq_len // (2 * cols)  # ring slice
        self.s4 = seq_len // cols  # tokens per column
        self.s32 = seq_len // (rows * cols)  # residual rows per device
        self.half_rows = rows // 2  # rows per ring slice of a column: S/8 = 4 * S/32

    def slices(self, c):
        return c, 2 * self.cols - 1 - c

    def local_to_natural(self, c: int, l: int) -> int:
        s0, s1 = self.slices(c)
        return s0 * self.sl + l if l < self.sl else s1 * self.sl + (l - self.sl)

    def block_start(self, r: int, c: int) -> int:
        """Natural first row of device (r, c)'s residual chunk (S/32 contiguous rows)."""
        return self.local_to_natural(c, r * self.s32)

    def owner(self, idx: int):
        """(row r, column c, row o in the chunk) of natural position idx."""
        s = idx // self.sl
        c = s if s < self.cols else 2 * self.cols - 1 - s
        local = (0 if s < self.cols else self.sl) + idx % self.sl
        return local // self.s32, c, local % self.s32

    def kv_perm(self, bs: int = KV_BLOCK):
        """Natural block (of bs rows) of each block of the column-gathered K ([col 0 local ; col 1 local ; ...])."""
        out = []
        for i in range(self.S // bs):
            c, w = divmod(i * bs, self.s4)
            out.append(self.local_to_natural(c, w) // bs)
        return out

    def exit_perm(self):
        """Natural block (of S/32 rows) of each block of the exit gather [r = 0: c' = 0..3 ; r = 1: ...]."""
        return [self.block_start(r, c) // self.s32 for r in range(self.rows) for c in range(self.cols)]

    # host helpers (tests)
    def to_cp(self, x: torch.Tensor, dim: int = -2):
        """Natural [.., S, ..] -> {(r, c): chunk} of the residual layout (chunk = S/32 rows of the natural x)."""
        return {
            (r, c): x.narrow(dim, self.block_start(r, c), self.s32) for r in range(self.rows) for c in range(self.cols)
        }

    def from_cp(self, chunks, dim: int = -2) -> torch.Tensor:
        order = sorted(chunks, key=lambda rc: self.block_start(*rc))
        return torch.cat([chunks[rc] for rc in order], dim=dim)


# ---------------------------------------------------------------------------------------------------------------
# CCL helpers on the model's prefill TT_CCL (same semaphore cycling as its own ring ops)
# ---------------------------------------------------------------------------------------------------------------
def _ag(tt_ccl, t, dim, axis, **kw):
    idx = tt_ccl.gather_idx[axis]
    out = ttnn.experimental.all_gather_async(
        input_tensor=t,
        persistent_output_buffer=None,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.gather_semaphore_handles[axis][idx],
        num_links=4,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(axis),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        subdevice_id=tt_ccl.worker_sub_device_id,
        cluster_axis=axis,
        **kw,
    )
    tt_ccl.gather_idx[axis] = (idx + 1) % tt_ccl.num_cbs
    return out


def _rs(tt_ccl, t, dim, axis, **kw):
    idx = tt_ccl.gather_idx[axis]
    out = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor=t,
        persistent_output_buffers=None,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.reduce_semaphore_handles[axis][idx],
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(axis),
        num_links=4,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        subdevice_id=tt_ccl.worker_sub_device_id,
        cluster_axis=axis,
        **kw,
    )
    tt_ccl.gather_idx[axis] = (idx + 1) % tt_ccl.num_cbs
    return out


# ---------------------------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------------------------
def stored_layer_weights(attention, feed_forward):
    """The stored prefill weight shards of one layer (bf8, DRAM interleaved, never modified)."""
    return {
        "wqkv": attention.wqkv_interleaved,
        "wo": attention.wo_interleaved,
        "w1": feed_forward.w1_interleaved,
        "w3": feed_forward.w3_interleaved,
        "w2": feed_forward.w2_interleaved,
    }


def gather_layer_weights(tt_ccl, attention, feed_forward, keys=("wqkv", "wo", "w1", "w3", "w2")):
    """All-gather the stored shards over the columns: per-row W_qkv [5120, 1280], W_o [1024, 5120],
    W1 / W3 [5120, 3200], W2 [3200, 5120] on every device of row r. Bit-exact concat of the row's shards."""
    stored = stored_layer_weights(attention, feed_forward)
    return {k: _ag(tt_ccl, stored[k], GATHER_DIMS[k], 1, **WEIGHT_AG_KW) for k in keys}


def free_weights(w):
    for t in w.values():
        ttnn.deallocate(t)
    w.clear()


# ---------------------------------------------------------------------------------------------------------------
# Per-length device context (built once per S and mesh, shared by all layers)
# ---------------------------------------------------------------------------------------------------------------
class CPContext:
    def __init__(self, mesh_device, seq_len):
        self.mesh = mesh_device
        self.lay = lay = CPLayout(seq_len)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.kv_page_table = ttnn.from_torch(
            torch.tensor([lay.kv_perm()], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        self.exit_page_table = ttnn.from_torch(
            torch.tensor([lay.exit_perm()], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        self.kv_scratch = None  # (K, V) [S/64, 1, 64, 128] bf8, allocated on first use

    def per_device_i32(self, fn):
        """1D int32 [4] per device, values fn(r, c)."""
        vals = [fn(r, c) for r in range(ROWS) for c in range(COLS)]
        return ttnn.from_torch(
            torch.tensor(vals, dtype=torch.int32).reshape(-1),
            dtype=ttnn.int32,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def replicated_i32(self, vals):
        return ttnn.from_torch(
            torch.tensor(vals, dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def identity_page_table(self):
        """[1, S/KV_BLOCK] int32 0, 1, 2, ..., replicated: the scratch K / V read as a paged cache (chunked SDPA)."""
        if getattr(self, "_identity_pt", None) is None:
            self._identity_pt = ttnn.from_torch(
                torch.arange(self.lay.S // KV_BLOCK, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        return self._identity_pt

    def scratch(self):
        if self.kv_scratch is None:
            S = self.lay.S
            self.kv_scratch = tuple(
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([S // KV_BLOCK, 1, KV_BLOCK, 128]),
                    ttnn.bfloat8_b,
                    ttnn.TILE_LAYOUT,
                    self.mesh,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
        return self.kv_scratch


def get_context(holder, mesh_device, seq_len) -> CPContext:
    """One CPContext per (S) cached on holder (the model's prefill TT_CCL)."""
    cache = holder.__dict__.setdefault("_qwen_cp_ctx", {})
    if seq_len not in cache:
        cache[seq_len] = CPContext(mesh_device, seq_len)
    return cache[seq_len]


def kv_to_natural(ctx: CPContext, tt_ccl, kv_local, scratch):
    """Column-local K or V [1, 1, S/4, 128] -> natural [1, 1, S, 128] (a view of scratch): gather over the columns,
    then a permuting paged_fill_cache into scratch (bit-exact, job139)."""
    g = _ag(tt_ccl, kv_local, 2, 1)
    ttnn.experimental.paged_fill_cache(scratch, g, ctx.kv_page_table, batch_idx=0)
    ttnn.deallocate(g)
    return ttnn.reshape(scratch, [1, 1, ctx.lay.S, scratch.shape[-1]])


# ---------------------------------------------------------------------------------------------------------------
# Entry / exit (once per forward)
# ---------------------------------------------------------------------------------------------------------------
def entry(ctx: CPContext, tt_ccl, x):
    """Natural [1, 1, S, 1280] (column c = hidden quarter c, replicated over rows) -> residual [1, 1, S/32, 5120].

    Device (r, c) slices the residual rows of devices (r, 0..3) in its hidden quarter, the column all-gather on the
    hidden dim gives those 4 blocks at full width, and column c keeps block c (mesh_partition)."""
    lay = ctx.lay
    W = int(x.shape[-1])
    if not hasattr(ctx, "entry_starts"):
        ctx.entry_starts = [ctx.per_device_i32(lambda r, c, k=k: [0, 0, lay.block_start(r, k), 0]) for k in range(COLS)]
        ctx.entry_ends = [
            ctx.per_device_i32(lambda r, c, k=k: [1, 1, lay.block_start(r, k) + lay.s32, W]) for k in range(COLS)
        ]
    parts = [
        ttnn.slice(x, starts=st, ends=en, slice_dim=2, num_devices=lay.S // lay.s32)
        for st, en in zip(ctx.entry_starts, ctx.entry_ends)
    ]
    cat = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in parts:
        ttnn.deallocate(t)
    g = _ag(tt_ccl, cat, 3, 1)
    ttnn.deallocate(cat)
    h = ttnn.mesh_partition(g, dim=2, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(g)
    return h


def exit_natural(ctx: CPContext, tt_ccl, h):
    """Residual [1, 1, S/32, 5120] -> natural [1, 1, S, 1280] (column c = hidden quarter c, replicated over rows),
    the tensor today's prefill forward returns.

    Column gather of the row's 4 blocks, keep hidden quarter c (mesh_partition), row gather of the 8 rows (block
    order (r, c')), then the 32 blocks back to the natural order."""
    lay = ctx.lay
    g = _ag(tt_ccl, h, 2, 1)
    q = ttnn.mesh_partition(g, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(g)
    a = _ag(tt_ccl, q, 2, 0)
    ttnn.deallocate(q)
    W = int(a.shape[-1])
    perm = lay.exit_perm()  # gathered block i holds natural block perm[i]
    src_of = {nb: i for i, nb in enumerate(perm)}
    blocks = [
        ttnn.slice(a, [0, 0, src_of[nb] * lay.s32, 0], [1, 1, (src_of[nb] + 1) * lay.s32, W]) for nb in range(len(perm))
    ]
    out = ttnn.concat(blocks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in blocks:
        ttnn.deallocate(t)
    ttnn.deallocate(a)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Layer pieces
# ---------------------------------------------------------------------------------------------------------------
def _mmc(m, k, n, sh, sw, gx=7, gy=10):
    return ttnn.MinimalMatmulConfig(
        M_block_size=m,
        K_block_size=k,
        N_block_size=n,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )


# Configs at the B shapes (M = S/4 = 10240). The K block sets the accumulation order and stays at the job139 pick;
# M block / subblock from cpb_sweep.py job168 (sha1 of all 32 device outputs equal to the job139 configs; host wall
# per call over 8 back-to-back calls; L1 held above 1335744 as in the model): QKV m8k8n4 s2x2 1.363 -> 1.327 ms,
# FF1/FF3 m8k8n10 s1x5 2.371 -> 2.025 ms, FF2 m8k10n8 s1x4 3.467 -> 3.357 ms, WO m16k8n4 s2x2 1.209 -> 1.190 ms.
# An FF1/FF3 N block that is not a multiple of 5 changes the bits; M block and subblock do not. WO m16k8n8 does not
# fit under the model's L1 buffer (job165).
QKV_CFG = _mmc(16, 8, 4, 1, 4)
QKV_FP16_CFG = _mmc(16, 8, 4, 2, 4)  # LoFi 16-bit dest (attn_mm_fp16_dest)
FF13_CFG = _mmc(16, 8, 10, 4, 2)
FF13_FP32_CFG = _mmc(8, 8, 8, 1, 4)  # HiFi2 fp32 acc fallback, 3.725 ms
FF2_CFG = _mmc(16, 10, 4, 1, 4)
FF2_FP16_CFG = _mmc(16, 10, 8, 2, 4)  # LoFi 16-bit dest (ff2_fp16_dest)
WO_CFG = _mmc(16, 8, 4, 1, 4)  # the seq-parallel path keeps WO_PREFILL_MINIMAL_PROGCFG (m16k8n4 s2x2)


def ff13_fp32_acc() -> bool:
    """Design fallback: FF1/FF3 HiFi2 with fp32 dest accumulation (error to the bf8 floor, +1.1 ms per op)."""
    return os.environ.get("QWEN_CP_FF13_FP32", "0") == "1"


def ff2_bf8_rs(layer_num=None, n_layers=None) -> bool:
    """FF2 output (and its row reduce-scatter) in bf8 instead of bf16: half the RS bytes (job163: RS [S/4, 5120]
    bf16 1.756 ms, bf8 0.942 ms). The residual add still writes bf16.
    QWEN_CP_FF2_BF16_LAST=N keeps bf16 on the last N layers (default 0: every layer bf8). The ring RS re-rounds the
    partial sum to bf8 at every hop (46% of layer 63's own error on the real 40k prompt, layer_err.py), but keeping
    bf16 on the last 8 / 16 layers (keeps 29 / 32) cost 7 / 11 ms b1 and did not raise device argmax agreement with
    the CPU reference on topk40k (990 with 16 layers, 993 with none; job296 / job299)."""
    if os.environ.get("QWEN_CP_FF2_BF8", "1") != "1":
        return False
    last = int(os.environ.get("QWEN_CP_FF2_BF16_LAST", "0"))
    return layer_num is None or n_layers is None or layer_num < n_layers - last


def attn_mm_fp16_dest() -> bool:
    """QKV / WO at LoFi with 16-bit dest accumulation (QWEN_CP_ATTN_MM_FP16, default 1; needs QWEN_CP_ATTN_MM_LOFI=1).
    cpb_sweep job266: QKV LoFi fp32 dest 1.058 -> 16-bit dest m16k8n4 s2x4 0.960 ms; WO 0.992 -> 0.929 ms (WO_CFG)."""
    return os.environ.get("QWEN_CP_ATTN_MM_LOFI", "1") == "1" and os.environ.get("QWEN_CP_ATTN_MM_FP16", "1") == "1"


def lofi_fp32_config(hifi2, env, fp16=False):
    """hifi2 (an fp32-dest HiFi2 config) at LoFi fidelity when env (default 1) is set, else hifi2 unchanged.
    Used for the QKV and WO matmuls (QWEN_CP_ATTN_MM_LOFI); cpb_sweep job266: QKV 1.353 -> 1.058, WO 1.194 -> 0.992 ms.
    """
    if os.environ.get(env, "1") != "1":
        return hifi2
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=hifi2.math_approx_mode,
        fp32_dest_acc_en=hifi2.fp32_dest_acc_en and not fp16,
        packer_l1_acc=hifi2.packer_l1_acc,
        dst_full_sync_en=hifi2.dst_full_sync_en,
    )


def ff2_fp16_dest() -> bool:
    """FF2 LoFi with 16-bit dest accumulation and the larger N block / 2x4 subblock that it allows (FF2_FP16_CFG)
    on the bf8-output layers (QWEN_CP_FF2_FP16=1, default). cpb_sweep job266: LoFi fp32 dest FF2_CFG 2.509 -> 16-bit dest m16k10n8 s2x4
    2.026 ms."""
    return os.environ.get("QWEN_CP_FF2_LOFI", "1") == "1" and os.environ.get("QWEN_CP_FF2_FP16", "1") == "1"


def ff2_compute_config(args, fp16=False):
    """FF2 matmul compute config: LoFi with fp32 dest accumulation (QWEN_CP_FF2_LOFI=1, default) or the model's HiFi2
    fp32-dest config (keep 17). cpb_sweep job266 (S=40960, FF2_CFG): HiFi2 3.28 -> LoFi 2.51 ms per call."""
    if os.environ.get("QWEN_CP_FF2_LOFI", "1") != "1":
        return args.compute_kernel_config_hifi2
    base = args.compute_kernel_config_hifi2
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=base.math_approx_mode,
        fp32_dest_acc_en=not fp16,
        packer_l1_acc=base.packer_l1_acc,
        dst_full_sync_en=base.dst_full_sync_en,
    )


def sdpa_fp16_acc() -> bool:
    """Ring SDPA with 16-bit dest accumulation (HiFi2 kept): sdpa_sweep e1-fid q320/k640 17.08 -> 16.16 ms,
    PCC vs HiFi4 fp32acc 0.99994 -> 0.99983. QWEN_CP_SDPA_FP16ACC=0 keeps the fp32 dest of keep 09."""
    return os.environ.get("QWEN_CP_SDPA_FP16ACC", "1") == "1"


def chunked_sdpa_enabled() -> bool:
    """Layout-B attention core as two stock chunked SDPA calls (one per owned ring slice) on the streaming compute
    instead of ring_distributed SDPA (legacy compute). Needs 16-bit dest. QWEN_CP_CHUNKED_SDPA=0 keeps the ring op.
    40k unit (cp_chunked_sdpa_unit.py job230): 16.04 -> 13.60 ms per layer; output scale vs fp32 +1.2% (ring
    +0.1..1.0%), pcc vs ring >= 0.99990."""
    return os.environ.get("QWEN_CP_CHUNKED_SDPA", "1") == "1"


def local_sdpa_enabled() -> bool:
    """Chunked-pair attention core as ONE model-local generic_op program per device (custom_ops/qwen_cp_sdpa): both
    owned slices' Q chunks spread over all grid cores by cost, each computed exactly as by its stock chunked call, the
    output written straight in the concat-heads layout WO reads (no concat, no nlp_concat_heads). Bit-identical to the
    pair. 40k unit (test_qwen_cp_sdpa_op.py, pipelined): 13.364 -> 12.476 ms per layer; b1 40k 1838.7 -> 1775.1 ms
    (job295), gate digit-identical (job296). QWEN_CP_LOCAL_SDPA=0 keeps the stock pair."""
    return os.environ.get("QWEN_CP_LOCAL_SDPA", "1") == "1"


def chunked_sdpa_config(ring_pc, sl):
    """SDPAProgramConfig of the chunked pair: the ring config's grid / exp mode / chunks, with each chunk reduced to
    a divisor of the slice length sl (the chunk start s * sl must be a multiple of both chunks). Env overrides
    QWEN_CP_CHUNKED_Q / QWEN_CP_CHUNKED_K (sweeps)."""

    def fit(c, cands=(640, 512, 384, 320, 256, 160, 128, 64, 32)):
        c = int(c)
        return c if sl % c == 0 else next(x for x in cands if x <= c and sl % x == 0)

    g = ring_pc.compute_with_storage_grid_size
    q = int(os.environ.get("QWEN_CP_CHUNKED_Q", 0)) or fit(ring_pc.q_chunk_size)
    k = int(os.environ.get("QWEN_CP_CHUNKED_K", 0)) or fit(ring_pc.k_chunk_size)
    assert sl % q == 0 and sl % k == 0, (sl, q, k)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y),
        exp_approx_mode=ring_pc.exp_approx_mode,
        q_chunk_size=q,
        k_chunk_size=k,
    )


def chunked_sdpa_pair(ctx: CPContext, q_nat, keys, values, page_table, scale, ckc, pc):
    """Causal attention of column c's two ring slices against the natural K / V of a paged cache, as two stock
    chunked SDPA calls, each one mesh-wide call with a per-device chunk start: call j runs slice lay.slices(c)[j] on
    column c. The start is a device tensor (chunk_start_idx_tensor, sharded so device (r, c) holds its own value)
    that the reader / compute / writer read at run time, so one program serves all four columns.

    q_nat: [1, H, S, D] with this column's slices at their natural rows (other rows unused), sliced per call; or a
    (q0, q1) pair of [1, H, S/8, D] buffers holding local slices 0 / 1 (qwen_q_prep q_out1), read directly (not
    freed here). Returns [1, H, S/4, D] in the column's local order [slice c ; slice 7 - c], as the ring op does.
    The output concat stays: a one-copy concat + nlp_concat_heads generic_op measured 0.294 vs 0.278 ms for the two
    stock ops (batch T, t2_unit.py job245), so it was dropped."""
    lay = ctx.lay
    split = isinstance(q_nat, (tuple, list))
    H, D = int(q_nat[0].shape[1] if split else q_nat.shape[1]), int((q_nat[0] if split else q_nat).shape[-1])
    cache = ctx.__dict__.setdefault("_chunked_sdpa", {})
    if (H, D) not in cache:
        tabs = []
        for j in range(2):
            st = ctx.per_device_i32(lambda r, c, j=j: [0, 0, lay.slices(c)[j] * lay.sl, 0])
            en = ctx.per_device_i32(lambda r, c, j=j: [1, H, (lay.slices(c)[j] + 1) * lay.sl, D])
            cs = ctx.per_device_i32(lambda r, c, j=j: [lay.slices(c)[j] * lay.sl])
            tabs.append((st, en, cs))
        cache[(H, D)] = tabs
        logger.info(f"Qwen CP prefill S={lay.S}: chunked SDPA pair, q{pc.q_chunk_size} k{pc.k_chunk_size}")
    scale32 = float(torch.tensor(scale, dtype=torch.float32))  # the binding takes an exact float32 (noconvert)
    outs = []
    for j, (st, en, cs) in enumerate(cache[(H, D)]):
        qs = q_nat[j] if split else ttnn.slice(q_nat, starts=st, ends=en, slice_dim=2, num_devices=2 * lay.cols)
        outs.append(
            ttnn.transformer.chunked_scaled_dot_product_attention(
                qs,
                keys,
                values,
                page_table,
                chunk_start_idx_tensor=cs,
                scale=scale32,
                compute_kernel_config=ckc,
                program_config=pc,
            )
        )
        if not split:
            ttnn.deallocate(qs)
    out = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in outs:
        ttnn.deallocate(t)
    return out


# Fused AG+matmul ops are measured and gated only for 32768 <= S <= 40960 (QWEN_OPT_MAX_SEQ_LEN); the CP path
# itself stops there too (qwen_ag_matmul's wave plan raises above 65536 rows, e.g. S=73728 in the 128k warmup sweep).
FUSED_AG_MIN_SEQ = 32768
FUSED_AG_MAX_SEQ = QWEN_OPT_MAX_SEQ_LEN


def fused_ag_enabled(seq_len) -> bool:
    """Attention-norm row all-gather + QKV matmul as one model-local program per device (custom_ops/qwen_ag_matmul):
    the gather runs on grid column 6 by fabric multicast while the matmul (columns 0-5, W_qkv resident in L1)
    consumes the slots as they land. Bit-identical to _row_ag + minimal_matmul. Used for S >= 32768 (job367, ms per
    pair, stock -> fused: 16k 0.895 -> 1.026, 24k 1.123 -> 1.381, 32k 1.507 -> 1.431, 40k 1.783 -> 1.603); b1 40k
    1739.0 -> 1727.0 ms (job372), gate digit-identical (job373). QWEN_CP_FUSED_AG=0 keeps the stock pair. The MLP
    uses the 7x10 form of the op instead (fused_ag_ff1_enabled): this 6x10 form is slower than the FF1 stock pair."""
    return os.environ.get("QWEN_CP_FUSED_AG", "1") == "1" and FUSED_AG_MIN_SEQ <= seq_len <= FUSED_AG_MAX_SEQ


def fused_ag_ff1_enabled(seq_len) -> bool:
    """FF-norm row all-gather + FF1 matmul as qwen_ag_matmul_v2 (the matmul on the full 7x10 grid, the fabric senders
    folded into eight in0 receiver cores; copy_local fills the own slot of the gathered buffer for FF3).
    Bit-identical. Used for S >= 32768 (job384, ms per pair, stock -> fused: 32k 2.451 -> 2.268, 40k 2.894 -> 2.678;
    b1 40k 1727.0 -> 1713.1 ms, job385; gate digit-identical, job386). QWEN_CP_FUSED_AG_FF1=0 keeps the stock pair."""
    return os.environ.get("QWEN_CP_FUSED_AG_FF1", "1") == "1" and FUSED_AG_MIN_SEQ <= seq_len <= FUSED_AG_MAX_SEQ


def release_fused_ag(mesh_device):
    """End of a prefill (also on an exception, from the callers' finally): free the fused ops' L1 semaphore words
    (see qwen_ag_matmul.release_semaphores). Both are freed even if the first release raises."""
    from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_ag_matmul import release_semaphores, release_semaphores_v2

    try:
        release_semaphores(mesh_device)
    finally:
        release_semaphores_v2(mesh_device)


def _row_ag(tt_ccl, model_config, t, seq_len):
    kw = dict(model_config["QWEN_PREFILL_RING_AG_PARAMS"](seq_len, "FF2") or {})
    return _ag(tt_ccl, t, 2, 0, **kw)


def _row_rs(tt_ccl, model_config, t, seq_len, key):
    kw = dict(model_config["QWEN_PREFILL_RING_RS_PARAMS"](seq_len, key) or {})
    return _rs(tt_ccl, t, 2, 0, **kw)


def attention_prefill(ctx: CPContext, tt_ccl, attn, x_local, rot_mats, w, user_id=0, page_table=None, kv_cache=None):
    """Design B attention (steps 2-12). x_local: attention-norm output [1, 1, S/32, 5120] bf8 (deallocated).
    Returns the WO output reduce-scattered over the rows, [1, 1, S/32, 5120] bf8. Fills the KV cache as today."""
    S = ctx.lay.S
    H, D = attn.n_local_heads, attn.head_dim
    mc = attn.model_config
    qkv_kw = dict(
        config=QKV_FP16_CFG if attn_mm_fp16_dest() else QKV_CFG,
        dtype=attn.ccl_dtype,
        compute_kernel_config=lofi_fp32_config(
            attn.compute_kernel_config_hifi2, "QWEN_CP_ATTN_MM_LOFI", fp16=attn_mm_fp16_dest()
        ),
    )
    if fused_ag_enabled(S):
        from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_ag_matmul import qwen_ag_matmul

        xqkv, xa = qwen_ag_matmul(x_local, w["wqkv"], **qkv_kw)  # [1, 1, S/4, (H + 2) * D]
        ttnn.deallocate(x_local)
    else:
        xa = _row_ag(tt_ccl, mc, x_local, S)  # [1, 1, S/4, 5120]
        ttnn.deallocate(x_local)
        xqkv = ttnn.experimental.minimal_matmul(
            input_tensor=xa, weight_tensor=w["wqkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG, **qkv_kw
        )  # [1, 1, S/4, (H + 2) * D]
    ttnn.deallocate(xa)

    # SDPA config (ring op, or the chunked pair on 16-bit dest).
    ring_cfg = mc.get("SDPA_RING_PROGCFG", mc["SDPA_PROGCFG"])
    ckc = attn.compute_kernel_config_hifi4
    if attn.compute_kernel_config_ring_sdpa_long is not None:
        ckc = attn.compute_kernel_config_ring_sdpa_long
        if sdpa_fp16_acc():
            ckc = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ckc.math_fidelity,
                math_approx_mode=ckc.math_approx_mode,
                fp32_dest_acc_en=False,
                packer_l1_acc=ckc.packer_l1_acc,
            )
    ring_pc = ring_cfg(S, chunk_start_idx=0)
    use_chunked = chunked_sdpa_enabled() and not ckc.fp32_dest_acc_en

    # Q on the owned slices, K over the local rows, V a column slice. Chunked pair: Q written straight into one
    # [1, H, S/8, D] buffer per owned slice (what each chunked call reads); ring op: the natural rows of a full Q.
    from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import qwen_q_prep

    trans = attn.transformation_mats["prefill"]
    q_rows = ctx.lay.sl if use_chunked else S
    q_8b = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, q_rows, D]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, attn.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(2 if use_chunked else 1)
    ]
    qwen_q_prep(
        q_8b[0],
        rot_mats[0],
        rot_mats[1],
        trans,
        attn.q_norm.weight,
        eps=prefill_norm_eps(attn.q_norm, S, getattr(attn, "model_config", None)),
        src=xqkv,
        src_col=0,
        src_local=True,
        q_out1=q_8b[1] if use_chunked else None,
    )
    rot_local = attn._qwen_local_q_setup(rot_mats, S)[3]
    k_loc = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, S // COLS, D]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, attn.mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    qwen_q_prep(
        k_loc,
        rot_local[0],
        rot_local[1],
        trans,
        attn.k_norm.weight,
        eps=prefill_norm_eps(attn.k_norm, S, getattr(attn, "model_config", None)),
        all_rows=True,
        src=xqkv,
        src_col=H * D,
    )
    v_loc = ttnn.slice(
        xqkv, [0, 0, 0, (H + 1) * D], [1, 1, S // COLS, (H + 2) * D], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(xqkv)

    # K / V to the natural order (views of the persistent scratch buffers: never deallocated here).
    ks, vs = ctx.scratch()
    k_nat = kv_to_natural(ctx, tt_ccl, k_loc, ks)
    v_nat = kv_to_natural(ctx, tt_ccl, v_loc, vs)
    ttnn.deallocate(k_loc)
    ttnn.deallocate(v_loc)

    # KV cache fill, as today's forward_prefill (no prefix cache here).
    keys, values = (kv_cache[0], kv_cache[1]) if kv_cache else (attn.layer_past[0], attn.layer_past[1])
    if page_table:
        ttnn.experimental.paged_fill_cache(keys, k_nat, page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(values, v_nat, page_table, batch_idx=0)
    else:
        k_fill = attn.prefill_prepare_tensor_for_kv_cache(k_nat, user_id)
        v_fill = attn.prefill_prepare_tensor_for_kv_cache(v_nat, user_id)
        ttnn.fill_cache(keys, k_fill, user_id % attn.batch_size_per_device_group)
        ttnn.fill_cache(values, v_fill, user_id % attn.batch_size_per_device_group)

    cat = None
    if use_chunked and local_sdpa_enabled():
        from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_cp_sdpa import qwen_cp_sdpa

        # Both slices in one program, written as [1, 1, S/4, H*D] in the local order (what the pair + concat +
        # nlp_concat_heads give, bit for bit).
        cat = qwen_cp_sdpa(
            q_8b[0],
            q_8b[1],
            ks,
            vs,
            ctx.identity_page_table(),
            scale=float(torch.tensor(attn.scale, dtype=torch.float32)),
            compute_kernel_config=ckc,
            program_config=chunked_sdpa_config(ring_pc, ctx.lay.sl),
            concat_heads=True,
        )
    elif use_chunked:
        # The scratch buffers [S/64, 1, 64, 128] read as a paged cache through the identity table.
        attn_out = chunked_sdpa_pair(
            ctx, q_8b, ks, vs, ctx.identity_page_table(), attn.scale, ckc, chunked_sdpa_config(ring_pc, ctx.lay.sl)
        )
    else:
        # Ring SDPA: Q owned slices, natural K / V -> [1, H, S/4, D] in the column's local order.
        attn_out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            q_8b[0],
            k_nat,
            v_nat,
            ring_size=COLS,
            scale=attn.scale,
            compute_kernel_config=ckc,
            program_config=ring_pc,
            page_table=None,
            chunk_start_idx=None,
        )
    for t in q_8b:
        ttnn.deallocate(t)
    if cat is None:
        cat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, S/4, H*D]
        ttnn.deallocate(attn_out)
    o = ttnn.experimental.minimal_matmul(
        input_tensor=cat,
        weight_tensor=w["wo"],
        config=WO_CFG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=lofi_fp32_config(
            attn.compute_kernel_config_hifi2, "QWEN_CP_ATTN_MM_LOFI", fp16=attn_mm_fp16_dest()
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1, 1, S/4, 5120]
    ttnn.deallocate(cat)
    out = _row_rs(tt_ccl, mc, o, S, "WO")
    ttnn.deallocate(o)
    return out


def mlp_prefill(ctx: CPContext, tt_ccl, mlp, x_local, w):
    """Design B MLP (steps 14-18). x_local: ff-norm output [1, 1, S/32, 5120] bf8 (deallocated).
    Returns FF2 reduce-scattered over the rows, [1, 1, S/32, 5120] in the model's prefill output dtype (bf16)."""
    S = ctx.lay.S
    args, mc = mlp.args, mlp.model_config
    fp32 = ff13_fp32_acc()
    ff13_kw = dict(
        config=FF13_FP32_CFG if fp32 else FF13_CFG,
        compute_kernel_config=args.compute_kernel_config_hifi2 if fp32 else args.compute_kernel_config_lofi,
        dtype=ttnn.bfloat8_b,
    )
    if fused_ag_ff1_enabled(S) and not fp32:
        from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_ag_matmul import qwen_ag_matmul_v2

        w1o, xa = qwen_ag_matmul_v2(x_local, w["w1"], copy_local=True, **ff13_kw)  # [1, 1, S/4, 3200]
    else:
        xa = _row_ag(tt_ccl, mc, x_local, S)  # [1, 1, S/4, 5120]
        w1o = ttnn.experimental.minimal_matmul(
            input_tensor=xa, weight_tensor=w["w1"], memory_config=ttnn.DRAM_MEMORY_CONFIG, **ff13_kw
        )  # [1, 1, S/4, 3200]
    ttnn.deallocate(x_local)
    ff13_kw["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    w3o = ttnn.experimental.minimal_matmul(input_tensor=xa, weight_tensor=w["w3"], **ff13_kw)
    ttnn.deallocate(xa)
    w2_in = ttnn.mul(
        w1o,
        w3o,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(w1o)
    ttnn.deallocate(w3o)
    bf8_out = ff2_bf8_rs(mlp.layer_num, args.full_model_n_layers)
    fp16 = bf8_out and ff2_fp16_dest()  # the bf16-output layers keep fp32 dest and FF2_CFG
    w2o = ttnn.experimental.minimal_matmul(
        input_tensor=w2_in,
        weight_tensor=w["w2"],
        config=FF2_FP16_CFG if fp16 else FF2_CFG,
        dtype=ttnn.bfloat8_b if bf8_out else mlp.prefill_output_dtype,
        compute_kernel_config=ff2_compute_config(args, fp16),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1, 1, S/4, 5120]
    ttnn.deallocate(w2_in)
    out = _row_rs(tt_ccl, mc, w2o, S, "FF2")
    ttnn.deallocate(w2o)
    return out


def _norm_gammas(dist_norm):
    """The layer norm's replicated gamma [1, 1, 160, 32] (ROW_MAJOR bf16) split into two width halves of 2560,
    cached on the module (a 10 KB copy per norm; the stored gamma is not touched)."""
    g = getattr(dist_norm, "_qwen_cp_gammas", None)
    if g is None:
        full = dist_norm.norm.weight
        n = int(full.shape[2])
        g = tuple(ttnn.slice(full, [0, 0, i * n // 2, 0], [1, 1, (i + 1) * n // 2, 32]) for i in range(2))
        dist_norm._qwen_cp_gammas = g
    return g


def width_norm_enabled() -> bool:
    """QWEN_CP_WIDTH_NORM=0 keeps the stock 2-slice form of local_norm instead of custom_ops.qwen_width_norm."""
    return os.environ.get("QWEN_CP_WIDTH_NORM", "1") == "1"


def local_norm(dist_norm, x, compute_kernel_config, out_dtype=ttnn.bfloat8_b):
    """RMSNorm of the residual [1, 1, S/32, 5120] over the full local width, no CCL.

    Default: custom_ops.qwen_width_norm (one generic_op: the width split over the 7 grid columns, the per-row partial
    sums exchanged within each grid row, HiFi4, fp32 partials; cpb job211 S=40960: 0.580 -> 0.296 ms per norm, rel_l2
    vs fp32 1.329e-2 -> 1.230e-2). Otherwise (or for a non-bf8 output) the stock ops: they overflow L1 at width 5120,
    so stats and apply run on two width halves of 2560 with the stats concatenated (job142: 0.737 ms, pcc 0.99996 vs
    fp32)."""
    # x holds S/32 rows
    eps = prefill_norm_eps(
        dist_norm.norm, int(x.shape[-2]) * ROWS * COLS, getattr(getattr(dist_norm, "args", None), "model_config", None)
    )
    if width_norm_enabled() and out_dtype == ttnn.bfloat8_b:
        from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_width_norm import qwen_width_norm

        return qwen_width_norm(x, dist_norm.norm.weight, eps=eps, fidelity=ttnn.MathFidelity.HiFi4)
    g = _norm_gammas(dist_norm)
    W = int(x.shape[-1])
    rows = int(x.shape[-2])
    xs = [ttnn.slice(x, [0, 0, 0, i * W // 2], [1, 1, rows, (i + 1) * W // 2]) for i in range(2)]
    sts = [
        ttnn.rms_norm_pre_all_gather(t, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16) for t in xs
    ]
    st = ttnn.concat(sts, dim=3)
    for t in sts:
        ttnn.deallocate(t)
    outs = [
        ttnn.rms_norm_post_all_gather(
            xs[i],
            st,
            epsilon=eps,
            weight=g[i],
            compute_kernel_config=compute_kernel_config,
            dtype=out_dtype,
        )
        for i in range(2)
    ]
    for t in xs + [st]:
        ttnn.deallocate(t)
    out = ttnn.concat(outs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in outs:
        ttnn.deallocate(t)
    return out


def decoder_prefill(ctx: CPContext, tt_ccl, layer, h, rot_mats, user_id=0, page_table=None, kv_cache=None, w=None):
    """One decoder layer in layout B. h: residual [1, 1, S/32, 5120] bf16 (deallocated). Returns the new residual.
    w: this layer's gathered weights (gather_layer_weights); gathered and freed here when None."""
    ckc = layer.args.compute_kernel_config_hifi2
    own_w = w is None
    if own_w:
        w = gather_layer_weights(tt_ccl, layer.attention, layer.feed_forward)
    xn = local_norm(layer.attention_norm, h, ckc)
    a = attention_prefill(ctx, tt_ccl, layer.attention, xn, rot_mats, w, user_id, page_table, kv_cache)
    h2 = ttnn.add(h, a, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(h)
    ttnn.deallocate(a)
    xn2 = local_norm(layer.ff_norm, h2, ckc)
    f = mlp_prefill(ctx, tt_ccl, layer.feed_forward, xn2, w)
    out = ttnn.add(h2, f, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    ttnn.deallocate(h2)
    ttnn.deallocate(f)
    if own_w:
        free_weights(w)
    return out


def model_prefill(model, x, rot_mats, user_id=0, page_table=None, kv_cache=None):
    """llama_model.forward body for the guarded case: natural embedding x [1, 1, S, 1280] -> natural pre-norm
    output [1, 1, S, 1280] (the same tensor today's prefill forward returns)."""
    S = int(x.shape[-2])
    ctx = get_context(model.tt_ccl, model.mesh_device, S)
    try:
        h = entry(ctx, model.tt_ccl, x)
        x.deallocate(True)
        for i, layer in enumerate(model.layers):
            h = decoder_prefill(
                ctx,
                model.tt_ccl,
                layer,
                h,
                rot_mats,
                user_id,
                page_table,
                kv_cache[i] if kv_cache is not None else None,
            )
        out = exit_natural(ctx, model.tt_ccl, h)
        ttnn.deallocate(h)
    finally:
        release_fused_ag(model.mesh_device)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Generator driver: layer-major over the users of one prefill call (design B slice 6)
# ---------------------------------------------------------------------------------------------------------------
def layer_major_enabled() -> bool:
    """QWEN_CP_LAYER_MAJOR=0 sends every user through the per-user forward (model_prefill) instead."""
    return os.environ.get("QWEN_CP_LAYER_MAJOR", "1") == "1"


def layer_major_applies(model, batch, prefill_seq_lens, num_cached_tokens, page_table, do_device_sampling) -> bool:
    """The generator's prefill_forward_text call can run layer-major: every user is a CP prefill of one padded
    length (> 8192, no prefix cache), paged KV cache, on-device sampling (the logits go to the sampling buffers)."""
    if not (layer_major_enabled() and do_device_sampling and page_table is not None and batch >= 1):
        return False
    if any(num_cached_tokens[:batch]) or len(set(prefill_seq_lens[:batch])) != 1:
        return False
    S = int(prefill_seq_lens[0])
    return (S // (ROWS * COLS)) % EXIT_WINDOW == 0 and cp_prefill_applies(
        model.args, model.use_prefetcher, model.prefetcher_setup, "prefill", 1, 0, S
    )


def model_entry(model, x):
    """Natural embedding x [1, 1, S, 1280] (deallocated) -> residual [1, 1, S/32, 5120]."""
    ctx = get_context(model.tt_ccl, model.mesh_device, int(x.shape[-2]))
    h = entry(ctx, model.tt_ccl, x)
    x.deallocate(True)
    return h


EXIT_WINDOW = 128  # rows of the last-token window: the smallest prefill length the model's output norm runs at


def exit_last_token(ctx: CPContext, tt_ccl, h, idx: int):
    """Residual [1, 1, S/32, 5120] -> the aligned EXIT_WINDOW-row window of natural positions that holds idx, hidden
    quarter c per column: [1, 1, EXIT_WINDOW, 1280], idx at row idx % EXIT_WINDOW (needs S/32 % EXIT_WINDOW == 0).
    Exact copies: the owner device's window is gathered over the rows and the columns and picked by static slices."""
    lay = ctx.lay
    n = EXIT_WINDOW
    assert lay.s32 % n == 0, (lay.S, n)
    r, c, o = lay.owner(idx)
    w0 = (o // n) * n
    W = int(h.shape[-1])
    win = ttnn.slice(h, [0, 0, w0, 0], [1, 1, w0 + n, W], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g0 = _ag(tt_ccl, win, 2, 0)  # [1, 1, ROWS * n, W]
    ttnn.deallocate(win)
    p0 = ttnn.slice(g0, [0, 0, n * r, 0], [1, 1, n * (r + 1), W], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(g0)
    g1 = _ag(tt_ccl, p0, 2, 1)  # [1, 1, COLS * n, W]
    ttnn.deallocate(p0)
    p1 = ttnn.slice(g1, [0, 0, n * c, 0], [1, 1, n * (c + 1), W], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(g1)
    out = ttnn.mesh_partition(p1, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(p1)
    return out


def layer_major_prefill(model, users, rot_mats, kv_cache):
    """users: list of dicts with h (model_entry residual), user_id, page_table (device, as prepare_inputs_prefill
    builds it), last_idx. Runs every layer for all users before the next layer: the layer's weights are gathered
    once per layer instead of once per user. Returns one exit_last_token window per user (h deallocated)."""
    S = int(users[0]["h"].shape[-2]) * ROWS * COLS
    ctx = get_context(model.tt_ccl, model.mesh_device, S)
    hs = [u["h"] for u in users]
    try:
        for i, layer in enumerate(model.layers):
            w = gather_layer_weights(model.tt_ccl, layer.attention, layer.feed_forward)
            for k, u in enumerate(users):
                hs[k] = decoder_prefill(
                    ctx,
                    model.tt_ccl,
                    layer,
                    hs[k],
                    rot_mats,
                    u["user_id"],
                    u["page_table"],
                    kv_cache[i] if kv_cache is not None else None,
                    w=w,
                )
            free_weights(w)
        windows = []
        for k, u in enumerate(users):
            windows.append(exit_last_token(ctx, model.tt_ccl, hs[k], int(u["last_idx"])))
            ttnn.deallocate(hs[k])
            u["h"] = None
    finally:
        release_fused_ag(model.mesh_device)
    return windows
