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
    permuting paged_fill_cache into a scratch buffer, which feeds the KV cache fill and ring SDPA.

Guard (cp_prefill_applies): Qwen, Wormhole, prefetcher, batch 1, start_pos 0 (no prefix cache), S > 8192,
S % 1024 == 0, env QWEN_CP_PREFILL=1 (the default). QWEN_CP_PREFILL=0 selects the keep-14 seq-parallel path.
"""

from __future__ import annotations

import os

import torch

import ttnn

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


# Best of the job139 variants at the B shapes (M = S/4 = 10240).
QKV_CFG = _mmc(8, 8, 4, 2, 2)  # 1.420 ms (today's cfg 1.541)
FF13_CFG = _mmc(8, 8, 10, 1, 5)  # 2.584 ms LoFi (today's cfg 2.928)
FF13_FP32_CFG = _mmc(8, 8, 8, 1, 4)  # HiFi2 fp32 acc fallback, 3.725 ms
FF2_CFG = _mmc(8, 10, 8, 1, 4)  # 3.667 ms


def ff13_fp32_acc() -> bool:
    """Design fallback: FF1/FF3 HiFi2 with fp32 dest accumulation (error to the bf8 floor, +1.1 ms per op)."""
    return os.environ.get("QWEN_CP_FF13_FP32", "0") == "1"


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
    xa = _row_ag(tt_ccl, mc, x_local, S)  # [1, 1, S/4, 5120]
    ttnn.deallocate(x_local)
    xqkv = ttnn.experimental.minimal_matmul(
        input_tensor=xa,
        weight_tensor=w["wqkv"],
        config=QKV_CFG,
        dtype=attn.ccl_dtype,
        compute_kernel_config=attn.compute_kernel_config_hifi2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1, 1, S/4, (H + 2) * D]
    ttnn.deallocate(xa)

    # Q on the owned slices (natural rows of the full Q buffer), K over the local rows, V a column slice.
    from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import qwen_q_prep

    trans = attn.transformation_mats["prefill"]
    q_8b = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H, S, D]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, attn.mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    qwen_q_prep(
        q_8b,
        rot_mats[0],
        rot_mats[1],
        trans,
        attn.q_norm.weight,
        eps=attn.q_norm.eps,
        src=xqkv,
        src_col=0,
        src_local=True,
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
        eps=attn.k_norm.eps,
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

    # Ring SDPA, today's call: Q owned slices, natural K / V -> [1, H, S/4, D] in the column's local order.
    ring_cfg = mc.get("SDPA_RING_PROGCFG", mc["SDPA_PROGCFG"])
    ckc = attn.compute_kernel_config_hifi4
    if attn.compute_kernel_config_ring_sdpa_long is not None:
        ckc = attn.compute_kernel_config_ring_sdpa_long
    attn_out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
        q_8b,
        k_nat,
        v_nat,
        ring_size=COLS,
        scale=attn.scale,
        compute_kernel_config=ckc,
        program_config=ring_cfg(S, chunk_start_idx=0),
        page_table=None,
        chunk_start_idx=None,
    )
    ttnn.deallocate(q_8b)
    cat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, S/4, H*D]
    ttnn.deallocate(attn_out)
    o = ttnn.experimental.minimal_matmul(
        input_tensor=cat,
        weight_tensor=w["wo"],
        config=mc["WO_PREFILL_MINIMAL_PROGCFG"](S),
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=attn.compute_kernel_config_hifi2,  # fp32 dest acc (keep 17)
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
    xa = _row_ag(tt_ccl, mc, x_local, S)  # [1, 1, S/4, 5120]
    ttnn.deallocate(x_local)
    fp32 = ff13_fp32_acc()
    ff13_kw = dict(
        config=FF13_FP32_CFG if fp32 else FF13_CFG,
        compute_kernel_config=args.compute_kernel_config_hifi2 if fp32 else args.compute_kernel_config_lofi,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w1o = ttnn.experimental.minimal_matmul(input_tensor=xa, weight_tensor=w["w1"], **ff13_kw)  # [1, 1, S/4, 3200]
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
    w2o = ttnn.experimental.minimal_matmul(
        input_tensor=w2_in,
        weight_tensor=w["w2"],
        config=FF2_CFG,
        dtype=mlp.prefill_output_dtype,
        compute_kernel_config=args.compute_kernel_config_hifi2,  # fp32 dest acc (keep 17)
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


def local_norm(dist_norm, x, compute_kernel_config, out_dtype=ttnn.bfloat8_b):
    """RMSNorm of the residual [1, 1, S/32, 5120] over the full local width, no CCL. The stock ops overflow L1 at
    width 5120, so stats and apply run on two width halves of 2560 with the stats concatenated (job142: 0.737 ms,
    pcc 0.99996 vs fp32)."""
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
            epsilon=dist_norm.norm.eps,
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


def decoder_prefill(ctx: CPContext, tt_ccl, layer, h, rot_mats, user_id=0, page_table=None, kv_cache=None):
    """One decoder layer in layout B. h: residual [1, 1, S/32, 5120] bf16 (deallocated). Returns the new residual."""
    ckc = layer.args.compute_kernel_config_hifi2
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
    free_weights(w)
    return out


def model_prefill(model, x, rot_mats, user_id=0, page_table=None, kv_cache=None):
    """llama_model.forward body for the guarded case: natural embedding x [1, 1, S, 1280] -> natural pre-norm
    output [1, 1, S, 1280] (the same tensor today's prefill forward returns)."""
    S = int(x.shape[-2])
    ctx = get_context(model.tt_ccl, model.mesh_device, S)
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
    return out
