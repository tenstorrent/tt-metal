# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused QKV → Q/K/V heads — Python wrappers.

Two variants:

  - ``bge_qkv_heads_stock``: thin shim around ``ttnn.experimental.nlp_create_qkv_heads``
    that gives the sweep harness a single function name to call. Equivalent
    to today's production path through ``BgeM3Attention``.

  - ``bge_qkv_heads_scatter``: (stub) custom path that should eventually fuse
    the matmul output writer with head split, deleting the
    ``NlpCreateHeadsDeviceOperation`` dispatch. Today raises NotImplementedError
    so the sweep records a `skipped` row instead of crashing.

The split is intentional: today's PR scaffolds the harness; the sweep
baseline test (`test_baseline_qkv_heads_timing`) must pass on silicon and
match the latest profile before we implement the scatter kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

# Path (relative to TT_METAL_HOME) where the future scatter-writer .cpp will live.
# Kept here as the single source of truth so the sweep file can reference it
# without re-hardcoding the path.
SCATTER_WRITER_KERNEL_REL_PATH = (
    "models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/kernels/" "writer_qkv_scatter.cpp"
)

# Track A optimized kernels (batched-barrier reader+writer).

# Head-split kernels (work units split by (batch, seq_tile, head_group)
# instead of just (batch, seq_tile)). Ported from Qwen3-Embedding-0.6B PR.
HEADSPLIT_READER_KERNEL_REL_PATH = (
    "models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/kernels/" "reader_qkv_heads_headsplit.cpp"
)
HEADSPLIT_WRITER_KERNEL_REL_PATH = (
    "models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/kernels/" "writer_qkv_heads_headsplit.cpp"
)

# Head-split K-BF4 fused-typecast kernels: K flows reader(BF8) -> compute
# (BF8->BF4 typecast) -> writer(BF4), folding the standalone ttnn.typecast(k,
# bfloat4_b) into the head-split op. Q/V stay on the direct BF8 path.

TILE_H = 32
TILE_W = 32


@dataclass(frozen=True)
class QkvHeadsShape:
    """Static shape info for the B1/S512 (and B32/S512) QKV → heads path."""

    batch: int
    seq_len: int
    hidden_size: int
    num_heads: int
    head_dim: int

    @property
    def qkv_width(self) -> int:
        return 3 * self.num_heads * self.head_dim


def _split_work_to_cores(num_blocks: int, grid_x: int, grid_y: int) -> tuple[int, list[tuple[int, int, int]]]:
    """Replicate tt::tt_metal::split_work_to_cores for the linear case.

    Mirrors the program factory: cores are addressed by ``(i / grid_y, i % grid_y)``
    in linear ID order, blocks are split as evenly as possible. Returns
    ``(num_cores, [(core_x, core_y, num_blocks_per_core), ...])``.
    """
    if num_blocks <= 0:
        return 0, []
    num_cores_total = grid_x * grid_y
    num_cores = min(num_cores_total, num_blocks)
    base = num_blocks // num_cores
    extra = num_blocks % num_cores
    cores: list[tuple[int, int, int]] = []
    for i in range(num_cores):
        n = base + (1 if i < extra else 0)
        cx, cy = divmod(i, grid_y)
        cores.append((cx, cy, n))
    return num_cores, cores


@dataclass(frozen=True)
class _TrackAPlan:
    """Pre-computed shape/layout knobs for one B1/S512-style call."""

    batch: int
    seq_len: int
    num_heads: int
    head_dim: int
    in_w: int
    in_w_tiles: int  # = 3 * num_heads * head_dim / TILE_W
    q_out_h_tiles: int  # = seq_len / TILE_H
    q_out_w_tiles: int  # = head_dim / TILE_W
    q_out_HtWt: int  # = q_out_h_tiles * q_out_w_tiles
    q_out_CHtWt: int  # = num_heads * q_out_HtWt
    kv_out_CHtWt: int  # = num_heads * q_out_HtWt  (BGE: kv heads == q heads)
    q_num_tiles: int  # = num_heads * q_out_w_tiles
    kv_num_tiles: int  # = num_heads * q_out_w_tiles
    num_blocks_total: int  # = batch * (seq_len / TILE_H)

    @classmethod
    def from_input(cls, qkv_fused: ttnn.Tensor, num_heads: int) -> "_TrackAPlan":
        shape = qkv_fused.padded_shape
        b, _, s, w = shape[0], shape[1], shape[2], shape[3]
        if s % TILE_H != 0:
            raise ValueError(f"seq_len {s} must be divisible by TILE_H={TILE_H}")
        if w % TILE_W != 0:
            raise ValueError(f"qkv_fused width {w} must be divisible by TILE_W={TILE_W}")
        if w != 3 * num_heads * (w // (3 * num_heads)):
            raise ValueError(f"qkv_fused width {w} not 3*num_heads={3*num_heads}-aligned")
        head_dim = w // (3 * num_heads)
        q_out_h_tiles = s // TILE_H
        q_out_w_tiles = head_dim // TILE_W
        q_out_HtWt = q_out_h_tiles * q_out_w_tiles
        return cls(
            batch=b,
            seq_len=s,
            num_heads=num_heads,
            head_dim=head_dim,
            in_w=w,
            in_w_tiles=w // TILE_W,
            q_out_h_tiles=q_out_h_tiles,
            q_out_w_tiles=q_out_w_tiles,
            q_out_HtWt=q_out_HtWt,
            q_out_CHtWt=num_heads * q_out_HtWt,
            kv_out_CHtWt=num_heads * q_out_HtWt,
            q_num_tiles=num_heads * q_out_w_tiles,
            kv_num_tiles=num_heads * q_out_w_tiles,
            num_blocks_total=b * q_out_h_tiles,
        )


def _tile_size_bytes(dtype: ttnn.DataType) -> int:
    return {
        ttnn.bfloat16: 2048,
        ttnn.bfloat8_b: 1088,
        ttnn.bfloat4_b: 576,
        ttnn.float32: 4096,
    }[dtype]


def bge_qkv_heads_headsplit(
    qkv_fused: ttnn.Tensor,
    *,
    num_heads: int,
    head_groups: int | None = None,
    out_memcfg: ttnn.MemoryConfig | None = None,
    k_out_dtype: ttnn.DataType | None = None,
    v_out_dtype: ttnn.DataType | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Head-split QKV head creation.

    Work decomposition splits each (batch, seq_tile) into ``head_groups`` sub-units,
    giving ``batch * seq_tiles * head_groups`` total work units. For BGE-M3 at
    B1/S512 with ``head_groups=num_heads=16`` that's 256 work units instead of
    Track A's 16, allowing all ~110 cores to participate.

    Args:
        qkv_fused: Tensor [B, 1, S, 3*num_heads*head_dim], TILE_LAYOUT.
        num_heads: BGE-M3 = 16.
        head_groups: How many slices to split heads into. Must divide num_heads.
            Default ``num_heads`` (max granularity: one KV head per work unit).
        out_memcfg: Output memcfg. Default DRAM.

    Returns:
        ``(q, k, v)`` each shape ``[B, num_heads, S, head_dim]``.
    """
    if out_memcfg is None:
        out_memcfg = ttnn.DRAM_MEMORY_CONFIG
    if head_groups is None:
        head_groups = num_heads
    if num_heads % head_groups != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by head_groups ({head_groups})")
    heads_per_group = num_heads // head_groups

    if k_out_dtype is not None and k_out_dtype != qkv_fused.dtype:
        raise ValueError(
            "k_out_dtype must match the input dtype. The QKV scatter emits BF4 K/V " "for the serving path."
        )

    device = qkv_fused.device()
    plan = _TrackAPlan.from_input(qkv_fused, num_heads)
    q_heads_per_kv = 1  # BGE-M3 MHA: Q heads == KV heads

    # ---- Pre-allocate Q/K/V outputs ----
    out_shape = (plan.batch, num_heads, plan.seq_len, plan.head_dim)
    out_dtype = qkv_fused.dtype
    q_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, out_memcfg)
    k_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, out_memcfg)
    v_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, out_memcfg)

    # ---- Work split: batch * seq_tiles * head_groups total units. ----
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_work_units_total = plan.num_blocks_total * head_groups
    num_cores, per_core = _split_work_to_cores(num_work_units_total, grid_x, grid_y)
    if num_cores == 0:
        raise RuntimeError("bge_qkv_heads_headsplit: nothing to do")

    used_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for (cx, cy, _) in per_core]
    )

    # ---- CB: shared between reader and writer. Size for one group_q chunk
    # (the largest of Q/K/V per work unit). Double-buffered. ----
    cb_id = 1
    group_q_tiles = heads_per_group * q_heads_per_kv * plan.q_out_w_tiles
    cb_total_tiles = group_q_tiles * 2  # double-buffer
    tile_size = _tile_size_bytes(out_dtype)
    cb_desc = ttnn.CBDescriptor(
        total_size=cb_total_tiles * tile_size,
        core_ranges=used_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_id,
                data_format=out_dtype,
                page_size=tile_size,
            )
        ],
    )

    # ---- Reader CT args ----
    reader_ct_args = [
        q_heads_per_kv,
        num_heads,  # num_kv_heads (BGE: same as num_q_heads)
        plan.q_out_w_tiles,  # head_dim_tiles
        plan.in_w_tiles,  # in0_w_tiles = 3 * num_heads * head_dim_tiles
        plan.q_out_h_tiles,  # seq_tiles
        head_groups,
        heads_per_group,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(qkv_fused).get_compile_time_args())

    reader_rt_per_core: list[tuple[tuple[int, int], list[int]]] = []
    work_unit_cursor = 0
    for cx, cy, n_units in per_core:
        reader_rt_per_core.append(
            (
                (cx, cy),
                [
                    qkv_fused.buffer_address(),  # in0_tensor_addr
                    n_units,  # num_work_units
                    work_unit_cursor,  # work_unit_start
                ],
            )
        )
        work_unit_cursor += n_units

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=HEADSPLIT_READER_KERNEL_REL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=used_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_per_core,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer CT args ----
    writer_ct_args = [
        plan.q_out_h_tiles,
        plan.q_out_w_tiles,
        plan.q_out_HtWt,
        num_heads,  # num_q_heads
        num_heads,  # num_kv_heads (BGE: same)
        q_heads_per_kv,
        head_groups,
        heads_per_group,
        plan.q_out_h_tiles,  # seq_tiles
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(q_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(k_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(v_tensor).get_compile_time_args())

    writer_rt_per_core: list[tuple[tuple[int, int], list[int]]] = []
    work_unit_cursor = 0
    for cx, cy, n_units in per_core:
        writer_rt_per_core.append(
            (
                (cx, cy),
                [
                    q_tensor.buffer_address(),
                    k_tensor.buffer_address(),
                    v_tensor.buffer_address(),
                    n_units,  # num_work_units
                    work_unit_cursor,  # work_unit_start
                ],
            )
        )
        work_unit_cursor += n_units

    writer_kd = ttnn.KernelDescriptor(
        kernel_source=HEADSPLIT_WRITER_KERNEL_REL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=used_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_per_core,
        config=ttnn.WriterConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kd, writer_kd],
        cbs=[cb_desc],
    )

    io_tensors = [qkv_fused, q_tensor, k_tensor, v_tensor]
    ttnn.generic_op(io_tensors, program_descriptor)
    return q_tensor, k_tensor, v_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Future Track B: fused matmul-with-scatter-writer (stub)
# ──────────────────────────────────────────────────────────────────────────────


def bge_qkv_heads_scatter(
    hidden_states: ttnn.Tensor,  # noqa: ARG001 (stub)
    wqkv: ttnn.Tensor,  # noqa: ARG001
    bqkv: ttnn.Tensor | None,  # noqa: ARG001
    *,
    num_heads: int,  # noqa: ARG001
    head_dim: int,  # noqa: ARG001
    qkv_compute_kernel_cfg,  # noqa: ARG001
    qkv_program_config,  # noqa: ARG001
    out_dtype: ttnn.DataType,  # noqa: ARG001
    out_memcfg: ttnn.MemoryConfig | None = None,  # noqa: ARG001
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Stub: fused QKV-matmul-with-scatter-writer.

    Not implemented in the scaffold pass. The sweep file calls this and
    catches NotImplementedError to record a `skipped` row.

    The implementation plan, once baseline timing is locked:
      1. Fork the matmul `in1_receiver_writer_padding_block_sharded.cpp`
         (the writer used by `MatmulMultiCoreReuseMultiCast2dProgramConfig`).
      2. Replace the single output `TensorAccessor` with three
         (Q, K, V), routing tiles by their N-column index.
      3. Wrap as a `ttnn.generic_op` with `ttnn.ProgramDescriptor` mirroring
         the production matmul's CB layout, reader CTs, and runtime args.
      4. Verify Q/K/V are bit-equivalent to
         `qkv_matmul → nlp_create_qkv_heads` on the same inputs.

    See `SCATTER_WRITER_KERNEL_REL_PATH` for the eventual kernel location.
    """
    raise NotImplementedError(
        "bge_qkv_heads_scatter: scatter writer not yet implemented. "
        "Implement after baseline sweep confirms timing baseline. "
        f"Future kernel path: {SCATTER_WRITER_KERNEL_REL_PATH}"
    )
