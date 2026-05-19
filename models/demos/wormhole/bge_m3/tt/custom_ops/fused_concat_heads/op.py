# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused concat-heads — stock baseline + Track A batched-barrier kernels.

Two variants:

  - ``bge_concat_heads_stock``: thin shim around
    ``ttnn.experimental.nlp_concat_heads``; matches the production
    ``BgeM3Attention.forward`` call. Used by sweeps as the timing baseline.

  - ``bge_concat_heads_tracka``: ``generic_op`` wrapper that wires a custom
    batched-barrier reader+writer pair. Bit-equivalent output (the math is a
    pure tile-copy reorder; only the dispatch pattern changes).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

# Track A optimized kernels (batched-barrier reader+writer).
TRACKA_READER_KERNEL_REL_PATH = (
    "models/demos/wormhole/bge_m3/tt/custom_ops/fused_concat_heads/kernels/" "reader_concat_heads_batched.cpp"
)
TRACKA_WRITER_KERNEL_REL_PATH = (
    "models/demos/wormhole/bge_m3/tt/custom_ops/fused_concat_heads/kernels/" "writer_concat_heads_batched.cpp"
)

# Head-split kernels (work units split by (batch, seq_tile, head_group)).
HEADSPLIT_READER_KERNEL_REL_PATH = (
    "models/demos/wormhole/bge_m3/tt/custom_ops/fused_concat_heads/kernels/" "reader_concat_heads_headsplit.cpp"
)
HEADSPLIT_WRITER_KERNEL_REL_PATH = (
    "models/demos/wormhole/bge_m3/tt/custom_ops/fused_concat_heads/kernels/" "writer_concat_heads_headsplit.cpp"
)

TILE_H = 32
TILE_W = 32


# ──────────────────────────────────────────────────────────────────────────────
# Stock baseline
# ──────────────────────────────────────────────────────────────────────────────


def bge_concat_heads_stock(
    context: ttnn.Tensor,
    *,
    out_memcfg: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Stock baseline: ``ttnn.experimental.nlp_concat_heads``.

    Args:
        context: head-laid-out tensor ``[B, num_heads, S, head_dim]``, TILE
            layout, BFP8/BF16, L1 or DRAM interleaved.
        out_memcfg: output memory config. Defaults to
            ``ttnn.L1_MEMORY_CONFIG`` (matches the B1/S512 production path).

    Returns:
        Concatenated tensor of shape ``[B, 1, S, num_heads*head_dim]``.
    """
    if out_memcfg is None:
        out_memcfg = ttnn.L1_MEMORY_CONFIG
    return ttnn.experimental.nlp_concat_heads(context, memory_config=out_memcfg)


# ──────────────────────────────────────────────────────────────────────────────
# Track A: batched-barrier reader + writer via generic_op
# ──────────────────────────────────────────────────────────────────────────────


def _split_work_to_cores(num_blocks: int, grid_x: int, grid_y: int) -> tuple[int, list[tuple[int, int, int]]]:
    """Replicate tt::tt_metal::split_work_to_cores for the linear case.

    Mirrors the program factory: cores are addressed in linear ID order with
    ``(i / grid_y, i % grid_y)``. Blocks are split as evenly as possible.
    Returns ``(num_cores, [(core_x, core_y, num_blocks_per_core), ...])``.
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


def _tile_size_bytes(dtype: ttnn.DataType) -> int:
    return {
        ttnn.bfloat16: 2048,
        ttnn.bfloat8_b: 1088,
        ttnn.bfloat4_b: 576,
        ttnn.float32: 4096,
    }[dtype]


@dataclass(frozen=True)
class _TrackAPlan:
    batch: int
    seq_len: int
    num_heads: int
    head_dim: int
    in0_h_tiles: int  # = seq_len / TILE_H
    in0_w_tiles: int  # = head_dim / TILE_W
    in0_HtWt: int  # = in0_h_tiles * in0_w_tiles
    in0_CHtWt: int  # = num_heads * in0_HtWt
    per_tensor_tiles: int  # = num_heads * in0_w_tiles  (≡ block size in tiles)
    num_blocks_total: int  # = batch * in0_h_tiles
    out_w_per_block: int  # tiles written per block per output (= per_tensor_tiles)

    @classmethod
    def from_input(cls, context: ttnn.Tensor) -> "_TrackAPlan":
        shape = context.padded_shape
        b, num_heads, s, head_dim = shape[0], shape[1], shape[2], shape[3]
        if s % TILE_H != 0:
            raise ValueError(f"seq_len {s} must be divisible by TILE_H={TILE_H}")
        if head_dim % TILE_W != 0:
            raise ValueError(f"head_dim {head_dim} must be divisible by TILE_W={TILE_W}")
        in0_h_tiles = s // TILE_H
        in0_w_tiles = head_dim // TILE_W
        in0_HtWt = in0_h_tiles * in0_w_tiles
        return cls(
            batch=b,
            seq_len=s,
            num_heads=num_heads,
            head_dim=head_dim,
            in0_h_tiles=in0_h_tiles,
            in0_w_tiles=in0_w_tiles,
            in0_HtWt=in0_HtWt,
            in0_CHtWt=num_heads * in0_HtWt,
            per_tensor_tiles=num_heads * in0_w_tiles,
            num_blocks_total=b * in0_h_tiles,
            out_w_per_block=num_heads * in0_w_tiles,
        )


def bge_concat_heads_tracka(
    context: ttnn.Tensor,
    *,
    out_memcfg: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Track A: stock-equivalent concat_heads with batched-barrier kernels.

    Drop-in replacement for ``bge_concat_heads_stock``. Same external contract;
    internally uses two custom .cpp kernels that batch CB reservations and NoC
    barriers per block (32 tiles) instead of per-tile.

    PCC must be bit-equivalent to the stock op — the math is a pure tile copy
    reorder; only the dispatch pattern differs.
    """
    if out_memcfg is None:
        out_memcfg = ttnn.L1_MEMORY_CONFIG

    device = context.device()
    plan = _TrackAPlan.from_input(context)
    out_dtype = context.dtype

    # ---- Pre-allocate the concatenated output ----
    out_shape = (plan.batch, 1, plan.seq_len, plan.num_heads * plan.head_dim)
    out_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, out_memcfg)

    # ---- Pick a core grid + partition like the stock op. ----
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores, per_core = _split_work_to_cores(plan.num_blocks_total, grid_x, grid_y)
    if num_cores == 0:
        raise RuntimeError("bge_concat_heads_tracka: nothing to do (num_blocks=0)")

    used_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for (cx, cy, _) in per_core]
    )

    # ---- CB: shared cb_id=0 between reader and writer. Size to hold one full
    # block (per_tensor_tiles tiles), double-buffered so reader can produce
    # the next block while writer drains the current one. ----
    cb_id = 0
    cb_total_tiles = plan.per_tensor_tiles * 2
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

    # ---- Reader kernel descriptor ----
    reader_ct_args = [plan.in0_h_tiles, plan.in0_w_tiles, plan.num_heads, plan.in0_HtWt]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(context).get_compile_time_args())

    reader_rt_per_core: list[tuple[tuple[int, int], list[int]]] = []
    num_blocks_written = 0
    for cx, cy, n_blocks in per_core:
        in0_h_dim = num_blocks_written % plan.in0_h_tiles
        in0_tensor_tile_id = (num_blocks_written // plan.in0_h_tiles) * plan.in0_CHtWt + in0_h_dim * plan.in0_w_tiles
        reader_rt_per_core.append(
            (
                (cx, cy),
                [
                    context.buffer_address(),
                    n_blocks,
                    in0_h_dim,
                    in0_tensor_tile_id,
                ],
            )
        )
        num_blocks_written += n_blocks

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=TRACKA_READER_KERNEL_REL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=used_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_per_core,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel descriptor ----
    writer_ct_args = [plan.per_tensor_tiles]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())

    writer_rt_per_core: list[tuple[tuple[int, int], list[int]]] = []
    num_blocks_written = 0
    for cx, cy, n_blocks in per_core:
        out_start_tile_id = num_blocks_written * plan.per_tensor_tiles
        writer_rt_per_core.append(
            (
                (cx, cy),
                [
                    out_tensor.buffer_address(),
                    n_blocks,
                    out_start_tile_id,
                ],
            )
        )
        num_blocks_written += n_blocks

    writer_kd = ttnn.KernelDescriptor(
        kernel_source=TRACKA_WRITER_KERNEL_REL_PATH,
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

    io_tensors = [context, out_tensor]
    ttnn.generic_op(io_tensors, program_descriptor)
    return out_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Head-split variant — finer work-unit decomposition for higher core util.
# ──────────────────────────────────────────────────────────────────────────────


def bge_concat_heads_headsplit(
    context: ttnn.Tensor,
    *,
    head_groups: int | None = None,
    out_memcfg: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Head-split concat-heads: split each (batch, s_tile) into ``head_groups``
    sub-units. For BGE-M3 B1/S512 with ``head_groups=num_heads=16`` that's 256
    work units instead of Track A's 16, lighting up the entire compute grid.
    """
    if out_memcfg is None:
        out_memcfg = ttnn.L1_MEMORY_CONFIG

    device = context.device()
    plan = _TrackAPlan.from_input(context)
    if head_groups is None:
        head_groups = plan.num_heads
    if plan.num_heads % head_groups != 0:
        raise ValueError(f"num_heads ({plan.num_heads}) must be divisible by head_groups ({head_groups})")
    heads_per_group = plan.num_heads // head_groups

    out_dtype = context.dtype
    out_shape = (plan.batch, 1, plan.seq_len, plan.num_heads * plan.head_dim)
    out_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, out_memcfg)

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_work_units_total = plan.num_blocks_total * head_groups
    num_cores, per_core = _split_work_to_cores(num_work_units_total, grid_x, grid_y)
    if num_cores == 0:
        raise RuntimeError("bge_concat_heads_headsplit: nothing to do")

    used_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for (cx, cy, _) in per_core]
    )

    cb_id = 0
    group_tiles = heads_per_group * plan.in0_w_tiles
    cb_total_tiles = group_tiles * 2
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
        plan.in0_h_tiles,
        plan.in0_w_tiles,
        plan.num_heads,
        plan.in0_HtWt,
        head_groups,
        heads_per_group,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(context).get_compile_time_args())

    reader_rt_per_core: list[tuple[tuple[int, int], list[int]]] = []
    work_cursor = 0
    for cx, cy, n_units in per_core:
        reader_rt_per_core.append(
            (
                (cx, cy),
                [
                    context.buffer_address(),
                    n_units,
                    work_cursor,
                ],
            )
        )
        work_cursor += n_units

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
        head_groups,
        heads_per_group,
        plan.in0_w_tiles,
        plan.per_tensor_tiles,
        plan.in0_h_tiles,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())

    writer_rt_per_core: list[tuple[tuple[int, int], list[int]]] = []
    work_cursor = 0
    for cx, cy, n_units in per_core:
        writer_rt_per_core.append(
            (
                (cx, cy),
                [
                    out_tensor.buffer_address(),
                    n_units,
                    work_cursor,
                ],
            )
        )
        work_cursor += n_units

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

    io_tensors = [context, out_tensor]
    ttnn.generic_op(io_tensors, program_descriptor)
    return out_tensor
