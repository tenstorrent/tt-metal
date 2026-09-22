# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-local head-split ``nlp_concat_heads`` for pplx-embed-v1-0.6B.

Stock ``ttnn.experimental.nlp_concat_heads`` splits work by ``(batch, seq_tile)``
only. At bs=1 / ISL=512 that is 16 work units on a 130-core Blackhole grid, so
the op runs at ~12% occupancy. This variant adds an inner head-group axis:
``(batch, seq_tile, head_group)``, giving ``16 * head_groups`` units.

The math is a pure tile-copy reorder, so output is bit-identical to the stock op.

Implemented as a ``ttnn.generic_op`` with kernels living next to this file, so
nothing under ``ttnn/`` or ``models/tt_transformers/`` is modified and the
optimization survives upstream rebases. This mirrors the approach taken by
``models/demos/wormhole/bge_m3/tt/custom_ops`` (whose kernels were themselves
adapted from the original in-tree Qwen3-Embedding head-split patch).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

READER_KERNEL = (
    "models/demos/blackhole/pplx_embed_0_6b/tt/custom_ops/fused_concat_heads/kernels/"
    "reader_concat_heads_headsplit.cpp"
)
WRITER_KERNEL = (
    "models/demos/blackhole/pplx_embed_0_6b/tt/custom_ops/fused_concat_heads/kernels/"
    "writer_concat_heads_headsplit.cpp"
)

TILE_H = 32
TILE_W = 32

_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}


def _split_work_to_cores(num_units: int, grid_x: int, grid_y: int):
    """Mirror ``tt::tt_metal::split_work_to_cores`` for the linear case.

    Cores are addressed in linear ID order as ``(i // grid_y, i % grid_y)`` and
    units are split as evenly as possible.
    """
    if num_units <= 0:
        return 0, []
    num_cores = min(grid_x * grid_y, num_units)
    base, extra = divmod(num_units, num_cores)
    cores = []
    for i in range(num_cores):
        n = base + (1 if i < extra else 0)
        cx, cy = divmod(i, grid_y)
        cores.append((cx, cy, n))
    return num_cores, cores


@dataclass(frozen=True)
class _Plan:
    batch: int
    num_heads: int
    seq_len: int
    head_dim: int
    in0_h_tiles: int
    in0_w_tiles: int
    in0_HtWt: int
    per_tensor_tiles: int
    num_blocks_total: int

    @classmethod
    def from_input(cls, context: ttnn.Tensor) -> "_Plan":
        b, num_heads, s, head_dim = (int(d) for d in context.padded_shape)
        in0_h_tiles = s // TILE_H
        in0_w_tiles = head_dim // TILE_W
        return cls(
            batch=b,
            num_heads=num_heads,
            seq_len=s,
            head_dim=head_dim,
            in0_h_tiles=in0_h_tiles,
            in0_w_tiles=in0_w_tiles,
            in0_HtWt=in0_h_tiles * in0_w_tiles,
            per_tensor_tiles=num_heads * in0_w_tiles,
            num_blocks_total=b * in0_h_tiles,
        )


def supported(context: ttnn.Tensor, head_groups: int | None = None) -> bool:
    """Can the head-split path handle this tensor? Callers fall back if not."""
    if context.is_sharded():
        return False
    if context.dtype not in _TILE_BYTES:
        return False
    if context.layout != ttnn.TILE_LAYOUT:
        return False
    shape = context.padded_shape
    if len(shape) != 4:
        return False
    _, num_heads, s, head_dim = (int(d) for d in shape)
    if s % TILE_H != 0 or head_dim % TILE_W != 0:
        return False
    groups = num_heads if head_groups is None else head_groups
    return groups > 0 and num_heads % groups == 0


def nlp_concat_heads_headsplit(
    context: ttnn.Tensor,
    *,
    head_groups: int | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Head-split concat-heads. Drop-in for ``ttnn.experimental.nlp_concat_heads``.

    Args:
        context: ``[B, num_heads, S, head_dim]``, TILE layout, interleaved.
        head_groups: how many slices to split the head axis into. Must divide
            ``num_heads``. Defaults to ``num_heads`` (finest granularity).
        memory_config: output memory config; defaults to L1 interleaved.

    Returns:
        ``[B, 1, S, num_heads * head_dim]``.
    """
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    device = context.device()
    plan = _Plan.from_input(context)
    if head_groups is None:
        head_groups = plan.num_heads
    if plan.num_heads % head_groups != 0:
        raise ValueError(f"num_heads ({plan.num_heads}) must be divisible by head_groups ({head_groups})")
    heads_per_group = plan.num_heads // head_groups

    out_dtype = context.dtype
    out_shape = (plan.batch, 1, plan.seq_len, plan.num_heads * plan.head_dim)
    out_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config
    )

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores, per_core = _split_work_to_cores(plan.num_blocks_total * head_groups, grid_x, grid_y)
    if num_cores == 0:
        raise RuntimeError("nlp_concat_heads_headsplit: nothing to do")

    used_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for (cx, cy, _) in per_core]
    )

    # Reader and writer share cb 0; double-buffered so the reader can stage the
    # next group while the writer drains the current one.
    cb_id = 0
    group_tiles = heads_per_group * plan.in0_w_tiles
    tile_size = _TILE_BYTES[out_dtype]
    cb_desc = ttnn.CBDescriptor(
        total_size=group_tiles * 2 * tile_size,
        core_ranges=used_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=out_dtype, page_size=tile_size)],
    )

    reader_ct = [
        plan.in0_h_tiles,
        plan.in0_w_tiles,
        plan.num_heads,
        plan.in0_HtWt,
        head_groups,
        heads_per_group,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(context).get_compile_time_args())

    writer_ct = [
        head_groups,
        heads_per_group,
        plan.in0_w_tiles,
        plan.per_tensor_tiles,
        plan.in0_h_tiles,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())

    reader_rt, writer_rt = [], []
    cursor = 0
    for cx, cy, n_units in per_core:
        reader_rt.append(((cx, cy), [context.buffer_address(), n_units, cursor]))
        writer_rt.append(((cx, cy), [out_tensor.buffer_address(), n_units, cursor]))
        cursor += n_units

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ],
        cbs=[cb_desc],
    )

    # io_tensors order binds buffer addresses to the TensorAccessor CT args.
    ttnn.generic_op([context, out_tensor], program_descriptor)
    return out_tensor
