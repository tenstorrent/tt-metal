# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-local head-split ``nlp_create_qkv_heads`` for pplx-embed-v1-4B.

Stock ``ttnn.experimental.nlp_create_qkv_heads`` splits work by
``(batch, seq_tile)``. At bs=1 / ISL=512 that is 16 work units on a 130-core
Blackhole grid. This variant adds a head-group axis so each unit handles
``heads_per_group`` KV heads (and their ``q_heads_per_kv`` Q heads) for one
``(batch, seq_tile)``.

pplx-embed-v1-4B is GQA: 16 Q heads over 8 KV heads, head_dim 128. With
``head_groups = num_kv_heads = 8`` that is 16 * 8 = 128 work units, each moving
8 Q + 4 K + 4 V tiles.

The math is a pure tile-copy reorder, so output is bit-identical to the stock op.

Implemented as a ``ttnn.generic_op`` so nothing under ``ttnn/`` or
``models/tt_transformers/`` is touched and the optimization survives upstream
rebases. Mirrors ``models/demos/wormhole/bge_m3/tt/custom_ops``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ttnn

READER_KERNEL = (
    "models/demos/blackhole/pplx_embed_4b/tt/custom_ops/fused_qkv_heads/kernels/" "reader_qkv_heads_headsplit.cpp"
)
WRITER_KERNEL = (
    "models/demos/blackhole/pplx_embed_4b/tt/custom_ops/fused_qkv_heads/kernels/" "writer_qkv_heads_headsplit.cpp"
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
    """Mirror ``tt::tt_metal::split_work_to_cores`` for the linear case."""
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
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    q_heads_per_kv: int
    head_dim: int
    in_w_tiles: int
    seq_tiles: int
    head_dim_tiles: int
    q_out_HtWt: int
    num_blocks_total: int

    @classmethod
    def from_input(cls, qkv_fused: ttnn.Tensor, num_heads: int, num_kv_heads: int) -> "_Plan":
        b, _, s, w = (int(d) for d in qkv_fused.padded_shape)
        head_dim = w // (num_heads + 2 * num_kv_heads)
        seq_tiles = s // TILE_H
        head_dim_tiles = head_dim // TILE_W
        return cls(
            batch=b,
            seq_len=s,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            q_heads_per_kv=num_heads // num_kv_heads,
            head_dim=head_dim,
            in_w_tiles=w // TILE_W,
            seq_tiles=seq_tiles,
            head_dim_tiles=head_dim_tiles,
            q_out_HtWt=seq_tiles * head_dim_tiles,
            num_blocks_total=b * seq_tiles,
        )


def supported(
    qkv_fused: ttnn.Tensor,
    num_heads: int,
    num_kv_heads: int,
    transpose_k_heads: bool,
    head_groups: int | None = None,
) -> bool:
    """Can the head-split path handle this call? Callers fall back if not.

    The kernels write Q/K/V in their natural ``[B, H, S, D]`` layout, so the
    transposed-K variant is out of scope and must use the stock op.
    """
    if transpose_k_heads:
        return False
    if qkv_fused.is_sharded():
        return False
    if qkv_fused.dtype not in _TILE_BYTES:
        return False
    if qkv_fused.layout != ttnn.TILE_LAYOUT:
        return False
    shape = qkv_fused.padded_shape
    if len(shape) != 4 or int(shape[1]) != 1:
        return False
    _, _, s, w = (int(d) for d in shape)
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return False
    total_heads = num_heads + 2 * num_kv_heads
    if w % total_heads != 0:
        return False
    head_dim = w // total_heads
    if s % TILE_H != 0 or head_dim % TILE_W != 0:
        return False
    groups = num_kv_heads if head_groups is None else head_groups
    return groups > 0 and num_kv_heads % groups == 0


def nlp_create_qkv_heads_headsplit(
    qkv_fused: ttnn.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_groups: int | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Head-split QKV head creation. Drop-in for the stock op with
    ``transpose_k_heads=False``.

    Args:
        qkv_fused: ``[B, 1, S, (num_heads + 2*num_kv_heads) * head_dim]``.
        num_heads: Q head count (pplx-embed-4B: 16).
        num_kv_heads: KV head count (pplx-embed-4B: 8).
        head_groups: slices of the KV-head axis. Must divide ``num_kv_heads``.
            Defaults to ``num_kv_heads``.
        memory_config: output memory config; defaults to DRAM interleaved.

    Returns:
        ``(q, k, v)`` with shapes ``[B, num_heads, S, head_dim]`` and
        ``[B, num_kv_heads, S, head_dim]`` for K/V.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    device = qkv_fused.device()
    plan = _Plan.from_input(qkv_fused, num_heads, num_kv_heads)
    if head_groups is None:
        _env = os.getenv("QWEN_HEADSPLIT_GROUPS_QKV")
        head_groups = int(_env) if _env else plan.num_kv_heads
    if plan.num_kv_heads % head_groups != 0:
        raise ValueError(f"num_kv_heads ({plan.num_kv_heads}) must be divisible by head_groups ({head_groups})")
    heads_per_group = plan.num_kv_heads // head_groups

    out_dtype = qkv_fused.dtype
    q_shape = (plan.batch, plan.num_q_heads, plan.seq_len, plan.head_dim)
    kv_shape = (plan.batch, plan.num_kv_heads, plan.seq_len, plan.head_dim)
    q_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(q_shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    k_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(kv_shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    v_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(kv_shape), out_dtype, ttnn.TILE_LAYOUT, device, memory_config)

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores, per_core = _split_work_to_cores(plan.num_blocks_total * head_groups, grid_x, grid_y)
    if num_cores == 0:
        raise RuntimeError("nlp_create_qkv_heads_headsplit: nothing to do")

    used_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for (cx, cy, _) in per_core]
    )

    # Reader pushes Q then K then V per work unit; the writer drains in the same
    # order. Size the CB for a whole unit (Q + K + V) double-buffered so the
    # reader never blocks mid-unit waiting on the writer.
    cb_id = 1
    group_q_tiles = heads_per_group * plan.q_heads_per_kv * plan.head_dim_tiles
    group_kv_tiles = heads_per_group * plan.head_dim_tiles
    unit_tiles = group_q_tiles + 2 * group_kv_tiles
    tile_size = _TILE_BYTES[out_dtype]
    cb_desc = ttnn.CBDescriptor(
        total_size=unit_tiles * 2 * tile_size,
        core_ranges=used_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=out_dtype, page_size=tile_size)],
    )

    reader_ct = [
        plan.q_heads_per_kv,
        plan.num_kv_heads,
        plan.head_dim_tiles,
        plan.in_w_tiles,
        plan.seq_tiles,
        head_groups,
        heads_per_group,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(qkv_fused).get_compile_time_args())

    writer_ct = [
        plan.seq_tiles,
        plan.head_dim_tiles,
        plan.q_out_HtWt,
        plan.num_q_heads,
        plan.num_kv_heads,
        plan.q_heads_per_kv,
        head_groups,
        heads_per_group,
        plan.seq_tiles,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(q_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(k_tensor).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(v_tensor).get_compile_time_args())

    reader_rt, writer_rt = [], []
    cursor = 0
    for cx, cy, n_units in per_core:
        reader_rt.append(((cx, cy), [qkv_fused.buffer_address(), n_units, cursor]))
        writer_rt.append(
            (
                (cx, cy),
                [
                    q_tensor.buffer_address(),
                    k_tensor.buffer_address(),
                    v_tensor.buffer_address(),
                    n_units,
                    cursor,
                ],
            )
        )
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
    ttnn.generic_op([qkv_fused, q_tensor, k_tensor, v_tensor], program_descriptor)
    return q_tensor, k_tensor, v_tensor
