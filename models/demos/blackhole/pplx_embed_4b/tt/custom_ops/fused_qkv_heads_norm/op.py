# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Head-split + per-head Q/K RMSNorm in one model-local ``ttnn.generic_op``.

Replaces three ops per layer (``nlp_create_qkv_heads`` + ``q_norm`` + ``k_norm``)
with one pass over the fused QKV activation. At bs=32 those three are 41 + 24 +
6 ms of kernel time per iteration, each a DRAM-bound pass over the same tensors.
Reader/writer reuse the head-split layout; a compute kernel normalises each Q
and K head over head_dim and applies gamma; V is copied through. Constants
(row-replicated gamma tiles, 1/head_dim scaler, eps) come from
``constants.make_norm_constants``.
"""
import os

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads.op import (
    _TILE_BYTES,
    _Plan,
    _split_work_to_cores,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
READER_KERNEL = os.path.join(_HERE, "kernels", "reader_qkv_heads_norm.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_qkv_heads_norm.cpp")
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_qkv_heads_norm.cpp")
_BF16_TILE = _TILE_BYTES[ttnn.bfloat16]


def nlp_create_qkv_heads_norm_headsplit(
    qkv_fused: ttnn.Tensor,
    gamma_q_tiles: ttnn.Tensor,
    gamma_k_tiles: ttnn.Tensor,
    scaler_tile: ttnn.Tensor,
    eps_tile: ttnn.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_groups: int | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """``qkv_fused``: ``[B, 1, S, (num_heads + 2*num_kv_heads) * head_dim]`` TILE bf16/bfp8.

    Returns ``(q, k, v)`` as ``[B, H, S, head_dim]`` with Q and K RMS-normalised per head.
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
    num_cores, per_core = _split_work_to_cores(plan.num_blocks_total * head_groups, int(grid.x), int(grid.y))
    if num_cores == 0:
        raise RuntimeError("nlp_create_qkv_heads_norm_headsplit: nothing to do")
    used_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for (cx, cy, _) in per_core]
    )

    Wt = plan.head_dim_tiles
    group_q_tiles = heads_per_group * plan.q_heads_per_kv * Wt
    group_kv_tiles = heads_per_group * Wt
    unit_tiles = group_q_tiles + 2 * group_kv_tiles
    tile_size = _TILE_BYTES[out_dtype]

    def cb(index, tiles, dtype, tsize):
        return ttnn.CBDescriptor(
            total_size=tiles * tsize,
            core_ranges=used_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tsize)],
        )

    cbs = [
        cb(0, unit_tiles * 2, out_dtype, tile_size),  # fused QKV in (double-buffered unit)
        cb(16, unit_tiles * 2, out_dtype, tile_size),  # normalised Q|K|V out
        cb(1, Wt, ttnn.bfloat16, _BF16_TILE),  # gamma_q tiles (resident)
        cb(2, Wt, ttnn.bfloat16, _BF16_TILE),  # gamma_k tiles (resident)
        cb(3, 1, ttnn.bfloat16, _BF16_TILE),  # 1/head_dim scaler (resident)
        cb(4, 1, ttnn.bfloat16, _BF16_TILE),  # eps (resident)
        cb(5, Wt, ttnn.bfloat16, _BF16_TILE),  # x^2
        cb(6, 1, ttnn.bfloat16, _BF16_TILE),  # mean-square (row values in col 0)
        cb(7, 1, ttnn.bfloat16, _BF16_TILE),  # rsqrt
        cb(8, Wt, ttnn.bfloat16, _BF16_TILE),  # x * inv
    ]

    reader_ct = [
        plan.q_heads_per_kv,
        plan.num_kv_heads,
        Wt,
        plan.in_w_tiles,
        plan.seq_tiles,
        head_groups,
        heads_per_group,
    ]
    for t in (qkv_fused, gamma_q_tiles, gamma_k_tiles, scaler_tile, eps_tile):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [plan.q_heads_per_kv, heads_per_group, Wt]
    writer_ct = [
        plan.seq_tiles,
        Wt,
        plan.q_out_HtWt,
        plan.num_q_heads,
        plan.num_kv_heads,
        plan.q_heads_per_kv,
        head_groups,
        heads_per_group,
        plan.seq_tiles,
    ]
    for t in (q_tensor, k_tensor, v_tensor):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = [], [], []
    cursor = 0
    for cx, cy, n_units in per_core:
        core = (cx, cy)
        reader_rt.append(
            (
                core,
                [
                    qkv_fused.buffer_address(),
                    gamma_q_tiles.buffer_address(),
                    gamma_k_tiles.buffer_address(),
                    scaler_tile.buffer_address(),
                    eps_tile.buffer_address(),
                    n_units,
                    cursor,
                ],
            )
        )
        compute_rt.append((core, [n_units]))
        writer_rt.append(
            (core, [q_tensor.buffer_address(), k_tensor.buffer_address(), v_tensor.buffer_address(), n_units, cursor])
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
                kernel_source=COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=used_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                    bfp8_pack_precise=False,
                ),
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
        cbs=cbs,
    )
    ttnn.generic_op(
        [qkv_fused, gamma_q_tiles, gamma_k_tiles, scaler_tile, eps_tile, q_tensor, k_tensor, v_tensor],
        program_descriptor,
    )
    return q_tensor, k_tensor, v_tensor
