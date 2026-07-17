# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 exact-shape encoder SDPA parity scaffold.

The experimental path mirrors the production ``SDPAProgramFactory`` for the
retained N300 DP=2 shape, while dispatching through ``ttnn.generic_op`` and
model-local JIT entrypoints.  It is intentionally not wired into attention.py;
the next owner must validate it on silicon before enabling it in the model.

No code in this directory requires rebuilding ``_ttnn.so``.  The C++ kernel
entrypoints are compiled by the normal device-kernel JIT on first use.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import ttnn

from .config import INACTIVE_CB, EncoderSDPAConfig, EncoderSDPAPlan, validate_encoder_sdpa_inputs

KERNEL_ROOT = "models/demos/wormhole/bge_m3/tt/custom_ops/encoder_sdpa/kernels"
READER_KERNEL = f"{KERNEL_ROOT}/reader.cpp"
WRITER_KERNEL = f"{KERNEL_ROOT}/writer.cpp"
COMPUTE_KERNEL = f"{KERNEL_ROOT}/compute.cpp"

# The locally-copied kernel bodies keep the production relative includes
# (dataflow_common.hpp, windowed_mask_gen.hpp, compute_common.hpp,
# compute_streaming.hpp). Point the JIT compiler at the production kernel dirs
# so those headers resolve from the model-local copies.
_SDPA_KERNEL_DIR = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"
DATAFLOW_INCLUDE_PATHS = [f"{_SDPA_KERNEL_DIR}/dataflow"]
COMPUTE_INCLUDE_PATHS = [f"{_SDPA_KERNEL_DIR}/compute"]

# Exact contiguous CB assignment for the unmasked, non-causal, FP32-dest path.
CB_Q = 0
CB_K = 1
CB_V = 2
CB_IDENTITY = 3
CB_COL_IDENTITY = 4
CB_QK = 5
CB_OUT_A = 6
CB_OUT_B = 7
CB_MAX_A = 8
CB_MAX_B = 9
CB_SUM_A = 10
CB_SUM_B = 11
CB_EXP_MAX_DIFF = 12
CB_OUT = 13
CB_RECIP_SCRATCH = 14  # streaming-only: 1-tile recip scratch for normalize_row_streaming


@dataclass(frozen=True)
class EncoderSDPABuild:
    descriptor: ttnn.ProgramDescriptor
    output: ttnn.Tensor
    io_tensors: list[ttnn.Tensor]
    plan: EncoderSDPAPlan


def _u32_from_float(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _tile_size_bytes(dtype: ttnn.DataType) -> int:
    return {
        ttnn.bfloat16: 2048,
        ttnn.bfloat8_b: 1088,
        ttnn.bfloat4_b: 576,
        ttnn.float32: 4096,
    }[dtype]


def _accessor_args(tensor: ttnn.Tensor) -> list[int]:
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def _cb_descriptor(
    cb_id: int,
    num_tiles: int,
    dtype: ttnn.DataType,
    core_grid: ttnn.CoreRangeSet,
) -> ttnn.CBDescriptor:
    page_size = _tile_size_bytes(dtype)
    return ttnn.CBDescriptor(
        total_size=num_tiles * page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_id,
                data_format=dtype,
                page_size=page_size,
            )
        ],
    )


def _compile_defines(plan: EncoderSDPAPlan) -> list[tuple[str, str]]:
    # Fully plan-derived from SDPAProgramFactory formulas (granularities scale
    # with q_chunk/k_chunk). At q128/k2048 this yields the parity-verified set
    # STATS=4 SUB_EXP=4 MUL_BCAST=4 DHT=2 REDUCE=2. exp_approx_mode=true.
    # KernelDescriptor(defines=...) expects a Sequence[tuple[str, str]], not a dict.
    defs = [
        ("STATS_GRANULARITY", str(plan.stats_granularity)),
        ("SUB_EXP_GRANULARITY", str(plan.sub_exp_granularity)),
        ("MUL_BCAST_GRANULARITY", str(plan.mul_bcast_granularity)),
        ("DHT_GRANULARITY", str(plan.dht_granularity)),
        ("REDUCE_GRANULARITY", str(plan.reduce_granularity)),
        ("EXP_APPROX_MODE", "1"),
    ]
    return defs


def _reader_compile_args(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    plan: EncoderSDPAPlan,
) -> list[int]:
    c = plan.config
    args = [
        c.batch,
        c.num_q_heads,
        c.num_kv_heads,
        c.num_kv_heads,  # NVH
        plan.sq_tiles,
        plan.sk_tiles,
        plan.sq_tiles,  # valid_Sqt
        plan.sk_tiles,  # valid_Skt
        plan.head_dim_tiles,
        plan.head_dim_tiles,  # vDHt
        plan.q_chunk_tiles,
        plan.q_num_chunks,
        plan.k_chunk_tiles,
        plan.k_num_chunks,
        plan.num_cores,
        0,  # is_causal
        0,  # use_provided_mask
        0,  # broadcast_provided_mask_batch
        0,  # broadcast_provided_mask_heads
        0,  # use_padded_mask
        0,  # is_chunked
        0,  # block_size_t
        0,  # page_table_stick_size
        0,  # use_attention_sink
        0,  # use_mla
        0,  # mla_kv_overlap
        plan.qk_out_subblock[0],  # qk_subblock_h (plan-derived)
        0,  # sliding_window_size
        int(plan.config.use_streaming),  # use_streaming_compute
        0,  # sender semaphore id; forwarding is inactive for this work split
        1,  # receiver semaphore id
        2,  # valid semaphore id
        0,  # mcast_enabled
        0,  # use_zigzag_balancing
    ]
    args.extend(_accessor_args(q))
    args.extend(_accessor_args(k))
    args.extend(_accessor_args(v))
    # Production passes null TensorAccessorArgs for optional inputs.  Reusing a
    # valid interleaved accessor layout preserves the compile-time offset chain;
    # runtime addresses remain zero and all corresponding paths are constexpr-off.
    for _ in range(4):  # mask, page table, attention sink, chunk-start tensor
        args.extend(_accessor_args(q))
    args.extend([CB_Q, CB_K, CB_V, INACTIVE_CB, INACTIVE_CB, INACTIVE_CB, INACTIVE_CB, INACTIVE_CB])
    return args


def _writer_compile_args(output: ttnn.Tensor, plan: EncoderSDPAPlan) -> list[int]:
    c = plan.config
    packed_bf16_one = 0x3F803F80
    args = [
        c.batch,
        c.num_q_heads,
        c.num_kv_heads,
        plan.sq_tiles,
        plan.sq_tiles,  # valid_Sqt
        c.kv_seq_len,  # unpadded Sk in elements
        plan.head_dim_tiles,
        plan.head_dim_tiles,
        plan.q_chunk_tiles,
        plan.q_num_chunks,
        plan.k_chunk_tiles,
        plan.k_num_chunks,
        packed_bf16_one,
        _u32_from_float(c.scale),
        plan.num_cores,
        0,  # is_causal
        0,  # use_provided_mask
        0,  # use_padded_mask
        0,  # is_chunked
        0,  # sliding_window_size
        0,  # use_lightweight_mask
        int(plan.config.use_streaming),  # use_streaming_compute
        plan.out_out_subblock[0],  # out_subblock_h (plan-derived)
        0,  # k_partial_col
        0,  # use_zigzag_balancing
        0,  # use_windowed_mask
    ]
    args.extend(_accessor_args(output))
    args.extend(_accessor_args(output))  # inactive cu_window accessor placeholder
    args.extend([INACTIVE_CB, CB_IDENTITY, CB_COL_IDENTITY, INACTIVE_CB, CB_OUT, CB_Q])
    return args


def _compute_compile_args(output: ttnn.Tensor, plan: EncoderSDPAPlan) -> list[int]:
    c = plan.config
    args = [
        c.batch,
        c.num_q_heads,
        c.num_kv_heads,
        plan.sk_tiles,
        plan.head_dim_tiles,
        plan.head_dim_tiles,
        plan.q_chunk_tiles,
        plan.q_num_chunks,
        plan.k_chunk_tiles,
        plan.k_num_chunks,
        plan.head_dim_tiles,  # qk_in0_block_w
        plan.qk_out_subblock[1],  # qk_out_subblock_w (plan returns (h, w))
        plan.qk_out_subblock[0],  # qk_out_subblock_h
        plan.qk_in0_num_subblocks,  # = q_chunk_tiles / qk_out_subblock_h
        plan.qk_in1_num_subblocks,  # = k_chunk_tiles / qk_out_subblock_w
        1,  # qk_num_blocks (DHt / qk_in0_block_w = 2/2)
        plan.k_chunk_tiles,  # out_in0_block_w
        plan.out_out_subblock[1],  # out_out_subblock_w
        plan.out_out_subblock[0],  # out_out_subblock_h
        plan.out_in0_num_subblocks,  # = q_chunk_tiles / out_out_subblock_h
        plan.out_in1_num_subblocks,  # = vDHt / out_out_subblock_w
        1,  # out_num_blocks (Sk_chunk_t / out_in0_block_w)
        plan.num_cores,
        0,  # is_causal
        0,  # use_provided_mask
        0,  # use_padded_mask
        0,  # is_chunked
        _u32_from_float(c.scale),
        0,  # sliding_window_size
        0,  # use_attention_sink
        int(plan.config.use_streaming),  # use_streaming_compute
        plan.sk_tiles,  # valid_Skt
        0,  # k_partial_col
        0,  # use_zigzag_balancing
        CB_Q,
        CB_K,
        CB_V,
        INACTIVE_CB,
        INACTIVE_CB,
        CB_IDENTITY,
        CB_COL_IDENTITY,
        INACTIVE_CB,
        CB_RECIP_SCRATCH if plan.config.use_streaming else INACTIVE_CB,  # cb_recip_scratch (offset+8)
        CB_OUT,
        CB_QK,
        CB_OUT_A,
        CB_OUT_B,
        CB_MAX_A,
        CB_MAX_B,
        CB_SUM_A,
        CB_SUM_B,
        CB_EXP_MAX_DIFF,
    ]
    args.extend(_accessor_args(output))
    return args


def _runtime_args(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    output: ttnn.Tensor,
    plan: EncoderSDPAPlan,
) -> tuple[list, list, list]:
    reader_args = []
    writer_args = []
    compute_args = []

    for core_id in range(plan.num_cores):
        x = core_id % plan.config.grid_x
        y = core_id // plan.config.grid_x
        global_q_start, global_q_count = plan.global_q_range(core_id)
        coord = (x, y)

        # Since every core owns exactly three complete Q heads, no head spans
        # cores and every KV chain field remains zero.
        chain_metadata = [0] * 14
        reader_args.append(
            (
                coord,
                [
                    q.buffer_address(),
                    k.buffer_address(),
                    v.buffer_address(),
                    0,  # mask
                    0,  # page table
                    0,  # attention sink
                    0,  # chunk-start tensor
                    core_id,
                    1,  # num_phases
                    0,  # chunked_q_chunk_offset
                    0,  # read_offset
                    *chain_metadata,
                    global_q_start,
                    global_q_count,
                ],
            )
        )
        writer_args.append(
            (
                coord,
                [
                    output.buffer_address(),
                    core_id,
                    1,  # num_phases
                    0,  # use_chunk_start_idx_tensor
                    0,  # phase-1 chunk offset
                    0,  # phase-1 write offset
                    0,  # phase-2 chunk offset (reserved slot)
                    0,  # phase-2 write offset (reserved slot)
                    global_q_start,
                    global_q_count,
                    0,  # cu_window_seqlens address
                    0,  # cu_window_seqlens elements
                ],
            )
        )
        compute_args.append(
            (
                coord,
                [
                    core_id,
                    1,  # num_phases
                    0,  # use_chunk_start_idx_tensor
                    0,  # phase-1 chunk offset
                    0,  # phase-2 chunk offset (reserved slot)
                    global_q_start,
                    global_q_count,
                ],
            )
        )
    return reader_args, writer_args, compute_args


def build_encoder_sdpa_descriptor(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    config: EncoderSDPAConfig = EncoderSDPAConfig(),
    output_mem_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> EncoderSDPABuild:
    """Build the unverified parity descriptor without launching it.

    The descriptor intentionally omits production's three KV-forwarding
    semaphores: exact BGE work partitioning gives each core three complete heads,
    so every runtime ``is_chain_participant`` flag is zero and no semaphore path
    executes.  The first silicon test must verify this assumption before model
    integration.
    """
    plan = validate_encoder_sdpa_inputs(q, k, v, config)
    import os as _os

    if _os.environ.get("BGE_SDPA_LOG_CFG", "0") == "1":
        from loguru import logger as _lg

        _lg.info(
            f"ENCODER_SDPA cfg: q_chunk={config.q_chunk_size} k_chunk={config.k_chunk_size} "
            f"score_cb={'bf8' if config.score_cb_bf8 else 'bf16'} fp32_dest={config.fp32_dest_acc_en} "
            f"full_sync={config.dst_full_sync_en} DST_SIZE={plan.DST_SIZE} "
            f"q_num_chunks={plan.q_num_chunks} k_num_chunks={plan.k_num_chunks} "
            f"qk_in1_nsb={plan.qk_in1_num_subblocks} streaming={config.use_streaming}"
        )
    device = q.device()
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(config.output_shape),
        q.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_mem_config,
    )

    core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(config.grid_x - 1, config.grid_y - 1),
            )
        ]
    )

    cbs = [
        # CB depths derive from the plan so q_chunk/k_chunk can be swept.
        # Defaults (q128/k2048): Q=16, K=V=256, QK=256 — identical to the
        # parity-verified sizes. K/V are double-buffered (x2), Q holds 2 chunks.
        _cb_descriptor(CB_Q, plan.config.q_buffer_depth * plan.q_chunk_tiles * plan.head_dim_tiles, ttnn.bfloat8_b, core_grid),
        _cb_descriptor(CB_K, plan.config.k_buffer_depth * plan.k_chunk_tiles * plan.head_dim_tiles, ttnn.bfloat4_b, core_grid),
        _cb_descriptor(CB_V, plan.config.v_buffer_depth * plan.k_chunk_tiles * plan.head_dim_tiles, ttnn.bfloat8_b, core_grid),
        _cb_descriptor(CB_IDENTITY, 1, ttnn.bfloat16, core_grid),
        _cb_descriptor(CB_COL_IDENTITY, 1, ttnn.bfloat16, core_grid),
        _cb_descriptor(
            CB_QK,
            plan.q_chunk_tiles * plan.k_chunk_tiles,
            ttnn.bfloat8_b if plan.config.score_cb_bf8 else ttnn.bfloat16,
            core_grid,
        ),
        # out_im/out = Sq_chunk_t*vDHt; max/sum/exp_max_diff = statistics_tiles
        # (=Sq_chunk_t). All plan-derived; q128 defaults = 8/8/4/4/4/4/4/8.
        _cb_descriptor(CB_OUT_A, plan.out_im_tiles, ttnn.bfloat16, core_grid),
        _cb_descriptor(CB_OUT_B, plan.out_im_tiles, ttnn.bfloat16, core_grid),
        _cb_descriptor(CB_MAX_A, plan.statistics_tiles, ttnn.bfloat16, core_grid),
        _cb_descriptor(CB_MAX_B, plan.statistics_tiles, ttnn.bfloat16, core_grid),
        _cb_descriptor(CB_SUM_A, plan.statistics_tiles, ttnn.bfloat16, core_grid),
        _cb_descriptor(CB_SUM_B, plan.statistics_tiles, ttnn.bfloat16, core_grid),
        _cb_descriptor(CB_EXP_MAX_DIFF, plan.statistics_tiles, ttnn.bfloat16, core_grid),
        _cb_descriptor(
            CB_OUT,
            plan.streaming_cb_out_tiles if plan.config.use_streaming else plan.out_im_tiles,
            ttnn.bfloat8_b,
            core_grid,
        ),
    ]
    if plan.config.use_streaming:
        # Streaming-only 1-tile recip scratch (im_df = Float16_b in the factory).
        cbs.append(_cb_descriptor(CB_RECIP_SCRATCH, 1, ttnn.bfloat16, core_grid))

    reader_rt, writer_rt, compute_rt = _runtime_args(q, k, v, output, plan)
    defines = _compile_defines(plan)

    reader = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=_reader_compile_args(q, k, v, plan),
        runtime_args=reader_rt,
        defines=defines,
        config=ttnn.ReaderConfigDescriptor(),
        compiler_include_paths=DATAFLOW_INCLUDE_PATHS,
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=_writer_compile_args(output, plan),
        runtime_args=writer_rt,
        defines=defines,
        config=ttnn.WriterConfigDescriptor(),
        compiler_include_paths=DATAFLOW_INCLUDE_PATHS,
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=_compute_compile_args(output, plan),
        runtime_args=compute_rt,
        defines=defines,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=plan.config.fp32_dest_acc_en,
            dst_full_sync_en=plan.config.dst_full_sync_en,
        ),
        compiler_include_paths=COMPUTE_INCLUDE_PATHS,
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs)
    return EncoderSDPABuild(
        descriptor=descriptor,
        output=output,
        io_tensors=[q, k, v, output],
        plan=plan,
    )


def bge_encoder_sdpa_experimental(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    config: EncoderSDPAConfig = EncoderSDPAConfig(),
    output_mem_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Launch the unverified model-local descriptor.

    This function is intentionally not imported or called by ``attention.py``.
    Use only from a dedicated parity probe until PCC, device time, repeat-cache,
    and trace replay all match production SDPA.
    """
    build = build_encoder_sdpa_descriptor(
        q,
        k,
        v,
        config=config,
        output_mem_config=output_mem_config,
    )
    ttnn.generic_op(build.io_tensors, build.descriptor)
    return build.output


def bge_encoder_sdpa_stock(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    compute_kernel_config,
    output_mem_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    config: EncoderSDPAConfig = EncoderSDPAConfig(),
) -> ttnn.Tensor:
    """Stock baseline with the same exact encoder contract."""
    validate_encoder_sdpa_inputs(q, k, v, config)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(config.grid_x, config.grid_y),
        q_chunk_size=config.q_chunk_size,
        k_chunk_size=config.k_chunk_size,
    )
    return ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        attn_mask=None,
        scale=config.scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=output_mem_config,
    )


__all__ = [
    "EncoderSDPABuild",
    "EncoderSDPAConfig",
    "bge_encoder_sdpa_experimental",
    "bge_encoder_sdpa_stock",
    "build_encoder_sdpa_descriptor",
]
