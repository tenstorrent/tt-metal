# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-input, rectangular adapter for the seven pinned numerical recipes.

Default blocking remains Q256/K512/D128. Optional chunk sizes and reader
barriers support controlled exact-shape tuning without changing numerical
recipes or input buffer depth. This is an experimental model adapter, not
production ring SDPA dispatch.
"""

import importlib.util
import math
from pathlib import Path
import struct

import ttnn

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "bfp4-lofi-v2"
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/fullchip/"
VARIANTS = {
    "A": "main",
    "B": "fast",
    "C": "balanced",
    "D": "accurate",
    "E": "lofi_fast_b8",
    "F": "lofi_fp32_b8",
    "G": "lofi_fast_b4",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = load_module("flux2_frontier_round", RESEARCH / "preprocess.py")
B4 = load_module("flux2_frontier_b4", RESEARCH / "bfp4_round.py")


def prepare(device, tensor, variant, *, is_q, cores):
    """Quantize on the owning rank, before KV communication. No CPU copy."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown attention variant: {variant}")
    if variant in "ABCD":
        return tensor
    if tensor.get_dtype() != ttnn.bfloat16:
        raise ValueError("Preprocessing requires original BF16 inputs")
    if variant == "G" and not is_q:
        out, invoke, _ = B4.build(device, tensor, ncores=cores)
    else:
        out, invoke, _ = PREP.build(
            device, tensor, bits=7 if is_q else 5, output_format="bf16" if is_q else "b8", ncores=cores
        )
    invoke()
    return out


def recipe(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown attention variant: {variant}")
    fp32, fast, lofi = variant in "CDF", variant in "BEG", variant in "EFG"
    defines = dict(
        EXP_APPROX_MODE="1",
        STATS_GRANULARITY="4" if fp32 else "8",
        SUB_EXP_GRANULARITY="4" if fp32 else "8",
        MUL_BCAST_GRANULARITY="4" if fp32 else "8",
        DHT_GRANULARITY="4",
        REDUCE_GRANULARITY="2" if fp32 else "4",
    )
    if variant == "A":
        defines["RESIDENT_MAIN"] = "1"
    if fast:
        defines.update(
            SDPA_STREAMING_ACCURACY="1",
            SDPA_STREAMING_NUMERATOR_COMPENSATION="1",
            SDPA_OUT_A_CB="8",
            SDPA_OUT_B_CB="9",
            SDPA_LOFI_FIX_CORRECTION="1",
        )
    if fp32:
        defines.update(SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1")
    if variant == "C":
        defines["SDPA_QK4"] = "1"
    if variant == "D":
        defines.update(
            {
                name: "1"
                for name in (
                    "SDPA_FP32_FUSED_EXP",
                    "SDPA_FP32_REUSE_EXP",
                    "SDPA_FP32_EXTRA_CONST",
                    "SDPA_FP32_PAIRED_UNPACK",
                    "SDPA_FP32_PAIRED_PACK",
                    "SDPA_FP32_L1_SUB",
                    "SDPA_FP32_L1_MACRO",
                    "SDPA_FP32_REFINE_MACRO",
                )
            }
        )
        defines.update(SDPA_DIAG_EXP_MODE="4", SDPA_FP32_SUB_BATCH="2", SDPA_DENOM_PHASES="2", SDPA_DIAG_SCORE_CB="7")
    if lofi:
        defines["SDPA_MATCH_HIFI2" if fp32 else "SDPA_LOFI_SAFE_RESCALE"] = "1"
    if variant == "F":
        defines["SDPA_LOFI_NATIVE_EXP"] = "1"
    fidelity = (
        ttnn.MathFidelity.LoFi if lofi else ttnn.MathFidelity.HiFi4 if variant == "D" else ttnn.MathFidelity.HiFi2
    )
    return fp32, fast, defines, fidelity


def attention(
    device,
    q,
    k,
    v,
    variant,
    *,
    max_cores=None,
    logical_k=None,
    q_chunk_size=256,
    k_chunk_size=512,
    reader_barrier_tiles=2,
):
    """Run prepared Q/K/V. Shapes are per-device, with no implicit padding."""
    if len(q.shape) != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("Only [1,H,Q,128] inputs are supported")
    heads, nq, nk = q.shape[1], q.shape[2], k.shape[2]
    if tuple(k.shape) != tuple(v.shape) or tuple(k.shape)[:2] != (1, heads) or k.shape[-1] != 128:
        raise ValueError("Q/K/V head and width mismatch")
    if q_chunk_size < 64 or k_chunk_size < 128 or q_chunk_size % 64 or k_chunk_size % 128:
        raise ValueError("Q chunks must be multiples of 64, K chunks multiples of 128")
    if nq <= 0 or nk <= 0 or nq % q_chunk_size or nk % k_chunk_size:
        raise ValueError(f"Unsupported Q/KV lengths: {nq}/{nk}; no padding or fallback")
    kv_dtype, kv_bytes = (
        (ttnn.bfloat4_b, 576)
        if variant == "G"
        else (ttnn.bfloat8_b, 1088) if variant in "EF" else (ttnn.bfloat16, 2048)
    )
    if q.get_dtype() != ttnn.bfloat16 or k.get_dtype() != kv_dtype or v.get_dtype() != kv_dtype:
        raise ValueError("Inputs do not match the selected recipe's prepared formats")
    for tensor in (q, k, v):
        if tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("Attention adapter requires interleaved tiled DRAM inputs")
    fp32, fast, defines, fidelity = recipe(variant)
    defines["SDPA_K_CHUNK_TILES"] = str(k_chunk_size // 32)
    padded = logical_k is not None and logical_k != nk
    if padded:
        if not 0 < nk - logical_k < 32:
            raise ValueError("Only a partially padded final K tile is supported")
        defines["SDPA_K_PARTIAL_COL"] = str(logical_k % 32)
    hardware = device.compute_with_storage_grid_size()
    jobs_per_head, chunks, qt = nq // q_chunk_size, nk // k_chunk_size, q_chunk_size // 32
    kt = k_chunk_size // 32
    budget = min(max_cores or hardware.x * (hardware.y - 1), hardware.x * hardware.y)
    chain = min(budget // heads, jobs_per_head)
    if chain < 1:
        raise ValueError("Not enough compute cores for a chain per head")
    cores = chain * heads
    coords = [ttnn.CoreCoord(i % hardware.x, i // hardware.x) for i in range(cores)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    physical = [device.worker_core_from_logical_core(c) for c in coords]
    out = ttnn.allocate_tensor_on_device(q.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    state_format, state_bytes = (ttnn.float32, 4096) if fp32 else (ttnn.bfloat16, 2048)
    slots, stride = (1 if fp32 else 2), (2 if fast else 1)
    specs = [
        (0, 2 * qt * 4, 2048, ttnn.bfloat16),
        (1, kt * 4 * slots, kv_bytes, kv_dtype),
        (2, kt * 4 * slots, kv_bytes, kv_dtype),
        (3, 1, 2048, ttnn.bfloat16),
        (4, 1, 2048, ttnn.bfloat16),
        (5, 1, state_bytes, state_format),
        (6, qt * kt, state_bytes, state_format),
        (8, qt * 4 * stride, state_bytes, state_format),
        (9, qt * 4 * stride, state_bytes, state_format),
        (10, qt, 2048, ttnn.bfloat16),
        (11, qt, 2048, ttnn.bfloat16),
        (12, qt * stride, state_bytes, state_format),
        (13, qt * stride, state_bytes, state_format),
        (14, qt, state_bytes, state_format),
        (16, 8 if fp32 else 16, 2048, ttnn.bfloat16),
    ]
    if padded:
        specs.append((15, 2, 2048, ttnn.bfloat16))
    cbs = []
    for index, count, size, fmt in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)]
        if index == 6 and fp32:
            formats.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=fmt, page_size=size))
        cbs.append(ttnn.CBDescriptor(total_size=count * size, core_ranges=grid, format_descriptors=formats))
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32, dst_full_sync_en=False, math_approx_mode=True
    )
    if fp32:
        pd = ttnn._ttnn.program_descriptor
        modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
        for cb in (5, 7, 8, 9, 12, 13, 14):
            modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    read, write, compute = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(coords):
        head, rank = divmod(i, chain)
        count = jobs_per_head // chain + (rank < jobs_per_head % chain)
        offset = head * jobs_per_head + rank * (jobs_per_head // chain) + min(rank, jobs_per_head % chain)
        prev = physical[i - 1] if rank else ttnn.CoreCoord(0, 0)
        following = physical[i + 1] if rank + 1 < chain else ttnn.CoreCoord(0, 0)
        next_count = (jobs_per_head // chain + (rank + 1 < jobs_per_head % chain)) if rank + 1 < chain else 0
        read[c.x][c.y] = [x.buffer_address() for x in (q, k, v)] + [
            offset,
            count,
            rank,
            chain,
            prev.x,
            prev.y,
            following.x,
            following.y,
            next_count,
        ]
        write[c.x][c.y] = [out.buffer_address(), offset, count]
        compute[c.x][c.y] = [count]
    reader_cta = [qt, chunks, jobs_per_head]
    for tensor in (q, k, v):
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[
            ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=value) for i, value in enumerate((0, 0, 1))
        ],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader_chain.cpp",
                core_ranges=grid,
                compile_time_args=reader_cta,
                runtime_args=read,
                defines=[("SDPA_READER_BARRIER_TILES", str(reader_barrier_tiles)), ("SDPA_K_CHUNK_TILES", str(kt))],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=("experiments/sdpa-l2/wan-frontier-v1/writer.cpp" if padded else PREFIX + "writer.cpp"),
                core_ranges=grid,
                compile_time_args=[qt] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=write,
                defines=[("SDPA_K_PARTIAL_COL", str(logical_k % 32))] if padded else [],
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=("experiments/sdpa-l2/wan-frontier-v1/compute.cpp" if padded else PREFIX + "compute.cpp"),
                core_ranges=grid,
                compile_time_args=[chunks, struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0], qt],
                runtime_args=compute,
                defines=list(defines.items()),
                config=config,
            ),
        ],
    )
    ttnn.generic_op([q, k, v, out], desc)
    return out
