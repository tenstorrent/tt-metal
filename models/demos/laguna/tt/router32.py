# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact fp32 top-K routing for one 32-token decode tile as one generic_op (Laguna). See kernels/router32_reader.cpp."""

import struct
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def route32(sel, scores, top_k, routed_scaling, norm_topk_prob, memory_config=ttnn.L1_MEMORY_CONFIG):
    """sel, scores: [1, 1, T, E] fp32 TILE interleaved, T <= 32. Returns the dense fp32 routing matrix [1, 1, T, E]
    (one core per token row)."""
    device = sel.device()
    T, E = sel.shape[-2], sel.shape[-1]
    assert T <= 32, sel.shape
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, E]), ttnn.float32, ttnn.TILE_LAYOUT, device, memory_config)
    grid_size = device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(T, grid_size, True)
    buf = ttnn.CBDescriptor(
        total_size=4 * E * 4 + 256,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4 * E * 4 + 256)],
    )
    args = []
    for t in (sel, scores, out, out):
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    scale_bits = struct.unpack("<I", struct.pack("<f", float(routed_scaling)))[0]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "router32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[E, int(top_k), grid_size.x, int(bool(norm_topk_prob)), scale_bits, 4096, 0, 0] + args,
        common_runtime_args=[sel.buffer_address(), scores.buffer_address(), out.buffer_address(), 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[buf])
    ttnn.generic_op([sel, scores, out], program)
    return out


def route_topk_rm(sel, scores, top_k, routed_scaling, norm_topk_prob, memory_config=ttnn.DRAM_MEMORY_CONFIG, bias=None):
    """Prefill: sel, scores [1, 1, T, E] fp32 TILE interleaved, T a multiple of 32. Returns (weights, indices): the
    [1, T, K] bf16 routing weights and uint16 expert ids, row-major (token dispatch's input layout). With ``bias``
    (the [1, 1, 1, E] fp32 TILE score-correction bias) ``scores`` is not read: a picked expert's score is sel - bias.
    See kernels/route_topk_reader.cpp."""
    device = sel.device()
    T, E = sel.shape[-2], sel.shape[-1]
    K = int(top_k)
    assert T % 32 == 0 and E % 32 == 0 and K <= 32, (sel.shape, K)
    idx = ttnn.allocate_tensor_on_device(ttnn.Shape([1, T, K]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, device, memory_config)
    wgt = ttnn.allocate_tensor_on_device(ttnn.Shape([1, T, K]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, memory_config)
    grid_size = device.compute_with_storage_grid_size()
    max_cores = grid_size.x * grid_size.y
    # rows per work unit: the size that minimises the rows the busiest RISC scans one after another (8-row units
    # left a 128-token prefill on 8 cores; at 1K, 4-row units gave some RISCs 8 rows, 1-row units give 5); ties
    # keep the larger unit (fewer NoC read batches)
    risc_n = 2 * max_cores
    R = min((8, 4, 2, 1), key=lambda r: (-(-(T // r) // risc_n) * r, -r))
    units = T // R
    cores = min(-(-units // 2), max_cores)  # two RISCs per core
    grid = ttnn.num_cores_to_corerangeset(cores, grid_size, True)
    stage = E // 32 * 2 * R * 64
    bufs = [
        ttnn.CBDescriptor(
            total_size=2 * stage + R * 128,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=2 * stage + R * 128)
            ],
        )
        for i in range(2)
    ]
    mode = 1 if bias is not None else 0
    second = bias if mode else scores
    args = []
    for t in (sel, second, idx, wgt):
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    scale_bits = struct.unpack("<I", struct.pack("<f", float(routed_scaling)))[0]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "route_topk_reader.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[E, K, grid_size.x, cores, int(bool(norm_topk_prob)), scale_bits, 4096, T // 32]
            + [K * 2, K * 2, risc, mode, R]
            + args,
            common_runtime_args=[
                sel.buffer_address(),
                second.buffer_address(),
                idx.buffer_address(),
                wgt.buffer_address(),
            ],
            config=ttnn.ReaderConfigDescriptor() if risc == 0 else ttnn.WriterConfigDescriptor(),
        )
        for risc in range(2)
    ]
    ttnn.generic_op([sel, second, idx, wgt], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=bufs))
    return wgt, idx


def route1_local(sel, scores, top_k, routed_scaling, norm_topk_prob, ep_off, local_e,
                 memory_config=ttnn.L1_MEMORY_CONFIG):
    """One decode token: sel, scores [1, 1, 1, E] fp32 TILE; ep_off: mesh-sharded uint32 holding each chip's first
    local expert. Returns this chip's [1, 1, 1, local_e] bf16 row-major routing row (normalized, scaled weights of
    its picked local experts, 0 elsewhere) -- the batch-1 MoE kernels' sparsity input."""
    device = sel.device()
    E = sel.shape[-1]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, local_e]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, memory_config
    )
    grid_size = device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(1, grid_size, True)
    buf = ttnn.CBDescriptor(
        total_size=4 * E * 4 + 256,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4 * E * 4 + 256)],
    )
    args = []
    for t in (sel, scores, out, ep_off):
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    scale_bits = struct.unpack("<I", struct.pack("<f", float(routed_scaling)))[0]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "router32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[E, int(top_k), grid_size.x, int(bool(norm_topk_prob)), scale_bits, 4096, int(local_e), 0] + args,
        common_runtime_args=[sel.buffer_address(), scores.buffer_address(), out.buffer_address(), ep_off.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([sel, scores, ep_off, out], ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[buf]))
    return out


def route_local_rows(sel, scores, top_k, routed_scaling, norm_topk_prob, ep_off, local_e,
                     memory_config=ttnn.L1_MEMORY_CONFIG):
    """2..32 decode tokens: sel, scores [1, 1, T, E] fp32 TILE; ep_off as in route1_local. Returns this chip's
    [1, 1, T, local_e] bf16 row-major routing rows (normalized, scaled weights of each token's picked local experts,
    0 elsewhere): moe_decode1.gate_up32 / down32 union them into the active set and gate_up32 builds each expert's
    per-token weight tile from them. One core per token replaces the dense routing tile, EP-select matmul, row sum,
    untilize and weight transpose."""
    device = sel.device()
    T, E = sel.shape[-2], sel.shape[-1]
    assert 1 < T <= 32, sel.shape
    rows = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, local_e]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, memory_config
    )
    grid_size = device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(T, grid_size, True)
    buf = ttnn.CBDescriptor(
        total_size=4 * E * 4 + 256,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4 * E * 4 + 256)],
    )
    args = []
    for t in (sel, scores, rows, ep_off):
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    scale_bits = struct.unpack("<I", struct.pack("<f", float(routed_scaling)))[0]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "router32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[E, int(top_k), grid_size.x, int(bool(norm_topk_prob)), scale_bits, 4096, int(local_e), T]
        + args,
        common_runtime_args=[sel.buffer_address(), scores.buffer_address(), rows.buffer_address(), ep_off.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([sel, scores, ep_off, rows], ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[buf]))
    return rows
