# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact residual writer: FP32 product, BF16 boundary, stream placement, FP32 add."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import GatedResidual

READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto ba = TensorAccessorArgs<0>();
    constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    const auto block = TensorAccessor(ba, get_arg_val<uint32_t>(0), 2048);
    const auto inject = TensorAccessor(ia, get_arg_val<uint32_t>(1), 2048);
    const auto residual = TensorAccessor(ra, get_arg_val<uint32_t>(2), 2048);
    const uint32_t start = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    const uint32_t mt = get_arg_val<uint32_t>(5);
    for (uint32_t tile = start; tile < start + count; ++tile) {
        const uint32_t row = tile / 320, col = tile % 320, stream = col / 80;
        cb_reserve_back(0, 1); cb_reserve_back(1, 1); cb_reserve_back(2, 1);
        noc_async_read(block.get_noc_addr(row * 80 + col % 80), get_write_ptr(0), 2048);
        noc_async_read(inject.get_noc_addr(stream * mt + row), get_write_ptr(1), 2048);
        noc_async_read(residual.get_noc_addr(tile), get_write_ptr(2), 2048);
        noc_async_read_barrier();
        cb_push_back(0, 1); cb_push_back(1, 1); cb_push_back(2, 1);
    }
}
"""
COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/typecast.h"
void kernel_main() {
    compute_kernel_hw_startup(0, 1, 3);
    const uint32_t count = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < count; ++i) {
        cb_wait_front(0, 1); cb_wait_front(1, 1); cb_wait_front(2, 1);
        tile_regs_acquire();
        copy_init(0); copy_tile(0, 0, 0);
        copy_init(1); copy_tile(1, 0, 1);
        copy_init(2); copy_tile(2, 0, 2);
        sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(0, 1);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        add_binary_tile_init(); add_binary_tile(0, 2, 0);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        cb_reserve_back(3, 1);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0, 3); tile_regs_release();
        cb_push_back(3, 1);
        cb_pop_front(0, 1); cb_pop_front(1, 1); cb_pop_front(2, 1);
    }
}
"""
WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t start = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    for (uint32_t tile = start; tile < start + count; ++tile) {
        cb_wait_front(3, 1);
        noc_async_write(get_read_ptr(3), output.get_noc_addr(tile), 2048);
        noc_async_write_barrier(); cb_pop_front(3, 1);
    }
}
"""


def fused_residual_write(residual, block, injection):
    tokens = residual.shape[-2]
    if (
        tuple(residual.shape) != (1, 1, tokens, 10240)
        or tuple(block.shape) != (1, 1, tokens, 2560)
        or tuple(injection.shape) != (1, 4, tokens, 1)
        or tokens < 1
        or any(
            t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            for t in (residual, block, injection)
        )
        or any(t.device() != residual.device() for t in (block, injection))
    ):
        raise ValueError("Residual write requires matching BF16 TILE DRAM four-stream inputs")
    mesh = residual.device()
    output = ttnn.empty_like(residual)
    mt = (tokens + 31) // 32
    total = mt * 320
    grid = mesh.compute_with_storage_grid_size()
    cores = min(total, grid.x * grid.y)
    ranges = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
            for i in range(cores)
        ]
    )
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(cores):
        x, y = i % grid.x, i // grid.x
        start, end = total * i // cores, total * (i + 1) // cores
        reader[x][y] = [
            block.buffer_address(),
            injection.buffer_address(),
            residual.buffer_address(),
            start,
            end - start,
            mt,
        ]
        compute[x][y] = [end - start]
        writer[x][y] = [output.buffer_address(), start, end - start]
    ct = []
    for t in (block, injection, residual):
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=args,
            runtime_args=rt,
            config=config,
        )
        for source, args, rt, config in [
            (READER, ct, reader, ttnn.ReaderConfigDescriptor()),
            (
                COMPUTE,
                [],
                compute,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                    dst_full_sync_en=False,
                ),
            ),
            (WRITER, ttnn.TensorAccessorArgs(output).get_compile_time_args(), writer, ttnn.WriterConfigDescriptor()),
        ]
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in range(4)
    ]
    ttnn.generic_op(
        [block, injection, residual, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-timing", action="store_true")
    parser.add_argument("--integrated", action="store_true")
    args = parser.parse_args()
    selected_write = fused_residual_write
    if args.integrated:
        from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_residual_write as selected_write
    assert bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING) != args.clean_timing
    torch.set_num_threads(8)
    torch.manual_seed(620)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    owner = GatedResidual.__new__(GatedResidual)
    owner.width, owner.streams = 2560, 4
    report = dict(
        rows=[],
        passed=False,
        clean_timing=args.clean_timing,
        watcher=os.environ.get("TT_METAL_WATCHER"),
        integrated=args.integrated,
    )
    trace = None
    try:
        for count in (1, 2, 3, 4, 7, 16, 17, 31, 32, 33, 65):
            inputs = []
            for case in range(2):
                residual = torch.randn(1, 1, count, 10240).bfloat16()
                block = torch.randn(1, 1, count, 2560).bfloat16()
                injection = torch.randn(1, 4, count, 1).bfloat16()
                if case:
                    vals = torch.tensor(
                        [1.0, 1.0078125, 1.9921875, -1.0, -1.0078125, -1.9921875, 0.00390625, -0.00390625],
                        dtype=torch.bfloat16,
                    )
                    block = vals.repeat(count * 320).reshape(1, 1, count, 2560)
                    injection = (
                        torch.tensor([0.5, 1.5, -0.5, -1.5], dtype=torch.bfloat16)
                        .reshape(1, 4, 1, 1)
                        .expand(1, 4, count, 1)
                        .contiguous()
                    )
                    update = (block * injection).permute(0, 2, 1, 3).reshape(1, 1, count, 10240)
                    residual = (-update + 0.00390625).bfloat16()
                inputs.append(
                    [
                        ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        for t in (residual, block, injection)
                    ]
                )
            expected = [ttnn.to_torch(owner.write(*values)) for values in inputs]
            actual = [ttnn.to_torch(selected_write(*values)) for values in inputs]
            row = dict(
                count=count,
                exact=[torch.equal(a, b) for a, b in zip(expected, actual)],
                max_abs=[float((a.float() - b.float()).abs().max()) for a, b in zip(expected, actual)],
                times={},
            )
            assert all(row["exact"]), row
            live = [ttnn.empty_like(t) for t in inputs[0]]
            for name, function in [("original", owner.write), ("fused", selected_write)]:
                for src, dst in zip(inputs[1], live):
                    ttnn.copy(src, dst)
                function(*live)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    out = function(*live)
                finally:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                    trace_allocation_tracker.acknowledge_corruptible(out)
                times = []
                for repeat in range(12):
                    case = repeat % 2
                    for src, dst in zip(inputs[case], live):
                        ttnn.copy(src, dst)
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    times.append((time.perf_counter() - start) * 1000)
                    assert torch.equal(ttnn.to_torch(out), expected[case])
                row["times"][name] = statistics.median(times[2:])
                ttnn.release_trace(mesh, trace)
                trace = None
            report["rows"].append(row)
            print(row, flush=True)
        report["passed"] = True
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
