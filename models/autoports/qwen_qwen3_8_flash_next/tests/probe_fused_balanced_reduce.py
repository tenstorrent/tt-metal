# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prototype exact half-split FP32 column tree for a future fused QKV kernel."""
import argparse
import json
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn

BALANCED_REDUCE = r"""
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_reduce.h"
void balanced_column_reduce_tile0() {
    MATH((::_llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>()));
    MATH((ckernel::sfpu::init_reduce<PoolType::SUM, DataFormat::Float32, true>()));
    MATH((::_llk_math_eltwise_sfpu_start_(0)));
    MATH(([&]() {
        constexpr uint32_t upper[4] = {0, 0, 16, 16};
        constexpr uint32_t lower[4] = {32, 32, 48, 48};
        constexpr uint32_t column[4] = {0, 2, 0, 2};
        for (uint32_t i=0;i<4;++i) {
            ckernel::sfpu::load_face_data<InstrModLoadStore::FP32, false>(upper[i], lower[i], column[i]);
            // Width32 ->16: corresponding upper/lower rows first.
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG4,p_sfpu::LREG0,0);
            TTI_SFPADD(p_sfpu::LREG1,p_sfpu::LCONST_1,p_sfpu::LREG5,p_sfpu::LREG1,0);
            TTI_SFPADD(p_sfpu::LREG2,p_sfpu::LCONST_1,p_sfpu::LREG6,p_sfpu::LREG2,0);
            TTI_SFPADD(p_sfpu::LREG3,p_sfpu::LCONST_1,p_sfpu::LREG7,p_sfpu::LREG3,0);
            // Width16 ->8 ->4 across the four row-group registers.
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG2,p_sfpu::LREG0,0);
            TTI_SFPADD(p_sfpu::LREG1,p_sfpu::LCONST_1,p_sfpu::LREG3,p_sfpu::LREG1,0);
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG1,p_sfpu::LREG0,0);
            TTI_SFPTRANSP(0,0,0,0);
            // Width4 ->2 ->1 across the four positions of the remaining row group.
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG2,p_sfpu::LREG0,0);
            TTI_SFPADD(p_sfpu::LREG1,p_sfpu::LCONST_1,p_sfpu::LREG3,p_sfpu::LREG1,0);
            TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG1,p_sfpu::LREG0,0);
            TT_SFPSTORE(p_sfpu::LREG0,static_cast<uint32_t>(InstrModLoadStore::FP32),ADDR_MOD_7,upper[i]+column[i]);
        }
    }()));
    MATH((::_llk_math_eltwise_sfpu_done_()));
}
"""
READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto a=TensorAccessorArgs<0>();
    const auto input=TensorAccessor(a,get_arg_val<uint32_t>(0),4096);
    cb_reserve_back(0,1);noc_async_read(input.get_noc_addr(0),get_write_ptr(0),4096);
    noc_async_read_barrier();cb_push_back(0,1);
}
"""
COMPUTE = (
    r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
"""
    + BALANCED_REDUCE
    + r"""
void kernel_main() {
    compute_kernel_hw_startup(0,1);
    cb_wait_front(0,1);cb_reserve_back(1,1);
    tile_regs_acquire();copy_init(0);copy_tile(0,0,0);
    balanced_column_reduce_tile0();
    tile_regs_commit();tile_regs_wait();pack_tile(0,1);tile_regs_release();
    cb_push_back(1,1);cb_pop_front(0,1);
}
"""
)
WRITER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto a=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(a,get_arg_val<uint32_t>(0),4096);
    cb_wait_front(1,1);noc_async_write(get_read_ptr(1),output.get_noc_addr(0),4096);
    noc_async_write_barrier();cb_pop_front(1,1);
}
"""


def balanced_reduce(x):
    assert tuple(x.shape) == (1, 1, 32, 32) and x.dtype == ttnn.float32
    out = ttnn.empty((1, 1, 1, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=x.device())
    core = ttnn.CoreCoord(0, 0)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [x.buffer_address()]
    writer[0][0] = [out.buffer_address()]
    config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_approx_mode=False)
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        for src, ct, rt, cfg in [
            (READER, ttnn.TensorAccessorArgs(x).get_compile_time_args(), reader, ttnn.ReaderConfigDescriptor()),
            (COMPUTE, [], ttnn.RuntimeArgs(), config),
            (WRITER, ttnn.TensorAccessorArgs(out).get_compile_time_args(), writer, ttnn.WriterConfigDescriptor()),
        ]
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i in [0, 1]
    ]
    ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(903)
    torch.set_num_threads(4)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    report = dict(rows=[], passed=False)
    try:
        for case in range(8):
            host = torch.randn(1, 1, 32, 32) * 10 ** (case - 4)
            if case >= 4:
                host[..., 16:, :] = -host[..., :16, :] + torch.randn(1, 1, 16, 32) * 0.001
            expected = host.clone()
            for width in (32, 16, 8, 4, 2):
                expected = expected[..., : width // 2, :] + expected[..., width // 2 : width, :]
            x = ttnn.from_torch(host, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            actual = ttnn.to_torch(balanced_reduce(x))
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = balanced_reduce(x)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(out)
            for _ in range(4):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            replay = ttnn.to_torch(out)
            ttnn.release_trace(mesh, trace)
            trace = None
            row = dict(
                case=case,
                exact=torch.equal(actual, expected),
                replay_exact=torch.equal(replay, actual),
                max_abs=float((actual - expected).abs().max()),
            )
            report["rows"].append(row)
            print(row, flush=True)
        report["passed"] = all(r["exact"] and r["replay_exact"] for r in report["rows"])
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
