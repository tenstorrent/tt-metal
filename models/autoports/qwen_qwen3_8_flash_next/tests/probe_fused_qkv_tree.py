# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental exact QKV product/tree fusion, isolated from the delivered decoder."""

import argparse
import json
import statistics
import time
from copy import copy
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_fused_balanced_reduce import BALANCED_REDUCE
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedExactQKV, FusedPreciseProjection
from models.autoports.qwen_qwen3_8_flash_next.tt.precise_gdn_projection import PreciseGDNProjection

READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto wa=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto w=TensorAccessor(wa,get_arg_val<uint32_t>(1),2048);
    const uint32_t first=get_arg_val<uint32_t>(2), count=get_arg_val<uint32_t>(3), nt=get_arg_val<uint32_t>(4);
    const uint32_t cache=get_write_ptr(6);
    for(uint32_t k=0;k<80;++k)noc_async_read(x.get_noc_addr(k),cache+k*2048,2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* input=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cache);
    // Both alternating query buffers start at zero; only column zero is consumed.
    volatile tt_l1_ptr uint32_t* qs=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    for(uint32_t i=0;i<1024;++i)qs[i]=0;
    for(uint32_t task=first;task<first+count;++task) {
        const uint32_t q=task/nt, outcol=task%nt;
        for(uint32_t j=0;j<5;++j)for(uint32_t b=0;b<16;++b) {
            cb_reserve_back(0,1);cb_reserve_back(1,1);
            noc_async_read(w.get_noc_addr((j*16+b)*nt+outcol),get_write_ptr(0),2048);
            volatile tt_l1_ptr uint16_t* query=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(1));
            for(uint32_t n=0;n<32;++n) {
                const uint32_t k=j+5*(b+16*n);
                const uint32_t offset=(k/32)*1024+(q/16)*512+((k%32)/16)*256+(q%16)*16+k%16;
                query[(n/16)*512+(n%16)*16]=input[offset];
            }
            noc_async_read_barrier();cb_push_back(0,1);cb_push_back(1,1);
        }
    }
}
"""
COMPUTE = (
    r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/typecast.h"
"""
    + BALANCED_REDUCE
    + r"""
void load(uint32_t cb,uint32_t index,uint32_t dst) {
    reconfig_data_format_srca(cb);copy_init(cb);copy_tile(cb,index,dst);
}
void pair_sum(uint32_t input,uint32_t left,uint32_t right,uint32_t output,uint32_t index) {
    tile_regs_acquire();load(input,left,0);load(input,right,1);
    add_binary_tile_init();add_binary_tile(0,1,0);
    tile_regs_commit();tile_regs_wait();pack_tile<true>(0,output,index);tile_regs_release();
}
void kernel_main() {
    compute_kernel_hw_startup(0,2);
    const uint32_t tasks=get_arg_val<uint32_t>(0);
    for(uint32_t task=0;task<tasks;++task) {
        pack_reconfig_data_format(2);
        cb_reserve_back(4,5);
        for(uint32_t j=0;j<5;++j) {
            cb_reserve_back(2,16);
            for(uint32_t b=0;b<16;++b) {
                cb_wait_front(0,1);cb_wait_front(1,1);
                tile_regs_acquire();load(0,0,0);load(1,0,1);
                sfpu_mul_bcast_col_init();sfpu_mul_bcast_col(0,1);
                balanced_column_reduce_tile0();
                tile_regs_commit();tile_regs_wait();pack_tile<true>(0,2,b);tile_regs_release();
                cb_pop_front(0,1);cb_pop_front(1,1);
            }
            cb_push_back(2,16);
            uint32_t input=2,output=3;
            for(uint32_t width=16;width>1;width/=2) {
                cb_wait_front(input,16);cb_reserve_back(output,16);
                for(uint32_t i=0;i<width/2;++i)pair_sum(input,i,i+width/2,output,i);
                cb_pop_front(input,16);cb_push_back(output,16);
                uint32_t swap=input;input=output;output=swap;
            }
            cb_wait_front(2,16);
            tile_regs_acquire();load(2,0,0);tile_regs_commit();tile_regs_wait();
            pack_tile<true>(0,4,j);tile_regs_release();cb_pop_front(2,16);
        }
        cb_push_back(4,5);cb_wait_front(4,5);cb_reserve_back(5,2);
        pair_sum(4,0,2,5,0);pair_sum(4,1,3,5,1);cb_push_back(5,2);cb_wait_front(5,2);
        tile_regs_acquire();load(5,0,0);load(5,1,1);
        add_binary_tile_init();add_binary_tile(0,1,0);load(4,4,1);add_binary_tile_init();add_binary_tile(0,1,0);
        typecast_tile_init<(uint32_t)DataFormat::Float32,(uint32_t)DataFormat::Float16_b>();typecast_tile<(uint32_t)DataFormat::Float32,(uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();tile_regs_wait();cb_reserve_back(7,1);
        pack_reconfig_data_format(7);pack_tile<true>(0,7);tile_regs_release();cb_push_back(7,1);
        cb_pop_front(4,5);cb_pop_front(5,2);
    }
}
"""
)
WRITER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto a=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(a,get_arg_val<uint32_t>(0),2048);
    const uint32_t first=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2), nt=get_arg_val<uint32_t>(3), tokens=get_arg_val<uint32_t>(4);
    const uint32_t zeros=get_write_ptr(8);
    volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros);
    for(uint32_t i=0;i<16;++i)z[i]=0;
    for(uint32_t task=first;task<first+count;++task) {
        const uint32_t q=task/nt,col=task%nt;
        cb_wait_front(7,1);const uint32_t src=get_read_ptr(7);
        const uint32_t offset=(q/16)*1024+(q%16)*32;
        noc_async_write(src,output.get_noc_addr(col,offset),32);
        noc_async_write(src+512,output.get_noc_addr(col,offset+512),32);
        if(q==0)for(uint32_t r=tokens;r<32;++r) {
            const uint32_t pad=(r/16)*1024+(r%16)*32;
            noc_async_write(zeros,output.get_noc_addr(col,pad),32);
            noc_async_write(zeros,output.get_noc_addr(col,pad+512),32);
        }
        noc_async_write_barrier();cb_pop_front(7,1);
    }
}
"""


class ExactQKV:
    def __init__(self, weight, device):
        self.output_width, self.input_width = weight.shape
        assert self.input_width == 2560 and self.output_width % 32 == 0
        order = torch.tensor([j + 5 * (b + 16 * n) for j in range(5) for b in range(16) for n in range(32)])
        self.weight = ttnn.from_torch(
            weight[:, order].T.contiguous().reshape(1, 1, 2560, -1),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x):
        count = x.shape[-2]
        assert tuple(x.shape) == (1, 1, count, 2560) and 1 <= count <= 32 and x.dtype == ttnn.bfloat16
        out = ttnn.empty(
            (1, 1, count, self.output_width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device()
        )
        grid = x.device().compute_with_storage_grid_size()
        nt = self.output_width // 32
        tasks = count * nt
        used = min(grid.x * grid.y, tasks)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(used)]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            start = tasks * i // used
            size = tasks * (i + 1) // used - start
            reader[c.x][c.y] = [x.buffer_address(), self.weight.buffer_address(), start, size, nt]
            compute[c.x][c.y] = [size]
            writer[c.x][c.y] = [out.buffer_address(), start, size, nt, count]
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
                (
                    READER,
                    ttnn.TensorAccessorArgs(x).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(self.weight).get_compile_time_args(),
                    reader,
                    ttnn.ReaderConfigDescriptor(),
                ),
                (COMPUTE, [], compute, config),
                (WRITER, ttnn.TensorAccessorArgs(out).get_compile_time_args(), writer, ttnn.WriterConfigDescriptor()),
            ]
        ]
        cbs = []
        for index, tiles, dtype, page in [
            (0, 2, ttnn.bfloat16, 2048),
            (1, 2, ttnn.bfloat16, 2048),
            (2, 16, ttnn.float32, 4096),
            (3, 16, ttnn.float32, 4096),
            (4, 5, ttnn.float32, 4096),
            (5, 2, ttnn.float32, 4096),
            (6, 80, ttnn.bfloat16, 2048),
            (7, 2, ttnn.bfloat16, 2048),
            (8, 1, ttnn.bfloat16, 64),
        ]:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=tiles * page,
                    core_ranges=ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
                )
            )
        ttnn.generic_op([x, self.weight, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--counts", type=int, nargs="+", default=[1, 2, 7, 16, 17, 31, 32])
    p.add_argument("--clean-timing", action="store_true")
    a = p.parse_args()
    assert bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING) != a.clean_timing
    torch.set_num_threads(8)
    torch.manual_seed(9384)
    name = "linear_attn.in_proj_qkv.weight"
    weights = load_weights(a.snapshot, 0, name)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    report = dict(
        rows=[],
        passed=False,
        clean_timing=a.clean_timing,
        trace_allocation_tracking=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
    )
    try:
        loader = WeightLoader(weights, mesh)
        original = PreciseGDNProjection(loader, name)
        ref = FusedPreciseProjection(original, loader.compute, 64, balanced_tree=True, widen_in_unpack=True)
        candidate = FusedExactQKV(copy(original), weights[name])
        report["candidate_class"] = type(candidate).__name__
        for count in a.counts:
            host = (torch.randn(1, 1, count, 2560) * 0.02).bfloat16()
            x = loader.tensor(host)
            expected = ttnn.to_torch(ref(x))
            actual = ttnn.to_torch(candidate(x))
            alternate = (-host.float() + torch.randn_like(host.float()) * 0.003).bfloat16()
            prepared = [ttnn.from_torch(v, layout=ttnn.TILE_LAYOUT) for v in (host, alternate)]
            ttnn.copy_host_to_device_tensor(prepared[1], x)
            alternate_expected = ttnn.to_torch(ref(x))
            alternate_actual = ttnn.to_torch(candidate(x))
            ttnn.copy_host_to_device_tensor(prepared[0], x)
            row = dict(
                alternate_exact=torch.equal(alternate_actual, alternate_expected),
                count=count,
                exact=torch.equal(actual, expected),
                equal_fraction=float((actual == expected).float().mean()),
                max_abs=float((actual.float() - expected.float()).abs().max()),
                times={},
            )
            for label, fn in [("tree", ref), ("fused", candidate)]:
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = fn(x)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                    trace_allocation_tracker.acknowledge_corruptible(out)
                samples = []
                for repeat in range(12):
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - start) * 1000)
                replay = ttnn.to_torch(out)
                row[label + "_replay_exact"] = torch.equal(replay, expected if label == "tree" else actual)
                row["times"][label] = statistics.median(samples[2:])
                changed = []
                for index in (1, 0, 1, 0):
                    ttnn.copy_host_to_device_tensor(prepared[index], x)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    changed.append(torch.equal(ttnn.to_torch(out), alternate_expected if index else expected))
                row[label + "_changed_input_exact"] = all(changed)
                ttnn.release_trace(mesh, trace)
                trace = None
            report["rows"].append(row)
            print(row, flush=True)
        report["passed"] = all(
            r["exact"]
            and r["alternate_exact"]
            and r["tree_replay_exact"]
            and r["fused_replay_exact"]
            and r["tree_changed_input_exact"]
            and r["fused_changed_input_exact"]
            for r in report["rows"]
        )
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
