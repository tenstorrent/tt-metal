# One-core MAX REDUCE_ROW program (reduce_init, reduce_tile per tile, reduce_uninit) for bits.py.
import torch
import ttnn

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto s = TensorAccessor(TensorAccessorArgs<xa.next_compile_time_args_offset()>(), get_arg_val<uint32_t>(1));
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t page = get_arg_val<uint32_t>(3);
    Noc noc;
    DataflowBuffer d_in(0), d_sc(1);
    for (uint32_t i = 0; i < n; ++i) {
        d_sc.reserve_back(1);
        noc.async_read(s, d_sc, page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        d_sc.push_back(1);
        d_in.reserve_back(1);
        noc.async_read(x, d_in, page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        d_in.push_back(1);
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto y = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0));
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t page = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer d_out(16);
    for (uint32_t i = 0; i < n; ++i) {
        d_out.wait_front(1);
        noc.async_write(d_out, y, page, {}, {.page_id = i});
        noc.async_write_barrier();
        d_out.pop_front(1);
    }
}
"""

COMPUTE = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t outputs = get_arg_val<uint32_t>(0);
    constexpr uint32_t in = 0, sc = 1, out = 16;
    DataflowBuffer d_in(in), d_sc(sc), d_out(out);
    compute_kernel_hw_startup(in, sc, out);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(in, sc, out);
    for (uint32_t o = 0; o < outputs; ++o) {
        d_sc.wait_front(1);
        d_in.wait_front(1);
        tile_regs_acquire();
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(in, sc, 0, 0, 0);
        tile_regs_commit();
        d_in.pop_front(1);
        d_sc.pop_front(1);
        d_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out);
        tile_regs_release();
        d_out.push_back(1);
    }
    reduce_uninit();
}
"""


def run_case(device, data, scal, fmt, fp32_dest, chunk=16384):
    dtype = ttnn.bfloat16 if fmt == "bf16" else ttnn.float32
    page = 4096 if dtype == ttnn.float32 else 2048
    odtype = ttnn.float32 if fp32_dest else dtype
    opage = 4096 if odtype == ttnn.float32 else 2048
    outs = []
    for c0 in range(0, data.shape[0], chunk):
        x, s = data[c0 : c0 + chunk], scal[c0 : c0 + chunk]
        n = x.shape[0]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        tx = ttnn.from_torch(x.reshape(1, n, 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        ts = ttnn.from_torch(s.reshape(1, n, 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        ty = ttnn.from_torch(torch.zeros(1, n, 32, 32), dtype=odtype, layout=ttnn.TILE_LAYOUT, device=device)

        def cb(i, pages, dt=dtype, pg=page):
            return ttnn.CBDescriptor(
                total_size=pages * pg,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=pg)],
            )

        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rd[0][0] = [tx.buffer_address(), ts.buffer_address(), n, page]
        wr[0][0] = [ty.buffer_address(), n, opage]
        cp[0][0] = [n]
        prog = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=READER,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=ttnn.TensorAccessorArgs(tx).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(ts).get_compile_time_args(),
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=WRITER,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=ttnn.TensorAccessorArgs(ty).get_compile_time_args(),
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=COMPUTE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=[],
                    defines=[],
                    runtime_args=cp,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_dest
                    ),
                ),
            ],
            semaphores=[],
            cbs=[cb(0, 2), cb(1, 2), cb(16, 2, odtype, opage)],
        )
        ttnn.generic_op([tx, ts, ty], prog)
        outs.append(ttnn.to_torch(ty).reshape(n, 32, 32))
        for t in (tx, ts, ty):
            ttnn.deallocate(t)
    return torch.cat(outs)
