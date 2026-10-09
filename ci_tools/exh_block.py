# Exhaustive bit identity of the block call against per tile calls for the callers #58820 switches (#58707): moreh softmax
# over H (REDUCE_COL, MAX and SUM, the column's Ht tiles into one output) and SDPA decode's row sum (REDUCE_ROW SUM, a row's
# Sk_chunk_t tiles into one output). Every bf16 pattern (65536) and every TF32 pattern (2^19), each once, k per output
# (one per tile, at a different row or column of each tile), and a dense layout of random values over 16 binades; k = 8, 16,
# 32; MAX, SUM at the kernel's HiFi4 and SUM with pow2_scaler; a 16-bit and an fp32 DEST.
# REDUCE_SCALAR with k = 8 tiles into one output: since the stride-0 SCALAR block runs per tile reduces, the same check.
# usage: python exh_block.py <out.txt> [dims]
import os as _os, sys as _sys

# 05 (12:35 UTC): the device only under scripts/hwlock.sh (it exports HWLOCK_HELD); the mock cluster (an existing descriptor) needs no lock
if not _os.environ.get("HWLOCK_HELD") and not _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    _sys.exit("not under hwlock")

import itertools
import sys

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
    const uint32_t k = get_arg_val<uint32_t>(4);   // tiles per output
    const uint32_t wt = get_arg_val<uint32_t>(5);  // 0: tiles in page order (ROW); else tiles per tile row (COL)
    Noc noc;
    DataflowBuffer d_in(0), d_sc(1);
    d_sc.reserve_back(1);
    noc.async_read(s, d_sc, page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    d_sc.push_back(1);
    for (uint32_t i = 0; i < n; ++i) {
        d_in.reserve_back(1);
        const uint32_t pid = wt == 0 ? i : (i % k) * wt + i / k;
        noc.async_read(x, d_in, page, {.page_id = pid}, {.offset_bytes = 0});
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
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

#ifdef POW2
constexpr bool pow2 = true;
#else
constexpr bool pow2 = false;
#endif

void kernel_main() {
    const uint32_t outputs = get_arg_val<uint32_t>(0);
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t in = 0, sc = 1, out = 16;
    DataflowBuffer d_in(in), d_sc(sc), d_out(out);
    compute_kernel_hw_startup(in, sc, out);
    reduce_init<POOL, DIM, DST_ACCUM_MODE, pow2>(in, sc, out);
    d_sc.wait_front(1);
    for (uint32_t o = 0; o < outputs; ++o) {
        d_in.wait_front(k);
        tile_regs_acquire();
#ifdef BLOCK
        reduce_block<POOL, DIM, DST_ACCUM_MODE, pow2>(in, sc, 0, 0, 0, k, 0);
#else
        for (uint32_t t = 0; t < k; ++t) {
            reduce_tile<POOL, DIM, DST_ACCUM_MODE, pow2>(in, sc, t, 0, 0);
        }
#endif
        tile_regs_commit();
        d_in.pop_front(k);
        d_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, out);
        tile_regs_release();
        d_out.push_back(1);
    }
    reduce_uninit();
}
"""

DIMS = {"row": "ReduceDim::REDUCE_ROW", "col": "ReduceDim::REDUCE_COL", "scalar": "ReduceDim::REDUCE_SCALAR"}
POOLS = {"max": "PoolType::MAX", "sum": "PoolType::SUM", "sum_pow2": "PoolType::SUM"}


def patterns(fmt):
    if fmt == "bf16":
        return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    return (torch.arange(0, 1 << 19, dtype=torch.int64) << 13).to(torch.int32).view(torch.float32)


def layout(fmt, dim, k, kind):
    """ROW: (R, 32k), output r sums its row over k tiles; COL: (32k, C), tile t of a column is rows 32t..32t+31."""
    if dim == "scalar":  # full tiles of distinct patterns (or random values), k consecutive tiles per output
        if kind == "exh":
            p = patterns(fmt)
            return p.reshape(1, p.numel() // 1024, 32, 32)
        g = torch.Generator().manual_seed(1000 * k + 7)
        shape = (64 * k, 32, 32)
        x = torch.randn(shape, generator=g) * torch.exp2(torch.randint(-8, 8, shape, generator=g).float())
        return x.reshape(1, 64 * k, 32, 32)
    if kind == "exh":
        p = patterns(fmt)
        n = p.numel() // k  # outputs (rows or columns), each with k distinct patterns, one per tile
        v = p.reshape(n, k)
        if dim == "row":
            x = torch.zeros(n, 32 * k)
            for t in range(k):
                x[:, 32 * t + (t * 7) % 32] = v[:, t]
            return x.reshape(1, 1, n, 32 * k)
        x = torch.zeros(32 * k, n)
        for t in range(k):
            x[32 * t + (t * 7) % 32, :] = v[:, t]
        return x.reshape(1, 1, 32 * k, n)
    g = torch.Generator().manual_seed(1000 * k + (dim == "col"))
    shape = (2048, 32 * k) if dim == "row" else (32 * k, 2048)
    x = torch.randn(shape, generator=g) * torch.exp2(torch.randint(-8, 8, shape, generator=g).float())
    return x.reshape(1, 1, *shape)


def run(device, x, dim, k, pool, fp32_dest, block, dtype):
    page = 4096 if dtype == ttnn.float32 else 2048
    odtype = ttnn.float32 if fp32_dest else dtype
    opage = 4096 if odtype == ttnn.float32 else 2048
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tx = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    n_in = x.numel() // 1024
    outputs = n_in // k
    wt = x.shape[-1] // 32 if dim == "col" else 0
    ts = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ty = ttnn.from_torch(torch.zeros(1, outputs, 32, 32), dtype=odtype, layout=ttnn.TILE_LAYOUT, device=device)
    cb = lambda i, pages, dt=dtype, pg=page: ttnn.CBDescriptor(
        total_size=pages * pg,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=pg)],
    )
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd[0][0] = [tx.buffer_address(), ts.buffer_address(), n_in, page, k, wt]
    wr[0][0] = [ty.buffer_address(), outputs, opage]
    cp[0][0] = [outputs]
    defines = [("DIM", DIMS[dim]), ("POOL", POOLS[pool])]
    defines += [("POW2", "1")] if pool == "sum_pow2" else []
    defines += [("BLOCK", "1")] if block else []
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
                compile_time_args=[k],
                defines=defines,
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_dest
                ),
            ),
        ],
        semaphores=[],
        cbs=[cb(0, 2 * k), cb(1, 1), cb(16, 2, odtype, opage)],
    )
    ttnn.generic_op([tx, ts, ty], prog)
    y = ttnn.to_torch(ty)
    for t in (tx, ts, ty):
        ttnn.deallocate(t)
    return y


def main():
    out = open(sys.argv[1], "w")
    device = ttnn.open_device(device_id=0)
    total_bad = 0
    cfgs = [("row", 8), ("row", 16), ("col", 8), ("col", 32), ("scalar", 8)]
    if len(sys.argv) > 2:  # dims to run, e.g. "scalar"
        cfgs = [c for c in cfgs if c[0] in sys.argv[2].split(",")]
    for (dim, k), pool, (fmt, fp32_dest), kind in itertools.product(
        cfgs, ("max", "sum", "sum_pow2"), (("bf16", False), ("bf16", True), ("tf32", True)), ("exh", "dense")
    ):
        if fmt == "tf32" and kind == "dense":
            continue  # the dense values are fp32 already in the bf16 run's DEST; TF32 coverage is the exhaustive layout
        dtype = ttnn.bfloat16 if fmt == "bf16" else ttnn.float32
        x = layout(fmt, dim, k, kind)
        name = f"{fmt} {kind} {dim} k={k} {pool} fp32_dest={fp32_dest}"
        try:
            a = run(device, x, dim, k, pool, fp32_dest, False, dtype)
            b = run(device, x, dim, k, pool, fp32_dest, True, dtype)
            ia = a.contiguous().view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32)
            ib = b.contiguous().view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32)
            diff = int((ia != ib).sum())
            total_bad += diff
            nan = int(torch.isnan(a.float()).sum())
            line = f"{name}: {a.numel()} outputs, {diff} differ, {nan} NaN"
        except Exception as e:
            line = f"{name}: ERROR {str(e).splitlines()[0][:200]}"
            total_bad += 1
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()
    out.write(f"TOTAL differing outputs: {total_bad}\n")
    print(f"TOTAL differing outputs: {total_bad}")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
