# Exhaustive bit identity of the power-of-two-scaler fidelity (#58706): every bf16 pattern (65536) and every TF32 pattern
# (2^19, the values an fp32 input reaches the FPU as) through REDUCE_ROW, REDUCE_COL (one value per output row or column,
# zeros elsewhere) and REDUCE_SCALAR (full tiles of distinct patterns, one output per tile) SUM, per tile and, for ROW and
# SCALAR, as one block call of 8 tiles (the value in the first tile), with the scaler 1.0 and 2^-9, a 16-bit and an fp32
# DEST, at the kernel's HiFi4 against the per-call fidelity (pow2_scaler: HiFi3 for ROW, HiFi2 for COL and SCALAR).
# usage: python exh_pow2.py <out.txt>
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
    Noc noc;
    DataflowBuffer d_in(0), d_sc(1);
    d_sc.reserve_back(1);
    noc.async_read(s, d_sc, page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    d_sc.push_back(1);
    for (uint32_t i = 0; i < n; ++i) {
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
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

#ifdef POW2
constexpr bool pow2 = true;
#else
constexpr bool pow2 = false;
#endif

void kernel_main() {
    constexpr uint32_t outputs = get_compile_time_arg_val(0);
    constexpr uint32_t per_out = get_compile_time_arg_val(1);
    constexpr uint32_t in = 0, sc = 1, out = 16;
    DataflowBuffer d_in(in), d_sc(sc), d_out(out);
    compute_kernel_hw_startup(in, sc, out);
    if constexpr (DIM == ReduceDim::REDUCE_ROW) {
        reconfig_data_format(sc, in);
    }
    reduce_init<PoolType::SUM, DIM, DST_ACCUM_MODE, pow2>(in, sc, out);
    d_sc.wait_front(1);
    for (uint32_t o = 0; o < outputs; ++o) {
        d_in.wait_front(per_out);
        tile_regs_acquire();
        if constexpr (per_out == 1) {
            reduce_tile<PoolType::SUM, DIM, DST_ACCUM_MODE, pow2>(in, sc, 0, 0, 0);
        } else {
            reduce_block<PoolType::SUM, DIM, DST_ACCUM_MODE, pow2>(in, sc, 0, 0, 0, per_out, 0);
        }
        tile_regs_commit();
        d_in.pop_front(per_out);
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


def patterns(fmt):
    if fmt == "bf16":
        return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    # TF32: sign, 8 exponent bits, 10 mantissa bits; the unpacker drops the low 13 bits of an fp32 mantissa
    return (torch.arange(0, 1 << 19, dtype=torch.int64) << 13).to(torch.int32).view(torch.float32)


def layout(p, dim, per_out):
    n = p.numel()
    if dim == "row":  # value i at row i, column 0 of the first tile of its group of per_out tiles
        x = torch.zeros(n, 32 * per_out)
        x[:, 0] = p
        return x.reshape(1, 1, n, 32 * per_out)
    if dim == "col":  # value j at row 0, column j
        x = torch.zeros(32, n)
        x[0, :] = p
        return x.reshape(1, 1, 32, n)
    tiles = n // 1024  # scalar: full tiles of distinct patterns, each followed by per_out - 1 zero tiles
    x = torch.zeros(tiles, per_out, 32, 32)
    x[:, 0] = p.reshape(tiles, 32, 32)
    return x.reshape(1, tiles * per_out, 32, 32)


def run(device, x, dim, per_out, scale, fp32_dest, pow2, dtype):
    page = 4096 if dtype == ttnn.float32 else 2048
    odtype = ttnn.float32 if fp32_dest else dtype  # an fp32 DEST packs to fp32 so the comparison sees every DEST bit
    opage = 4096 if odtype == ttnn.float32 else 2048
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tx = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    n_in = x.numel() // 1024
    outputs = n_in // per_out
    ts = ttnn.from_torch(torch.full((1, 1, 32, 32), scale), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ty = ttnn.from_torch(torch.zeros(1, outputs, 32, 32), dtype=odtype, layout=ttnn.TILE_LAYOUT, device=device)
    cb = lambda i, pages, dt=dtype, pg=page: ttnn.CBDescriptor(
        total_size=pages * pg,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=pg)],
    )
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd[0][0] = [tx.buffer_address(), ts.buffer_address(), n_in, page]
    wr[0][0] = [ty.buffer_address(), outputs, opage]
    defines = [("DIM", DIMS[dim])] + ([("POW2", "1")] if pow2 else [])
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
                compile_time_args=[outputs, per_out],
                defines=defines,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_dest
                ),
            ),
        ],
        semaphores=[],
        cbs=[cb(0, 2 * per_out), cb(1, 1), cb(16, 2, odtype, opage)],
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
    for fmt, dim, per_out, scale, fp32_dest in itertools.product(
        ("bf16", "tf32"), ("row", "col", "scalar"), (1, 8), (1.0, 2.0**-9), (False, True)
    ):
        if per_out == 8 and dim == "col":
            continue
        if fmt == "tf32" and (not fp32_dest or (dim == "row" and per_out == 8)):
            continue  # fp32 data runs with an fp32 DEST; its ROW block input would be 512 MB
        dtype = ttnn.bfloat16 if fmt == "bf16" else ttnn.float32
        x = layout(patterns(fmt), dim, per_out)
        try:
            a = run(device, x, dim, per_out, scale, fp32_dest, False, dtype)
            b = run(device, x, dim, per_out, scale, fp32_dest, True, dtype)
            ia = a.contiguous().view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32)
            ib = b.contiguous().view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32)
            diff = int((ia != ib).sum())
            total_bad += diff
            nan = int(torch.isnan(a.float()).sum())
            line = (
                f"{fmt} {dim} per_out={per_out} scale={scale:g} fp32_dest={fp32_dest}: "
                f"{a.numel()} outputs, {diff} differ, {nan} NaN at HiFi4"
            )
        except Exception as e:
            line = f"{fmt} {dim} per_out={per_out} scale={scale:g} fp32_dest={fp32_dest}: ERROR {str(e).splitlines()[0][:200]}"
            total_bad += 1
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()
    out.write(f"TOTAL differing outputs: {total_bad}\n")
    print(f"TOTAL differing outputs: {total_bad}")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
