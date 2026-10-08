# MAX REDUCE_ROW, every value: main against a branch form, bit for bit (run once per farm; compare with --cmp a.pt b.pt).
# Every bf16 pattern (65536) and every TF32 pattern (2^19), one per row at a varying column, the rest of the row the lowest
# finite value, then the same with the rest of the row 0, and a dense layout of random values over 16 binades; scaler 1.0;
# bf16 with a 16-bit and an fp32 DEST, fp32 with an fp32 DEST.
# (Kept from maxrow_denorm.py:) MAX REDUCE_ROW with denormal scalers, main against the branch (#58705's change runs the face-row pools under the preserve
# value of the Src zero flag; main ran them under the operand default): every bf16 denormal scaler (254), the zeros and
# a few normal and special scalers, one per output tile, each against its own data tile (random normals of both signs with
# denormals and zeros mixed in), bf16 with a 16-bit and an fp32 DEST, and fp32 data with 256 fp32 denormal scalers.
# usage: python maxrow_denorm.py <out.pt>   (run once per farm; compare with --cmp a.pt b.pt)
import os
import sys

import torch

if sys.argv[1] == "--cmp":
    a, b = torch.load(sys.argv[2], weights_only=True), torch.load(sys.argv[3], weights_only=True)
    tot = 0
    for k in a:
        x, y = a[k]["out"], b[k]["out"]
        xi = x.contiguous().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)
        yi = y.contiguous().view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)
        per_tile = (xi != yi).reshape(-1, 1024).any(dim=1)
        sc = a[k]["scalers"]
        bad = [hex(int(v)) for v, d in zip(sc, per_tile) if d]
        tot += len(bad)
        print(f"{k}: {per_tile.numel()} scalers, {len(bad)} with a differing output tile" + (f": {bad[:20]}" if bad else ""))
    print(f"TOTAL scalers with a differing output: {tot}")
    sys.exit(0)

# 05 (12:35 UTC): the device only under scripts/hwlock.sh (it exports HWLOCK_HELD); the mock cluster (an existing descriptor) needs no lock
if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")

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



def layouts(fmt):
    if fmt == "bf16":
        p = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
        low = -3.3895313892515355e38
    else:
        p = (torch.arange(0, 1 << 19, dtype=torch.int64) << 13).to(torch.int32).view(torch.float32)
        low = -3.4028234663852886e38
    n = p.numel()
    col = torch.arange(n) % 32
    out = {}
    for name, fill in (("low", low), ("zero", 0.0)):
        x = torch.full((n, 32), fill)
        x[torch.arange(n), col] = p
        out[name] = x.reshape(n // 32, 32, 32)
    g = torch.Generator().manual_seed(23)
    shape = (2048, 32, 32)
    out["dense"] = torch.randn(shape, generator=g) * torch.exp2(torch.randint(-8, 8, shape, generator=g).float())
    return out


def run(device, data, fmt, fp32_dest):
    dtype = ttnn.bfloat16 if fmt == "bf16" else ttnn.float32
    page = 4096 if dtype == ttnn.float32 else 2048
    odtype = ttnn.float32 if fp32_dest else dtype
    opage = 4096 if odtype == ttnn.float32 else 2048
    n = data.shape[0]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tx = ttnn.from_torch(data.reshape(1, n, 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ts = ttnn.from_torch(torch.ones(1, n, 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ty = ttnn.from_torch(torch.zeros(1, n, 32, 32), dtype=odtype, layout=ttnn.TILE_LAYOUT, device=device)
    cb = lambda i, pages, dt=dtype, pg=page: ttnn.CBDescriptor(
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
    y = ttnn.to_torch(ty)
    for t in (tx, ts, ty):
        ttnn.deallocate(t)
    return y


def main():
    device = ttnn.open_device(device_id=0)
    res = {}
    for fmt, fp32_dest in (("bf16", False), ("bf16", True), ("tf32", True)):
        for name, data in layouts(fmt).items():
            k = f"{fmt} {name} fp32_dest={fp32_dest}"
            res[k] = {"out": run(device, data, fmt, fp32_dest), "scalers": torch.zeros(data.shape[0], dtype=torch.int64)}
            print(f"CASE {k} ok", flush=True)
    torch.save(res, sys.argv[1])
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
