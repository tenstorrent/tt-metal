# MAX REDUCE_ROW with denormal scalers, main against the branch (#58705's change runs the face-row pools under the preserve
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


def scalers(fmt):
    if fmt == "bf16":
        den = list(range(0x0001, 0x0080)) + list(range(0x8001, 0x8080))
        other = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x3F00, 0x0080, 0x8080, 0x7F80, 0xFF80, 0x7FC0, 0x7F7F]
        bits = torch.tensor(den + other, dtype=torch.int32)
        return bits, bits.to(torch.int16).view(torch.bfloat16).float()
    g = torch.Generator().manual_seed(5)
    den = torch.randint(1, 1 << 23, (256,), generator=g, dtype=torch.int64)
    den[128:] |= 1 << 31
    other = torch.tensor([0, 1 << 31, 0x3F800000, 0xBF800000, 0x00800000, 0x7F800000], dtype=torch.int64)
    bits = torch.cat([den, other]).to(torch.int64)
    return bits, bits.to(torch.int32).view(torch.float32)


def run(device, fmt, fp32_dest):
    dtype = ttnn.bfloat16 if fmt == "bf16" else ttnn.float32
    page = 4096 if dtype == ttnn.float32 else 2048
    odtype = ttnn.float32 if fp32_dest else dtype
    opage = 4096 if odtype == ttnn.float32 else 2048
    bits, sv = scalers(fmt)
    n = sv.numel()
    g = torch.Generator().manual_seed(11)
    data = torch.randn(n, 32, 32, generator=g) * torch.exp2(torch.randint(-6, 6, (n, 32, 32), generator=g).float())
    mask = torch.rand(n, 32, 32, generator=g)
    data[mask < 0.05] = 0.0
    data[(mask >= 0.05) & (mask < 0.10)] = 1e-39 if fmt == "bf16" else 1e-40  # denormal data
    sc = sv.reshape(n, 1, 1).expand(n, 32, 32).contiguous()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tx = ttnn.from_torch(data.reshape(1, n, 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ts = ttnn.from_torch(sc.reshape(1, n, 32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
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
    return {"out": y, "scalers": bits}


def main():
    device = ttnn.open_device(device_id=0)
    res = {}
    for fmt, fp32_dest in (("bf16", False), ("bf16", True), ("fp32", True)):
        k = f"{fmt} fp32_dest={fp32_dest}"
        res[k] = run(device, fmt, fp32_dest)
        print(f"CASE {k} ok", flush=True)
    torch.save(res, sys.argv[1])
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
