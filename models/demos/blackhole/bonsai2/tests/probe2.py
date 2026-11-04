# Where do a copied [1,32] tile and a custom_mm [1,32] output land in DEST (packed out as a full 32x32 fp32 tile)?
import torch

import ttnn

K = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"
device = ttnn.open_device(device_id=0)
try:
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    x = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32).bfloat16()
    w = torch.eye(32).bfloat16()
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    wt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for mode in (0, 1):
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, device)
        tiny = ttnn.TileDescriptor(1, 32, False)
        cbs = [
            ttnn.CBDescriptor(
                total_size=64,
                core_ranges=core,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=64, tile=tiny)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=2048,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=2048)],
            ),
            ttnn.CBDescriptor(
                total_size=4096,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.float32, page_size=4096)],
            ),
        ]
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rd[0][0] = [xt.buffer_address(), wt.buffer_address()]
        wr[0][0] = [out.buffer_address(), 1]
        reader = """
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t xa = get_arg_val<uint32_t>(0), wa = get_arg_val<uint32_t>(1);
    constexpr auto xs = TensorAccessorArgs<0>();
    constexpr auto ws = TensorAccessorArgs<xs.next_compile_time_args_offset()>();
    const auto X = TensorAccessor(xs, xa, 64);
    const auto W = TensorAccessor(ws, wa, 2048);
    cb_reserve_back(0, 1); noc_async_read(X.get_noc_addr(0), get_write_ptr(0), 64);
    cb_reserve_back(1, 1); noc_async_read(W.get_noc_addr(0), get_write_ptr(1), 2048);
    noc_async_read_barrier(); cb_push_back(0, 1); cb_push_back(1, 1);
}"""
        rct = ttnn.TensorAccessorArgs(xt).get_compile_time_args() + ttnn.TensorAccessorArgs(wt).get_compile_time_args()
        prog = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=reader,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=core,
                    compile_time_args=rct,
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=K + "probe_writer.cpp",
                    core_ranges=core,
                    compile_time_args=ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=K + "probe2_compute.cpp",
                    core_ranges=core,
                    compile_time_args=[mode],
                    runtime_args=[],
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        o = ttnn.to_torch(ttnn.generic_op([xt, wt, out], prog)).float()
        nz = (o != 0).nonzero().tolist()
        print("mode", mode, "nonzero count", len(nz))
        rows = sorted(set(r for r, c in nz))
        for r in rows[:6]:
            print("  row", r, o[r].tolist())
finally:
    ttnn.close_device(device)
