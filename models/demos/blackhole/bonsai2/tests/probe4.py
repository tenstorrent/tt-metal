import torch

import ttnn

K = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"
device = ttnn.open_device(device_id=0)
try:
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    z = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, device)
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i in (0, 16)
    ]
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd[0][0] = [z.buffer_address(), 0, 0]
    wr[0][0] = [out.buffer_address(), 1]
    reader = """
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t za = get_arg_val<uint32_t>(0);
    constexpr auto zs = TensorAccessorArgs<0>();
    const auto Z = TensorAccessor(zs, za, 4096);
    cb_reserve_back(0, 1); noc_async_read(Z.get_noc_addr(0), get_write_ptr(0), 4096); noc_async_read_barrier(); cb_push_back(0, 1);
}"""
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=reader,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=core,
                compile_time_args=ttnn.TensorAccessorArgs(z).get_compile_time_args(),
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
                kernel_source=K + "probe4_compute.cpp",
                core_ranges=core,
                compile_time_args=[],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    o = ttnn.to_torch(ttnn.generic_op([z, out], prog)).float()
    for v in (1, 2, 3, 9, 10):
        cells = (o == v).nonzero().tolist()
        rows = sorted(set(r for r, c in cells))
        cols = sorted(set(c for r, c in cells))
        print(f"dst_reg[{ {1:0,2:1,3:2,9:8,10:9}[v] }] -> {len(cells)} cells rows {rows} cols {cols}")
finally:
    ttnn.close_device(device)
