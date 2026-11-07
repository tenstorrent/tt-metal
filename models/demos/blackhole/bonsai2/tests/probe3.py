import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.pack import pack_ternary_tiles

K = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"
device = ttnn.open_device(device_id=0)
try:
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rng = np.random.default_rng(3)
    t = rng.integers(-1, 2, size=(32, 32)).astype(np.int8)
    for name, x in (("ones", torch.ones(1, 32)), ("onehot5", torch.eye(32)[5:6]), ("onehot20", torch.eye(32)[20:21])):
        xt = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        wt = ttnn.from_torch(
            torch.from_numpy(pack_ternary_tiles(t).view(np.int32)).reshape(1, 64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
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
                total_size=256,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat8_b, page_size=256)],
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
    const auto W = TensorAccessor(ws, wa, 256);
    cb_reserve_back(0, 1); noc_async_read(X.get_noc_addr(0), get_write_ptr(0), 64);
    cb_reserve_back(1, 1); noc_async_read(W.get_noc_addr(0), get_write_ptr(1), 256);
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
                    kernel_source=K + "probe3_compute.cpp",
                    core_ranges=core,
                    compile_time_args=[],
                    runtime_args=[],
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        o = ttnn.to_torch(ttnn.generic_op([xt, wt, out], prog)).float()[0]
        exp = (x.float() @ torch.from_numpy(t).float())[0]
        print(name, "match cols0-15", torch.equal(o[:16], exp[:16]), "cols16-31", torch.equal(o[16:], exp[16:]))
        print("  got  ", o[14:22].tolist())
        print("  exp  ", exp[14:22].tolist())
        # which weight row/col does got[16:] match?
        if not torch.equal(o[16:], exp[16:]):
            tt_ = torch.from_numpy(t).float()
            for k0 in range(32):
                if name.startswith("onehot") and torch.equal(o[16:], tt_[k0, 16:]):
                    print("  cols16-31 equal t[%d,16:]" % k0)
                if name.startswith("onehot") and torch.equal(o[16:], tt_[k0, :16]):
                    print("  cols16-31 equal t[%d,:16]" % k0)
finally:
    ttnn.close_device(device)
