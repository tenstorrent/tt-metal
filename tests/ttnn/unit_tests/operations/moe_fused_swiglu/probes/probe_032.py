import sys

sys.stdout.reconfigure(line_buffering=True)
import ttnn

READER = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_local = get_compile_time_arg_val(0);
    constexpr uint32_t cb_incoming = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    // seed both CBs with SOMETHING (garbage is fine -- we only care whether the kernel completes)
    cb_reserve_back(cb_local, num_tiles);
    cb_push_back(cb_local, num_tiles);
    cb_reserve_back(cb_incoming, num_tiles);
    cb_push_back(cb_incoming, num_tiles);
}
"""

COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
using namespace compute_kernel_lib;
void kernel_main() {
    constexpr uint32_t cb_local = get_compile_time_arg_val(0);
    constexpr uint32_t cb_incoming = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cb_local, cb_incoming, cb_local);
    add<input(cb_local), input(cb_incoming), output(cb_local)>(EltwiseShape::tiles(num_tiles));
}
"""

WRITER = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {}
"""


def make_cb(idx, core_range, n, page_bytes=1088):
    return ttnn.CBDescriptor(
        total_size=n * page_bytes,
        core_ranges=core_range,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat8_b, page_size=page_bytes)
        ],
    )


device = ttnn.open_device(device_id=0)
try:
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    N = 12
    cbs = [make_cb(2, core, N), make_cb(3, core, N)]
    reader = ttnn.KernelDescriptor(
        kernel_source=READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[2, 3, N],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[2, 3, N],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            bfp8_pack_precise=True,
        ),
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    print("about to run minimal in-place add probe", flush=True)
    # generic_op needs at least one tensor arg per its signature; use a tiny dummy DRAM tensor.
    import torch

    dummy = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.generic_op([dummy], descriptor)
    ttnn.synchronize_device(device)
    print("MINIMAL PROBE COMPLETED WITHOUT HANGING", flush=True)
finally:
    ttnn.close_device(device)
