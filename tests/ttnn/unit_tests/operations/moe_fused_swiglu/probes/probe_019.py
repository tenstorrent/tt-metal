import ttnn
import torch


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_cfg(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_b = 2, cb_out = 4;
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    // trivial: nothing else. If this alone hangs, the bug is structural (program descriptor / CB setup),
    // not in the op body.
}
"""

device = ttnn.open_device(device_id=0)
try:
    m, n = 32, 192
    mem = sharded_cfg((m, n))
    a = ttnn.from_torch(
        torch.zeros(m, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )
    b = ttnn.from_torch(
        torch.zeros(m, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([m, n]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, mem)

    compute = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, a),
        ttnn.cb_descriptor_from_sharded_tensor(2, b),
        ttnn.cb_descriptor_from_sharded_tensor(4, out),
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
    result = ttnn.generic_op([a, b, out], descriptor)
    ttnn.synchronize_device(device)
    print("MINIMAL KERNEL: OK, no hang")
finally:
    ttnn.close_device(device)
