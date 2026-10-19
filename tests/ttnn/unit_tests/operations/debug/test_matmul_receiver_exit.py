# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the production matmul receiver exit with ordinary and fused signaling."""
import os
from pathlib import Path

import pytest
import torch
import ttnn

KERNELS = Path(__file__).parent / "kernels/matmul_receiver_exit"


def _cores(*cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})


def _cb(index, size, cores, page=None):
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=_cores(*cores),
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page or size)
        ],
    )


def _kernel(name, cores, values=(), compile_args=(), **kwargs):
    args = ttnn.RuntimeArgs()
    for c in cores:
        args[c.x][c.y] = list(values)
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / name),
        core_ranges=_cores(*cores),
        runtime_args=args,
        compile_time_args=list(compile_args),
        config=ttnn.WriterConfigDescriptor(),
        **kwargs,
    )


@pytest.mark.parametrize("fused", [False, True], ids=["ordinary", "fused"])
def test_matmul_receiver_retires_atomics(device, fused):
    assert os.getenv("TT_METAL_WATCHER"), "enable watcher to validate every kernel exit"
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("one-card Blackhole component reproduction")
    grid = device.compute_with_storage_grid_size()
    a, f, b = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    pa, pb = device.worker_core_from_logical_core(a), device.worker_core_from_logical_core(b)
    output = ttnn.from_torch(
        torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    access = ttnn.TensorAccessorArgs(output).get_compile_time_args()
    cbs = [_cb(0, 16384, [a, b, f]), _cb(1, 2048, [a]), _cb(16, 2048, [a])]
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=_cores(a, b), initial_value=0) for i in range(4)]
    flood = _kernel("flood_neighbour.cpp", [f], [pb.x, pb.y, 100000, 4096])
    args = [pb.x, pb.y, output.buffer_address(), 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    compile_args = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0] + access
    if fused:
        # One master signals semaphore 3 on the remote consumer after output becomes ready.
        compile_args[18] = 1
        args += [1, 0, 2, pa.x, pa.y, 1, pb.x, pb.y, 3, 1]
    producer = _kernel(
        "matmul_receiver.cpp",
        [a],
        args,
        compile_args,
        defines=[("OUT_SHARDED", "1")],
        named_compile_time_args=[("cb_in1", 1), ("cb_out", 16)],
    )
    target = _kernel("matmul_peer.cpp", [b], [pa.x, pa.y])
    program = ttnn.ProgramDescriptor(kernels=[producer, target, flood], cbs=cbs, semaphores=semaphores)
    kind = "fused" if fused else "ordinary"
    count = int(os.getenv("TT_METAL_KERNEL_EXIT_DRAIN_ITERATIONS", "64"))
    for iteration in range(count):
        # generic_op takes input/output handles; all component traffic uses the explicit CBs.
        ttnn.generic_op([output, output], program)
        ttnn.synchronize_device(device)
        print(f"{kind}: completed_exits={iteration + 1}", flush=True)
    print(f"{kind}: completed {count} kernel exits with watcher enabled")
