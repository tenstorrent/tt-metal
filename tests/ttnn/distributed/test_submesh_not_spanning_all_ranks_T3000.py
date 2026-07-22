# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest


# This test is meant to run on a T3000 system, with devices split across two ranks.
def test_submesh_not_spanning_all_ranks_T3000():
    if not ttnn.using_distributed_env():
        pytest.skip("This test only makes sense in a distributed environment")

    # Create a mesh device that fits on one rank
    mesh_shape = ttnn.MeshShape(2, 2)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    # Allocate tensors on the mesh device, confirming that allocation works on active and inactive ranks
    a_0 = ttnn.from_torch(torch.randn(1024, 1024), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)
    b_0 = ttnn.from_torch(torch.randn(1024, 1024), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)

    # Perform any simple operation on the tensors, the operation should execute on the active ranks and be a nop on the
    # inactive ranks
    res = ttnn.experimental.minimal_matmul(a_0, b_0)
    ttnn.synchronize_device(mesh_device)

    # Perform another simple operation using the result of the first operation. The operation is a nop on the inactive
    # ranks, but it needs to return a result that can be used by following operations (they'll also be a nop on the
    # inactive ranks).
    ttnn.experimental.minimal_matmul(res, res)

    # PreparedGenericOp must follow the same inactive-rank contract: the active rank dispatches the program while the
    # rank without local devices retains the tensors and treats construction, dispatch, and synchronization as no-ops.
    output = ttnn.allocate_tensor_on_device(a_0.spec, mesh_device)
    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    kernel = ttnn.KernelDescriptor(
        kernel_source="void kernel_main() {}",
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_set,
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
    mesh_range = ttnn.MeshCoordinateRange(mesh_shape)
    mesh_program = ttnn.MeshProgramDescriptor({mesh_range: program})

    prepared = ttnn.prepare_generic_op([a_0, output], mesh_program, cq_id=0)
    assert prepared.output_tensor.buffer_address() == output.buffer_address()
    prepared.dispatch()
    prepared.dispatch()
    prepared.synchronize()
