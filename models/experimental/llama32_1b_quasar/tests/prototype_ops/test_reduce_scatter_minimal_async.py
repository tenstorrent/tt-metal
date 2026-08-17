# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.reduce_scatter_minimal_async``  (MULTI-DEVICE).

Model call sites (modules/attention/attention_1d.py):
  * L1121  _all_reduce_output_decode  — reduce-scatter of the WO output (decode)
  * L1164  _all_reduce_output_prefill — reduce-scatter of the WO output (prefill)

Kwarg structure copied from _all_reduce_output_decode (attention_1d.py:1121):
    persistent_output_buffers=None, dim=3, multi_device_global_semaphore=<handle>,
    barrier_semaphore=<handle>, num_links=..., memory_config=...,
    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=...,
    chunks_per_sync=..., num_workers_per_link=..., num_buffers_per_channel=...

Semaphore handles come from a TT-CCL manager in the model
(tt_ccl.get_and_cycle_rs_semaphore_handles / _barrier_semaphore_handle). Here we
create them directly via ttnn.create_global_semaphore over the worker sub-device.
reduce_scatter sums the replicated input across devices and scatters the result
along dim 3, so the per-device output width is input_width / num_devices. We
assert shape/dtype.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# CCL collective — needs a multi-device mesh. The (1, 2) parametrization below is
# skipped automatically by the ttnn_mesh_device fixture on single-device systems
# (N150 / 2-node emulator present as one device), so it stays out of the emulator run
# while still exercising the op on real multi-device systems (N300 / T3K).


def _worker_crs(mesh):
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 2)], indirect=True)
def test_reduce_scatter_minimal_async(ttnn_mesh_device, reset_seeds):
    mesh = ttnn_mesh_device
    if tuple(mesh.shape) == (1, 1):
        pytest.skip("multi-device op")

    num_devices = max(tuple(mesh.shape))

    # WO output replicated across devices (each device contributes a full-width partial).
    shape = (1, 1, U.MAX_BATCH, U.DIM)
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # replicated

    crs = _worker_crs(mesh)
    rs_semaphore = ttnn.create_global_semaphore(mesh, crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(mesh, crs, 0)

    reduced = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=rs_semaphore,
        barrier_semaphore=barrier_semaphore,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # reduce-scatter sums the replicated input across devices then scatters along dim 3 -> width/num_devices
    # per device. Every device was fed the same x, so the composed (concatenated) result is num_devices * x.
    assert reduced.shape[-1] == U.DIM // num_devices, f"scattered width {reduced.shape[-1]} != {U.DIM // num_devices}"
    U.assert_pcc(x_torch.float() * num_devices, reduced, pcc=0.999, mesh_device=mesh)
