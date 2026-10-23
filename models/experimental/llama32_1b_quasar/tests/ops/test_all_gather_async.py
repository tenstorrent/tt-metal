# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.all_gather_async``  (MULTI-DEVICE).

Model call sites:
  * models/llama32_1b/model.py:227  _all_gather_rmsnorm_tensor
  * models/llama32_1b/model.py:975  gather_and_untilize_logits
  * modules/attention/attention_1d.py:925  _all_gather_before_wo_prefill_fused
  * modules/sampling/sampling_1d.py:319/333  _argmax_all_gather

Kwarg structure copied from _all_gather_rmsnorm_tensor (model.py:227):
    persistent_output_buffer=None, dim=3, multi_device_global_semaphore=<handle>,
    num_links=1, topology=..., memory_config=..., barrier_semaphore=<handle>,
    chunks_per_sync=24, num_workers_per_link=4, num_buffers_per_channel=2.

In the model the semaphore handles come from a TT-CCL manager
(tt_ccl.get_and_cycle_ag_semaphore_handles / _barrier_semaphore_handle). Here we
create them directly via ttnn.create_global_semaphore over the worker sub-device,
mirroring the CCL unit tests. We assert shape/dtype of the gathered output.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# CCL collective — needs a multi-device mesh. The (1, 2) parametrization below is
# skipped automatically by the ttnn_mesh_device fixture on single-device systems
# (N150 / 2-node emulator present as one device), so it stays out of the emulator run
# while still exercising the op on real multi-device systems (N300 / T3K).


def _worker_crs(mesh):
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 2)], indirect=True)
def test_all_gather_async(ttnn_mesh_device, reset_seeds):
    mesh = ttnn_mesh_device
    if tuple(mesh.shape) == (1, 1):
        pytest.skip("multi-device op")

    num_devices = max(tuple(mesh.shape))

    shape = (1, 1, U.MAX_BATCH, U.DIM)
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, mesh_mapper=3)  # shard last dim across devices

    crs = _worker_crs(mesh)
    ag_semaphore = ttnn.create_global_semaphore(mesh, crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(mesh, crs, 0)

    gathered = ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=ag_semaphore,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=barrier_semaphore,
        chunks_per_sync=24,
        num_workers_per_link=4,
        num_buffers_per_channel=2,
    )

    assert gathered.shape[-1] == (U.DIM // num_devices) * num_devices
    U.assert_shape_dtype(gathered, dtype=ttnn.bfloat16, finite=False)
