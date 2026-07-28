# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.all_gather``  (MULTI-DEVICE).

Model call site (modules/sampling/sampling_1d.py):
  * L624  _perform_all_gather — ttnn.all_gather(tensor, dim=dim, num_links=num_links,
              memory_config=..., cluster_axis=cluster_axis, topology=ttnn.Topology.Linear)

This is the non-async all-gather used to reconstruct the full vocab from
per-device top-k shards. The isolation reproducer
(tests/modules/sampling/test_sampling_1d.py:test_topk_allgather_isolation, L1029)
uses dim=3 and cluster_axis = None if 1 in cluster_shape else 0.

Here we shard a tensor along the last dim across the mesh and all-gather it back;
the gathered tensor should span the full (num_devices * shard) width. We assert
shape/dtype (the emulator default mesh (1,1) is skipped — this op only runs
multi-device).

# TODO: verify on device.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# CCL collective — multi-device Ring topology only (T3K/Galaxy). NOT called by the
# model on a single-device N150 or the 2-node emulator (get_num_devices() == 1 takes
# the plain non-CCL path). Skipped so it drops cleanly out of an emulator run.
# Remove this mark to exercise it on a real multi-device Ring mesh.
pytestmark = pytest.mark.skip(reason="CCL op: multi-device Ring only; not used on single-device N150/emulator")


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 2)], indirect=True)
def test_all_gather(ttnn_mesh_device, reset_seeds):
    mesh = ttnn_mesh_device
    if tuple(mesh.shape) == (1, 1):
        pytest.skip("multi-device op")

    cluster_shape = tuple(mesh.shape)
    num_devices = max(cluster_shape)
    cluster_axis = None if 1 in cluster_shape else 0

    # Full tensor sharded along the last dim across devices (each device holds DIM/num_devices).
    shape = (1, 1, U.MAX_BATCH, U.DIM)
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, mesh_mapper=3)  # shard along torch dim 3

    gathered = ttnn.all_gather(
        x,
        dim=3,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
    )

    # After gather each device holds the full width again.
    assert (
        gathered.shape[-1] == (U.DIM // num_devices) * num_devices
    ), f"gathered width {gathered.shape[-1]} != full {U.DIM}"
    U.assert_shape_dtype(gathered, dtype=ttnn.bfloat16, finite=False)
