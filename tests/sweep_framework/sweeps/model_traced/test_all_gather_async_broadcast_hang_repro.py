# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Repro for all_gather_async use_broadcast=True hang (CI run 25358960768).

The first use_broadcast=True config (hash 1f53dcab) caused a fetch-queue
timeout on CI runner g04glx03, cascading into 26 further device timeouts.
Passes locally on UF-EV-B5-GWH02 in main-process mode (--vector-id /
--main-proc-verbose). This pytest target allows isolated dispatch to
determine if the hang is child-process / runner-environment specific.

Usage:
    pytest tests/sweep_framework/sweeps/model_traced/test_all_gather_async_broadcast_hang_repro.py -v
"""

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_blackhole


@skip_for_blackhole("Galaxy 4x8 (Wormhole) only")
@pytest.mark.parametrize(
    "mesh_device, device_params, all_gather_topology",
    [
        (
            (4, 8),
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["mesh_device", "device_params"],
    ids=["fabric2d_linear"],
)
def test_all_gather_async_broadcast_hang(mesh_device, all_gather_topology):
    """Exact config from CI failure: hash 1f53dcab, deepseek_v3 demo.py.

    Input: (1,1,4,16384) BF16 ROW_MAJOR L1_INTERLEAVED
    Placement: PlacementShard(1), PlacementShard(-1) on 4x8 mesh
    Gather: dim=2, cluster_axis=1, use_broadcast=True, topology=Linear
    """
    if mesh_device.shape[0] < 4 or mesh_device.shape[1] < 8:
        pytest.skip(f"requires Galaxy 4x8 mesh, got {mesh_device.shape}")

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))
    run_all_gather_impl(
        submesh_device,
        all_gather_topology=all_gather_topology,
        num_devices=8,
        ag_output_shape=[1, 1, 32, 16384],
        dim=2,
        num_links=4,
        ag_input_dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mem_config_input=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        mem_config_ag=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        cluster_axis=1,
        use_broadcast=True,
        use_explicit_subdevice_id=False,
        enable_trace=False,
        num_iters=1,
    )
