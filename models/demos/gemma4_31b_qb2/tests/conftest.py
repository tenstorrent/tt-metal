# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM
from models.demos.utils.trace_region_sizes import build_trace_device_params


@pytest.fixture
def qb2_mesh():
    if (
        ttnn.get_num_devices() != 4
        or ttnn.get_arch_name() != "blackhole"
        or ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.P300_X2
    ):
        pytest.skip("Requires a four-device Blackhole P300_X2 QB2")
    ttnn.set_fabric_config(**Gemma4ForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4), l1_small_size=16384, **build_trace_device_params("gemma4-31b-qb2-decoder")
    )
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
