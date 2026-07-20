# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("ag_output_shape", [[1, 1, 256, 256]])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
)
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
    ],
    indirect=True,
)
def test_all_gather_2d_fabric(
    bh_2d_mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    # Reshape mesh_device to match its default MeshGraphDescriptor device_topology to prevent hang
    system_mesh_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()
    bh_2d_mesh_device.reshape(system_mesh_desc.local_shape())

    run_all_gather_impl(
        bh_2d_mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        allowed_pcc=0.9999,
    )
