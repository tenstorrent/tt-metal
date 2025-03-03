# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.utility_functions import comp_pcc, skip_for_blackhole, run_for_wormhole_b0
import ttnn.experimental


@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_sanity(mesh_device):
    pass


@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_reduce_scatter(n300_mesh_device):
    dim = 3
    input = torch.rand((1, 1, 32, 64), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        input,
        mesh_mapper=ttnn.ShardTensorToMesh(n300_mesh_device, dim),
        device=n300_mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    print(tt_input)
    output = ttnn.experimental.llama_reduce_scatter(tt_input, dim)
    print(output)


@skip_for_blackhole()
# @pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_reduce_scatter_sd(device):
    dim = 3
    input = torch.rand((1, 1, 32, 64), dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(input, device=device, layout=ttnn.TILE_LAYOUT)
    print(tt_input)
    tt_output = ttnn.experimental.llama_reduce_scatter(tt_input, dim)
    print(tt_output)
