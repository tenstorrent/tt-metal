# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import torch.nn.functional as F
from models.experimental.vovnet.tt.vovnet import TtVoVNet
from models.experimental.vovnet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.vovnet.common import load_torch_model, VOVNET_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "model_name",
    (("hf_hub:timm/ese_vovnet19b_dw.ra_in1k"),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": VOVNET_L1_SMALL_SIZE}], indirect=True)
def test_vovnet_model_inference(device, model_name, reset_seeds, model_location_generator):
    model = load_torch_model(model_location_generator)

    parameters = custom_preprocessor(device, model.state_dict())

    tt_model = TtVoVNet(
        device=device,
        parameters=parameters,
        base_address="",
    )

    input = torch.rand(1, 3, 224, 224)
    model_output = model(input)
    core_grid = ttnn.CoreGrid(y=8, x=8)
    n, c, h, w = input.shape
    num_cores = core_grid.x * core_grid.y
    shard_h = (n * w * h + num_cores - 1) // num_cores
    grid_size = core_grid
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input = input.permute(0, 2, 3, 1)
    input = input.reshape(1, 1, h * w * n, c)

    min_channels = 16
    if c < min_channels:
        padding_c = min_channels - c
    input = F.pad(input, (0, padding_c), "constant", 0)
    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input = tt_input.to(device, input_mem_config)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    # Low pcc is expected and is tracked here - https://github.com/tenstorrent/tt-metal/issues/23474
    assert_with_pcc(model_output, tt_output_torch, 0.79)
