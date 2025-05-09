# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn

from models.experimental.functional_unet.tests.common import verify_with_pcc, UNET_L1_SMALL_REGION_SIZE


@pytest.mark.parametrize("batch, groups", [(1, 4)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_output_layer(batch, groups, device, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(batch, groups, input_channels=16)
    torch_output = model.output_layer(torch_input)

    # TODO: Either infer these for get them from the model implementation
    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
        }
    )
    input_shard_shape = (2688, 16 * groups)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device, memory_config=input_memory_config)

    ttnn_output = ttnn_model.output_layer(ttnn_input)
    ttnn_output = ttnn_model.postprocess_output_tensor(ttnn_output)

    B, C, H, W = torch_output.shape
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert list(ttnn_output.shape) == [B, 1, C, H * W], "Expected output layer to be [1, 1, BHW, C]"
    ttnn_output = ttnn_output.reshape(B, C, H, W)
    verify_with_pcc(torch_output, ttnn_output, 0.9999)
