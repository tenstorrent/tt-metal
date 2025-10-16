# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vanilla_unet_new.tt.common import (
    VANILLA_UNET_L1_SMALL_SIZE,
    VANILLA_UNET_PCC_WH,
    create_unet_preprocessor,
    load_reference_model,
)
from models.demos.vanilla_unet_new.tt.config import create_unet_configs_from_parameters
from models.demos.vanilla_unet_new.tt.model import create_unet_from_configs
from models.experimental.functional_unet.tt.model_preprocessing import create_unet_input_tensors
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch, input_channels, input_height, input_width", [(1, 3, 480, 640)])
def test_vanilla_unet_model(
    batch, input_channels, input_height, input_width, device, reset_seeds, model_location_generator
):
    reference_model = load_reference_model(model_location_generator)

    input_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
        }
    )
    input_shard_shape = (input_channels, input_height * input_width // 64)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_sharded_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(
        batch=batch,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        channel_order="first",
        pad=False,
        fold=True,
        device=device,
        memory_config=input_sharded_memory_config,
    )
    torch_output_tensor = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_unet_preprocessor(device), device=None
    )
    configs = create_unet_configs_from_parameters(
        parameters=parameters, input_height=input_height, input_width=input_width, batch_size=batch
    )
    model = create_unet_from_configs(configs, device)

    ttnn_output = model(ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)  # NHWC -> NCHW
    ttnn_output = ttnn_output.reshape(torch_output_tensor.shape)

    assert ttnn_output.shape == torch_output_tensor.shape
    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output, pcc=VANILLA_UNET_PCC_WH)
    logger.info(f"PCC check was successful ({pcc_message:.5f} > {VANILLA_UNET_PCC_WH:.5f})")
