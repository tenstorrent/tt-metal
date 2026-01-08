# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import torch_random
from models.demos.blackhole.vit.tt import ttnn_optimized_sharded_vit_hiRes_bh as ttnn_optimized_sharded_vit
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1024, 2048, 3072])
@pytest.mark.parametrize("hidden_size", [512, 1024, 1536, 2304])
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 0}],
    indirect=True,
)
def test_vit_layer(device, model_name, batch_size, sequence_size, hidden_size, model_location_generator):
    """
    Test ViT encoder layer with variable sequence lengths and hidden dimensions.

    Supported configurations:
        - batch: 1, 4, 8
        - sequence_length: 1024, 2048, 3072
        - hidden_size: 512, 1024, 1536, 2304
        - heads: 16 (12 for hidden_size >= 2048)
        - layers: 12
        - MLP intermediate size: 4x hidden_size
    """
    torch.manual_seed(0)

    config = transformers.ViTConfig.from_pretrained(model_name)
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size * 4
    config.num_attention_heads = 16
    if config.hidden_size >= 2048:
        config.num_attention_heads = 12
    config.num_hidden_layers = 12
    config = ttnn_optimized_sharded_vit.update_model_config(config, batch_size, sequence_size)

    model = transformers.models.vit.modeling_vit.ViTLayer(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -1, 1, dtype=torch.float32)
    torch_output, *_ = model(torch_hidden_states)
    torch_hidden_states = torch_hidden_states.unsqueeze(1)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_sharded_vit.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    encoder_input = ttnn.to_memory_config(
        hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            hidden_states.shape,
            core_grid=config.core_grid_BLOCK_SHARDED,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(hidden_states)

    output = ttnn_optimized_sharded_vit.vit_layer(
        config,
        encoder_input,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    output = output.squeeze(1)
    assert_with_pcc(torch_output, output, 0.985)
