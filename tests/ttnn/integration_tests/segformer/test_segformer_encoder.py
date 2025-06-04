# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import (
    preprocess_layernorm_parameter,
    preprocess_model_parameters,
    ParameterDict,
    ParameterList,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.integration_tests.segformer.test_segformer_overlap_path_embeddings import (
    create_custom_preprocessor as create_customer_preprocessor_overlap_path,
)
from tests.ttnn.integration_tests.segformer.test_segformer_layer import (
    create_custom_preprocessor as create_customer_preprocessor_layer,
)
from models.utility_functions import skip_for_grayskull

from transformers import SegformerModel, SegformerConfig
import pytest
from models.demos.segformer.tt.ttnn_segformer_encoder import (
    TtSegformerEncoder,
)
from models.demos.segformer.reference.segformer_encoder import SegformerEncoder


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerEncoder):
            parameters["patch_embeddings"] = {}
            for i in range(4):
                parameters["patch_embeddings"][i] = {}
                overlap_path_embedding_preprocess = create_customer_preprocessor_overlap_path(device)
                parameters["patch_embeddings"][i] = overlap_path_embedding_preprocess(
                    model.patch_embeddings[i], None, None
                )

            # block starts
            parameters["block"] = {}
            for i in range(4):
                parameters["block"][i] = {}
                for j in range(2):
                    parameters["block"][i][j] = {}
                    layer_preprocess = create_customer_preprocessor_layer(device)
                    parameters["block"][i][j] = layer_preprocess(model.block[i][j], None, None)

            parameters["layer_norm"] = {}
            for i in range(4):
                parameters["layer_norm"][i] = {}
                parameters["layer_norm"][i]["weight"] = preprocess_layernorm_parameter(
                    model.layer_norm[i].weight, dtype=ttnn.bfloat8_b
                )
                parameters["layer_norm"][i]["bias"] = preprocess_layernorm_parameter(
                    model.layer_norm[i].bias, dtype=ttnn.bfloat8_b
                )

        return parameters

    return custom_preprocessor


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["sr", "proj", "dwconv"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, num_channels, height, width",
    [
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_encoder(batch_size, num_channels, height, width, device, reset_seeds, is_ci_env):
    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config

    torch_model = torch_model.encoder
    config = SegformerConfig()
    state_dict = torch_model.state_dict()

    reference_model = SegformerEncoder(config)

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )
    parameters = move_to_device(parameters, device)

    ttnn_model = TtSegformerEncoder(config, parameters)

    sharded_input_enabled = 0

    if not sharded_input_enabled:
        torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_permuted,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    else:
        torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        N, H, W, C = torch_input_tensor.shape
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 7),
                ),
            }
        )
        n_cores = 64
        shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )

    ttnn_output = ttnn_model(device, ttnn_input_tensor, parameters=parameters)

    ttnn_final_output = ttnn.to_torch(ttnn_output.last_hidden_state)
    torch_final_output = torch.permute(torch_output.last_hidden_state, (0, 2, 3, 1))

    assert_with_pcc(torch_final_output, ttnn_final_output, pcc=0.929)
