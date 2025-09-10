# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import ParameterDict, ParameterList, preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_encoder import SegformerEncoder
from models.demos.segformer.tests.pcc.test_segformer_layer import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_layer,
)
from models.demos.segformer.tests.pcc.test_segformer_overlap_path_embeddings import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_overlap_path,
)
from models.demos.segformer.tt.common import preprocess_layernorm_parameter
from models.demos.segformer.tt.ttnn_segformer_encoder import TtSegformerEncoder
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerEncoder):
            parameters["patch_embeddings"] = {}
            for i in range(4):
                parameters["patch_embeddings"][i] = {}
                overlap_path_embedding_preprocess = create_customer_preprocessor_overlap_path(mesh_mapper)
                parameters["patch_embeddings"][i] = overlap_path_embedding_preprocess(
                    model.patch_embeddings[i], None, None, None
                )

            # block starts
            parameters["block"] = {}
            for i in range(4):
                parameters["block"][i] = {}
                for j in range(2):
                    parameters["block"][i][j] = {}
                    layer_preprocess = create_customer_preprocessor_layer(mesh_mapper)
                    parameters["block"][i][j] = layer_preprocess(model.block[i][j], None, None, None)

            parameters["layer_norm"] = {}
            for i in range(4):
                parameters["layer_norm"][i] = {}
                parameters["layer_norm"][i]["weight"] = preprocess_layernorm_parameter(
                    model.layer_norm[i].weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
                )
                parameters["layer_norm"][i]["bias"] = preprocess_layernorm_parameter(
                    model.layer_norm[i].bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
                )

        return parameters

    return custom_mesh_preprocessor


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
def test_segformer_encoder(batch_size, num_channels, height, width, device, model_location_generator):
    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)

    config = load_config("configs/segformer_encoder_config.json")
    reference_model = SegformerEncoder(config)
    target_prefix = f"encoder."
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_output = reference_model(torch_input_tensor)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
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
