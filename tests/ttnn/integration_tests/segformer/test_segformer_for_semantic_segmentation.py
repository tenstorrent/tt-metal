# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from PIL import Image
import torch
import math

import requests
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters, ParameterDict, ParameterList
from models.demos.segformer.tt.ttnn_segformer_for_semantic_segmentation import (
    TtSegformerForSemanticSegmentation,
)
from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
    SegformerForSemanticSegmentationReference,
)
from tests.ttnn.integration_tests.segformer.test_segformer_model import (
    create_custom_preprocessor as create_custom_preprocessor_model,
)
from tests.ttnn.integration_tests.segformer.test_segformer_decode_head import (
    create_custom_preprocessor as create_custom_preprocessor_deocde_head,
)
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerForSemanticSegmentationReference):
            parameters["segformer"] = {}
            segformer_preprocess = create_custom_preprocessor_model(device)
            parameters["segformer"] = segformer_preprocess(model.segformer, None, None)
            parameters["decode_head"] = {}
            deocde_preprocess = create_custom_preprocessor_deocde_head(device)
            parameters["decode_head"] = deocde_preprocess(model.decode_head, None, None)

        return parameters

    return custom_preprocessor


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["sr", "proj", "dwconv", "linear_fuse", "classifier"]:
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_for_semantic_segmentation(device, is_ci_env):
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")
    config = torch_model.config

    reference_model = SegformerForSemanticSegmentationReference(config=config)
    state_dict = torch_model.state_dict()
    inputs = processor(images=image, return_tensors="pt")

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_output = reference_model(inputs.pixel_values)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )
    parameters = move_to_device(parameters, device)

    for i in range(4):
        parameters["decode_head"]["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["weight"], device=device
        )
        parameters["decode_head"]["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
            parameters["decode_head"]["linear_c"][i]["proj"]["bias"], device=device
        )

    ttnn_model = TtSegformerForSemanticSegmentation(config, parameters)

    sharded_input_enabled = 1

    if not sharded_input_enabled:
        torch_input_tensor_permuted = torch.permute(inputs.pixel_values, (0, 2, 3, 1))
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_permuted,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        torch_input_tensor_permuted = torch.permute(inputs.pixel_values, (0, 2, 3, 1))
        N, H, W, C = torch_input_tensor_permuted.shape
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 7),
                ),
            }
        )
        n_cores = 64
        shard_spec = ttnn.ShardSpec(
            shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardMode.PHYSICAL
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
        ttnn_input_tensor_unpadded = ttnn.from_torch(
            torch_input_tensor_permuted,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )
        ttnn_input_tensor = ttnn.pad(ttnn_input_tensor_unpadded, [N, H, W, 8], [0, 0, 0, 0], 0)

    ttnn_output = ttnn_model(
        device,
        ttnn_input_tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parameters=parameters,
    )

    ttnn_output = ttnn.to_torch(ttnn_output.logits)
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
    h = w = int(math.sqrt(ttnn_output.shape[-1]))
    ttnn_final_output = torch.reshape(ttnn_output, (ttnn_output.shape[0], ttnn_output.shape[1], h, w))

    assert_with_pcc(torch_output.logits, ttnn_final_output, pcc=0.984)
