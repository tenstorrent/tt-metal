# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from PIL import Image

import requests
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters, ParameterDict, ParameterList
from models.experimental.functional_segformer.tt.ttnn_segformer_for_semantic_segmentation import (
    TtSegformerForSemanticSegmentation,
)
from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from models.experimental.functional_segformer.reference.segformer_for_semantic_segmentation import (
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
    if is_ci_env:
        pytest.skip("Skip in CI, model is WIP, issue# 13357")

    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")
    config = torch_model.config

    reference_model = SegformerForSemanticSegmentationReference(config=config)
    state_dict = torch_model.state_dict()
    inputs = processor(images=image, return_tensors="pt")

    ttnn_input_tensor = ttnn.from_torch(
        inputs.pixel_values,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

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

    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parameters=parameters,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output.logits)

    assert_with_pcc(torch_output.logits, ttnn_final_output, pcc=0.99)
