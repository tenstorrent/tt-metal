# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from loguru import logger
from datasets import load_dataset
from transformers import SegformerForImageClassification, AutoImageProcessor
from ttnn.model_preprocessing import preprocess_model
from models.experimental.functional_segformer.tt.ttnn_segformer_image_classification import (
    TtSegformerForImageClassification,
)
from datasets import load_dataset
from transformers import SegformerForImageClassification, AutoImageProcessor
from models.experimental.functional_segformer.reference.segformer_image_classification import (
    SegformerForImageClassificationReference,
)
from tests.ttnn.integration_tests.segformer.test_segformer_image_classification import create_custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_demo_ima(device):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    torch_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
    config = torch_model.config
    reference_model = SegformerForImageClassificationReference(config=config)
    state_dict = torch_model.state_dict()
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
    inputs = image_processor(image, return_tensors="pt")
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

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(inputs.pixel_values),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerForImageClassification(config, parameters, reference_model)

    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parameters=parameters,
        model=reference_model,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output.logits)

    predicted_label = ttnn_final_output.argmax(-1).item()

    logger.info("Output")
    logger.info(reference_model.config.id2label[predicted_label])
