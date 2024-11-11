# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
import pytest


from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)
import ttnn

from models.demos.wormhole.swin.demo_utils import get_data_loader, get_batch, preprocess
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.swin.tt import ttnn_optimized_swin
from transformers import SwinForImageClassification as HF_SwinForImageClassification
from models.demos.wormhole.swin.tt.swin_utils import get_relative_position, get_attn_mask


def run_swin_imagenet_inference(
    batch_size,
    iterations,
    imagenet_label_dict,
    model_location_generator,
    mesh_device,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    # Setup model
    torch_model = HF_SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    config = torch_model.config
    torch_model.to(torch.bfloat16)
    torch_model.eval()

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            model_name=torch_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
            custom_preprocessor=ttnn_optimized_swin.custom_preprocessor,
        )

    # load inputs
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    bias_table = get_relative_position(
        torch_model.config, parameters.swin, inputs_mesh_mapper, mesh_device, output_mesh_composer
    )
    attention_mask_list = get_attn_mask(
        torch_model.config, inputs_mesh_mapper, mesh_device, output_mesh_composer, batch_size
    )

    # load ImageNet batch by batch
    # and run inference
    correct = 0
    torch_ttnn_correct = 0
    torch_correct = 0
    for iter in range(iterations):
        predictions = []
        torch_predictions = []
        inputs, labels = get_batch(data_loader)
        torch_outputs = torch_model(inputs)
        tt_batched_input_tensor = ttnn.from_torch(
            inputs, mesh_mapper=inputs_mesh_mapper, device=mesh_device, layout=ttnn.TILE_LAYOUT
        )
        tt_output = ttnn_optimized_swin.swin_for_image_classification(
            torch_model.config,
            pixel_values=tt_batched_input_tensor,
            parameters=parameters,
            device=mesh_device,
            bias_table=bias_table,
            attention_mask_list=attention_mask_list,
            mesh_mapper=inputs_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )
        tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
        prediction = tt_output.argmax(dim=-1)
        torch_prediction = torch_outputs[0].argmax(dim=-1)
        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            torch_predictions.append(imagenet_label_dict[torch_prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- \n Torch Predicted label:{predictions[-1]} \tPredicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
            if imagenet_label_dict[labels[i]] == torch_predictions[-1]:
                torch_correct += 1
            if predictions[-1] == torch_predictions[-1]:
                torch_ttnn_correct += 1

        del tt_output, tt_batched_input_tensor, inputs, labels, predictions
    accuracy = correct / (batch_size * iterations)
    torch_accuracy = torch_correct / (batch_size * iterations)
    torch_ttnn_accuracy = torch_ttnn_correct / (batch_size * iterations)

    logger.info(f"Model Swin for Image Classification")
    logger.info(f"TTNN Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
    logger.info(f"Torch Accuracy for {batch_size}x{iterations} inputs: {torch_accuracy}")
    logger.info(f"Torch vs TTNN Accuracy for {batch_size}x{iterations} inputs: {torch_ttnn_accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((8, 5),),
)
def test_demo_imagenet(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device):
    run_swin_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device)
