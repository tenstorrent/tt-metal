# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.classification_models_evaluation.classification_eval_utils import get_data_loader


def evaluation(
    device=None,
    model=None,
    inputs=None,
    parameters=None,
    model_name=None,
    model_type=None,
    model_location_generator=None,
    image_processor=None,
    config=None,
    reference_model=None,
    get_batch=None,
    imagenet_label_dict=None,
    batch_size=None,
):
    print("Inside evaluation")

    # Loading the dataset
    input_loc = str(model_location_generator("ImageNet_data"))
    iterations = 100
    # iteration dataset, preprocessing
    data_loader = get_data_loader(input_loc, batch_size, iterations)
    gt_id = []
    pred_id = []
    for i in range(iterations // batch_size):
        if model_name == "vit":
            inputs, labels = get_batch(data_loader, image_processor)
        else:
            inputs, labels = get_batch(data_loader)
            inputs = image_processor(inputs, return_tensors="pt")

        # preprocess
        if model_name == "Segformer":
            torch_input_tensor = inputs.pixel_values
            torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor_permuted,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            inputs = torch.permute(inputs, (0, 2, 3, 1))
            inputs = torch.nn.functional.pad(inputs, (0, 1, 0, 0, 0, 0, 0, 0))
            batch_size, img_h, img_w, img_c = inputs.shape  # permuted input NHWC
            patch_size = config.patch_size
            inputs = inputs.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
            print("inputs", inputs.shape)

            tt_inputs_host = ttnn.from_torch(
                inputs,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=model.test_infra.inputs_mesh_mapper,
            )

        # Inference
        if model_type == "tt_model":
            if model_name == "Segformer":
                output = model(
                    ttnn_input_tensor,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                    parameters=parameters,
                    model=None,
                )
            elif model_name == "vit":
                output = model.execute_vit_trace_2cqs_inference(tt_inputs_host)

        # post_process
        if model_name == "Segformer":
            final_output = ttnn.to_torch(output.logits)
            predicted_id = final_output.argmax(-1).item()
            predicted_label = imagenet_label_dict[predicted_id]

            print("predicted_id", predicted_id, "predicted_label", predicted_label)

            pred_id.append(predicted_id)
            gt_id.append(labels[0])
        elif model_name == "vit":
            final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            predicted_id = final_output[:, 0, :1000].argmax(dim=-1)

            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                print("pred_id", predicted_id, "labels", labels[i])
                gt_id.append(labels[i])

    # Evaluation : Here we use correct_predection/total items
    correct_prediction = 0
    correct_prediction = sum(1 for a, b in zip(gt_id, pred_id) if a == b)
    accuracy = correct_prediction / len(pred_id)

    print("pred_id", pred_id)
    print("gt_id", gt_id)
    logger.info(f"accuracy: {accuracy:.2f}%")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_classification(device, model_location_generator, imagenet_label_dict):
    from transformers import AutoImageProcessor, SegformerForImageClassification
    from models.demos.segformer.demo.classification_demo_utils import get_batch
    from ttnn.model_preprocessing import preprocess_model_parameters

    from models.demos.segformer.reference.segformer_for_image_classification import (
        SegformerForImageClassificationReference,
    )
    from models.demos.segformer.tt.ttnn_segformer_for_image_classification import TtSegformerForImageClassification
    from tests.ttnn.integration_tests.segformer.test_segformer_for_image_classification import (
        create_custom_preprocessor,
    )
    from tests.ttnn.integration_tests.segformer.test_segformer_model import move_to_device

    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
    torch_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0").to(torch.bfloat16)
    reference_model = SegformerForImageClassificationReference(config=torch_model.config)

    reference_model.load_state_dict(torch_model.state_dict())
    reference_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_preprocessor(device),
        device=None,
    )
    parameters = move_to_device(parameters, device)
    ttnn_model = TtSegformerForImageClassification(torch_model.config, parameters)

    evaluation(
        device=device,
        model=ttnn_model,
        model_location_generator=model_location_generator,
        parameters=parameters,
        model_type="tt_model",
        model_name="Segformer",
        image_processor=image_processor,
        config=reference_model.config,
        reference_model=reference_model,
        get_batch=get_batch,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=1,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1700000}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((8),),
)
def test_vit_classification(
    mesh_device, use_program_cache, batch_size_per_device, imagenet_label_dict, model_location_generator
):
    from models.demos.vit.tests.vit_performant_imagenet import VitTrace2CQ
    from transformers import AutoImageProcessor
    import transformers
    from models.demos.wormhole.vit.demo.vit_helper_funcs import get_batch

    batch_size = batch_size_per_device * mesh_device.get_num_devices()

    vit_trace_2cq = VitTrace2CQ()

    vit_trace_2cq.initialize_vit_trace_2cqs_inference(
        mesh_device,
        batch_size_per_device,
    )

    model_version = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(model_version)
    config = transformers.ViTConfig.from_pretrained(model_version)

    # input_loc = str(model_location_generator("ImageNet_data"))
    # data_loader = get_data_loader(input_loc, batch_size, 1, False)

    # inputs, labels = get_batch(data_loader, image_processor)
    # # preprocessing
    # inputs = torch.permute(inputs, (0, 2, 3, 1))
    # inputs = torch.nn.functional.pad(inputs, (0, 1, 0, 0, 0, 0, 0, 0))
    # batch_size, img_h, img_w, img_c = inputs.shape  # permuted input NHWC
    # patch_size = config.patch_size
    # inputs = inputs.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)
    # print("inputs",inputs.shape)

    # tt_inputs_host = ttnn.from_torch(
    #             inputs,
    #             dtype=ttnn.bfloat16,
    #             layout=ttnn.ROW_MAJOR_LAYOUT,
    #             mesh_mapper=vit_trace_2cq.test_infra.inputs_mesh_mapper,
    #         )

    # output = vit_trace_2cq.execute_vit_trace_2cqs_inference(tt_inputs_host)
    # output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # prediction = output[:, 0, :1000].argmax(dim=-1)
    # predictions=[]
    # for i in range(8):
    #     predictions.append(prediction[i].item())
    # print("predictions",predictions)
    # print("labels",labels)

    evaluation(
        device=mesh_device,
        model=vit_trace_2cq,
        model_location_generator=model_location_generator,
        parameters=None,
        model_type="tt_model",
        model_name="vit",
        image_processor=image_processor,
        config=config,
        reference_model=None,
        get_batch=get_batch,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=batch_size,
    )


# def test_rn50_classification():
#     print("will work")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weight", [True], ids=["pretrained_weight_true"])
@pytest.mark.parametrize("batch_size, res, iterations", [[8, 224, 1]])
def test_mobilenetv2_classification(
    device, use_pretrained_weight, batch_size, res, iterations, imagenet_label_dict, model_location_generator
):
    from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
    from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
    import os
    from models.demos.mobilenetv2.tt.model_preprocessing import create_mobilenetv2_model_parameters
    from models.demos.mobilenetv2.demo.demo_utils import get_batch, get_data_loader

    weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/mobilenetv2/weights_download.sh")

    torch_model = Mobilenetv2()
    if use_pretrained_weight:
        state_dict = torch.load(weights_path)
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {
            name1: parameter2
            for (name1, _), (_, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
            if isinstance(parameter2, torch.FloatTensor)
        }
        torch_model.load_state_dict(new_state_dict)
    else:
        state_dict = torch_model.state_dict()

    torch_model.eval()

    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(parameters, device, batchsize=batch_size)
    correct = 0

    for iter in range(iterations):
        predictions = []
        inputs, labels = get_batch(data_loader, res)
        torch_input_tensor = inputs.reshape(batch_size, 3, res, res)
        torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

        ttnn_input_tensor = torch_input_tensor.reshape(
            1,
            1,
            torch_input_tensor.shape[0] * torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
        )

        ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)
        tt_output = ttnn_model(ttnn_input_tensor)
        tt_output = ttnn.from_device(tt_output, blocking=True).to_torch().to(torch.float)
        prediction = tt_output.argmax(dim=-1)

        for i in range(batch_size):
            predicted_label = imagenet_label_dict[prediction[i].item()]
            true_label = imagenet_label_dict[labels[i]]
            predictions.append(predicted_label)
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {true_label} -- Predicted Label: {predicted_label}"
            )
            if true_label == predicted_label:
                correct += 1

        del tt_output, inputs, labels, predictions

    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
