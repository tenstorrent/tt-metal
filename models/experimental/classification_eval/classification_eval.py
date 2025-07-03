# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from loguru import logger

import torch
import torchvision
import pytest
import transformers
from transformers import AutoImageProcessor

import ttnn
from models.experimental.classification_eval.classification_eval_utils import get_data_loader


def evaluation(
    device=None,
    model=None,
    inputs=None,
    model_name=None,
    model_type=None,
    model_location_generator=None,
    image_processor=None,
    config=None,
    get_batch=None,
    batch_size=None,
    res=None,
):
    # Loading the dataset
    input_loc = str(model_location_generator("ImageNet_data"))
    iterations = 512
    # iteration dataset, preprocessing
    data_loader = get_data_loader(input_loc, batch_size, iterations // batch_size)
    gt_id = []
    pred_id = []
    for i in range(iterations // batch_size):
        if model_name in ["vit", "resnet50", "mobilenetv2"]:
            inputs, labels = get_batch(data_loader, image_processor)
        else:
            inputs, labels = get_batch(data_loader)
            inputs = image_processor(inputs, return_tensors="pt")

        # preprocess
        if model_name == "mobilenetv2":
            n, c, h, w = inputs.shape
            torch_input_tensor = inputs.reshape(batch_size, 3, res, res)
            tt_inputs = torch.permute(inputs, (0, 2, 3, 1))
            ttnn_input_tensor = tt_inputs.reshape(
                1,
                1,
                n * h * w,
                c,
            )

            ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn_input_tensor = ttnn.pad(ttnn_input_tensor, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        elif model_name == "resnet50":
            torch_input_tensor = inputs
            if model_type == "tt_model":
                tt_inputs_host = ttnn.from_torch(
                    inputs,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=model.test_infra.inputs_mesh_mapper,
                )
        else:
            torch_input_tensor = inputs
            if model_type == "tt_model":
                tt_inputs = torch.permute(inputs, (0, 2, 3, 1))
                tt_inputs = torch.nn.functional.pad(tt_inputs, (0, 1, 0, 0, 0, 0, 0, 0))
                batch_size, img_h, img_w, img_c = tt_inputs.shape  # permuted input NHWC
                patch_size = config.patch_size
                tt_inputs = tt_inputs.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

                tt_inputs_host = ttnn.from_torch(
                    tt_inputs,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=model.test_infra.inputs_mesh_mapper,
                )

        # Inference
        if model_type == "tt_model":
            if model_name == "mobilenetv2":
                output = model.execute_mobilenetv2_trace_2cqs_inference(ttnn_input_tensor)
            elif model_name == "vit":
                output = model.execute_vit_trace_2cqs_inference(tt_inputs_host)
            elif model_name == "resnet50":
                output = model.execute_resnet50_trace_2cqs_inference(tt_inputs_host)
        elif model_type == "torch_model":
            output = model(torch_input_tensor)

        # post_process
        if model_name == "mobilenetv2":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output)
            else:
                final_output = output
            prediction = final_output.argmax(dim=-1)

            for i in range(batch_size):
                pred_id.append(prediction[i].item())
                gt_id.append(labels[i])

            del output, final_output

        elif model_name == "vit":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                predicted_id = final_output[:, 0, :1000].argmax(dim=-1)
            else:
                final_output = output.logits
                predicted_id = final_output.argmax(dim=-1)

            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                gt_id.append(labels[i])
        elif model_name == "resnet50":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                predicted_id = final_output[:, 0, 0, :].argmax(dim=-1)
            else:
                final_output = output
                predicted_id = final_output.argmax(dim=-1)

            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                gt_id.append(labels[i])

    if model_type == "tt_model":
        if model_name == "mobilenetv2":
            model.release_mobilenetv2_trace_2cqs_inference()
        elif model_name == "resnet50":
            model.release_resnet50_trace_2cqs_inference()
        elif model_name == "vit":
            model.release_vit_trace_2cqs_inference()

    # Evaluation : Here we use correct_predection/total items
    correct_prediction = 0
    correct_prediction = sum(1 for a, b in zip(gt_id, pred_id) if a == b)
    accuracy = (correct_prediction / len(pred_id)) * 100

    logger.info(f"accuracy: {accuracy:.2f}%")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1700000}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((8),),
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
def test_vit_image_classification_eval(
    mesh_device,
    model_type,
    batch_size_per_device,
    model_location_generator,
):
    from models.demos.vit.tests.vit_performant_imagenet import VitTrace2CQ
    from transformers import ViTForImageClassification
    from models.demos.wormhole.vit.demo.vit_helper_funcs import get_batch

    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    if model_type == "torch_model":
        torch_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    else:
        vit_trace_2cq = VitTrace2CQ()

        vit_trace_2cq.initialize_vit_trace_2cqs_inference(
            mesh_device,
            batch_size_per_device,
        )

    model_version = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(model_version)
    config = transformers.ViTConfig.from_pretrained(model_version)

    evaluation(
        device=mesh_device,
        model=vit_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="vit",
        image_processor=image_processor,
        config=config,
        get_batch=get_batch,
        batch_size=batch_size,
    )


@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_resnet50_image_classification_eval(
    device,
    model_type,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
):
    from models.demos.ttnn_resnet.tests.resnet50_performant_imagenet import ResNet50Trace2CQ
    from models.demos.ttnn_resnet.tests.demo_utils import get_batch

    if model_type == "torch_model":
        torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet50_trace_2cq = ResNet50Trace2CQ()

        resnet50_trace_2cq.initialize_resnet50_trace_2cqs_inference(
            device,
            batch_size,
            act_dtype,
            weight_dtype,
        )

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    evaluation(
        device=device,
        model=resnet50_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="resnet50",
        image_processor=image_processor,
        config=None,
        get_batch=get_batch,
        batch_size=batch_size,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("batch_size, res", [[8, 224]])
def test_mobilenetv2_image_classification_eval(
    device, model_type, batch_size, res, model_location_generator, reset_seeds
):
    from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
    from models.demos.mobilenetv2.tests.mobilenetv2_e2e_performant import MobileNetV2Trace2CQ
    import torchvision.models as models
    from models.demos.ttnn_resnet.tests.demo_utils import get_batch

    weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"

    model_version = "microsoft/resnet-50"
    image_processor = AutoImageProcessor.from_pretrained(model_version)

    if model_type == "torch_model":
        torch_model = models.mobilenet_v2(pretrained=True)
    else:
        if not os.path.exists(weights_path):
            os.system("bash models/demos/mobilenetv2/weights_download.sh")

        reference_model = Mobilenetv2()

        state_dict = torch.load(weights_path)
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {
            name1: parameter2
            for (name1, _), (_, parameter2) in zip(reference_model.state_dict().items(), ds_state_dict.items())
            if isinstance(parameter2, torch.FloatTensor)
        }
        reference_model.load_state_dict(new_state_dict)

        reference_model.eval()

        mobilenetv2_trace_2cq = MobileNetV2Trace2CQ()

        mobilenetv2_trace_2cq.initialize_mobilenetv2_trace_2cqs_inference(
            device,
            batch_size,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        )

    evaluation(
        device=device,
        model=mobilenetv2_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="mobilenetv2",
        image_processor=image_processor,
        config=None,
        get_batch=get_batch,
        batch_size=batch_size,
        res=res,
    )
