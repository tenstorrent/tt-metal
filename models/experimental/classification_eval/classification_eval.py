# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from loguru import logger
import torch
import torchvision
import pytest
import transformers
from transformers import AutoImageProcessor
from models.demos.mobilenetv2.tests.perf.mobilenetv2_common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE
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
        if model_name in ["vit", "resnet50", "mobilenetv2", "vovnet"]:
            inputs, labels = get_batch(data_loader, image_processor)
        elif model_name in ["efficientnet_b0", "swin_v2"]:
            inputs, labels = get_batch(data_loader)
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
        elif model_name in ["swin_v2", "vovnet"]:
            torch_input_tensor = inputs
        elif model_name == "resnet50":
            torch_input_tensor = inputs
            if model_type == "tt_model":
                tt_inputs_host = ttnn.from_torch(
                    inputs,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=model.test_infra.inputs_mesh_mapper,
                )
        elif model_name in ["vovnet", "efficientnet_b0"]:
            torch_input_tensor = inputs
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
                output = model.run(inputs)
            elif model_name == "vit":
                output = model.execute_vit_trace_2cqs_inference(tt_inputs_host)
            elif model_name == "resnet50":
                output = model.execute_resnet50_trace_2cqs_inference(tt_inputs_host)
            elif model_name in ["swin_v2", "vovnet"]:
                output = model.run(torch_input_tensor)
            elif model_name == "efficientnet_b0":
                output = model.run(inputs)
        elif model_type == "torch_model":
            output = model(torch_input_tensor)

        # post_process
        if model_name == "mobilenetv2":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=model.test_infra.output_mesh_composer)
            else:
                final_output = output
            prediction = final_output.argmax(dim=-1)

        elif model_name == "vovnet":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=model.runner_infra.output_mesh_composer)
            else:
                final_output = output
            prediction = final_output.argmax(dim=-1)

        elif model_name == "swin_v2":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=model.runner_infra.output_composer)
            else:
                final_output = output
            prediction = final_output.argmax(dim=-1)

        elif model_name == "vit":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                predicted_id = final_output[:, 0, :1000].argmax(dim=-1)
            else:
                final_output = output.logits
                predicted_id = final_output.argmax(dim=-1)

        elif model_name == "resnet50":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                predicted_id = final_output[:, 0, 0, :].argmax(dim=-1)
            else:
                final_output = output
                predicted_id = final_output.argmax(dim=-1)

        if model_name in ["mobilenetv2", "vovnet", "swin_v2"]:
            for i in range(batch_size):
                pred_id.append(prediction[i].item())
                gt_id.append(labels[i])

            del output, final_output
        elif model_name in ["vit", "resnet50"]:
            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                gt_id.append(labels[i])
        elif model_name == "efficientnet_b0":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output)
                probabilities = torch.nn.functional.softmax(final_output[0], dim=0)
                top_prob, predicted_id = torch.topk(probabilities, 1)
            else:
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, predicted_id = torch.topk(probabilities, 1)
            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                gt_id.append(labels[i])

    if model_type == "tt_model":
        if model_name == "mobilenetv2":
            model.release_mobilenetv2_trace_2cqs_inference()
        elif model_name in ["vovnet", "swin_v2", "efficientnet_b0"]:
            model.release()
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


def run_mobilenetv2_image_classification_eval(
    device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
    from models.demos.mobilenetv2.runner.performant_runner import MobileNetV2Trace2CQ
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
            device_batch_size,
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
        batch_size=device_batch_size * device.get_num_devices(),
        res=res,
    )


def run_swin_v2_image_classification_eval(device, model_type, batch_size, res, model_location_generator, reset_seeds):
    from models.experimental.swin_v2.runner.performant_runner import SwinV2PerformantRunner
    from models.experimental.swin_v2.reference.swin_transformer import SwinTransformer
    from models.experimental.swin_v2.demo.demo_utils import get_batch
    from models.experimental.swin_v2.common import load_torch_model

    total_batch_size = batch_size * device.get_num_devices()
    if model_type == "torch_model":
        torch_model = SwinTransformer(
            patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[8, 8]
        )
        torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)
    else:
        swinv2_trace_2cq = SwinV2PerformantRunner(
            device=device, device_batch_size=batch_size, model_location_generator=model_location_generator
        )

    evaluation(
        device=device,
        model=swinv2_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="swin_v2",
        image_processor=None,
        config=None,
        get_batch=get_batch,
        batch_size=total_batch_size,
        res=res,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
def test_mobilenetv2_image_classification_eval(
    device, model_type, batch_size, model_location_generator, reset_seeds, res=224
):
    run_mobilenetv2_image_classification_eval(
        device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
def test_mobilenetv2_image_classification_eval_dp(
    mesh_device, model_type, batch_size, model_location_generator, reset_seeds, res=224
):
    run_mobilenetv2_image_classification_eval(
        mesh_device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


def run_vovnet_image_classification_eval(
    device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    from models.experimental.vovnet.runner.performant_runner import VovnetPerformantRunner
    from models.demos.ttnn_resnet.tests.demo_utils import get_batch
    from models.experimental.vovnet.common import load_torch_model

    model_version = "microsoft/resnet-50"
    image_processor = AutoImageProcessor.from_pretrained(model_version)
    batch_size = device_batch_size * device.get_num_devices()
    if model_type == "torch_model":
        torch_model = load_torch_model(model_location_generator).eval()
    else:
        vovnet_trace_2cq = VovnetPerformantRunner(
            device,
            device_batch_size,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            resolution=(224, 224),
            model_location_generator=model_location_generator,
        )

        vovnet_trace_2cq._capture_vovnet_trace_2cqs()
    evaluation(
        device=device,
        model=vovnet_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="vovnet",
        image_processor=image_processor,
        config=None,
        get_batch=get_batch,
        batch_size=batch_size,
        res=res,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1752064, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("batch_size, res", [[1, 224]])
def test_vovnet_image_classification_eval(device, model_type, batch_size, res, model_location_generator, reset_seeds):
    return run_vovnet_image_classification_eval(
        device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1752064, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("device_batch_size, res", [[1, 224]])
def test_vovnet_image_classification_eval_dp(
    mesh_device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    return run_vovnet_image_classification_eval(
        mesh_device, model_type, device_batch_size, res, model_location_generator, reset_seeds
    )


def run_efficientnetb0_image_classification_eval(
    device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    from models.experimental.efficientnetb0.reference import efficientnetb0
    from efficientnet_pytorch import EfficientNet
    from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner
    from models.experimental.efficientnetb0.demo.demo_utils import get_batch
    from models.experimental.efficientnetb0.tt.model_preprocessing import get_mesh_mappers

    if model_type == "torch_model":
        model = EfficientNet.from_pretrained("efficientnet-b0").eval()
        state_dict = model.state_dict()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        torch_model = efficientnetb0.Efficientnetb0()

        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()
    else:
        inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
        performant_runner = EfficientNetb0PerformantRunner(
            device,
            device_batch_size,
            ttnn.bfloat16,
            ttnn.bfloat16,
            resolution=(res, res),
            mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            mesh_composer=outputs_mesh_composer,
            model_location_generator=model_location_generator,
        )

    evaluation(
        device=device,
        model=performant_runner if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="efficientnet_b0",
        batch_size=device_batch_size * device.get_num_devices(),
        res=res,
        get_batch=get_batch,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 7 * 1024, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("batch_size, res", [[1, 224]])
def test_efficientnetb0_image_classification_eval(
    device, model_type, batch_size, res, model_location_generator, reset_seeds
):
    run_efficientnetb0_image_classification_eval(
        device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 7 * 1024, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("batch_size, res", [[1, 224]])
def test_efficientnetb0_image_classification_eval_dp(
    mesh_device, model_type, batch_size, res, model_location_generator, reset_seeds
):
    run_efficientnetb0_image_classification_eval(
        mesh_device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 16998400, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("batch_size, res", [[1, 512]])
def test_swin_v2_image_classification_eval(device, model_type, batch_size, res, model_location_generator, reset_seeds):
    return run_swin_v2_image_classification_eval(
        device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 16998400, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("device_batch_size, res", [[1, 512]])
def test_swin_v2_image_classification_eval_dp(
    mesh_device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    return run_swin_v2_image_classification_eval(
        mesh_device, model_type, device_batch_size, res, model_location_generator, reset_seeds
    )
