# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torchvision
import transformers
from loguru import logger
from transformers import AutoImageProcessor

import ttnn
from models.demos.mobilenetv2.common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE
from models.demos.utils.common_demo_utils import get_data_loader, load_imagenet_dataset
from models.experimental.efficientnetb0.common import EFFICIENTNETB0_L1_SMALL_SIZE
from models.experimental.swin_s.common import SWIN_S_L1_SMALL_SIZE
from models.experimental.swin_v2.common import SWIN_V2_L1_SMALL_SIZE
from models.experimental.vovnet.common import VOVNET_L1_SMALL_SIZE


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
    entire_imagenet_dataset=False,
    inputs_mesh_mapper=None,
    output_mesh_composer=None,
):
    # Loading the dataset
    input_loc = load_imagenet_dataset(model_location_generator)
    iterations = 512
    # iteration dataset, preprocessing
    data_loader = get_data_loader(input_loc, batch_size, iterations // batch_size, entire_imagenet_dataset)
    gt_id = []
    pred_id = []
    input_tensors_all = []
    input_labels_all = []
    tt_inputs_host_all = []
    for i in range(iterations // batch_size):
        if model_name == "vit":
            inputs, labels = get_batch(data_loader, image_processor)
        elif model_name in ["mobilenetv2", "swin_s", "resnet50", "efficientnet_b0", "swin_v2", "vovnet"]:
            inputs, labels = get_batch(data_loader, res)

        # preprocess
        if model_name in ["mobilenetv2", "resnet50"]:
            torch_input_tensor = inputs
            if model_type == "tt_model":
                if model_name == "mobilenetv2":
                    ttnn_input = torch.permute(inputs, (0, 2, 3, 1))
                    ttnn_input = torch.nn.functional.pad(ttnn_input, (0, 16 - ttnn_input.shape[-1]), value=0)
                    ttnn_input = ttnn.from_torch(
                        ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=inputs_mesh_mapper
                    )
                    ttnn_input = ttnn.reshape(
                        ttnn_input,
                        (1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]),
                    )
                    input_tensors_all.append(ttnn_input)
                if model_name == "resnet50":
                    input_tensors_all.append(inputs)
                input_labels_all.append(labels)
        elif model_name in ["swin_s", "vovnet", "efficientnet_b0", "swin_v2"]:
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
            if model_name == "vit":
                output = model.execute_vit_trace_2cqs_inference(tt_inputs_host)
            elif model_name in ["swin_v2", "vovnet", "swin_s", "efficientnet_b0"]:
                output = model.run(torch_input_tensor)
        elif model_type == "torch_model":
            output = model(torch_input_tensor)

        if model_name in ["mobilenetv2", "resnet50"]:
            if model_type == "torch_model":
                final_output = output
                prediction = final_output.argmax(dim=-1)
                for i in range(batch_size):
                    pred_id.append(prediction[i].item())
                    gt_id.append(labels[i])

                del output, final_output

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

        elif model_name == "swin_s":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=model.runner_infra.mesh_composer).to(torch.float)
                predicted_id = final_output.argmax(dim=-1)
            else:
                predicted_id = output.argmax(dim=-1)

        elif model_name == "vit":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                predicted_id = final_output[:, 0, :1000].argmax(dim=-1)
            else:
                final_output = output.logits
                predicted_id = final_output.argmax(dim=-1)

        elif model_name == "efficientnet_b0":
            if model_type == "tt_model":
                output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, predicted_id = torch.topk(probabilities, 1)

        if model_name in ["vovnet", "swin_v2"]:
            for i in range(batch_size):
                pred_id.append(prediction[i].item())
                gt_id.append(labels[i])
            del output, final_output
        elif model_name in ["vit", "efficientnet_b0", "swin_s"]:
            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                gt_id.append(labels[i])

    if model_type == "tt_model":
        if model_name in ["mobilenetv2", "resnet50"]:
            if model_name == "resnet50":
                for inputs in input_tensors_all:
                    tt_inputs_host = ttnn.from_torch(
                        inputs,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=inputs_mesh_mapper,
                    )
                    tt_inputs_host_all.append(tt_inputs_host)

                outputs = model.enqueue(tt_inputs_host_all).pop_all()
            else:
                outputs = model.enqueue(input_tensors_all).pop_all()
            for iter in range(iterations // batch_size):
                predictions = []
                output = outputs[iter]
                labels = input_labels_all[iter]
                output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
                if model_name == "resnet50":
                    output = torch.reshape(output, (output.shape[0], 1000))
                prediction = output.argmax(dim=-1)
                for i in range(batch_size):
                    pred_id.append(prediction[i].item())
                    gt_id.append(labels[i])
                del output, prediction

    if model_type == "tt_model":
        if model_name in ["mobilenetv2", "resnet50"]:
            model.cleanup()
        elif model_name in ["vovnet", "swin_v2", "efficientnet_b0", "swin_s"]:
            model.release()
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
    from transformers import ViTForImageClassification

    from models.demos.vit.tests.vit_performant_imagenet import VitTrace2CQ
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


def run_resnet50_image_classification_eval(
    device, model_type, batch_size_per_device, act_dtype, weight_dtype, model_location_generator, resolution=224
):
    from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
    from models.demos.utils.common_demo_utils import get_batch
    from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

    batch_size = batch_size_per_device * device.get_num_devices()
    if model_type == "torch_model":
        torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        test_infra = create_test_infra(
            device,
            batch_size_per_device,
            act_dtype,
            weight_dtype,
            ttnn.MathFidelity.LoFi,
            use_pretrained_weight=True,
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
            model_location_generator=model_location_generator,
        )
        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

        def model_wrapper(l1_input_tensor):
            test_infra.input_tensor = l1_input_tensor
            return test_infra.run()

        pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
            model=model_wrapper,
            device=device,
            dram_input_memory_config=sharded_mem_config_DRAM,
            l1_input_memory_config=input_mem_config,
        )
        pipeline.compile(tt_inputs_host)

    evaluation(
        device=device,
        model=pipeline if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="resnet50",
        image_processor=None,
        config=None,
        get_batch=get_batch,
        batch_size=batch_size,
        res=resolution,
        inputs_mesh_mapper=test_infra.inputs_mesh_mapper if model_type == "tt_model" else None,
        output_mesh_composer=test_infra.output_mesh_composer if model_type == "tt_model" else None,
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
    "batch_size_per_device, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_resnet50_image_classification_eval(
    device, model_type, batch_size_per_device, act_dtype, weight_dtype, model_location_generator
):
    return run_resnet50_image_classification_eval(
        device, model_type, batch_size_per_device, act_dtype, weight_dtype, model_location_generator
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
    "batch_size_per_device, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_resnet50_image_classification_eval_dp(
    mesh_device, model_type, batch_size_per_device, act_dtype, weight_dtype, model_location_generator
):
    return run_resnet50_image_classification_eval(
        mesh_device, model_type, batch_size_per_device, act_dtype, weight_dtype, model_location_generator
    )


def run_mobilenetv2_image_classification_eval(
    device, model_type, batch_size_per_device, res, model_location_generator, reset_seeds
):
    from models.demos.mobilenetv2.common import load_torch_model
    from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
    from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
    from models.demos.mobilenetv2.tt.model_preprocessing import (
        create_mobilenetv2_input_tensors,
        create_mobilenetv2_model_parameters,
    )
    from models.demos.utils.common_demo_utils import get_batch, get_mesh_mappers
    from models.tt_cnn.tt.pipeline import (
        PipelineConfig,
        create_pipeline_from_config,
        get_memory_config_for_persistent_dram_tensor,
    )

    batch_size = batch_size_per_device * device.get_num_devices()
    with torch.no_grad():
        inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
        reference_model = Mobilenetv2()
        reference_model = load_torch_model(reference_model, model_location_generator)
        model_parameters = create_mobilenetv2_model_parameters(reference_model, device=device)
        ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size_per_device)

        _, host_input_tensor = create_mobilenetv2_input_tensors(
            batch=batch_size, input_height=res, input_width=res, pad_channels=16, mesh_mapper=inputs_mesh_mapper
        )
        input_dram_mem_config = get_memory_config_for_persistent_dram_tensor(
            host_input_tensor.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
        )
        logger.info(
            f"Auto-selected persistent DRAM tensor memory config: shape={host_input_tensor.shape}, shard_shape={input_dram_mem_config.shard_spec.shape}, grid={input_dram_mem_config.shard_spec.grid}"
        )

        input_l1_core_grid = ttnn.CoreGrid(x=8, y=8)
        assert (
            host_input_tensor.shape[-2] % input_l1_core_grid.num_cores == 0
        ), "Expecting even sharding on L1 input tensor"
        input_l1_mem_config = ttnn.create_sharded_memory_config(
            shape=(host_input_tensor.shape[2] // input_l1_core_grid.num_cores, host_input_tensor.shape[-1]),
            core_grid=input_l1_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
        pipe = create_pipeline_from_config(
            config,
            ttnn_model,
            device,
            dram_input_memory_config=input_dram_mem_config,
            l1_input_memory_config=input_l1_mem_config,
        )
        pipe.compile(host_input_tensor)

    evaluation(
        device=device,
        model=pipe if model_type == "tt_model" else reference_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="mobilenetv2",
        image_processor=None,
        config=None,
        get_batch=get_batch,
        batch_size=batch_size_per_device * device.get_num_devices(),
        res=res,
        inputs_mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
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


def run_swin_v2_image_classification_eval(device, model_type, batch_size, res, model_location_generator, reset_seeds):
    from models.demos.utils.common_demo_utils import get_batch
    from models.experimental.swin_v2.common import load_torch_model
    from models.experimental.swin_v2.reference.swin_transformer import SwinTransformer
    from models.experimental.swin_v2.runner.performant_runner import SwinV2PerformantRunner

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
    [{"l1_small_size": SWIN_V2_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
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
    "device_params",
    [{"l1_small_size": SWIN_V2_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
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


def run_vovnet_image_classification_eval(
    device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    from models.demos.utils.common_demo_utils import get_batch
    from models.experimental.vovnet.common import load_torch_model
    from models.experimental.vovnet.runner.performant_runner import VovnetPerformantRunner

    batch_size = device_batch_size * device.get_num_devices()
    if model_type == "torch_model":
        torch_model = load_torch_model(model_location_generator).eval()
    else:
        vovnet_trace_2cq = VovnetPerformantRunner(
            device,
            device_batch_size,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            resolution=(res, res),
            model_location_generator=model_location_generator,
        )

        vovnet_trace_2cq._capture_vovnet_trace_2cqs()
    evaluation(
        device=device,
        model=vovnet_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="vovnet",
        image_processor=None,
        config=None,
        get_batch=get_batch,
        batch_size=batch_size,
        res=res,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VOVNET_L1_SMALL_SIZE, "trace_region_size": 1752064, "num_command_queues": 2}],
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
def test_vovnet_image_classification_eval(device, model_type, batch_size, res, model_location_generator, reset_seeds):
    return run_vovnet_image_classification_eval(
        device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VOVNET_L1_SMALL_SIZE, "trace_region_size": 1752064, "num_command_queues": 2}],
    indirect=True,
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
    from models.demos.utils.common_demo_utils import get_batch, get_mesh_mappers
    from models.experimental.efficientnetb0.common import load_torch_model
    from models.experimental.efficientnetb0.runner.performant_runner import EfficientNetb0PerformantRunner

    if model_type == "torch_model":
        torch_model = load_torch_model(model_location_generator=model_location_generator)
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
        output_mesh_composer=outputs_mesh_composer if model_type == "tt_model" else None,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
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
    [{"l1_small_size": EFFICIENTNETB0_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
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


def run_swin_s_image_classification_eval(
    device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    from models.demos.utils.common_demo_utils import get_batch
    from models.experimental.swin_s.common import load_torch_model
    from models.experimental.swin_s.reference.swin_transformer import SwinTransformer
    from models.experimental.swin_s.runner.performant_runner import SwinSPerformantRunner

    batch_size = device_batch_size * device.get_num_devices()
    if model_type == "torch_model":
        torch_model = SwinTransformer(
            patch_size=[4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=[7, 7]
        )
        torch_model = load_torch_model(torch_model=torch_model, model_location_generator=model_location_generator)
    else:
        swin_s_trace_2cq = SwinSPerformantRunner(
            device,
            device_batch_size,
            ttnn.bfloat16,
            ttnn.bfloat16,
            resolution=(res, res),
            model_location_generator=model_location_generator,
        )

    evaluation(
        device=device,
        model=swin_s_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        model_type=model_type,
        model_name="swin_s",
        get_batch=get_batch,
        batch_size=batch_size,
        res=res,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SWIN_S_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("batch_size, res", [[1, 512]])
def test_swin_s_image_classification_eval(device, model_type, batch_size, res, model_location_generator, reset_seeds):
    return run_swin_s_image_classification_eval(
        device, model_type, batch_size, res, model_location_generator, reset_seeds
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SWIN_S_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("device_batch_size, res", [[1, 512]])
def test_swin_s_image_classification_eval_dp(
    mesh_device, model_type, device_batch_size, res, model_location_generator, reset_seeds
):
    return run_swin_s_image_classification_eval(
        mesh_device, model_type, device_batch_size, res, model_location_generator, reset_seeds
    )
