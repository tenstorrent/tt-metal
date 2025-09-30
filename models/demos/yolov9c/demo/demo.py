# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import LoadImages, get_mesh_mappers, load_coco_class_names
from models.demos.utils.common_demo_utils import postprocess as obj_postprocess
from models.demos.utils.common_demo_utils import preprocess, save_yolo_predictions_by_model
from models.demos.yolov9c.common import YOLOV9C_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov9c.demo.demo_utils import postprocess, save_seg_predictions_by_model
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner


def init_model_and_runner(
    model_location_generator, device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_task
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")

    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)
    enable_segment = model_task == "segment"

    if use_weights_from_ultralytics:
        model = load_torch_model(model_task, model_location_generator)
    else:
        model = yolov9c.YoloV9(enable_segment=enable_segment)
        state_dict = state_dict if state_dict else model.state_dict()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        model.load_state_dict(new_state_dict)
        model.eval()

    performant_runner = None
    if model_type == "tt_model":
        performant_runner = YOLOv9PerformantRunner(
            device,
            batch_size_per_device,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            model_task=model_task,
            resolution=(640, 640),
            mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            mesh_composer=outputs_mesh_composer,
            model_location_generator=model_location_generator,
        )

    return model, performant_runner, outputs_mesh_composer, batch_size, enable_segment


def process_images(dataset, res, batch_size):
    torch_images, orig_images, paths_images = [], [], []

    for batch in dataset:
        paths, im0s, _ = batch
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        paths_images.extend(paths)
        orig_images.extend(im0s)

        for idx, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[idx]}")
            tensor = preprocess([img], res=res)
            torch_images.append(tensor)

        if len(torch_images) >= batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    return torch_input_tensor, orig_images, paths_images, paths


def run_inference_and_save(
    model,
    runner,
    model_type,
    outputs_mesh_composer,
    im_tensor,
    orig_images,
    paths_images,
    save_dir,
    names,
    enable_segment,
    dataset,
    batch_size,
):
    if model_type == "torch_model":
        preds = model(im_tensor)
        if enable_segment:
            results = postprocess(preds, im_tensor, orig_images, dataset)
            for result, image_path in zip(results, paths_images):
                save_seg_predictions_by_model(result, save_dir, image_path, model_type)
        else:
            results = obj_postprocess(preds, im_tensor, orig_images, dataset, load_coco_class_names())
            for result, image_path in zip(results, paths_images):
                save_yolo_predictions_by_model(result, save_dir, image_path, model_type)

    else:
        preds = runner.run(im_tensor)
        preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32, mesh_composer=outputs_mesh_composer)

        if enable_segment:
            detect_outputs = [
                ttnn.to_torch(tensor, dtype=torch.float32, mesh_composer=outputs_mesh_composer)
                for tensor in preds[1][0]
            ]
            mask = ttnn.to_torch(preds[1][1], dtype=torch.float32, mesh_composer=outputs_mesh_composer)
            proto = ttnn.to_torch(preds[1][2], dtype=torch.float32, mesh_composer=outputs_mesh_composer)
            proto = proto.reshape((batch_size, 160, 160, 32)).permute((0, 3, 1, 2))
            preds[1] = [detect_outputs, mask, proto]
            results = postprocess(preds, im_tensor, orig_images, dataset)
            for result, image_path in zip(results, paths_images):
                save_seg_predictions_by_model(result, save_dir, image_path, model_type)
        else:
            results = obj_postprocess(preds, im_tensor, orig_images, dataset, load_coco_class_names())
            for result, image_path in zip(results, paths_images):
                save_yolo_predictions_by_model(result, save_dir, image_path, model_type)

    if model_type == "tt_model":
        runner.release()

    logger.info("Inference done")


def run_yolov9c_demo(
    model_location_generator,
    device,
    model_type,
    use_weights_from_ultralytics,
    res,
    input_loc,
    batch_size_per_device,
    model_task,
):
    model, runner, mesh_composer, batch_size, enable_segment = init_model_and_runner(
        model_location_generator, device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_task
    )

    dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
    im_tensor, orig_images, paths_images, dataset = process_images(dataset, res, batch_size)
    names = load_coco_class_names()
    save_dir = "models/demos/yolov9c/demo/runs"

    run_inference_and_save(
        model,
        runner,
        model_type,
        mesh_composer,
        im_tensor,
        orig_images,
        paths_images,
        save_dir,
        names,
        enable_segment,
        dataset,
        batch_size,
    )

    if runner:
        runner.release()
    logger.info("Inference done")


def run_yolov9c_demo_dataset(
    model_location_generator, device, model_type, use_weights_from_ultralytics, res, batch_size_per_device, model_task
):
    model, runner, mesh_composer, batch_size, enable_segment = init_model_and_runner(
        model_location_generator, device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_task
    )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    filepaths = [sample["filepath"] for sample in dataset]
    image_loader = LoadImages(filepaths, batch=batch_size)
    im_tensor, orig_images, paths_images, dataset = process_images(image_loader, res, batch_size)

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json") as f:
        names = json.load(f)["classes"]

    save_dir = "models/demos/yolov9c/demo/runs"
    run_inference_and_save(
        model,
        runner,
        model_type,
        mesh_composer,
        im_tensor,
        orig_images,
        paths_images,
        save_dir,
        names,
        enable_segment,
        dataset,
        batch_size,
    )

    if runner:
        runner.release()
    logger.info("Inference done")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device",
    [
        (
            "models/demos/yolov9c/demo/images",
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_detect(
    model_location_generator,
    device,
    model_type,
    use_weights_from_ultralytics,
    res,
    input_loc,
    batch_size_per_device,
    reset_seeds,
):
    run_yolov9c_demo(
        model_location_generator,
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device",
    [
        (
            "models/demos/yolov9c/demo/images",
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to th,e run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_detect_dp(
    model_location_generator,
    mesh_device,
    model_type,
    use_weights_from_ultralytics,
    res,
    input_loc,
    batch_size_per_device,
    reset_seeds,
):
    run_yolov9c_demo(
        model_location_generator,
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_detect_dataset(
    model_location_generator,
    device,
    model_type,
    use_weights_from_ultralytics,
    res,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        model_location_generator,
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_detect_dataset_dp(
    model_location_generator,
    mesh_device,
    model_type,
    use_weights_from_ultralytics,
    res,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        model_location_generator,
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device",
    [
        (
            "models/demos/yolov9c/demo/images",
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_segment(
    model_location_generator,
    device,
    model_type,
    use_weights_from_ultralytics,
    res,
    input_loc,
    batch_size_per_device,
    reset_seeds,
):
    run_yolov9c_demo(
        model_location_generator,
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_task="segment",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device",
    [
        (
            "models/demos/yolov9c/demo/images",
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_segment_dp(
    model_location_generator,
    mesh_device,
    model_type,
    use_weights_from_ultralytics,
    res,
    input_loc,
    batch_size_per_device,
    reset_seeds,
):
    run_yolov9c_demo(
        model_location_generator,
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_task="segment",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_segment_dataset(
    model_location_generator,
    device,
    model_type,
    use_weights_from_ultralytics,
    res,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        model_location_generator,
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_task="segment",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_segment_dataset_dp(
    model_location_generator,
    mesh_device,
    model_type,
    use_weights_from_ultralytics,
    res,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        model_location_generator,
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_task="segment",
    )
