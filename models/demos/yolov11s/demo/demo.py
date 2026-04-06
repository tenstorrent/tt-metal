# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.utils.common_demo_utils import (
    LoadImages,
    get_mesh_mappers,
    load_coco_class_names,
    postprocess,
    preprocess,
    save_yolo_predictions_by_model,
)
from models.demos.yolov11s.common import YOLOV11S_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov11s.reference import yolov11s
from models.demos.yolov11s.runner.performant_runner import YOLOv11sPerformantRunner


def init_model_and_runner(
    device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")

    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    if use_weights_from_ultralytics:
        torch_model = load_torch_model(model_location_generator)
        model = torch_model.eval()
    else:
        model = yolov11s.YoloV11s()

    performant_runner = None
    if model_type == "tt_model":
        performant_runner = YOLOv11sPerformantRunner(
            device,
            batch_size_per_device,
            model_location_generator=model_location_generator,
            inputs_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            outputs_mesh_composer=outputs_mesh_composer,
        )

    return model, performant_runner, outputs_mesh_composer, batch_size


def process_images(dataset, res, batch_size):
    """One batched preprocess (faster than per-image) and optional pad to ``batch_size``."""
    torch_images, orig_images, paths_images = [], [], []

    for paths, im0s, _ in dataset:
        if not im0s:
            continue
        for idx, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[idx]}")
        paths_images.extend(paths)
        orig_images.extend(im0s)
        torch_images.append(preprocess(im0s, res=res))

        if sum(t.shape[0] for t in torch_images) >= batch_size:
            break

    if not torch_images:
        raise ValueError("No images loaded from dataset")

    torch_input_tensor = torch.cat(torch_images, dim=0)
    if torch_input_tensor.shape[0] > batch_size:
        torch_input_tensor = torch_input_tensor[:batch_size]
        orig_images = orig_images[:batch_size]
        paths_images = paths_images[:batch_size]

    n = torch_input_tensor.shape[0]
    if n < batch_size:
        logger.warning(
            f"Only {n} image(s) for batch_size={batch_size}; padding by repeating the last image "
            "(needed for multi-device sharded batch)."
        )
        last = torch_input_tensor[-1:].clone()
        pad_orig = orig_images[-1].copy()
        pad_path = paths_images[-1]
        while torch_input_tensor.shape[0] < batch_size:
            torch_input_tensor = torch.cat([torch_input_tensor, last], dim=0)
            orig_images.append(pad_orig.copy())
            paths_images.append(pad_path)

    return torch_input_tensor, orig_images, paths_images


def run_inference_and_save(
    model, runner, model_type, outputs_mesh_composer, im_tensor, orig_images, paths_images, save_dir, names
):
    if model_type == "torch_model":
        with torch.inference_mode():
            preds = model(im_tensor)
    else:
        preds = runner.run(im_tensor)
        preds = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=outputs_mesh_composer)

    results = postprocess(preds, im_tensor, orig_images, paths_images, names)

    for result, image_path in zip(results, paths_images):
        save_yolo_predictions_by_model(result, save_dir, image_path, model_type)


def run_yolov11s_demo(
    device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device, model_location_generator
):
    model, runner, mesh_composer, batch_size = init_model_and_runner(
        device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
    )

    dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(dataset, res, batch_size)
    names = load_coco_class_names()
    save_dir = "models/demos/yolov11s/demo/runs"

    run_inference_and_save(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, names
    )

    if runner:
        runner.release()
    logger.info("Inference done")


def run_yolov11s_demo_dataset(
    device, model_type, use_weights_from_ultralytics, res, batch_size_per_device, model_location_generator
):
    model, runner, mesh_composer, batch_size = init_model_and_runner(
        device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
    )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    filepaths = [sample["filepath"] for sample in dataset]
    image_loader = LoadImages(filepaths, batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(image_loader, res, batch_size)

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json") as f:
        names = json.load(f)["classes"]

    save_dir = "models/demos/yolov11s/demo/runs"
    run_inference_and_save(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, names
    )

    if runner:
        runner.release()
    logger.info("Inference done")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device ",
    [
        (
            "models/demos/yolov11s/demo/images",
            1,
        ),
    ],
)
def test_demo(
    device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device, model_location_generator
):
    # pytest.skip("Skipping yolov11s demo test: https://github.com/tenstorrent/tt-metal/issues/30568")
    run_yolov11s_demo(
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device ",
    [
        (
            "models/demos/yolov11s/demo/images",
            1,
        ),
    ],
)
def test_demo_dp(
    mesh_device,
    model_type,
    use_weights_from_ultralytics,
    res,
    input_loc,
    batch_size_per_device,
    model_location_generator,
):
    run_yolov11s_demo(
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_dataset(device, model_type, use_weights_from_ultralytics, res, model_location_generator):
    run_yolov11s_demo_dataset(
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_location_generator=model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_dataset_dp(mesh_device, model_type, use_weights_from_ultralytics, res, model_location_generator):
    run_yolov11s_demo_dataset(
        mesh_device,
        model_type,
        use_weights_from_ultralytics,
        res,
        batch_size_per_device=1,
        model_location_generator=model_location_generator,
    )
