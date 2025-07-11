# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov9c.demo.demo_utils import (
    LoadImages,
    load_coco_class_names,
    load_torch_model,
    postprocess,
    save_seg_predictions_by_model,
)
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.demos.yolov9c.tt.model_preprocessing import get_mesh_mappers
from models.experimental.yolo_eval.evaluate import save_yolo_predictions_by_model
from models.experimental.yolo_eval.utils import postprocess as obj_postprocess
from models.experimental.yolo_eval.utils import preprocess
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


def ttnn_preds_postprocess(
    batch_size,
    preds,
    paths,
    im,
    im0s,
    batch,
    names,
    enable_segment,
    output_mesh_composer,
    save_dir,
    model_type,
):
    preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32, mesh_composer=output_mesh_composer)

    if enable_segment:
        detect_outputs = [
            ttnn.to_torch(tensor, dtype=torch.float32, mesh_composer=output_mesh_composer) for tensor in preds[1][0]
        ]
        mask = ttnn.to_torch(preds[1][1], dtype=torch.float32, mesh_composer=output_mesh_composer)
        proto = ttnn.to_torch(preds[1][2], dtype=torch.float32, mesh_composer=output_mesh_composer)
        proto = proto.reshape((batch_size, 160, 160, 32)).permute((0, 3, 1, 2))
        preds[1] = [detect_outputs, mask, proto]
        results = postprocess(preds, im, im0s, batch)
        for result, image_path in zip(results, paths):
            save_seg_predictions_by_model(result, save_dir, image_path, model_type)
    else:
        results = obj_postprocess(preds, im, im0s, batch, names)
        for result, image_path in zip(results, paths):
            save_yolo_predictions_by_model(result, save_dir, image_path, model_type)


def run_yolov9c_demo(
    device,
    batch_size_per_device,
    input_loc,
    use_weights_from_ultralytics,
    model_type,
    model_task,
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    enable_segment = model_task == "segment"

    if model_type == "torch_model":
        model = load_torch_model(use_weights_from_ultralytics=use_weights_from_ultralytics, model_task=model_task)
        logger.info("Inferencing [Torch] Model")
    else:
        performant_runner = YOLOv9PerformantRunner(
            device,
            batch_size_per_device,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            model_task=model_task,
            resolution=(640, 640),
            mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/demos/yolov9c/demo/runs"
    input_loc = os.path.abspath(input_loc)
    dataset = LoadImages(path=input_loc, batch=batch_size)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, _ = batch
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        preprocessed_im = []
        for i, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[i]}")
            processed = preprocess([img], res=(640, 640))
            preprocessed_im.append(processed)

        if len(preprocessed_im) >= batch_size:
            break

        im = torch.cat(preprocessed_im, dim=0)
        if model_type == "torch_model":
            preds = model(im)
            if enable_segment:
                results = postprocess(preds, im, im0s, batch)
                for result, image_path in zip(results, paths):
                    save_seg_predictions_by_model(result, save_dir, image_path, model_type)
            else:
                results = obj_postprocess(preds, im, im0s, batch, names)
                for result, image_path in zip(results, paths):
                    save_yolo_predictions_by_model(result, save_dir, image_path, model_type)

        else:
            preds = performant_runner.run(torch_input_tensor=im)
            ttnn_preds_postprocess(
                batch_size,
                preds,
                paths,
                im,
                im0s,
                batch,
                names,
                enable_segment,
                output_mesh_composer,
                save_dir,
                model_type,
            )

    if model_type == "tt_model":
        performant_runner.release()

    logger.info("Inference done")


def run_yolov9c_demo_dataset(
    device,
    use_weights_from_ultralytics,
    model_type,
    batch_size_per_device,
    model_task,
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size_per_device = 1
    batch_size = batch_size_per_device * num_devices
    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    enable_segment = model_task == "segment"

    if model_type == "torch_model":
        model = load_torch_model(use_weights_from_ultralytics=use_weights_from_ultralytics, model_task=model_task)
        logger.info("Inferencing [Torch] Model")
    else:
        performant_runner = YOLOv9PerformantRunner(
            device,
            batch_size_per_device,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            model_task=model_task,
            resolution=(640, 640),
            mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )
        logger.info("Inferencing [TTNN] Model")

    names = load_coco_class_names()
    save_dir = "models/demos/yolov9c/demo/runs"
    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    data_set = LoadImages([sample["filepath"] for sample in dataset], batch=batch_size)
    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json", "r") as file:
        coco_info = json.load(file)
        class_names = coco_info["classes"]

    torch_images = []
    orig_images = []
    paths_images = []

    for batch in data_set:
        paths, im0s, _ = batch
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        paths_images.extend(paths)
        orig_images.extend(im0s)

        for i, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[i]}")
            tensor = preprocess([img], res=(640, 640))
            torch_images.append(tensor)

        if len(torch_images) >= batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    if model_type == "torch_model":
        preds = model(torch_input_tensor)
        if enable_segment:
            results = postprocess(preds, torch_input_tensor, im0s, batch)
            for result, image_path in zip(results, paths):
                save_seg_predictions_by_model(result, save_dir, image_path, model_type)
        else:
            results = obj_postprocess(preds, torch_input_tensor, im0s, batch, names)
            for result, image_path in zip(results, paths):
                save_yolo_predictions_by_model(result, save_dir, image_path, model_type)
    else:
        preds = performant_runner.run(torch_input_tensor)
        ttnn_preds_postprocess(
            batch_size,
            preds,
            paths,
            torch_input_tensor,
            im0s,
            batch,
            names,
            enable_segment,
            output_mesh_composer,
            save_dir,
            model_type,
        )

    if model_type == "tt_model":
        performant_runner.release()

    logger.info("Inference done")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov9c/demo/images",
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
def test_demo_detect(
    device,
    batch_size_per_device,
    input_loc,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo(
        device,
        batch_size_per_device,
        input_loc,
        use_weights_from_ultralytics,
        model_type,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
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
def test_demo_detect_dataset(
    device,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        device,
        use_weights_from_ultralytics,
        model_type,
        batch_size_per_device=1,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov9c/demo/images",
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
def test_demo_detect_dp(
    mesh_device,
    batch_size_per_device,
    input_loc,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo(
        mesh_device,
        batch_size_per_device,
        input_loc,
        use_weights_from_ultralytics,
        model_type,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
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
def test_demo_detect_dataset_dp(
    mesh_device,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        mesh_device,
        use_weights_from_ultralytics,
        model_type,
        batch_size_per_device=1,
        model_task="detect",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov9c/demo/images",
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
def test_demo_segment(
    device,
    batch_size_per_device,
    input_loc,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo(
        device,
        batch_size_per_device,
        input_loc,
        use_weights_from_ultralytics,
        model_type,
        model_task="segment",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
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
def test_demo_segment_dataset(
    device,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        device,
        use_weights_from_ultralytics,
        model_type,
        batch_size_per_device=1,
        model_task="segment",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov9c/demo/images",
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
def test_demo_segment_dp(
    mesh_device,
    batch_size_per_device,
    input_loc,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo(
        mesh_device,
        batch_size_per_device,
        input_loc,
        use_weights_from_ultralytics,
        model_type,
        model_task="segment",
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
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
def test_demo_segment_dataset_dp(
    mesh_device,
    use_weights_from_ultralytics,
    model_type,
    reset_seeds,
):
    run_yolov9c_demo_dataset(
        mesh_device,
        use_weights_from_ultralytics,
        model_type,
        batch_size_per_device=1,
        model_task="segment",
    )
