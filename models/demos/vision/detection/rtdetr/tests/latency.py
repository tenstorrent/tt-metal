# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import statistics
import time
from types import SimpleNamespace

import pytest
import requests
import torch
from loguru import logger
from transformers import RTDetrImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.detection.rtdetr.common.preprocessing import custom_preprocessor
from models.demos.vision.detection.rtdetr.demo import MODEL_CONFIGS, SAMPLE_TEST_IMAGES, load_image
from models.demos.vision.detection.rtdetr.tt.model import TtRTDetrModel

WARMUP_IMAGE = SAMPLE_TEST_IMAGES[0]
TIMED_IMAGES = SAMPLE_TEST_IMAGES[1:9]


def prepare_latency_test(device, model_version):
    model_name, model_class = MODEL_CONFIGS[model_version]

    try:
        images = [load_image(source) for source in [WARMUP_IMAGE, *TIMED_IMAGES]]
    except requests.RequestException as error:
        pytest.skip(f"COCO validation images could not be downloaded: {error}")

    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    pixel_values = [image_processor(images=image, return_tensors="pt").pixel_values for image in images]
    _, _, input_height, input_width = pixel_values[0].shape

    torch_model = model_class.from_pretrained(model_name).eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.model,
        custom_preprocessor=custom_preprocessor,
    )
    tt_model = TtRTDetrModel(
        config=torch_model.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        input_height=input_height,
        input_width=input_width,
    )
    device.enable_program_cache()

    return images, image_processor, pixel_values, tt_model


def log_latencies(model_version, latency_name, latencies):
    latencies_ms = [latency * 1000 for latency in latencies]
    logger.info(
        f"RT-DETR v{model_version} {latency_name} latencies: " f"{[round(latency, 2) for latency in latencies_ms]} ms"
    )
    logger.info(f"RT-DETR v{model_version} {latency_name} mean latency: {statistics.mean(latencies_ms):.2f} ms")
    logger.info(f"RT-DETR v{model_version} {latency_name} median latency: {statistics.median(latencies_ms):.2f} ms")


def run_rtdetr_batch_1_trace_latency(device, model_version):
    _, _, pixel_values, tt_model = prepare_latency_test(device, model_version)
    host_inputs = [
        ttnn.from_torch(
            torch_pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        for torch_pixel_values in pixel_values
    ]
    tt_pixel_values = ttnn.from_torch(
        pixel_values[0],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    warmup_outputs = tt_model(tt_pixel_values)
    ttnn.synchronize_device(device)
    del warmup_outputs

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_outputs = tt_model(tt_pixel_values)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    latencies = []
    try:
        for host_input in host_inputs[1:]:
            ttnn.copy_host_to_device_tensor(host_input, tt_pixel_values, cq_id=0)
            ttnn.synchronize_device(device)

            start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            latencies.append(time.perf_counter() - start)
    finally:
        ttnn.release_trace(device, trace_id)
        del trace_outputs

    log_latencies(model_version, "batch 1 trace replay", latencies)


def run_rtdetr_batch_1_e2e_latency(device, model_version):
    images, image_processor, pixel_values, tt_model = prepare_latency_test(device, model_version)
    tt_pixel_values = ttnn.from_torch(
        pixel_values[0],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    warmup_outputs = tt_model(tt_pixel_values)
    ttnn.synchronize_device(device)
    del warmup_outputs

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_outputs = tt_model(tt_pixel_values)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    latencies = []
    try:
        for run, image in enumerate(images[1:], start=1):
            start = time.perf_counter()

            torch_pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values
            target_sizes = torch.tensor([(image.height, image.width)])
            host_input = ttnn.from_torch(
                torch_pixel_values,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_input, tt_pixel_values, cq_id=0)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

            tt_outputs = SimpleNamespace(
                logits=ttnn.to_torch(trace_outputs[4]).float(),
                pred_boxes=ttnn.to_torch(trace_outputs[5]).float(),
            )
            detections = image_processor.post_process_object_detection(
                tt_outputs,
                threshold=0.5,
                target_sizes=target_sizes,
            )

            latency = time.perf_counter() - start
            latencies.append(latency)
            logger.info(f"RT-DETR v{model_version} traced end-to-end run {run}: {latency * 1000:.2f} ms")
            del detections
            del host_input
    finally:
        ttnn.release_trace(device, trace_id)
        del trace_outputs

    log_latencies(model_version, "batch 1 traced end-to-end", latencies)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1 << 26}],
    indirect=True,
)
def test_rtdetr_batch_1_trace_latency(device):
    run_rtdetr_batch_1_trace_latency(device, model_version=1)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1 << 26}],
    indirect=True,
)
def test_rtdetr_v2_batch_1_trace_latency(device):
    run_rtdetr_batch_1_trace_latency(device, model_version=2)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1 << 26}],
    indirect=True,
)
def test_rtdetr_batch_1_e2e_latency(device):
    run_rtdetr_batch_1_e2e_latency(device, model_version=1)
