# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI U.S. Corp.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.vision.segmentation.sam2.common import load_sam2_model_and_processor
from models.demos.vision.segmentation.sam2.tt.tt_sam2_video import SAM2_L1_SMALL_SIZE, build_tt_sam2_model

VIDEO_WARMUP_FRAMES = 32
VIDEO_MEASUREMENT_FRAMES = 100
EXPECTED_VIDEO_STEADY_LATENCY_S = 0.05
N300_VIDEO_DEVICE_PARAMS = {
    "l1_small_size": SAM2_L1_SMALL_SIZE,
    "require_exact_physical_num_devices": True,
    "num_command_queues": 2,
}


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [N300_VIDEO_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_sam2_sustained_video_latency(mesh_device, model_location_generator):
    hf_model, processor = load_sam2_model_and_processor(model_location_generator)
    image = np.random.default_rng(0).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    pixels = processor(images=image, return_tensors="pt").pixel_values
    prompts = {
        "input_points": torch.tensor([[[[512.0, 512.0]]]]),
        "input_labels": torch.tensor([[[1]]], dtype=torch.int32),
    }

    model = build_tt_sam2_model(hf_model, mesh_device, bridge_upload_cq_id=1)
    session = None
    try:
        session = model.start_video_session()
        frame_latencies = []
        start = time.perf_counter()
        frame_count = VIDEO_WARMUP_FRAMES + VIDEO_MEASUREMENT_FRAMES + 1
        for output in session.run([pixels] * frame_count, prompts):
            mask = ttnn.to_torch(output["pred_masks_high_res"])
            now = time.perf_counter()
            frame_latencies.append(now - start)
            start = now

        measured_latencies = frame_latencies[VIDEO_WARMUP_FRAMES:-1]
        mean_latency = sum(measured_latencies) / len(measured_latencies)
        logger.info(
            "SAM2 sustained video: {:.3f} ms/frame, {:.3f} frames/s",
            mean_latency * 1000,
            1.0 / mean_latency,
        )
        assert tuple(mask.shape) == (1, 1, 1024, 1024), f"unexpected mask shape {tuple(mask.shape)}"
        assert (
            len(measured_latencies) == VIDEO_MEASUREMENT_FRAMES
        ), f"expected {VIDEO_MEASUREMENT_FRAMES} measured frames, got {len(measured_latencies)}"
        assert mean_latency <= EXPECTED_VIDEO_STEADY_LATENCY_S * 1.10, (
            f"video latency {mean_latency * 1000:.3f} ms exceeded "
            f"{EXPECTED_VIDEO_STEADY_LATENCY_S * 1.10 * 1000:.3f} ms regression limit"
        )
    finally:
        if session is not None:
            session.close()
        model.close()
