# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device performance test for RF-DETR Medium.

Measures wall-clock latency per component and end-to-end using
ttnn.synchronize_device() for accurate device timing.

Usage:
    pytest models/experimental/rfdetr_medium/tests/perf/test_device_perf.py -v -s
"""

import time

import pytest
import torch

import ttnn

from models.experimental.rfdetr_medium.common import (
    RESOLUTION,
    RFDETR_MEDIUM_L1_SMALL_SIZE,
)


@pytest.fixture(scope="module")
def torch_model():
    from models.experimental.rfdetr_medium.common import load_torch_model

    return load_torch_model()


@pytest.fixture(scope="module")
def sample_image():
    torch.manual_seed(42)
    return torch.randn(1, 3, RESOLUTION, RESOLUTION)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_rfdetr_medium_device_perf(device, torch_model, sample_image):
    """
    Measure per-component and end-to-end device latency.

    Reports:
      - Backbone (DINOv2-ViT-S)
      - Projector (MultiScaleProjector)
      - Two-stage proposal (enc_output + top-K)
      - Decoder (4 layers)
      - Detection heads
      - Post-processing
      - Total E2E
    """
    from models.experimental.rfdetr_medium.tt.tt_rfdetr import TtRFDETR
    from models.experimental.rfdetr_medium.tt.model_preprocessing import (
        load_backbone_weights,
        load_projector_weights,
        load_decoder_weights,
        load_detection_head_weights,
    )

    backbone_params = load_backbone_weights(torch_model, device)
    projector_params = load_projector_weights(torch_model, device)
    decoder_params = load_decoder_weights(torch_model, device)
    head_params = load_detection_head_weights(torch_model, device)

    tt_model = TtRFDETR(
        device=device,
        torch_model=torch_model,
        backbone_params=backbone_params,
        projector_params=projector_params,
        decoder_params=decoder_params,
        head_params=head_params,
    )

    batch_size = sample_image.shape[0]
    num_warmup = 2
    num_iterations = 5

    # --- Warmup ---
    print(f"\nWarmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        _ = tt_model.forward(sample_image)
        ttnn.synchronize_device(device)

    # --- Per-component timing ---
    print(f"\nPer-component timing ({num_iterations} iterations)...")
    timings = {
        "preprocess": [],
        "backbone": [],
        "projector": [],
        "two_stage": [],
        "decoder": [],
        "heads": [],
        "postprocess": [],
        "e2e": [],
    }

    for i in range(num_iterations):
        ttnn.synchronize_device(device)

        t_e2e_start = time.perf_counter()

        # Preprocess
        t0 = time.perf_counter()
        pixel_values = tt_model.preprocess_image(sample_image)
        ttnn.synchronize_device(device)
        timings["preprocess"].append(time.perf_counter() - t0)

        # Backbone
        t0 = time.perf_counter()
        feature_maps = tt_model.forward_backbone(pixel_values, batch_size)
        ttnn.synchronize_device(device)
        timings["backbone"].append(time.perf_counter() - t0)

        # Projector
        t0 = time.perf_counter()
        projected = tt_model.forward_projector(feature_maps, batch_size)
        ttnn.synchronize_device(device)
        timings["projector"].append(time.perf_counter() - t0)

        # Two-stage
        t0 = time.perf_counter()
        refpoint_ts, memory, spatial_shapes, level_start_index = tt_model.forward_two_stage(projected, batch_size)
        ttnn.synchronize_device(device)
        timings["two_stage"].append(time.perf_counter() - t0)

        # Decoder
        t0 = time.perf_counter()
        hs, references = tt_model.forward_decoder(memory, refpoint_ts, spatial_shapes, level_start_index, batch_size)
        ttnn.synchronize_device(device)
        timings["decoder"].append(time.perf_counter() - t0)

        # Detection heads
        t0 = time.perf_counter()
        outputs_class, outputs_coord = tt_model.forward_heads(hs, references)
        ttnn.synchronize_device(device)
        timings["heads"].append(time.perf_counter() - t0)

        # Post-processing
        t0 = time.perf_counter()
        img_h, img_w = sample_image.shape[-2:]
        detections = tt_model.postprocess(outputs_class, outputs_coord, (img_h, img_w))
        timings["postprocess"].append(time.perf_counter() - t0)

        timings["e2e"].append(time.perf_counter() - t_e2e_start)

    # --- Report ---
    print("\n" + "=" * 70)
    print(f"RF-DETR Medium Device Performance (batch_size={batch_size})")
    print(f"  Iterations: {num_iterations} (after {num_warmup} warmup)")
    print("=" * 70)
    print(f"{'Component':<20} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 70)

    total_device_ms = 0.0
    for name, times in timings.items():
        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        print(f"{name:<20} {avg_ms:>10.2f} {min_ms:>10.2f} {max_ms:>10.2f}")
        if name != "e2e":
            total_device_ms += avg_ms

    e2e_avg_ms = sum(timings["e2e"]) / len(timings["e2e"]) * 1000
    fps = 1000.0 / e2e_avg_ms

    print("-" * 70)
    print(f"{'Sum of parts':<20} {total_device_ms:>10.2f}")
    print(f"{'E2E avg':<20} {e2e_avg_ms:>10.2f}")
    print(f"{'FPS':<20} {fps:>10.1f}")
    print("=" * 70)

    # Also run a tight e2e loop without per-component sync overhead
    print(f"\nTight E2E loop ({num_iterations} iterations, no intermediate syncs)...")
    tight_times = []
    for _ in range(num_iterations):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        _ = tt_model.forward(sample_image)
        ttnn.synchronize_device(device)
        tight_times.append(time.perf_counter() - t0)

    tight_avg_ms = sum(tight_times) / len(tight_times) * 1000
    tight_min_ms = min(tight_times) * 1000
    tight_fps = 1000.0 / tight_avg_ms

    print(f"  Avg: {tight_avg_ms:.2f} ms  Min: {tight_min_ms:.2f} ms  FPS: {tight_fps:.1f}")
    print()
