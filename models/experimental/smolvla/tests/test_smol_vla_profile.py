#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLA Performance Profiling Test

Benchmark command:
    ITERATIONS=100 pytest models/experimental/smolvla/tests/test_smol_vla_profile.py -v -s

Tracy profiling command:
    python3 ./tools/tracy/profile_this.py -c "pytest models/experimental/smolvla/tests/test_smol_vla_profile.py -v -s"

Current baseline (Dec 22, 2025): E2E=8.2 FPS | Device=37 FPS | TT-only=24 FPS
"""

import os
import time

import numpy as np
import pytest
import torch
from PIL import Image


# Get iterations from environment variable (default 10)
ITERATIONS = int(os.environ.get("ITERATIONS", 10))


def create_test_image(size: int = 512, seed: int = 42) -> Image.Image:
    """Create a deterministic test image."""
    np.random.seed(seed)
    img_array = np.zeros((size, size, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.tile(np.linspace(50, 200, size), (size, 1)).astype(np.uint8)
    img_array[:, :, 1] = np.tile(np.linspace(100, 150, size), (size, 1)).T.astype(np.uint8)
    img_array[:, :, 2] = 128
    return Image.fromarray(img_array)


class TestSmolVLAProfile:
    """Minimal profiling test for SmolVLA TT model only."""

    @pytest.fixture(scope="class")
    def tt_model(self):
        """Load TT model only (no CPU model for comparison)."""
        import ttnn
        from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction

        print("\nLoading TT model for profiling...")
        device = ttnn.open_device(device_id=0)
        model_tt = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=device)
        model_tt.processor.image_processor.do_image_splitting = False
        model_tt.eval()

        yield {"tt": model_tt, "device": device}

        print("\nClosing TT device...")
        ttnn.close_device(device)

    def test_tt_inference_perf(self, tt_model):
        """Profile TT inference over multiple iterations."""
        from models.experimental.smolvla.tt.smol_vla import CHECKPOINTS

        img = create_test_image()
        instruction = "pick up the red block"

        iterations = ITERATIONS

        print(f"\n[Running {iterations} iterations...]")

        # Time the entire loop
        start_total = time.perf_counter()
        for i in range(iterations):
            torch.manual_seed(42)
            np.random.seed(42)

            actions_tt = tt_model["tt"].predict_action(
                images=[img], instruction=instruction, num_inference_steps=1, action_dim=6
            )
        end_total = time.perf_counter()

        total_time = end_total - start_total
        avg_time = total_time / iterations

        # Get last iteration's checkpoint breakdown
        results = CHECKPOINTS.analyze()

        # Extract timing components
        vision_time = 0.0
        vlmkv_time = 0.0
        flow_time = 0.0
        preprocess_time = 0.0

        for name, duration in results.items():
            if "VISIONFORWARD" in name and "EXPERT" not in name:
                vision_time = duration
            elif "VLMKVCOMPUTE" in name:
                vlmkv_time = duration
            elif "FLOWMATCHING" in name and "EXPERT" not in name:
                flow_time = duration
            elif "PREPROCESS" in name and "PROCESSOR" not in name:
                preprocess_time = duration

        tt_only_time = vision_time + vlmkv_time + flow_time

        print("\n" + "=" * 70)
        print(f"PERFORMANCE RESULTS ({iterations} iterations)")
        print("=" * 70)
        print(f"  Total time:         {total_time*1000:.2f} ms")
        print(f"  Avg E2E:            {avg_time*1000:.2f} ms")
        print(f"  E2E FPS:            {iterations/total_time:.1f} FPS")
        print("-" * 70)
        print(f"  Last iter breakdown (checkpoint timing):")
        print(f"    VISION:           {vision_time*1000:.2f} ms")
        print(f"    VLM K/V:          {vlmkv_time*1000:.2f} ms")
        print(f"    FLOWMATCH:        {flow_time*1000:.2f} ms")
        print(f"    CPU PREPROC:      {preprocess_time*1000:.2f} ms")
        print(f"    TT-ONLY:          {tt_only_time*1000:.2f} ms ({1.0/tt_only_time:.1f} FPS)")
        print("=" * 70)

        print(f"\nTT output shape: {actions_tt.shape}")

        # Basic sanity check
        assert actions_tt.shape == (50, 6), f"Expected (50, 6), got {actions_tt.shape}"
