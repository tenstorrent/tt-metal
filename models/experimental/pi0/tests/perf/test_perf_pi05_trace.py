# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 Performance Benchmark with 2CQ + Trace

Captures the denoising loop in a Metal Trace for zero host overhead.
"""

import sys
import os
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN

CHECKPOINT_PATH = os.path.join(
    os.environ.get("TT_METAL_HOME", "/home/ttuser/experiments/pi0_5/tt-metal"),
    "models/experimental/pi0/weights/pi05_base",
)


def create_pi05_config():
    config = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig()
    return config


def main():
    print("=" * 60)
    print("  PI0.5 Performance: 2CQ + Trace")
    print("=" * 60)

    config = create_pi05_config()
    weight_loader = PI0WeightLoader(CHECKPOINT_PATH)

    # Open device with trace support
    device = ttnn.open_device(
        device_id=0,
        l1_small_size=24576,
        trace_region_size=80_000_000,  # 80MB trace buffer (denoising loop needs ~71MB)
        num_command_queues=2,
    )
    print(f"Device: {device}")

    try:
        torch.manual_seed(42)
        model = PI0ModelTTNN(config, weight_loader, device)

        # Create inputs
        images = [torch.randn(1, 3, 224, 224) for _ in range(2)]
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(2)]
        lang_tokens = torch.randint(0, 256000, (1, 32))
        lang_masks = torch.ones(1, 32, dtype=torch.bool)
        state = torch.randn(1, 32)

        images_ttnn = [
            ttnn.from_torch(
                img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for img in images
        ]
        lang_tokens_ttnn = ttnn.from_torch(lang_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # --- Non-traced baseline ---
        print("\n--- Non-traced baseline ---")
        # Warmup
        with torch.no_grad():
            _ = model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )

        baseline_times = []
        for i in range(3):
            start = time.time()
            with torch.no_grad():
                _ = model.sample_actions(
                    images=images_ttnn,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens_ttnn,
                    lang_masks=lang_masks_ttnn,
                    state=state_ttnn,
                )
            elapsed = (time.time() - start) * 1000
            baseline_times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.1f}ms")
        baseline_best = min(baseline_times)
        print(f"  Best: {baseline_best:.1f}ms ({50.0/(baseline_best/1000):.1f} actions/sec)")

        # --- Traced version ---
        print("\n--- 2CQ + Trace ---")
        # First call compiles + captures trace
        print("  Compiling + capturing trace...")
        with torch.no_grad():
            start = time.time()
            result = model.sample_actions_traced(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )
            compile_time = (time.time() - start) * 1000
        print(f"  Compile + capture: {compile_time:.1f}ms")

        # Subsequent calls execute trace
        trace_times = []
        for i in range(5):
            start = time.time()
            with torch.no_grad():
                result = model.sample_actions_traced(
                    images=images_ttnn,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens_ttnn,
                    lang_masks=lang_masks_ttnn,
                    state=state_ttnn,
                )
            elapsed = (time.time() - start) * 1000
            trace_times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.1f}ms")

        trace_best = min(trace_times)
        trace_avg = sum(trace_times) / len(trace_times)

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Baseline best:    {baseline_best:.1f}ms ({50.0/(baseline_best/1000):.1f} actions/sec)")
        print(f"  Traced best:      {trace_best:.1f}ms ({50.0/(trace_best/1000):.1f} actions/sec)")
        print(f"  Traced avg:       {trace_avg:.1f}ms")
        if trace_best < baseline_best:
            speedup = baseline_best / trace_best
            print(f"  Speedup:          {speedup:.2f}x")
        print(f"{'='*60}")

        model.release_trace()

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
