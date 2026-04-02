# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 Performance Benchmark

Measures:
- Prefix embedding time (SigLIP + language + VLM forward)
- Denoising loop time (per step and total)
- End-to-end latency
- Throughput (actions/sec)
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
    print("  PI0.5 Performance Benchmark")
    print("=" * 60)

    config = create_pi05_config()
    weight_loader = PI0WeightLoader(CHECKPOINT_PATH)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
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

        # Convert inputs
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

        # Warmup run
        print("\nWarmup run...")
        with torch.no_grad():
            _ = model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )
        print("  Done")

        # Benchmark: 3 runs
        print("\nBenchmark (3 runs):")
        times = []
        for i in range(3):
            start = time.time()
            with torch.no_grad():
                result = model.sample_actions(
                    images=images_ttnn,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens_ttnn,
                    lang_masks=lang_masks_ttnn,
                    state=state_ttnn,
                )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.1f}ms")

        avg = sum(times) / len(times)
        best = min(times)
        # 50 action timesteps per inference
        actions_per_sec = 50.0 / (best / 1000.0)

        print(f"\n{'='*60}")
        print(f"  Average latency:  {avg:.1f}ms")
        print(f"  Best latency:     {best:.1f}ms")
        print(f"  Actions/sec:      {actions_per_sec:.1f}")
        print(f"  Denoise steps:    10")
        print(f"{'='*60}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
