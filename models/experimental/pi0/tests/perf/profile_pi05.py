# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile Pi0.5 inference with Tracy."""

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


def main():
    config = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True, num_denoising_steps=10)
    config.siglip_config = SigLIPConfig()
    weight_loader = PI0WeightLoader(CHECKPOINT_PATH)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    device.enable_program_cache()

    torch.manual_seed(42)
    model = PI0ModelTTNN(config, weight_loader, device)

    images_ttnn = [
        ttnn.from_torch(
            torch.randn(1, 3, 224, 224),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    lang_tokens_ttnn = ttnn.from_torch(
        torch.randint(0, 256000, (1, 32)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    lang_masks_ttnn = ttnn.from_torch(
        torch.ones(1, 32).float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    state_ttnn = ttnn.from_torch(torch.randn(1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    img_masks = [torch.ones(1, dtype=torch.bool)] * 2

    # Warmup (compiles all kernels)
    print("Warmup...", flush=True)
    with torch.no_grad():
        for _ in range(2):
            _ = model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )
    print(f"Program cache: {device.num_program_cache_entries()} entries", flush=True)

    # Profiled run
    print("Profiled run...", flush=True)
    ttnn.tracy_frame()
    with torch.no_grad():
        ttnn.tracy_message("inference_start")
        t0 = time.time()
        result = model.sample_actions(
            images=images_ttnn,
            img_masks=img_masks,
            lang_tokens=lang_tokens_ttnn,
            lang_masks=lang_masks_ttnn,
            state=state_ttnn,
        )
        elapsed = (time.time() - t0) * 1000
        ttnn.tracy_message("inference_end")
    ttnn.tracy_frame()

    print(f"Inference: {elapsed:.1f}ms", flush=True)

    ttnn.close_device(device)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
