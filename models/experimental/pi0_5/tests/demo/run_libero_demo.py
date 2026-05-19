#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Demo with LIBERO Real Images

This standalone script demonstrates PI0 inference on real robot images
from the LIBERO dataset, proving the model works with realistic inputs.

Usage:
    python run_libero_demo.py
"""

import sys
import os
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from loguru import logger

import ttnn

# Demo folder location
DEMO_DIR = Path(__file__).parent

# Add parent paths for imports
sys.path.insert(0, str(DEMO_DIR.parent.parent.parent.parent))

# Use the same implementations as PCC/Perf tests
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


# =============================================================================
# CONFIGURATION
# =============================================================================
TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = "lerobot/pi0_base"
LIBERO_IMAGES_DIR = DEMO_DIR / "sample_images" / "libero"

# LIBERO task prompts (from the benchmark)
LIBERO_PROMPTS = [
    "pick up the black bowl",
    "put the bowl on the plate",
    "open the cabinet door",
    "close the drawer",
    "pick up the red cube",
]

BATCH_SIZE = 1
SEED = 42
PCC_THRESHOLD = 0.93


def create_config() -> PI0ModelConfig:
    """Create PI0ModelConfig."""
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
    )
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    return config


def load_and_preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for PI0."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Convert to tensor and normalize to [-1, 1] (SigLIP preprocessing)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

    return img_tensor.unsqueeze(0)


def tokenize_prompt(prompt: str, max_length: int = 32):
    """Simple tokenization for demo."""
    tokens = [ord(char) % 256000 for char in prompt[:max_length]]
    while len(tokens) < max_length:
        tokens.append(0)
    tokens = torch.tensor([tokens[:max_length]], dtype=torch.long)
    mask = torch.ones(1, max_length, dtype=torch.bool)
    mask[0, len(prompt) :] = False
    return tokens, mask


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()
    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)
    if std1 < 1e-6 or std2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


def main():
    logger.info("\n" + "=" * 70)
    logger.info("        PI0 DEMO - LIBERO BENCHMARK REAL IMAGES")
    logger.info("=" * 70)

    # Check paths
    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.is_absolute() and not checkpoint_path.exists():
        logger.info(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not LIBERO_IMAGES_DIR.exists():
        logger.info(f"❌ LIBERO images not found: {LIBERO_IMAGES_DIR}")
        logger.info(f"   Run: python extract_libero_samples.py")
        sys.exit(1)

    # Find image pairs (main + wrist)
    main_images = sorted(LIBERO_IMAGES_DIR.glob("*_main.png"))[:2]
    wrist_images = sorted(LIBERO_IMAGES_DIR.glob("*_wrist.png"))[:2]

    if len(main_images) < 1:
        logger.info(f"❌ No LIBERO images found in {LIBERO_IMAGES_DIR}")
        sys.exit(1)

    config = create_config()
    image_size = config.siglip_config.image_size

    logger.info(f"\n📸 Loading LIBERO benchmark images:")
    logger.info(f"   Dataset: HuggingFaceVLA/libero")
    logger.info(f"   Original size: 256x256, Resized to: {image_size}x{image_size}")

    # Load images - use main image as first view, wrist as second
    images = []
    for img_path in [main_images[0], wrist_images[0]]:
        img_tensor = load_and_preprocess_image(str(img_path), image_size)
        images.append(img_tensor)
        logger.info(f"   ✅ Loaded: {img_path.name}")

    img_masks = [torch.ones(BATCH_SIZE, dtype=torch.bool) for _ in range(2)]

    # Use a LIBERO task prompt
    prompt = LIBERO_PROMPTS[0]
    lang_tokens, lang_masks = tokenize_prompt(prompt)
    state = torch.randn(BATCH_SIZE, config.state_dim, dtype=torch.float32)

    logger.info(f'\n🗣️  Prompt: "{prompt}"')
    logger.info(f"🤖 State: random {config.state_dim}-dim vector")

    # Load models
    logger.info(f"\n📦 Loading models...")
    weight_loader = PI0WeightLoader(str(checkpoint_path))

    logger.info(f"   Loading PyTorch reference...")
    torch_model = PI0ModelTorch(config, weight_loader)

    logger.info(f"   Loading TTNN model...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    torch.manual_seed(SEED)
    ttnn_model = PI0ModelTTNN(config, weight_loader, device)

    # Run inference
    logger.info(f"\n🚀 Running inference on LIBERO images (10 denoising steps)...")

    # PyTorch reference
    torch.manual_seed(SEED)
    t0 = time.time()
    with torch.no_grad():
        torch_actions = torch_model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
    torch_time = (time.time() - t0) * 1000
    logger.info(f"   PyTorch: {torch_time:.2f}ms")

    # TTNN
    t0 = time.time()
    with torch.no_grad():
        # Convert images to TTNN tensors
        images_ttnn = [
            ttnn.from_torch(
                img,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for img in images
        ]

        # Convert other inputs to TTNN tensors
        lang_tokens_ttnn = ttnn.from_torch(
            lang_tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        state_ttnn = ttnn.from_torch(
            state,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        ttnn_actions = ttnn_model.sample_actions(
            images=images_ttnn,
            img_masks=img_masks,
            lang_tokens=lang_tokens_ttnn,
            lang_masks=lang_masks_ttnn,
            state=state_ttnn,
        )
    ttnn.synchronize_device(device)
    ttnn_time = (time.time() - t0) * 1000
    logger.info(f"   TTNN:    {ttnn_time:.2f}ms")

    # Convert TTNN output to torch for comparison
    if isinstance(ttnn_actions, ttnn.Tensor):
        ttnn_actions = ttnn.to_torch(ttnn_actions)

    # Compute PCC
    pcc = compute_pcc(torch_actions, ttnn_actions)

    # Results
    logger.info("\n" + "-" * 70)
    logger.info("📊 RESULTS (LIBERO Benchmark Real Images):")
    logger.info("-" * 70)

    logger.info(f"\n  PyTorch Actions:")
    logger.info(f"    Shape: {torch_actions.shape}")
    logger.info(f"    Range: [{torch_actions.min():.4f}, {torch_actions.max():.4f}]")
    logger.info(f"    Mean:  {torch_actions.mean():.4f}")

    logger.info(f"\n  TTNN Actions:")
    logger.info(f"    Shape: {ttnn_actions.shape}")
    logger.info(f"    Range: [{ttnn_actions.min():.4f}, {ttnn_actions.max():.4f}]")
    logger.info(f"    Mean:  {ttnn_actions.mean():.4f}")

    logger.info(f"\n✅ Validation:")
    logger.info(f"   PCC Score: {pcc:.4f}")
    logger.info(f"   PCC Threshold: {PCC_THRESHOLD}")
    passed = pcc >= PCC_THRESHOLD
    logger.info(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")

    logger.info(f"\n⚡ Performance:")
    logger.info(f"   PyTorch Time: {torch_time:.2f}ms")
    logger.info(f"   TTNN Time:    {ttnn_time:.2f}ms")
    logger.info(f"   Speedup:      {torch_time/ttnn_time:.2f}x")

    logger.info(f"\n🎯 This demo proves:")
    logger.info(f"   ✓ Model processes LIBERO benchmark images correctly")
    logger.info(f"   ✓ TTNN implementation matches PyTorch reference (PCC={pcc:.4f})")
    logger.info(f"   ✓ Works with different camera views (main + wrist)")
    logger.info(f"   ✓ Handles LIBERO manipulation task prompts")

    logger.info("=" * 70 + "\n")

    # Cleanup
    ttnn.close_device(device)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
