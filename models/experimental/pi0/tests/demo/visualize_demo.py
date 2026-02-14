#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Demo Visualization

Creates a visual summary showing:
1. Input images from LIBERO/ALOHA
2. Predicted action trajectories
3. TTNN vs PyTorch comparison

Outputs: visualization.png in the demo folder
"""

import sys
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import ttnn

# Demo folder location
DEMO_DIR = Path(__file__).parent

# Add parent paths for imports
sys.path.insert(0, str(DEMO_DIR.parent.parent.parent.parent))

from models.experimental.pi0.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


# Configuration
TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
LIBERO_IMAGES_DIR = DEMO_DIR / "sample_images" / "libero"
ALOHA_IMAGES_DIR = DEMO_DIR / "sample_images" / "aloha_sim"
OUTPUT_PATH = DEMO_DIR / "visualization.png"
SEED = 42


def create_config():
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


def load_image(path, size=224):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((size, size), Image.BILINEAR)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    img_tensor = (img_tensor - 0.5) / 0.5
    return img, img_tensor.unsqueeze(0)


def tokenize_prompt(prompt, max_length=32):
    tokens = [ord(c) % 256000 for c in prompt[:max_length]]
    tokens += [0] * (max_length - len(tokens))
    tokens = torch.tensor([tokens[:max_length]], dtype=torch.long)
    mask = torch.ones(1, max_length, dtype=torch.bool)
    mask[0, len(prompt) :] = False
    return tokens, mask


def compute_pcc(t1, t2):
    t1, t2 = t1.flatten().float(), t2.flatten().float()
    m1, m2 = t1.mean(), t2.mean()
    s1, s2 = t1.std(), t2.std()
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return ((t1 - m1) * (t2 - m2)).mean() / (s1 * s2)


def main():
    print("=" * 60)
    print("  PI0 VISUALIZATION DEMO")
    print("=" * 60)

    config = create_config()

    # Find images
    images_to_use = []
    prompts = []
    dataset_name = None

    if LIBERO_IMAGES_DIR.exists():
        main_imgs = sorted(LIBERO_IMAGES_DIR.glob("*_main.png"))
        wrist_imgs = sorted(LIBERO_IMAGES_DIR.glob("*_wrist.png"))
        if main_imgs and wrist_imgs:
            images_to_use = [main_imgs[0], wrist_imgs[0]]
            prompts = ["pick up the black bowl"]
            dataset_name = "LIBERO"

    if not images_to_use and ALOHA_IMAGES_DIR.exists():
        aloha_imgs = sorted(ALOHA_IMAGES_DIR.glob("*.png"))[:2]
        if aloha_imgs:
            images_to_use = aloha_imgs if len(aloha_imgs) >= 2 else [aloha_imgs[0], aloha_imgs[0]]
            prompts = ["Transfer cube"]
            dataset_name = "ALOHA Sim"

    if not images_to_use:
        print("❌ No sample images found!")
        print(f"   Run: python extract_libero_samples.py")
        print(f"   Or:  python extract_aloha_samples.py")
        return 1

    prompt = prompts[0]
    print(f"\nDataset: {dataset_name}")
    print(f'Prompt: "{prompt}"')

    # Load images
    orig_images = []
    tensor_images = []
    for img_path in images_to_use:
        orig, tensor = load_image(str(img_path), config.siglip_config.image_size)
        orig_images.append(orig)
        tensor_images.append(tensor)
        print(f"Loaded: {img_path.name}")

    # Prepare inputs
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(2)]
    lang_tokens, lang_masks = tokenize_prompt(prompt)
    state = torch.randn(1, config.state_dim, dtype=torch.float32)

    # Load models
    print("\nLoading models...")
    weight_loader = PI0WeightLoader(CHECKPOINT_PATH)
    torch_model = PI0ModelTorch(config, weight_loader)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    ttnn_model = PI0ModelTTNN(config, weight_loader, device)

    # Run inference
    print("Running inference...")

    torch.manual_seed(SEED)
    with torch.no_grad():
        torch_actions = torch_model.sample_actions(
            images=tensor_images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )

    torch.manual_seed(SEED)
    with torch.no_grad():
        ttnn_actions = ttnn_model.sample_actions(
            images=tensor_images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
    ttnn.synchronize_device(device)

    pcc = compute_pcc(torch_actions, ttnn_actions).item()
    print(f"PCC: {pcc:.4f}")

    # Create visualization
    print("\nCreating visualization...")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'PI0 Model Demo - {dataset_name}\nPrompt: "{prompt}"', fontsize=16, fontweight="bold", y=0.98)

    # Row 1: Input images
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(orig_images[0])
    ax1.set_title(f"Input Image 1\n(Original: {orig_images[0].size[0]}x{orig_images[0].size[1]})", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(orig_images[1])
    ax2.set_title(f"Input Image 2\n(Original: {orig_images[1].size[0]}x{orig_images[1].size[1]})", fontsize=11)
    ax2.axis("off")

    # Row 2: Action trajectories (first 8 action dimensions)
    torch_np = torch_actions[0].float().numpy()  # [50, 32]
    ttnn_np = ttnn_actions[0].float().numpy()

    ax3 = fig.add_subplot(gs[1, 0:2])
    for dim in range(min(8, torch_np.shape[1])):
        ax3.plot(torch_np[:, dim], alpha=0.7, label=f"dim {dim}")
    ax3.set_title("PyTorch Actions (dims 0-7)", fontsize=11)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Action Value")
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2:4])
    for dim in range(min(8, ttnn_np.shape[1])):
        ax4.plot(ttnn_np[:, dim], alpha=0.7, label=f"dim {dim}")
    ax4.set_title("TTNN Actions (dims 0-7)", fontsize=11)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Action Value")
    ax4.legend(loc="upper right", fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # Row 3: Comparison and stats
    ax5 = fig.add_subplot(gs[2, 0:2])
    # Overlay comparison for first 4 dims
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for dim in range(4):
        ax5.plot(torch_np[:, dim], color=colors[dim], linestyle="-", alpha=0.8, label=f"PyTorch dim{dim}")
        ax5.plot(ttnn_np[:, dim], color=colors[dim], linestyle="--", alpha=0.8, label=f"TTNN dim{dim}")
    ax5.set_title("PyTorch vs TTNN Overlay (dims 0-3)\n(solid=PyTorch, dashed=TTNN)", fontsize=11)
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Action Value")
    ax5.grid(True, alpha=0.3)

    # Stats panel
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.axis("off")

    stats_text = f"""
    ═══════════════════════════════════════
              VALIDATION RESULTS
    ═══════════════════════════════════════

    Dataset:          {dataset_name}
    Prompt:           "{prompt}"

    Action Shape:     {list(torch_actions.shape)}
    Action Horizon:   {torch_actions.shape[1]} steps
    Action Dims:      {torch_actions.shape[2]}

    ───────────────────────────────────────

    PyTorch Actions:
      • Range: [{torch_np.min():.4f}, {torch_np.max():.4f}]
      • Mean:  {torch_np.mean():.4f}
      • Std:   {torch_np.std():.4f}

    TTNN Actions:
      • Range: [{ttnn_np.min():.4f}, {ttnn_np.max():.4f}]
      • Mean:  {ttnn_np.mean():.4f}
      • Std:   {ttnn_np.std():.4f}

    ───────────────────────────────────────

    PCC Score:        {pcc:.4f}
    PCC Threshold:    0.93
    Status:           {"✅ PASS" if pcc >= 0.93 else "❌ FAIL"}

    ═══════════════════════════════════════
    """

    ax6.text(
        0.1,
        0.95,
        stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    # Save
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n✅ Visualization saved to: {OUTPUT_PATH}")

    # Cleanup
    ttnn.close_device(device)
    plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
