#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GR00T N1.6 Action Visualization

Creates a visual summary showing:
1. Predicted action trajectories across time horizon
2. Action dimension heatmap
3. Per-step statistics

Outputs: actions_visualization.png in the demo folder

Usage:
    cd /home/ttuser/experiments/pi0/tt-metal
    export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
    python models/experimental/gr00t_n1_6/tests/demo/visualize_actions.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

DEMO_DIR = Path(__file__).parent
OUTPUT_PATH = DEMO_DIR / "actions_visualization.png"


def main():
    print("=" * 60)
    print("  GR00T N1.6 Action Visualization")
    print("=" * 60)

    import ttnn
    from models.experimental.gr00t_n1_6.common.configs import Gr00tN16Config
    from models.experimental.gr00t_n1_6.common.weight_loader import Gr00tN16WeightLoader
    from models.experimental.gr00t_n1_6.tt.ttnn_groot_n16_model import Gr00tN16ModelTTNN
    from models.experimental.gr00t_n1_6.tt.ttnn_common import to_tt_tensor

    config = Gr00tN16Config.default()
    loader = Gr00tN16WeightLoader()
    loader.load()

    device = ttnn.open_device(device_id=0)

    try:
        model = Gr00tN16ModelTTNN(config, loader, device)

        # Run inference with two different seeds
        results = {}
        for seed in [42, 123]:
            torch.manual_seed(seed)
            pixel_values = torch.randn(1, 3, 224, 224)
            state = torch.randn(1, config.embodiment.max_state_dim)

            t0 = time.time()
            img = model.encode_vision(pixel_values)
            backbone = to_tt_tensor(ttnn.to_torch(img), device)
            actions = model.run_flow_matching(backbone, state, embodiment_id=0)
            elapsed = (time.time() - t0) * 1000

            results[seed] = {"actions": actions[0].numpy(), "latency": elapsed}
            print(f"  Seed {seed}: {actions.shape}, {elapsed:.1f}ms")

        # Create visualization
        print("\nCreating visualization...")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.3)

        fig.suptitle("GR00T N1.6-3B Action Predictions (Tenstorrent Blackhole)", fontsize=14, fontweight="bold")

        actions_42 = results[42]["actions"]  # [50, 128]
        actions_123 = results[123]["actions"]

        # Row 1: Action trajectories (first 8 dims)
        ax1 = fig.add_subplot(gs[0, 0])
        for d in range(min(8, actions_42.shape[1])):
            ax1.plot(actions_42[:, d], alpha=0.7, label=f"dim {d}")
        ax1.set_title("Action Trajectories (seed=42, dims 0-7)", fontsize=11)
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Action Value")
        ax1.legend(loc="upper right", fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        for d in range(min(8, actions_123.shape[1])):
            ax2.plot(actions_123[:, d], alpha=0.7, label=f"dim {d}")
        ax2.set_title("Action Trajectories (seed=123, dims 0-7)", fontsize=11)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Action Value")
        ax2.legend(loc="upper right", fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)

        # Row 2: Heatmaps
        ax3 = fig.add_subplot(gs[1, 0])
        # Show first 32 action dims for readability
        n_show = min(32, actions_42.shape[1])
        im = ax3.imshow(actions_42[:, :n_show].T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax3.set_title(f"Action Heatmap (seed=42, dims 0-{n_show-1})", fontsize=11)
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Action Dimension")
        plt.colorbar(im, ax=ax3, shrink=0.8)

        ax4 = fig.add_subplot(gs[1, 1])
        im2 = ax4.imshow(actions_123[:, :n_show].T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax4.set_title(f"Action Heatmap (seed=123, dims 0-{n_show-1})", fontsize=11)
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Action Dimension")
        plt.colorbar(im2, ax=ax4, shrink=0.8)

        # Row 3: Stats
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        stats_text = (
            f"{'='*70}\n"
            f"{'GR00T N1.6-3B Performance Summary':^70}\n"
            f"{'='*70}\n\n"
            f"  Action Shape:     [{actions_42.shape[0]}, {actions_42.shape[1]}] "
            f"(horizon={actions_42.shape[0]}, dims={actions_42.shape[1]})\n\n"
            f"  Seed 42:   range=[{actions_42.min():.4f}, {actions_42.max():.4f}]  "
            f"mean={actions_42.mean():.4f}  std={actions_42.std():.4f}  "
            f"latency={results[42]['latency']:.1f}ms\n"
            f"  Seed 123:  range=[{actions_123.min():.4f}, {actions_123.max():.4f}]  "
            f"mean={actions_123.mean():.4f}  std={actions_123.std():.4f}  "
            f"latency={results[123]['latency']:.1f}ms\n\n"
            f"  Device:           Tenstorrent Blackhole p150a\n"
            f"  Vision Encoder:   SigLIP2 (27 layers, 1152 dim)\n"
            f"  Action Head:      AlternateVLDiT (32 layers, 1536 dim)\n"
            f"  Flow Steps:       4 (Euler integration)\n"
            f"{'='*70}"
        )

        ax5.text(
            0.05,
            0.95,
            stats_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"\n  Saved: {OUTPUT_PATH}")

    finally:
        ttnn.close_device(device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
