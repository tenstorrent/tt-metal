# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Throughput scaling benchmark (Scenario 4).

Measures PI0 and SmolVLA inference throughput from 1 to N chips,
generates scaling charts, and produces a latency breakdown waterfall.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def run_pi0_benchmark(
    model,
    device,
    num_iterations: int = 20,
    warmup: int = 3,
) -> Dict:
    """
    Benchmark a single PI0 replica.

    Returns dict with latency stats and FPS.
    """
    from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
    config = model.config

    images = [torch.randn(1, 3, 224, 224) for _ in range(2)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(2)]
    lang_tokens = torch.randint(0, 256000, (1, 32))
    lang_masks = torch.ones(1, 32, dtype=torch.bool)
    state = torch.randn(1, 32)

    import ttnn
    images_tt = [
        ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for img in images
    ]
    lang_tt = ttnn.from_torch(lang_tokens, dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    lmask_tt = ttnn.from_torch(lang_masks.float(), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)
    state_tt = ttnn.from_torch(state, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            model.sample_actions(images=images_tt, img_masks=img_masks,
                                 lang_tokens=lang_tt, lang_masks=lmask_tt, state=state_tt)
        ttnn.synchronize_device(device)

    latencies = []
    for _ in range(num_iterations):
        t0 = time.time()
        with torch.no_grad():
            model.sample_actions(images=images_tt, img_masks=img_masks,
                                 lang_tokens=lang_tt, lang_masks=lmask_tt, state=state_tt)
        ttnn.synchronize_device(device)
        latencies.append((time.time() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "model": "PI0",
        "num_iterations": num_iterations,
        "avg_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": 1000.0 / np.mean(arr),
        "latencies": latencies,
    }


def run_smolvla_benchmark(
    model,
    num_iterations: int = 20,
    warmup: int = 3,
) -> Dict:
    """Benchmark a single SmolVLA replica."""
    from PIL import Image
    test_img = Image.new("RGB", (512, 512), color=(128, 128, 128))

    for _ in range(warmup):
        with torch.no_grad():
            model.sample_actions(images=[test_img], instruction="pick up object",
                                 num_inference_steps=10, action_dim=6)

    latencies = []
    for _ in range(num_iterations):
        t0 = time.time()
        with torch.no_grad():
            model.sample_actions(images=[test_img], instruction="pick up object",
                                 num_inference_steps=10, action_dim=6)
        latencies.append((time.time() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "model": "SmolVLA",
        "num_iterations": num_iterations,
        "avg_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": 1000.0 / np.mean(arr),
        "latencies": latencies,
    }


def generate_scaling_chart(
    results: List[Dict],
    title: str = "Throughput Scaling",
    save_path: str = "scaling_chart.png",
) -> Optional[str]:
    """
    Generate a bar chart showing FPS scaling from 1 to N chips.

    results: list of dicts, each with 'num_chips', 'total_fps', 'model'.
    """
    if not _HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    models = sorted(set(r["model"] for r in results))
    colors = {"PI0": "#6c5ce7", "SmolVLA": "#00cec9"}
    bar_width = 0.35

    for midx, model_name in enumerate(models):
        model_results = [r for r in results if r["model"] == model_name]
        model_results.sort(key=lambda r: r["num_chips"])
        chips = [r["num_chips"] for r in model_results]
        fps_vals = [r["total_fps"] for r in model_results]

        x = np.arange(len(chips)) + midx * bar_width
        ax.bar(x, fps_vals, bar_width, label=model_name,
               color=colors.get(model_name, "#dfe6e9"), edgecolor="white", linewidth=0.5)

        for xi, fv in zip(x, fps_vals):
            ax.text(xi, fv + 0.1, f"{fv:.1f}", ha="center", va="bottom",
                    color="white", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Blackhole Chips", color="white", fontsize=12)
    ax.set_ylabel("Total Throughput (FPS)", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(len(chips)) + bar_width / 2)
    ax.set_xticklabels([str(c) for c in chips], color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#16213e", edgecolor="white", labelcolor="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return save_path


def generate_latency_waterfall(
    breakdown: Dict[str, float],
    title: str = "Latency Breakdown",
    save_path: str = "latency_waterfall.png",
) -> Optional[str]:
    """
    Generate a horizontal waterfall chart showing latency per pipeline stage.

    breakdown: dict like {'Vision': 23, 'VLM Prefill': 9, 'Denoising': 121, ...}
    """
    if not _HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    stages = list(breakdown.keys())
    values = list(breakdown.values())
    cumulative = np.cumsum([0] + values[:-1])
    palette = ["#6c5ce7", "#a29bfe", "#00cec9", "#81ecec", "#fdcb6e", "#fab1a0"]

    for i, (stage, val) in enumerate(zip(stages, values)):
        color = palette[i % len(palette)]
        ax.barh(0, val, left=cumulative[i], height=0.5, color=color,
                edgecolor="white", linewidth=0.5, label=f"{stage}: {val:.0f}ms")

    ax.set_yticks([])
    ax.set_xlabel("Latency (ms)", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#16213e", edgecolor="white", labelcolor="white",
              loc="upper right", fontsize=9)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return save_path
