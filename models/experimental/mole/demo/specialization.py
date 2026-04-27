# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import torch

from models.experimental.mole.demo.run import (
    add_dataset_arguments,
    add_model_arguments,
    CHECKPOINT_BASE_DIR,
    CheckpointEndpointOptions,
    CheckpointInferenceEndpoint,
    close_ttnn_device,
    config_from_checkpoint_resolution,
    open_ttnn_device,
    resolve_mole_checkpoint,
    set_random_seed,
    unpack_batch,
)
from models.experimental.mole.reference.config import MoLEConfig

COLORS = ("#0B6E4F", "#C84C09", "#2F5DBE", "#B02E63", "#0E7490", "#8A6F00", "#6D28D9", "#4B5563")
DEFAULT_ROUTER_IMAGE_PATH = str(Path.home() / ".cache" / "tt-metal" / "mole" / "router_weights.png")
EVAL_BATCH_SIZE = 32


@dataclass(frozen=True)
class VisualizationOptions:
    checkpoint_path: str
    dataset_csv_path: str
    dataset: str
    image_path: str = DEFAULT_ROUTER_IMAGE_PATH
    max_eval_batches: int | None = None
    plot_max_samples: int = 1200
    seed: int = 2021
    dataset_dir: str | None = None


def _save_png(path: Path, router_weights: torch.Tensor, *, plot_max_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = router_weights.to(dtype=torch.float32).clamp(0.0, 1.0)
    num_samples, num_experts = weights.shape

    if plot_max_samples <= 0:
        raise ValueError("plot_max_samples must be > 0")

    if num_samples > plot_max_samples:
        reduced = weights[:plot_max_samples]
        start_index = 0
    else:
        reduced = weights
        start_index = 0

    x = torch.arange(start_index, start_index + reduced.shape[0], dtype=torch.float32)
    x_min = float(x[0].item())
    x_max = float(x[-1].item())

    winner_idx = torch.argmax(reduced, dim=1)
    fig = plt.figure(figsize=(14, 8.8), facecolor="#F7F4EC")
    grid = fig.add_gridspec(2, 1, height_ratios=(0.55, 2.45), hspace=0.18)
    ax_winners = fig.add_subplot(grid[0, 0])
    ax_lines = fig.add_subplot(grid[1, 0])

    cmap = ListedColormap([COLORS[i % len(COLORS)] for i in range(num_experts)])
    norm = BoundaryNorm(boundaries=list(range(num_experts + 1)), ncolors=num_experts)
    ax_winners.imshow(
        winner_idx.unsqueeze(0).numpy(),
        cmap=cmap,
        norm=norm,
        extent=(x_min, x_max, 0.0, 1.0),
        aspect="auto",
        interpolation="nearest",
    )
    ax_winners.set_yticks([])
    ax_winners.set_xticks([])
    ax_winners.set_title("Top expert", fontsize=12, pad=8)
    ax_winners.set_xlim(x_min, x_max)
    ax_winners.margins(x=0)

    winner_handles = [
        Patch(facecolor=COLORS[i % len(COLORS)], edgecolor="none", label=f"Expert {i}") for i in range(num_experts)
    ]
    ax_winners.legend(
        handles=winner_handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
    )

    for expert_index in range(num_experts):
        color = COLORS[expert_index % len(COLORS)]
        ax_lines.plot(
            x.numpy(),
            reduced[:, expert_index].numpy(),
            color=color,
            linewidth=1.8,
            label=f"Expert {expert_index}",
        )
    ax_lines.set_xlim(x_min, x_max)
    ax_lines.margins(x=0)

    flattened = reduced.reshape(-1)
    low_q = float(torch.quantile(flattened, 0.01).item())
    high_q = float(torch.quantile(flattened, 0.99).item())
    y_min = max(0.0, low_q - 0.04)
    y_max = min(1.0, high_q + 0.04)
    if y_max - y_min < 0.12:
        center = 0.5 * (y_min + y_max)
        y_min = max(0.0, center - 0.06)
        y_max = min(1.0, center + 0.06)
    ax_lines.set_ylim(y_min, y_max)
    ax_lines.set_xlabel(
        f"Sample index ({reduced.shape[0]} of {num_samples} shown"
        + (f", start={start_index}" if num_samples > reduced.shape[0] else "")
        + ")"
    )
    ax_lines.set_ylabel("Router weight")
    ax_lines.grid(True, alpha=0.25)
    ax_lines.set_title("Per-expert weights", fontsize=11, pad=6)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_router_weights(
    model_config: MoLEConfig,
    *,
    options: VisualizationOptions | None = None,
) -> None:
    if options is None:
        raise ValueError("VisualizationOptions are required")
    visualization_options = options

    set_random_seed(visualization_options.seed)

    collected_samples = 0
    device = open_ttnn_device()
    try:
        endpoint = CheckpointInferenceEndpoint(
            device=device,
            options=CheckpointEndpointOptions(
                checkpoint_path=visualization_options.checkpoint_path,
                dataset_csv_path=visualization_options.dataset_csv_path,
                assets_root=visualization_options.dataset_dir or CHECKPOINT_BASE_DIR,
                dataset=visualization_options.dataset,
            ),
        )
        loaders, model_config = endpoint.resolve_dataset(
            model_config,
            eval_batch_size=EVAL_BATCH_SIZE,
        )
        model = endpoint.build_mole_ttnn(model_config)

        router_weights_list = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loaders["test"]):
                if (
                    visualization_options.max_eval_batches is not None
                    and batch_index >= visualization_options.max_eval_batches
                ):
                    break
                inputs, _, input_marks, _ = unpack_batch(batch)
                if input_marks is None:
                    raise ValueError("specialization requires x_mark time features")
                _, gating_weights = endpoint.predict_from_torch(
                    model=model,
                    torch_input=inputs,
                    torch_input_mark=input_marks,
                    return_router_output=True,
                )
                router_weights = gating_weights.mean(dim=2)
                router_weights_list.append(router_weights)
                collected_samples += int(router_weights.shape[0])
                if collected_samples >= visualization_options.plot_max_samples:
                    break
    finally:
        close_ttnn_device(device)

    if not router_weights_list:
        raise ValueError("No evaluation batches produced")

    router_weights = torch.cat(router_weights_list, dim=0)
    _save_png(
        Path(visualization_options.image_path),
        router_weights,
        plot_max_samples=visualization_options.plot_max_samples,
    )
    print(f"checkpoint_path={endpoint.options.checkpoint_path}")
    print(f"dataset_csv_path={endpoint.options.dataset_csv_path}")
    print(f"Saved {visualization_options.image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MoLE router weights as a PNG image.")
    add_dataset_arguments(parser)
    add_model_arguments(parser)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--image-path", type=str, default=DEFAULT_ROUTER_IMAGE_PATH, help="Output PNG path")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--plot-max-samples",
        type=int,
        default=1200,
        help="Maximum number of contiguous test samples to visualize",
    )
    args = parser.parse_args()
    resolution = resolve_mole_checkpoint(
        dataset=args.dataset,
        base_model_type=args.base_model_type,
        num_experts=args.num_experts,
        assets_root=args.dataset_dir,
    )
    options = VisualizationOptions(
        checkpoint_path=resolution.checkpoint_path,
        dataset_csv_path=resolution.dataset_csv_path,
        dataset=args.dataset,
        image_path=args.image_path,
        max_eval_batches=args.max_eval_batches,
        plot_max_samples=args.plot_max_samples,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
    )

    visualize_router_weights(
        model_config=config_from_checkpoint_resolution(
            resolution,
            base_model_type=args.base_model_type,
            num_experts=args.num_experts,
        ),
        options=options,
    )
