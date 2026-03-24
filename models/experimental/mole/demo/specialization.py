# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from models.experimental.mole.demo.core import (
    TrainingConfig,
    add_dataset_arguments,
    add_model_arguments,
    add_training_arguments,
    model_config_from_args,
    resolve_dataset_config,
    set_random_seed,
    training_config_from_args,
    train_model_on_dataloader,
    unpack_batch,
)
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.mole import MixtureOfLinearExperts
from models.experimental.mole.utils.datasets import create_real_dataset_loaders

COLORS = ("#0b6e4f", "#c84c09", "#2f5dbe", "#b02e63", "#6c4ab6", "#007f8c", "#8a6f00", "#5c4033")
W, H = 1040, 420
ML, MR, MT, MB = 72, 24, 32, 60
DEFAULT_ROUTER_IMAGE_PATH = str(Path.home() / ".cache" / "tt-metal" / "mole" / "router_weights.png")


@dataclass(frozen=True)
class VisualizationOptions:
    image_path: str = DEFAULT_ROUTER_IMAGE_PATH
    max_eval_batches: int | None = None
    seed: int = 2021
    dataset_path: str | None = None


def _save_png(path: Path, router_weights: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n, e = router_weights.shape
    pw, ph = W - ML - MR, H - MT - MB

    def sx(i):
        return ML + pw * i / max(1, n - 1)

    def wy(v):
        return MT + ph * (1.0 - v)

    img = Image.new("RGB", (W, H), "#fffdf8")
    draw = ImageDraw.Draw(img)

    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = wy(v)
        draw.line((ML, y, W - MR, y), fill="#d9d9d9", width=1)
        draw.text((ML - 32, y - 6), f"{v:.2f}", fill="#444")

    draw.line((ML, MT, ML, H - MB), fill="#444", width=2)
    draw.line((ML, H - MB, W - MR, H - MB), fill="#444", width=2)
    draw.text((W // 2 - 140, 8), f"MoLE Router Weights  (samples={n}, experts={e})", fill="#111")

    for expert_index in range(e):
        color = COLORS[expert_index % len(COLORS)]
        pts = [(sx(i), wy(router_weights[i, expert_index].item())) for i in range(n)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
        ly = MT + 18 * expert_index
        draw.line((W - 160, ly, W - 136, ly), fill=color, width=3)
        draw.text((W - 128, ly - 6), f"Expert {expert_index}", fill="#222")

    img.save(path)


def visualize_router_weights(
    model_config: MoLEConfig,
    training_config: TrainingConfig,
    *,
    dataset_name: str,
    options: VisualizationOptions | None = None,
) -> None:
    visualization_options = options or VisualizationOptions()

    set_random_seed(visualization_options.seed)

    loaders, input_dim = create_real_dataset_loaders(
        dataset_name,
        visualization_options.dataset_path,
        seq_len=model_config.seq_len,
        pred_len=model_config.pred_len,
        batch_size=training_config.batch_size,
        eval_batch_size=training_config.eval_batch_size,
        freq=model_config.freq,
    )
    model_config = resolve_dataset_config(model_config, input_dim=input_dim)

    model = MixtureOfLinearExperts(model_config)
    training_summary = train_model_on_dataloader(model, loaders, training_config, return_summary=True)
    model = training_summary["trained_model"]

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
            _, gating_weights = model(inputs, input_marks)
            router_weights_list.append(gating_weights.mean(dim=2))

    if not router_weights_list:
        raise ValueError("No evaluation batches produced")

    router_weights = torch.cat(router_weights_list, dim=0)
    _save_png(Path(visualization_options.image_path), router_weights)
    print(f"Saved {visualization_options.image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MoLE router weights as a PNG image.")
    add_dataset_arguments(parser, dataset_help="Dataset for visualization", dataset_path_help="Optional dataset path")
    add_model_arguments(parser)
    add_training_arguments(parser, default_batch_size=16, default_eval_batch_size=32, default_steps=80)
    parser.add_argument("--image-path", type=str, default=DEFAULT_ROUTER_IMAGE_PATH, help="Output PNG path")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.set_defaults(seed=2021)
    args = parser.parse_args()
    options = VisualizationOptions(
        image_path=args.image_path,
        max_eval_batches=args.max_eval_batches,
        seed=args.seed,
        dataset_path=args.dataset_path,
    )

    visualize_router_weights(
        model_config=model_config_from_args(args),
        training_config=training_config_from_args(args),
        dataset_name=args.dataset_name,
        options=options,
    )
