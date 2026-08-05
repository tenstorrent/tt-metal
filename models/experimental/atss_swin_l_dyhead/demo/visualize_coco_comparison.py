#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Render COCO ground truth and model detections as 2x2 comparison images.

Each output contains:
  - COCO ground truth
  - PyTorch reference predictions
  - Vanilla TTNN predictions
  - Stage-2 high-precision TTNN predictions

Prediction files use the standard COCO result format emitted by
``evaluate_coco_slice_4dev.py``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PANEL_WIDTH = 960
TITLE_HEIGHT = 46


def _load_json(path: Path):
    with path.open() as file:
        return json.load(file)


def _index_by_image(items):
    indexed = defaultdict(list)
    for item in items:
        indexed[int(item["image_id"])].append(item)
    return indexed


def _draw_xywh(image, items, color, *, score_threshold=None, show_scores=False):
    rendered = image.copy()
    line_width = max(2, round(min(image.shape[:2]) / 450))
    font_scale = max(0.55, min(image.shape[:2]) / 1500)
    kept = 0

    for item in items:
        score = float(item.get("score", 1.0))
        if score_threshold is not None and score < score_threshold:
            continue

        x, y, width, height = (float(value) for value in item["bbox"])
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + width))
        y2 = int(round(y + height))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, line_width)

        label = f"{score:.2f}" if show_scores else "boat"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width)
        label_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            rendered,
            (x1, label_top),
            (x1 + text_width + 6, label_top + text_height + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            rendered,
            label,
            (x1 + 3, label_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )
        kept += 1

    return rendered, kept


def _make_panel(image, title, count, color):
    scale = PANEL_WIDTH / image.shape[1]
    panel = cv2.resize(image, (PANEL_WIDTH, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    title_bar = np.full((TITLE_HEIGHT, PANEL_WIDTH, 3), 28, dtype=np.uint8)
    cv2.rectangle(title_bar, (0, 0), (10, TITLE_HEIGHT), color, -1)
    cv2.putText(
        title_bar,
        f"{title} ({count} boxes)",
        (24, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((title_bar, panel))


def _pad_to_shape(image, height, width):
    bottom = height - image.shape[0]
    right = width - image.shape[1]
    return cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(18, 18, 18))


def _make_grid(panels):
    max_height = max(panel.shape[0] for panel in panels)
    max_width = max(panel.shape[1] for panel in panels)
    panels = [_pad_to_shape(panel, max_height, max_width) for panel in panels]
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))


def _make_contact_sheet(comparisons, output_path):
    if not comparisons:
        return

    thumbnail_width = 760
    thumbnails = []
    for comparison in comparisons:
        scale = thumbnail_width / comparison.shape[1]
        thumbnails.append(
            cv2.resize(
                comparison,
                (thumbnail_width, round(comparison.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        )

    if len(thumbnails) % 2:
        thumbnails.append(np.full_like(thumbnails[0], 245))
    rows = [np.hstack(thumbnails[index : index + 2]) for index in range(0, len(thumbnails), 2)]
    cv2.imwrite(str(output_path), np.vstack(rows))


def render(args):
    annotations_path = Path(args.annotations).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_json(annotations_path)
    image_infos = sorted(dataset["images"], key=lambda image: int(image["id"]))
    ground_truth = _index_by_image(
        annotation for annotation in dataset["annotations"] if int(annotation["category_id"]) == args.category_id
    )

    prediction_paths = {
        "pytorch": Path(args.pytorch_predictions).resolve(),
        "ttnn_vanilla": Path(args.ttnn_vanilla_predictions).resolve(),
        "ttnn_stage2_hp": Path(args.ttnn_stage2_hp_predictions).resolve(),
    }
    predictions = {
        name: _index_by_image(
            prediction for prediction in _load_json(path) if int(prediction["category_id"]) == args.category_id
        )
        for name, path in prediction_paths.items()
    }

    colors = {
        "ground_truth": (70, 190, 70),
        "pytorch": (220, 120, 30),
        "ttnn_vanilla": (20, 165, 255),
        "ttnn_stage2_hp": (205, 70, 205),
    }
    titles = {
        "ground_truth": "COCO ground truth",
        "pytorch": "PyTorch reference",
        "ttnn_vanilla": "TTNN vanilla",
        "ttnn_stage2_hp": "TTNN stage-2 precision",
    }

    manifest = {
        "score_threshold": args.score_threshold,
        "annotations": str(annotations_path),
        "prediction_files": {name: str(path) for name, path in prediction_paths.items()},
        "images": [],
    }
    comparisons = []

    for image_info in image_infos:
        image_id = int(image_info["id"])
        image_path = images_dir / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        gt_image, gt_count = _draw_xywh(
            image,
            ground_truth[image_id],
            colors["ground_truth"],
            show_scores=False,
        )
        panels = [_make_panel(gt_image, titles["ground_truth"], gt_count, colors["ground_truth"])]
        counts = {"ground_truth": gt_count}

        for model_name in ("pytorch", "ttnn_vanilla", "ttnn_stage2_hp"):
            prediction_image, count = _draw_xywh(
                image,
                predictions[model_name][image_id],
                colors[model_name],
                score_threshold=args.score_threshold,
                show_scores=True,
            )
            panels.append(_make_panel(prediction_image, titles[model_name], count, colors[model_name]))
            counts[model_name] = count

        comparison = _make_grid(panels)
        output_path = output_dir / f"{image_id:03d}_{Path(image_info['file_name']).stem}_comparison.jpg"
        if not cv2.imwrite(str(output_path), comparison):
            raise RuntimeError(f"Failed to write comparison image: {output_path}")
        comparisons.append(comparison)
        manifest["images"].append(
            {
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "comparison": output_path.name,
                "counts": counts,
            }
        )

    manifest["totals"] = {
        name: sum(image["counts"][name] for image in manifest["images"])
        for name in ("ground_truth", "pytorch", "ttnn_vanilla", "ttnn_stage2_hp")
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _make_contact_sheet(comparisons, output_dir / "contact_sheet.jpg")
    print(f"Rendered {len(comparisons)} comparisons in {output_dir}")
    print(f"Totals: {manifest['totals']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COCO ground truth and three ATSS model outputs")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--pytorch-predictions", required=True)
    parser.add_argument("--ttnn-vanilla-predictions", required=True)
    parser.add_argument("--ttnn-stage2-hp-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--category-id", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
