# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.atss_swin_l_dyhead.demo.evaluate_coco_slice_4dev import (
    remap_results_to_original,
    results_to_coco,
)


def test_remap_results_to_original_1920x1080():
    results = {
        "bboxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
        "scores": torch.tensor([0.75]),
        "labels": torch.tensor([0]),
    }

    remapped = remap_results_to_original(results, original_width=1920, original_height=1080)

    torch.testing.assert_close(remapped["bboxes"], torch.tensor([[150.0, 168.75, 450.0, 337.5]]))
    torch.testing.assert_close(results["bboxes"], torch.tensor([[100.0, 200.0, 300.0, 400.0]]))


def test_results_to_coco_maps_category_and_xywh():
    results = {
        "bboxes": torch.tensor([[10.0, 20.0, 40.0, 70.0], [0.0, 0.0, 5.0, 5.0]]),
        "scores": torch.tensor([0.9, 0.01]),
        "labels": torch.tensor([0, 0]),
    }

    converted = results_to_coco(
        results,
        image_id=17,
        category_map={0: 1},
        score_threshold=0.05,
    )

    assert converted == [
        {
            "image_id": 17,
            "category_id": 1,
            "bbox": [10.0, 20.0, 30.0, 50.0],
            "score": pytest.approx(0.9),
        }
    ]


def test_empty_results_are_supported():
    results = {
        "bboxes": torch.zeros((0, 4)),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
    }

    remapped = remap_results_to_original(results, original_width=1920, original_height=1080)

    assert remapped["bboxes"].shape == (0, 4)
    assert results_to_coco(remapped, image_id=3, category_map={0: 1}) == []


def test_results_to_coco_requires_category_mapping():
    results = {
        "bboxes": torch.tensor([[10.0, 20.0, 40.0, 70.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([2]),
    }

    with pytest.raises(KeyError, match="model label 2"):
        results_to_coco(results, image_id=17, category_map={0: 1})
