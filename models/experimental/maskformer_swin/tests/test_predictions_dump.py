# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import torch

from models.experimental.maskformer_swin.eval_utils import dump_predictions_json, summarize_predictions


def test_dump_predictions_json(tmp_path: Path):
    """Ensure prediction dump helper writes expected fields."""

    class_logits = torch.tensor([[[2.0, -1.0, 0.5, -3.0], [0.1, 1.5, -0.2, -0.5]]], dtype=torch.float32)
    id2label = {"0": "person", "1": "car", "2": "tree"}

    out_path = tmp_path / "preds.json"
    dump_predictions_json(class_logits=class_logits, id2label=id2label, output_path=out_path, task_type="instance")
    payload = json.loads(out_path.read_text())

    assert payload["num_queries"] == 2
    assert payload["task_type"] == "instance"
    assert len(payload["predictions"]) == 2
    first = payload["predictions"][0]
    assert set(first.keys()) == {"mask_index", "class_id", "class_label", "confidence", "task_type"}


def test_summarize_predictions_shapes():
    class_logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    preds = summarize_predictions(class_logits=class_logits, id2label={}, task_type="semantic")
    assert len(preds) == 3
    assert all(p["task_type"] == "semantic" for p in preds)
