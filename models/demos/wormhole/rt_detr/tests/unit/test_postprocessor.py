# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
rtdetr_pytorch_path = Path(__file__).parent.parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(rtdetr_pytorch_path))

from tt.postprocessor import postprocess

from models.common.utility_functions import comp_pcc

# so the logic should be mathematically identical to the reference.
PCC_THRESHOLD = 0.999


@pytest.fixture(scope="module")
def device():
    mesh_shape = ttnn.MeshShape(1, 1)
    dev = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)
    yield dev
    ttnn.close_mesh_device(dev)


def _to_device(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


class TestPostProcessor:
    def _reference_postprocess(self, logits, boxes, orig_sizes, score_threshold=0.3):
        """Golden PyTorch reference logic for bounding box decoding."""
        scores = logits.sigmoid()
        scores, labels = scores.max(dim=-1)

        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        results = []
        for i in range(logits.shape[0]):
            img_h, img_w = orig_sizes[i].tolist()
            scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

            keep = scores[i] > score_threshold
            results.append(
                {
                    "scores": scores[i][keep],
                    "labels": labels[i][keep],
                    "boxes": boxes_xyxy[i][keep] * scale,
                }
            )
        return results

    def _get_mock_inputs(self):
        """Generates deterministic mock decoder outputs to test postprocessing math."""
        torch.manual_seed(42)
        batch_size = 1
        num_queries = 300
        num_classes = 80

        # Simulate unnormalized logits and normalized [0, 1] boxes
        logits = torch.randn(batch_size, num_queries, num_classes)
        boxes = torch.rand(batch_size, num_queries, 4)
        orig_sizes = torch.tensor([[480, 640]])

        return logits, boxes, orig_sizes

    def test_postprocessor_logic(self, device):
        logits_torch, boxes_torch, orig_sizes = self._get_mock_inputs()
        score_threshold = 0.3

        # truncate PyTorch floats to bfloat16
        logits_torch = logits_torch.bfloat16().float()
        boxes_torch = boxes_torch.bfloat16().float()

        # 2. Prepare TTNN inputs
        logits_tt = _to_device(logits_torch.unsqueeze(1), device)
        boxes_tt = _to_device(boxes_torch.unsqueeze(1), device)

        # 3. pytorch reference
        gold_results = self._reference_postprocess(logits_torch, boxes_torch, orig_sizes, score_threshold)

        # 4. TT Postprocessor
        tt_results = postprocess(logits_tt, boxes_tt, orig_sizes, score_threshold)

        assert len(gold_results) == len(tt_results), "Batch size mismatch in outputs"

        for i in range(len(gold_results)):
            gold = gold_results[i]
            tt = tt_results[i]

            assert len(gold["scores"]) == len(tt["scores"]), f"Mismatch in number of detected objects for batch {i}"

            # Scores PCC
            score_pcc, score_msg = comp_pcc(gold["scores"], tt["scores"], PCC_THRESHOLD)
            print(f"\n[Batch {i} Scores] PCC: {score_pcc:.6f}")
            assert score_pcc >= PCC_THRESHOLD, f"Scores PCC failed: {score_msg}"

            # Labels Match (Exact match for categorical data)
            assert torch.equal(gold["labels"], tt["labels"]), f"Labels mismatch in batch {i}"

            # Boxes PCC
            box_pcc, box_msg = comp_pcc(gold["boxes"], tt["boxes"], PCC_THRESHOLD)
            print(f"[Batch {i} Boxes]  PCC: {box_pcc:.6f}")
            assert box_pcc >= PCC_THRESHOLD, f"Boxes PCC failed: {box_msg}"
