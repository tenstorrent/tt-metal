# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.config.decoder_config import CODE_SIZE, REG_XY, REG_Z
from models.experimental.bevformer.config.head_config import NUM_CLASSES, POST_CENTER_RANGE
from models.experimental.bevformer.reference.nms_free_coder import NMSFreeCoder
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.decoder_common import NUM_LAYERS, NUM_QUERY
from models.experimental.bevformer.tests.head_common import assert_boxes_close
from models.experimental.bevformer.tt.tt_nms_free_coder import SCORE_DTYPE, TtNMSFreeCoder

# Logits spread evenly over [-LOGIT_RANGE, LOGIT_RANGE]: adjacent sigmoid scores differ by
# 1e-5 or more, well above the device sigmoid's error, so the top-k must match exactly.
LOGIT_RANGE = 4.0


def _head_outputs(batch_size, generator):
    """Head-like ``(L, bs, num_query, *)`` class logits, all distinct, and box codes whose
    centres spread past ``POST_CENTER_RANGE`` on every axis, so the range filter drops some
    of the top-k boxes."""
    shape = (NUM_LAYERS, batch_size, NUM_QUERY)
    num_scores = NUM_QUERY * NUM_CLASSES
    logits = torch.linspace(-LOGIT_RANGE, LOGIT_RANGE, num_scores)
    cls_scores = torch.stack([logits[torch.randperm(num_scores, generator=generator)] for _ in range(batch_size)])
    cls_scores = cls_scores.view(1, batch_size, NUM_QUERY, NUM_CLASSES).expand(NUM_LAYERS, -1, -1, -1).contiguous()

    bbox_preds = torch.randn(*shape, CODE_SIZE, generator=generator)
    centre_limit = torch.tensor(POST_CENTER_RANGE[3:]) * 1.2
    centres = (torch.rand(*shape, 3, generator=generator) * 2 - 1) * centre_limit
    bbox_preds[..., REG_XY] = centres[..., 0:2]
    bbox_preds[..., REG_Z] = centres[..., 2:3]
    return cls_scores, bbox_preds


@pytest.mark.parametrize("batch_size", [1, 2], ids=["bs1", "bs2"])
def test_nms_free_coder(device, reset_seeds, batch_size):
    cls_scores, bbox_preds = _head_outputs(batch_size, torch.Generator().manual_seed(0))
    reference = NMSFreeCoder()
    tt_coder = TtNMSFreeCoder()
    tt_cls_scores, tt_bbox_preds = (
        ttnn.from_torch(t, dtype=SCORE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device) for t in (cls_scores, bbox_preds)
    )

    tt_scores, tt_labels, tt_query_index, tt_boxes = (
        ttnn.to_torch(t) for t in tt_coder.topk(tt_cls_scores[-1], tt_bbox_preds[-1])
    )
    for i in range(batch_size):
        scores, labels, query_index, boxes = reference.topk(cls_scores[-1, i], bbox_preds[-1, i])
        assert torch.equal(tt_labels[i].long(), labels), f"sample {i}: top-k labels differ"
        assert torch.equal(tt_query_index[i, :, 0].long(), query_index), f"sample {i}: top-k queries differ"
        assert_pcc(scores, tt_scores[i].float(), 0.99)
        assert_boxes_close(boxes, tt_boxes[i].float())

    expected = reference.decode(cls_scores, bbox_preds)
    actual = tt_coder.decode(tt_cls_scores, tt_bbox_preds)
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        assert 0 < len(e["bboxes"]) < reference.max_num, "the range filter must drop some boxes and keep some"
        assert torch.equal(a["labels"], e["labels"]), f"sample {i}: kept labels differ"
        assert_pcc(e["scores"], a["scores"], 0.99)
        assert_boxes_close(e["bboxes"], a["bboxes"])
