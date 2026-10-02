# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.config.decoder_config import CODE_SIZE, CODE_XY, CODE_Z
from models.experimental.bevformer.config.head_config import NUM_CLASSES, POST_CENTER_RANGE
from models.experimental.bevformer.reference.nms_free_coder import NMSFreeCoder
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.decoder_common import NUM_LAYERS, NUM_QUERY
from models.experimental.bevformer.tests.head_common import assert_boxes_close
from models.experimental.bevformer.tt.tt_common import SCORE_DTYPE
from models.experimental.bevformer.tt.tt_decoder import GRID_DTYPE
from models.experimental.bevformer.tt.tt_nms_free_coder import TtNMSFreeCoder

NUM_SCORES = NUM_QUERY * NUM_CLASSES
NUM_FOREGROUND = 50
# The scores' relative tolerance: the device sigmoid stays within about two float32 ulps of
# torch's on these logits; twice that.
SCORE_RTOL = 4 * torch.finfo(torch.float32).eps
# All distinct in float32, shuffled into the class logits.
LOGITS = {
    # Spread evenly over [-4, 4].
    "spread": torch.linspace(-4.0, 4.0, NUM_SCORES),
    # A trained head's shape: a few confident (query, class) pairs and a dense band of
    # background ones the top-k boundary falls into. The whole band rounds to one bfloat16
    # value, -7.96875, so bfloat16 ties it unless the coder shifts it by its pivot first.
    "background": torch.cat(
        [torch.linspace(-7.983, -7.955, NUM_SCORES - NUM_FOREGROUND), torch.linspace(1.0, 3.0, NUM_FOREGROUND)]
    ),
}


def _head_outputs(logits, batch_size, generator):
    """Head-like ``(L, bs, num_query, *)`` class logits, ``logits`` shuffled apart per layer and
    sample, so decoding the wrong layer shows, and box predictions whose centers spread past
    ``POST_CENTER_RANGE`` on every axis, so the range filter drops some of the top-k boxes."""
    shape = (NUM_LAYERS, batch_size, NUM_QUERY)
    cls_scores = torch.stack(
        [logits[torch.randperm(NUM_SCORES, generator=generator)] for _ in range(NUM_LAYERS * batch_size)]
    ).view(*shape, NUM_CLASSES)

    bbox_preds = torch.randn(*shape, CODE_SIZE, generator=generator)
    center_limit = torch.tensor(POST_CENTER_RANGE[3:]) * 1.2
    centers = (torch.rand(*shape, 3, generator=generator) * 2 - 1) * center_limit
    bbox_preds[..., CODE_XY] = centers[..., 0:2]
    bbox_preds[..., CODE_Z] = centers[..., 2:3]
    return cls_scores, bbox_preds


@pytest.mark.parametrize(
    "logits, batch_size",
    [("spread", 1), ("spread", 2), ("background", 2)],
    ids=["spread-bs1", "spread-bs2", "background-bs2"],
)
def test_nms_free_coder(device, reset_seeds, logits, batch_size):
    cls_scores, bbox_preds = _head_outputs(LOGITS[logits], batch_size, torch.Generator().manual_seed(0))
    reference = NMSFreeCoder()
    tt_coder = TtNMSFreeCoder()
    # The dtypes the head emits.
    tt_cls_scores = ttnn.from_torch(cls_scores, dtype=SCORE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bbox_preds = ttnn.from_torch(bbox_preds, dtype=GRID_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)

    tt_scores, tt_labels, tt_query_index, tt_boxes = (
        ttnn.to_torch(t) for t in tt_coder.topk(tt_cls_scores[-1], tt_bbox_preds[-1])
    )
    num_programs = device.num_program_cache_entries()
    for i in range(batch_size):
        scores, labels, query_index, boxes = reference.topk(cls_scores[-1, i], bbox_preds[-1, i])
        assert torch.equal(tt_labels[i].long(), labels), f"sample {i}: top-k labels differ"
        assert torch.equal(tt_query_index[i, :, 0].long(), query_index), f"sample {i}: top-k queries differ"
        # PCC alone would pass a constant offset or scale.
        assert_pcc(scores, tt_scores[i].float(), 0.99)
        torch.testing.assert_close(tt_scores[i].float(), scores, rtol=SCORE_RTOL, atol=0)
        assert_boxes_close(boxes, tt_boxes[i].float())

    expected = reference.decode(cls_scores, bbox_preds)
    actual = tt_coder.decode(tt_cls_scores, tt_bbox_preds)
    # decode runs topk again on the same shapes, so it must only hit the program cache.
    assert device.num_program_cache_entries() == num_programs
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        assert 0 < len(e["bboxes"]) < reference.max_num, "the range filter must drop some boxes and keep some"
        assert torch.equal(a["labels"], e["labels"]), f"sample {i}: kept labels differ"
        assert_pcc(e["scores"], a["scores"], 0.99)
        torch.testing.assert_close(a["scores"], e["scores"], rtol=SCORE_RTOL, atol=0)
        assert_boxes_close(e["bboxes"], a["bboxes"])
