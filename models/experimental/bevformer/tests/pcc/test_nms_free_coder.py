# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.bevformer.model_config import (
    BOX_DTYPE,
    CODE_SIZE,
    CODE_XY,
    CODE_Z,
    DECODER_NUM_LAYERS,
    MAX_NUM,
    NUM_CLASSES,
    NUM_QUERY,
    POST_CENTER_RANGE,
    SCORE_DTYPE,
)
from models.experimental.bevformer.reference.nms_free_coder import NMSFreeCoder, bottom_center
from models.experimental.bevformer.tests.common import assert_boxes_close, assert_pcc, tt_coder_detections
from models.experimental.bevformer.tt.tt_nms_free_coder import TTNMSFreeCoder

NUM_SCORES = NUM_QUERY * NUM_CLASSES
NUM_FOREGROUND = 50
# The scores' relative tolerance: the device sigmoid stays within about two float32 ulps of
# torch's on these logits; twice that.
SCORE_RTOL = 4 * torch.finfo(torch.float32).eps
# Shuffled into the class logits; the top 300 of each are distinct in float32.
LOGITS = {
    # Spread evenly over [-4, 4].
    "spread": torch.linspace(-4.0, 4.0, NUM_SCORES),
    # A detection head's shape: a few confident (query, class) pairs and a dense band of
    # background ones the top-k boundary falls into. The whole band rounds to one bfloat16
    # value, -7.96875, so bfloat16 ties it unless the coder shifts it by its pivot first.
    "background": torch.cat(
        [torch.linspace(-7.983, -7.955, NUM_SCORES - NUM_FOREGROUND), torch.linspace(1.0, 3.0, NUM_FOREGROUND)]
    ),
    # The 300th logit alone just above a band of equal ones, all of which round to its bfloat16
    # value, also after a shift by the bfloat16 pivot: a ranking that keeps a fixed number of
    # bfloat16 candidates past the pivot can lose it.
    "cutoff-collision": torch.cat(
        [
            torch.linspace(1.0, 3.0, MAX_NUM - 1),
            torch.tensor([-7.98002]),
            torch.full((NUM_SCORES - MAX_NUM,), -7.98003),
        ]
    ),
}


def _head_outputs(logits, batch_size, generator):
    """Head-like ``(L, bs, num_query, *)`` class logits, ``logits`` shuffled apart per layer and
    sample, so decoding the wrong layer shows, and box predictions whose centers spread past
    ``POST_CENTER_RANGE`` on every axis, so the range filter drops some of the top-k boxes."""
    shape = (DECODER_NUM_LAYERS, batch_size, NUM_QUERY)
    cls_scores = torch.stack(
        [logits[torch.randperm(NUM_SCORES, generator=generator)] for _ in range(DECODER_NUM_LAYERS * batch_size)]
    ).view(*shape, NUM_CLASSES)

    bbox_preds = torch.randn(*shape, CODE_SIZE, generator=generator)
    center_limit = torch.tensor(POST_CENTER_RANGE[3:]) * 1.2
    centers = (torch.rand(*shape, 3, generator=generator) * 2 - 1) * center_limit
    bbox_preds[..., CODE_XY] = centers[..., 0:2]
    bbox_preds[..., CODE_Z] = centers[..., 2:3]
    return cls_scores, bbox_preds


@pytest.mark.parametrize(
    "logits, batch_size, score_threshold",
    [
        ("spread", 1, None),
        ("spread", 2, None),
        ("background", 2, None),
        ("cutoff-collision", 2, None),
        # Some of the top-k scores, 0.977 to 0.982, pass the threshold.
        ("spread", 1, 0.98),
        # Some of the 50 foreground scores, 0.73 to 0.95, pass the threshold.
        ("background", 2, 0.5),
        # No score passes, so upstream's fallback lowers the threshold to 0.891, which only some
        # of the foreground scores reach.
        ("background", 2, 0.99),
    ],
    ids=[
        "spread-bs1",
        "spread-bs2",
        "background-bs2",
        "cutoff-collision-bs2",
        "spread-bs1-threshold",
        "background-bs2-threshold",
        "background-bs2-threshold-fallback",
    ],
)
def test_nms_free_coder(device, reset_seeds, logits, batch_size, score_threshold):
    cls_scores, bbox_preds = _head_outputs(LOGITS[logits], batch_size, torch.Generator().manual_seed(0))
    reference = NMSFreeCoder(score_threshold=score_threshold)
    tt_coder = TTNMSFreeCoder(device, score_threshold=score_threshold)
    # The dtypes the head emits.
    tt_cls_scores = ttnn.from_torch(cls_scores, dtype=SCORE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bbox_preds = ttnn.from_torch(bbox_preds, dtype=BOX_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)

    tt_scores, tt_labels, tt_query_index, tt_boxes = (
        ttnn.to_torch(t) for t in tt_coder.topk(tt_cls_scores[-1], tt_bbox_preds[-1])
    )
    for i in range(batch_size):
        scores, labels, query_index, boxes = reference.topk(cls_scores[-1, i], bbox_preds[-1, i])
        assert torch.equal(tt_labels[i].long(), labels), f"sample {i}: top-k labels differ"
        assert torch.equal(tt_query_index[i].long(), query_index), f"sample {i}: top-k queries differ"
        # PCC alone would pass a constant offset or scale.
        assert_pcc(scores, tt_scores[i].float(), 0.99)
        torch.testing.assert_close(tt_scores[i].float(), scores, rtol=SCORE_RTOL, atol=0)
        assert_boxes_close(boxes, tt_boxes[i].float())

    expected = reference.decode(cls_scores, bbox_preds)
    # The first decode compiles the range test; a second one must only hit the program cache.
    tt_coder_detections(tt_coder, tt_cls_scores, tt_bbox_preds)
    num_programs = device.num_program_cache_entries()
    actual = tt_coder_detections(tt_coder, tt_cls_scores, tt_bbox_preds)
    assert device.num_program_cache_entries() == num_programs
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        assert 0 < len(e["bboxes"]) < reference.max_num, "the filters must drop some boxes and keep some"
        assert torch.equal(a["labels"], e["labels"]), f"sample {i}: kept labels differ"
        assert_pcc(e["scores"], a["scores"], 0.99)
        torch.testing.assert_close(a["scores"], e["scores"], rtol=SCORE_RTOL, atol=0)
        assert_boxes_close(e["bboxes"], a["bboxes"])


@pytest.mark.parametrize("score_threshold", [None, 0.99], ids=["no-threshold", "threshold-fallback"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_nms_free_coder_traced(device, reset_seeds, score_threshold):
    """The coder's device part, the top-k and ``finalize``, as a trace: capture fails on any host
    read or write, so the score threshold's fallback runs on device. The replay's boxes, kept by
    the validity mask, are upstream ``get_bboxes``' boxes."""
    cls_scores, bbox_preds = _head_outputs(LOGITS["background"], 2, torch.Generator().manual_seed(0))
    reference = NMSFreeCoder(score_threshold=score_threshold)
    tt_coder = TTNMSFreeCoder(device, score_threshold=score_threshold)
    tt_cls_scores = ttnn.from_torch(cls_scores, dtype=SCORE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bbox_preds = ttnn.from_torch(bbox_preds, dtype=BOX_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)

    def run():
        scores, labels, _, boxes = tt_coder.topk(tt_cls_scores[-1], tt_bbox_preds[-1])
        return (scores, labels, *tt_coder.finalize(scores, boxes))

    # Compiles the programs before the capture.
    run()
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_outputs = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        actual = tt_coder.compact(*tt_outputs)
    finally:
        ttnn.release_trace(device, trace_id)

    for i, (e, a) in enumerate(zip(reference.decode(cls_scores, bbox_preds), actual, strict=True)):
        assert 0 < len(e["bboxes"]) < reference.max_num, "the filters must drop some boxes and keep some"
        assert torch.equal(a["labels"], e["labels"]), f"sample {i}: kept labels differ"
        torch.testing.assert_close(a["scores"], e["scores"], rtol=SCORE_RTOL, atol=0)
        assert_boxes_close(bottom_center(e["bboxes"]), a["bboxes"])
