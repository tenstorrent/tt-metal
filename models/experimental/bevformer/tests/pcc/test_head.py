# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.config.head_config import NUM_CLASSES
from models.experimental.bevformer.reference.nms_free_coder import NMSFreeCoder, denormalize_bbox
from models.experimental.bevformer.tests.backbone_common import assert_pcc
from models.experimental.bevformer.tests.decoder_common import BEV_SHAPES, assert_channels_close, random_bev_features
from models.experimental.bevformer.tests.head_common import assert_boxes_close, build_reference_head, centre_channels
from models.experimental.bevformer.tt.model_preprocessing_head import create_head_parameters
from models.experimental.bevformer.tt.tt_head import TtBEVFormerHead
from models.experimental.bevformer.tt.tt_nms_free_coder import TtNMSFreeCoder

CASES = [
    # (name, bev_shape, batch_size, traced)
    pytest.param("tiny", BEV_SHAPES["tiny"], 1, False, id="tiny"),
    pytest.param("tiny-traced", BEV_SHAPES["tiny"], 1, True, id="tiny-traced"),
    pytest.param(
        "base",
        BEV_SHAPES["base"],
        1,
        False,
        id="base",
        marks=pytest.mark.xfail(
            reason="decoder accuracy on the 200x200 grid: its last layer's output reaches PCC 0.98, "
            "and some box channels computed from it fall below 0.99 (the centres do not)",
            raises=AssertionError,
            strict=True,
        ),
    ),
    # bs=2, so a mix-up in the queries repeated over the batch shows.
    pytest.param("tiny-bs2", BEV_SHAPES["tiny"], 2, False, id="tiny-bs2"),
]

# Headroom for the decoder's and branches' recorded commands, not a measured size.
DEVICE_PARAMS = [{"trace_region_size": 32 * 1024 * 1024}]


def _bev_embed(bev_shape, batch_size, seed):
    """Batch-first ``(bs, bev_h * bev_w, C)`` BEV features, as the encoder emits them."""
    generator = torch.Generator().manual_seed(seed)
    return random_bev_features(bev_shape, batch_size, generator).permute(1, 0, 2).contiguous()


def _models(bev_shape, device):
    torch_model = build_reference_head(bev_shape)
    return torch_model, TtBEVFormerHead(create_head_parameters(torch_model, device), device, bev_shape)


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _check(torch_outputs, tt_outputs):
    tt_outputs = tuple(ttnn.to_torch(t).float() for t in tt_outputs)
    # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
    for name, tensor in zip(("class logits", "box predictions"), tt_outputs):
        assert torch.isfinite(tensor).all(), f"non-finite values in the head {name}"
    centres = centre_channels()
    centre_error = (torch_outputs[1][..., centres] - tt_outputs[1][..., centres]).abs()
    logger.info(f"box centre error: mean {centre_error.mean():.4f} m, max {centre_error.max():.4f} m")
    assert_pcc(torch_outputs[0], tt_outputs[0], 0.99)
    # Channel by channel: a joint PCC would be the centres' alone, as x and y span 102.4 m
    # against z's 8 m and the other channels' O(1).
    assert_channels_close(torch_outputs[1], tt_outputs[1])


@torch.no_grad()
@pytest.mark.parametrize("name, bev_shape, batch_size, traced", CASES)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_head(device, reset_seeds, name, bev_shape, batch_size, traced):
    torch_model, tt_model = _models(bev_shape, device)
    bev_embed = _bev_embed(bev_shape, batch_size, seed=0)
    tt_bev_embed = _to_device(bev_embed, device)

    # The first run compiles; a second one must only hit the program cache.
    tt_model(tt_bev_embed)
    num_programs = device.num_program_cache_entries()

    if not traced:
        tt_outputs = tt_model(tt_bev_embed)
        assert device.num_program_cache_entries() == num_programs
        _check(torch_model(bev_embed), tt_outputs)
        return

    # Capture fails on any host read or write in forward. Replaying on fresh features shows
    # the replay, not a leftover eager result, fills the captured output buffers.
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_outputs = tt_model(tt_bev_embed)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    replay_bev_embed = _bev_embed(bev_shape, batch_size, seed=1)
    host = ttnn.from_torch(replay_bev_embed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(host, tt_bev_embed)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    try:
        _check(torch_model(replay_bev_embed), tt_outputs)
    finally:
        ttnn.release_trace(device, trace_id)


@torch.no_grad()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_head_detections(device, reset_seeds):
    """The coder on the head's own outputs: it must select a top-k of the reference head's
    scores, and the scores and boxes it returns must be the reference head's at the same
    (query, class) pairs.

    The head's logits differ from the reference's slightly, so near-equal scores may rank in
    a different order and the two top-k sets differ at their tail; the overlap is logged.
    """
    bev_shape, batch_size = BEV_SHAPES["tiny"], 2
    torch_model, tt_model = _models(bev_shape, device)
    bev_embed = _bev_embed(bev_shape, batch_size, seed=0)
    cls_scores, bbox_preds = torch_model(bev_embed)
    tt_cls_scores, tt_bbox_preds = tt_model(_to_device(bev_embed, device))

    reference = NMSFreeCoder()
    tt_scores, tt_labels, tt_query_index, tt_boxes = (
        ttnn.to_torch(t) for t in TtNMSFreeCoder().topk(tt_cls_scores[-1], tt_bbox_preds[-1])
    )
    # The largest score error over all num_query * num_classes pairs, not only the selected.
    score_error = (cls_scores[-1].sigmoid() - ttnn.to_torch(tt_cls_scores[-1]).float().sigmoid()).abs().amax((1, 2))
    for i in range(batch_size):
        labels, query_index = tt_labels[i].long(), tt_query_index[i, :, 0].long()
        ref_scores, ref_labels, ref_query_index, _ = reference.topk(cls_scores[-1, i], bbox_preds[-1, i])
        assert (tt_scores[i][1:] <= tt_scores[i][:-1]).all(), f"sample {i}: top-k scores are not sorted"
        # A pair the device ranks above a reference top-k pair is off by at most twice the
        # score error, so every selected pair scores at least the reference k-th minus that.
        selected = cls_scores[-1, i, query_index, labels].sigmoid()
        kth = ref_scores[-1] - 2 * score_error[i]
        assert (selected >= kth).all(), f"sample {i}: a selected pair scores below the top-k bound {kth:.5f}"
        pairs = set((query_index * NUM_CLASSES + labels).tolist())
        ref_pairs = set((ref_query_index * NUM_CLASSES + ref_labels).tolist())
        logger.info(f"sample {i}: top-{reference.max_num} overlap {len(pairs & ref_pairs) / len(ref_pairs):.3f}")

        assert_pcc(selected, tt_scores[i].float(), 0.99)
        assert_boxes_close(denormalize_bbox(bbox_preds[-1, i, query_index]), tt_boxes[i].float())
