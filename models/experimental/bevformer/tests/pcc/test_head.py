# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    assert_topk_close,
    assert_channels_close,
    assert_pcc,
    build_reference_head,
    center_channels,
    random_bev,
)
from models.experimental.bevformer.tt.model_preprocessing import create_head_parameters
from models.experimental.bevformer.tt.tt_head import TtBEVFormerHead
from models.experimental.bevformer.tt.tt_nms_free_coder import TtNMSFreeCoder

CASES = [
    # (name, bev_shape, batch_size, traced)
    pytest.param("tiny", BEV_SHAPES["tiny"], 1, False, id="tiny"),
    pytest.param("tiny-traced", BEV_SHAPES["tiny"], 1, True, id="tiny-traced"),
    pytest.param("base", BEV_SHAPES["base"], 1, False, id="base"),
    # bs=2, so a mix-up in the queries repeated over the batch shows.
    pytest.param("tiny-bs2", BEV_SHAPES["tiny"], 2, False, id="tiny-bs2"),
]

# Headroom for the decoder's and branches' recorded commands, not a measured size.
DEVICE_PARAMS = [{"trace_region_size": 32 * 1024 * 1024}]


def _bev_embed(bev_shape, batch_size, seed):
    """Batch-first ``(bs, bev_h * bev_w, C)`` BEV features, as the encoder emits them."""
    generator = torch.Generator().manual_seed(seed)
    return random_bev(bev_shape, batch_size, generator).permute(1, 0, 2).contiguous()


def _models(bev_shape, device):
    torch_model = build_reference_head(bev_shape)
    return torch_model, TtBEVFormerHead(create_head_parameters(torch_model, device), device)


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _check(torch_outputs, tt_outputs):
    tt_outputs = tuple(ttnn.to_torch(t).float() for t in tt_outputs)
    # comp_pcc zeroes NaN and Inf before correlating, so they must be ruled out here.
    for name, tensor in zip(("class logits", "box predictions"), tt_outputs):
        assert torch.isfinite(tensor).all(), f"non-finite values in the head {name}"
    assert_centers_close(torch_outputs[1], tt_outputs[1])
    # Over all layers, then the last alone, which the coder reads and where bfloat16 error has built up most.
    for layers in (slice(None), -1):
        assert_pcc(torch_outputs[0][layers], tt_outputs[0][layers], 0.99)
        # Channel by channel: a joint PCC would be the centers' alone, as x and y span 102.4 m
        # against z's 8 m and the other channels' O(1).
        assert_channels_close(torch_outputs[1][layers], tt_outputs[1][layers])


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
    """The coder's top-k on the head's own outputs, against the reference head's (``assert_topk_close``)."""
    bev_shape, batch_size = BEV_SHAPES["tiny"], 2
    torch_model, tt_model = _models(bev_shape, device)
    bev_embed = _bev_embed(bev_shape, batch_size, seed=0)
    cls_scores, bbox_preds = torch_model(bev_embed)
    tt_cls_scores, tt_bbox_preds = tt_model(_to_device(bev_embed, device))
    tt_topk = tuple(ttnn.to_torch(t) for t in TtNMSFreeCoder().topk(tt_cls_scores[-1], tt_bbox_preds[-1]))
    assert_topk_close(
        cls_scores[-1], bbox_preds[-1], ttnn.to_torch(tt_cls_scores[-1]), ttnn.to_torch(tt_bbox_preds[-1]), tt_topk
    )
