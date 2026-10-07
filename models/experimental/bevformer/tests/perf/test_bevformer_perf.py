"""Tracy harness for the detector end to end: images to class logits, boxes and the frame's BEV.

Dummy weights, ``BEVFORMER_CHECKPOINT`` or not, and ``test_bevformer``'s inputs over three frames.
The first frame, which has no previous BEV, runs eagerly; the second is the warmup and the PCC
gate, then is captured as a trace and replayed once in a signposted region, so the report covers
already-compiled programs with no host dispatch in between. The capture is also the check that the
forward has no host operations: any read or write between host and device inside a capture region
fails it.

The third frame replays the same trace, as a deployment would: ``prepare_frame`` refills the
frame's buffers in place on the host, and the second frame's BEV is copied into the captured
previous-BEV input. The camera rig stays the same, so the plan sized for the first frame covers
every refill. The ego drives the third step backwards and twice as far, turning the other way,
so its shift and rotation differ from the second frame's.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.bevformer.tests.common import (
    BEV_SHAPES,
    assert_channels_close,
    assert_pcc,
    build_reference_bevformer,
    frame_metas,
    random_image_batch,
    signposted_trace,
    to_conv_layout,
)
from models.experimental.bevformer.tt.model_preprocessing import create_bevformer_parameters
from models.experimental.bevformer.tt.tt_bevformer import TtBEVFormer

# test_bevformer gates at 0.95 for the trained weights; the dummy ones stay above 0.99 there.
PCC_THRESHOLD = 0.99


def _check(expected, tt_outputs):
    cls_scores, bbox_preds, bev = expected
    tt_cls_scores, tt_bbox_preds, tt_bev = (ttnn.to_torch(t).float() for t in tt_outputs)
    assert_pcc(bev, tt_bev.reshape(bev.shape), PCC_THRESHOLD)
    assert_pcc(cls_scores[-1], tt_cls_scores[-1], PCC_THRESHOLD)
    assert_channels_close(bbox_preds[-1], tt_bbox_preds[-1], PCC_THRESHOLD)


@torch.no_grad()
# The reference runs the backbone on six 928x1600 images on the CPU four times: once to record
# the conv shapes and once per frame.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    # Headroom for the detector's recorded commands, not a measured size.
    [{"l1_small_size": 32 * 1024, "trace_region_size": 128 * 1024 * 1024}],
    indirect=True,
)
def test_bevformer_perf(device, reset_seeds):
    torch_model = build_reference_bevformer(BEV_SHAPES["base"])
    img = random_image_batch()[None]
    frames = frame_metas(1, 3, torch.Generator().manual_seed(0), motion=(1, -2))

    expected, torch_prev = [], None
    for metas in frames:
        outputs = torch_model(img, metas, prev_bev=torch_prev)
        expected.append(outputs)
        *_, torch_prev = outputs

    tt_model = TtBEVFormer(create_bevformer_parameters(torch_model, img, device), device)
    # Uploaded once so the profiled replay measures the detector, not the transfer.
    tt_img = to_conv_layout(img.flatten(0, 1), device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Later frames refill this frame's buffers in place, the ones the trace records.
    frame = tt_model.prepare_frame(frames[0])
    *_, prev_bev = tt_model(tt_img, frame)
    tt_model.prepare_frame(frames[1], frame)
    # Doubles as the warmup of the path with a previous BEV, the one the trace records.
    _check(expected[1], tt_model(tt_img, frame, prev_bev=prev_bev))

    with signposted_trace(device, lambda: tt_model(tt_img, frame, prev_bev=prev_bev)) as (trace_id, tt_outputs):
        _check(expected[1], tt_outputs)
        start = time.perf_counter()
        tt_model.prepare_frame(frames[2], frame)
        logger.info(f"prepare_frame refill on the host: {(time.perf_counter() - start) * 1e3:.1f} ms")
        *_, tt_bev = tt_outputs
        # Into the buffer the trace reads as the previous BEV; rebinding the name would not reach it.
        ttnn.copy(tt_bev, prev_bev)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        _check(expected[2], tt_outputs)
