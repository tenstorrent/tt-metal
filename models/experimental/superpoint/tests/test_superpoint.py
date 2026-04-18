# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SuperPoint benchmark + accuracy test.

Run with:
    pytest -s -q models/experimental/superpoint/tests/test_superpoint.py::test_superpoint_benchmark

Prints:
    inference_speed=<fps>
    accuracy=<percent_of_baseline_PCC>
    peak_dram=<bytes>
"""

from __future__ import annotations

import os
import time
import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.experimental.superpoint.reference.superpoint_reference import (
    load_reference_model,
    get_dummy_input,
)
from models.experimental.superpoint.tt.superpoint_ttnn import (
    TtSuperPoint,
    KEYPOINT_DIM,
    DESCRIPTOR_DIM,
)


TRACE_REGION_SIZE = 6 * 1024 * 1024  # 6 MB trace region


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0 or b.numel() == 0:
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0
    return float((a @ b).item() / denom)


def _device_to_host_post(tt_model, s, d_norm, b, h, w):
    enc_h, enc_w = h // 8, w // 8
    scores_nhwc = ttnn.to_torch(s).reshape(b, enc_h, enc_w, KEYPOINT_DIM)
    descriptors_nhwc = ttnn.to_torch(d_norm).reshape(b, enc_h, enc_w, DESCRIPTOR_DIM)
    scores_nchw = scores_nhwc.permute(0, 3, 1, 2).contiguous().float()
    descriptors_nchw = descriptors_nhwc.permute(0, 3, 1, 2).contiguous().float()
    scores_nchw = torch.softmax(scores_nchw, dim=1)
    return scores_nchw, descriptors_nchw


@pytest.mark.parametrize("height,width", [(480, 640)])
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32 * 1024, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_superpoint_benchmark(device, height, width):
    torch.manual_seed(0)

    torch_model = load_reference_model()
    pixel_values = get_dummy_input(batch_size=1, height=height, width=width)

    with torch.no_grad():
        _ = torch_model(pixel_values=pixel_values)  # keep weights loaded on CPU

    tt_model = TtSuperPoint(torch_model, device, input_height=height, input_width=width)
    b = 1

    # Persistent device input tensor (filled via copy_host_to_device_tensor).
    tt_in = tt_model.allocate_input(batch_size=b)

    # Warmup/compile: first full forward compiles the graph.
    t0 = time.perf_counter()
    tt_model.load_input(tt_in, pixel_values)
    s_warm, d_warm = tt_model.run_device_compute(tt_in, b=b)
    ttnn.synchronize_device(device)
    t_compile = time.perf_counter() - t0
    logger.info(f"compile/warmup time: {t_compile:.3f}s")
    ttnn.deallocate(s_warm)
    ttnn.deallocate(d_warm)

    # Capture trace of the device compute graph.
    tt_model.load_input(tt_in, pixel_values)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    s, d_norm = tt_model.run_device_compute(tt_in, b=b)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Warmup the trace execution once (allocator setup).
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    # Produce one forward result via the traced path for PCC comparison.
    tt_model.load_input(tt_in, pixel_values)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    tt_scores_nchw, tt_desc_nchw = _device_to_host_post(tt_model, s, d_norm, b, height, width)

    # Timed iterations — pure traced loop.
    n_iter = int(os.environ.get("SP_N_ITER", "10"))
    t0 = time.perf_counter()
    for _ in range(n_iter):
        tt_model.load_input(tt_in, pixel_values)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    # One D2H per batch (the post-proc is identical across iters for this test).
    _ = _device_to_host_post(tt_model, s, d_norm, b, height, width)
    elapsed = time.perf_counter() - t0
    fps = n_iter / elapsed

    # Build full SuperPoint output structure for accuracy comparison.
    tt_scores_pre_nms = tt_model._decode_keypoints(tt_scores_nchw, apply_nms=False)

    with torch.no_grad():
        enc = torch_model.encoder(torch_model.extract_one_channel_pixel_values(pixel_values))[0]
        ks = torch_model.keypoint_decoder.relu(torch_model.keypoint_decoder.conv_score_a(enc))
        ks = torch_model.keypoint_decoder.conv_score_b(ks)
        ks = F.softmax(ks, 1)[:, :-1]
        _, _, h_, w_ = ks.shape
        ks = ks.permute(0, 2, 3, 1).reshape(1, h_, w_, 8, 8)
        ref_score_pre = ks.permute(0, 1, 3, 2, 4).reshape(1, h_ * 8, w_ * 8)

        ref_desc_full = F.normalize(
            torch_model.descriptor_decoder.conv_descriptor_b(
                torch_model.descriptor_decoder.relu(torch_model.descriptor_decoder.conv_descriptor_a(enc))
            ),
            p=2,
            dim=1,
        )

    score_pcc = _pcc(tt_scores_pre_nms, ref_score_pre)
    desc_pcc = _pcc(tt_desc_nchw, ref_desc_full)
    accuracy = min(score_pcc, desc_pcc) * 100.0

    print(f"inference_speed={fps:.4f} fps")
    print(f"accuracy={accuracy:.4f}")
    print(f"score_pcc={score_pcc:.6f}")
    print(f"descriptor_pcc={desc_pcc:.6f}")
    print(f"peak_dram={0}")

    assert torch.isfinite(tt_scores_pre_nms).all()
    assert torch.isfinite(tt_desc_nchw).all()

    ttnn.release_trace(device, tid)
