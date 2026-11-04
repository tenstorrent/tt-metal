# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Wall time of the MiniMax-M3 vision encoder on one 672x672 image (h48w48: 2304 patches, 576 vision tokens): its two
speed numbers, so anyone can rerun them from the repository.

Opt-in, as it builds the 32-layer encoder; skipped unless M3_VISION_TIMING=1:
    M3_VISION_TIMING=1 pytest -s models/demos/minimax_m3/tests/perf/test_vision_encoder_timing.py

It prints two lines, each with the median of 10 calls, the fastest and the slowest of them, and the 10 in the order
they ran. Each line's calls follow 2 warm-up calls and a pause of IDLE_S (3 s), and each call is timed up to the end of
ttnn.synchronize_device after it. The pause matters: after about a second of back-to-back forwards the board lowers the
chip clock, and about a second of idle brings it back. So the 10 calls, about 0.7 s of work at 672x672, run at full
clock, and each line is the time of one image on an idle chip. Both lines come from one process. The device forward
runs first: its first warm-up compiles the programs and pushes the grid's RoPE tables, which the end-to-end calls then
reuse.
  - "device forward X ms": M3VLVisionModel.forward_device on pixel values pushed to the device once, before the calls:
    every device op of the encoder (patch embed, the input LayerNorm, the blocks and the projector), their dispatch
    from the host included. This is the encoder's time for one image on an idle chip, the number to compare with its
    roofline, which assumes the full clock.
  - "end-to-end forward Y ms": model(pixel_values, image_grid_thw) on the pixel values on the host, what a caller sees
    for one image: the input check and the upload (push_pixel_values), then forward_device. As both lines run at full
    clock, Y - X is the time of the check and the upload.
Neither line is the rate of images sent back to back: on the P150 where this was measured, 672x672 forwards run back to
back for a few seconds settled about 8% slower each, about 76.5 ms against 70.9 ms.
It asserts that the two outputs have the shape of the 576 tokens, are finite and are equal bit for bit (the same device
ops on the same input), never a time.

The model has the checkpoint's 32 layers and projector at its dimensions, with the seeded random weights of the vision
unit tests (the vision_reference fixture of tests/conftest.py): the time does not depend on the weights' values,
so the test needs no checkpoint and no HF_MODEL. It opens one card (1x1), as the vision unit tests do.
"""
from __future__ import annotations

import math
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.image_input import num_tokens
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel

from ..test_factory import parametrize_mesh_with_fabric
from ..unit.vision_helpers import VISION_GRIDS, to_host

# A skip of the module, not a skipif mark: parametrize_mesh_with_fabric counts the chips when the test is defined, which
# fails on a host with none.
if os.getenv("M3_VISION_TIMING") != "1":
    pytest.skip("builds the 32-layer vision encoder; set M3_VISION_TIMING=1", allow_module_level=True)

WARMUP_CALLS = 2
TIMED_CALLS = 10
# Seconds of idle between the warm-ups and the timed calls. After about a second of back-to-back forwards the board
# lowers the chip clock, and back to back a 672x672 forward then takes about 8% longer; about a second of idle brings
# the clock back.
IDLE_S = 3.0


def _time_calls(call, mesh_device):
    """Runs call() WARMUP_CALLS times, waits IDLE_S, then runs it TIMED_CALLS times, each call followed by
    ttnn.synchronize_device, and frees each output before the next call. Returns the timed calls' wall times in ms and
    the last output."""
    times_ms, out = [], None
    for i in range(WARMUP_CALLS + TIMED_CALLS):
        if out is not None:
            ttnn.deallocate(out)
        if i == WARMUP_CALLS:
            time.sleep(IDLE_S)
        start = time.perf_counter()
        out = call()
        ttnn.synchronize_device(mesh_device)
        times_ms.append((time.perf_counter() - start) * 1e3)
    return times_ms[WARMUP_CALLS:], out


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_vision_encoder_timing(mesh_device, device_params, vision_reference):
    """Prints the median wall time of forward_device and of model(pixel_values, image_grid_thw) on one 672x672 image,
    and checks that both give the same finite vision tokens of the right shape."""
    reference = vision_reference(MiniMaxM3VisionConfig().num_hidden_layers)
    model = M3VLVisionModel.from_torch(mesh_device, reference, dtype=ttnn.bfloat16)
    grid = VISION_GRIDS["h48w48"]
    grid_thw = torch.tensor([grid])
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(math.prod(grid), reference.config.patch_dim, generator=generator)

    x_tt = model.push_pixel_values(pixel_values, grid_thw)
    runs = {
        "device forward": _time_calls(lambda: model.forward_device(x_tt, grid_thw), mesh_device),
        "end-to-end forward": _time_calls(lambda: model(pixel_values, grid_thw), mesh_device),
    }
    print()
    for name, (times_ms, _) in runs.items():
        median, fastest, slowest = statistics.median(times_ms), min(times_ms), max(times_ms)
        calls = " ".join(f"{t:.2f}" for t in times_ms)
        print(f"{name} {median:.2f} ms (min {fastest:.2f}, max {slowest:.2f}), calls {calls}")

    shape = (num_tokens(grid_thw), reference.config.text_hidden_size)
    tokens = []
    for name, (_, out) in runs.items():
        assert tuple(out.shape) == (1, 1, *shape), f"{name}: vision tokens of shape {tuple(out.shape)}"
        tokens.append(to_host(out, mesh_device, shape))
        num_bad = int((~torch.isfinite(tokens[-1])).sum())
        assert not num_bad, f"{name}: {num_bad} of the {tokens[-1].numel()} vision token values are NaN or Inf"
    num_diff = int((tokens[0].view(torch.int32) != tokens[1].view(torch.int32)).sum())
    assert not num_diff, f"{num_diff} of the {tokens[0].numel()} vision token values differ between the two calls"
