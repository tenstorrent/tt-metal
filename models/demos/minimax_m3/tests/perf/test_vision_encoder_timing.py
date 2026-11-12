# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Wall time of the MiniMax-M3 vision encoder on one 672x672 image (2304 patches).

Opt-in, as it builds all 32 layers:
    M3_VISION_TIMING=1 pytest -s models/demos/minimax_m3/tests/perf/test_vision_encoder_timing.py

It prints the median, fastest and slowest of 10 calls, and all 10, for two kinds of call:
  - "device forward": forward_device on pixel values already on the chip;
  - "end-to-end forward": model(pixel_values, image_grid_thw) from the host, with the input check and the upload.
Each set of calls follows 2 warm-ups and an IDLE_S pause, so it runs at full chip clock. Back to back, images run
slower. The test asserts only that both outputs are equal, finite and of the right shape. It uses seeded random
weights, so it needs no checkpoint.
M3_VISION_ACTIVATIONS sets M3VLVisionModel.activations, "l1", "l1_interleaved" or "dram", and the model's default
when unset. Both lines print it and the placement the forward takes. The test fails if the L1 check before a forward
sends it to DRAM.
"""
from __future__ import annotations

import math
import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.minimax_m3.tt.vision.common import block_placement
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.image_input import num_tokens
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel

from ..test_factory import parametrize_mesh_with_fabric
from ..unit.vision_checkpoint import random_vision_weights
from ..unit.vision_helpers import VISION_GRIDS, to_host

# A skip of the module, not a skipif mark: parametrize_mesh_with_fabric counts the chips when the test is defined, which
# fails on a host with none.
if os.getenv("M3_VISION_TIMING") != "1":
    pytest.skip("builds the 32-layer vision encoder; set M3_VISION_TIMING=1", allow_module_level=True)

WARMUP_CALLS = 2
TIMED_CALLS = 10
# Seconds of idle before the timed calls. The board lowers the chip clock after about a second of back-to-back
# forwards, and about a second of idle brings it back.
IDLE_S = 3.0


def _time_calls(call, mesh_device):
    """Time call(): WARMUP_CALLS warm-ups, an IDLE_S pause, then TIMED_CALLS timed calls.

    Each call is timed up to ttnn.synchronize_device after it, and its output is freed before the next call. Returns
    the timed calls' wall times in ms and the last output.
    """
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
def test_vision_encoder_timing(mesh_device, device_params):
    """Time forward_device and the whole forward on one 672x672 image; both must give the same finite tokens."""
    config = MiniMaxM3VisionConfig()
    model = M3VLVisionModel.from_state_dict(mesh_device, random_vision_weights(config), config, dtype=ttnn.bfloat16)
    model.activations = os.getenv("M3_VISION_ACTIVATIONS", model.activations)
    grid = VISION_GRIDS["h48w48"]
    grid_thw = torch.tensor([grid])
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(math.prod(grid), config.patch_dim, generator=generator)

    x_tt = model.push_pixel_values(pixel_values, grid_thw)
    placement = model.placement(x_tt)
    wanted = block_placement(model.blocks[0], x_tt.padded_shape[-2], model.activations)
    assert placement is wanted, f"activations {model.activations} take placement {placement.name}, not {wanted.name}"
    runs = {
        "device forward": _time_calls(lambda: model.forward_device(x_tt, grid_thw), mesh_device),
        "end-to-end forward": _time_calls(lambda: model(pixel_values, grid_thw), mesh_device),
    }
    # The L1 check must still give the same placement after the timed calls. A tensor left in L1 would have sent
    # them, and the times printed for them, to DRAM.
    after = model.placement(x_tt)
    assert after is placement, f"after the timed calls the forward takes placement {after.name}, not {placement.name}"
    print()
    for name, (times_ms, _) in runs.items():
        median, fastest, slowest = statistics.median(times_ms), min(times_ms), max(times_ms)
        calls = " ".join(f"{t:.2f}" for t in times_ms)
        print(
            f"{name} {median:.2f} ms (min {fastest:.2f}, max {slowest:.2f}), activations {model.activations}"
            f" in placement {placement.name}, calls {calls}"
        )

    shape = (num_tokens(grid_thw), config.text_hidden_size)
    tokens = []
    for name, (_, out) in runs.items():
        assert tuple(out.shape) == (1, 1, *shape), f"{name}: vision tokens of shape {tuple(out.shape)}"
        tokens.append(to_host(out, mesh_device, shape))
        num_bad = int((~torch.isfinite(tokens[-1])).sum())
        assert not num_bad, f"{name}: {num_bad} of the {tokens[-1].numel()} vision token values are NaN or Inf"
    num_diff = int((tokens[0].view(torch.int32) != tokens[1].view(torch.int32)).sum())
    assert not num_diff, f"{num_diff} of the {tokens[0].numel()} vision token values differ between the two calls"
