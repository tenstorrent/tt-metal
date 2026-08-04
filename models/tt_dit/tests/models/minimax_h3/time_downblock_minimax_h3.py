# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock per down_block, **without** the profiler, to validate the profiled budget.

Tracy's per-op ``DEVICE FW DURATION`` agrees with an independent trace measurement for
Conv3d (18.5 ms profiled vs 18.25 ms from the blocking sweep's trace timer), but appears to
disagree wildly for the data-movement ops: it reports 21.3 ms for a tilize that measures
3.0 ms standalone at the identical shape, dtype and memory config, from the identical
producer. Only one of those can be right, and which one decides where the remaining encoder
time actually is.

The arbiter: sum the profiled per-op times for a block and compare against the block's
wall clock with no profiler attached. Same block, same shapes, one number each.

    pytest models/tt_dit/tests/models/minimax_h3/time_downblock_minimax_h3.py -s
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3 import encoder_minimax_h3 as enc_mod
from .profile_downblock_minimax_h3 import LEVELS

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
ITERS = 5


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_time_downblocks(mesh_device):
    total = 0.0
    for level, (in_ch, out_ch, t, h, w, t_stride, s_stride) in enumerate(LEVELS):
        torch.manual_seed(0)
        block = enc_mod.MiniMaxH3DownBlock3d(
            in_ch,
            out_ch,
            num_layers=2,
            num_frames=t,
            height=h,
            width=w,
            temporal_downsample_factor=t_stride,
            spatial_downsample_factor=s_stride,
            temporal_taps=3,
            mesh_device=mesh_device,
        )
        state = {}
        for i in range(2):
            c_in = in_ch if i == 0 else out_ch
            state[f"resnets.{i}.norm1.weight"] = torch.ones(c_in)
            state[f"resnets.{i}.norm1.bias"] = torch.zeros(c_in)
            state[f"resnets.{i}.norm2.weight"] = torch.ones(out_ch)
            state[f"resnets.{i}.norm2.bias"] = torch.zeros(out_ch)
            state[f"resnets.{i}.conv1.weight"] = torch.randn(out_ch, c_in, 3, 3, 3) * 0.02
            state[f"resnets.{i}.conv1.bias"] = torch.zeros(out_ch)
            state[f"resnets.{i}.conv2.weight"] = torch.randn(out_ch, out_ch, 3, 3, 3) * 0.02
            state[f"resnets.{i}.conv2.bias"] = torch.zeros(out_ch)
            if c_in != out_ch:
                state[f"resnets.{i}.conv_shortcut.weight"] = torch.randn(out_ch, c_in, 1, 1, 1) * 0.02
                state[f"resnets.{i}.conv_shortcut.bias"] = torch.zeros(out_ch)
        if t_stride * s_stride > 1:
            state["downsamplers.0.conv.weight"] = torch.randn(out_ch, out_ch, 3, 3, 3) * 0.02
            state["downsamplers.0.conv.bias"] = torch.zeros(out_ch)
        block.load_torch_state_dict(state)

        x = ttnn.from_torch(
            torch.randn(1, t, h, w, in_ch), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        block(x)
        ttnn.synchronize_device(mesh_device)
        best = float("inf")
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = block(x)
            ttnn.synchronize_device(mesh_device)
            best = min(best, time.perf_counter() - t0)
            del out
        total += best
        print(f"  level {level}  {in_ch:4d}->{out_ch:4d}  T{t} {h}x{w}   {best * 1e3:8.2f} ms", flush=True)
        del x, block

    print(f"  {'SUM of 6 down_blocks':34s} {total * 1e3:8.2f} ms", flush=True)
