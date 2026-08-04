# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Which GroupNorm is actually fastest at the encoder's real norm shapes?

Three candidates, all timed **ROW_MAJOR in, ROW_MAJOR out** -- which is the honest
comparison, because ``conv3d`` sits on both sides of every norm and requires ROW_MAJOR, so
whatever tilize/untilize a candidate needs is part of its cost:

* :class:`MiniMaxH3FrameGroupNorm` -- ``ttnn.group_norm`` with T as the batch axis, and the
  pinned ``determine_expected_group_norm_dram_grid_size`` grid. This is what the encoder ran
  before amendment 52.
* :class:`MiniMaxH3DistributedFrameGroupNorm` -- the shipping hand-written stats norm.
* ``ttnn.experimental.dit_fused_distributed_groupnorm`` -- rejects ``N > 1``, so it is called
  once per frame; the question is whether lifting that restriction is worth a kernel change.
  It gets the *same pinned grid* as the group_norm path, not an arbitrary 8x8.

Levels 0 and 1 are 92 % of encoder device time, so those two shapes decide it.
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn

from ....models.vae.minimax_h3.encoder_minimax_h3 import (
    MINIMAX_H3_VAE_NORM_EPS,
    MINIMAX_H3_VAE_NUM_GROUPS,
    MiniMaxH3DistributedFrameGroupNorm,
    MiniMaxH3FrameGroupNorm,
)

SINGLE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
ITERS = 5

# (T, H, W, C) at the two levels that dominate.
CASES = [(17, 256, 256, 128), (17, 128, 128, 256)]


def _best(fn, device) -> float:
    out = fn()
    ttnn.synchronize_device(device)
    del out
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
        del out
    return best * 1e3


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE, indirect=["mesh_device", "device_params"])
def test_groupnorm_candidates(mesh_device):
    for T, H, W, C in CASES:
        mb = T * H * W * C * 2 / 1e6
        print(f"\n=== T={T} {H}x{W} C={C}  ({mb:.0f} MB activation) ===", flush=True)
        torch.manual_seed(0)
        state = {"weight": torch.randn(C), "bias": torch.randn(C)}
        kwargs = dict(num_frames=T, height=H, width=W, mesh_device=mesh_device)

        x_rm = ttnn.from_torch(
            torch.randn(1, T, H, W, C), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        plain = MiniMaxH3FrameGroupNorm(C, **kwargs)
        plain.load_torch_state_dict(dict(state))
        print(f"  {'ttnn.group_norm (T as batch)':34s} {_best(lambda: plain(x_rm), mesh_device):8.2f} ms", flush=True)

        stats = MiniMaxH3DistributedFrameGroupNorm(C, **kwargs)
        stats.load_torch_state_dict(dict(state))
        print(f"  {'stats norm (shipping)':34s} {_best(lambda: stats(x_rm), mesh_device):8.2f} ms", flush=True)

        # The fused op, with the same pinned grid the group_norm path uses. One frame at a
        # time (N==1 is all it accepts), so multiply by T for the whole site -- plus the
        # tilize/untilize the encoder would still have to pay around it.
        grid = plain.core_grid
        mask = plain.mask.data
        gamma = plain.weight.data
        beta = plain.bias.data
        x_one = ttnn.from_torch(
            torch.randn(1, 1, H * W, C), dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT
        )

        def fused_one():
            return ttnn.experimental.dit_fused_distributed_groupnorm(
                x_one,
                num_groups=MINIMAX_H3_VAE_NUM_GROUPS,
                epsilon=MINIMAX_H3_VAE_NORM_EPS,
                cluster_axis=0,
                mesh_device=mesh_device,
                multi_device_global_semaphore=[],
                topology=ttnn.Topology.Linear,
                input_mask=mask,
                weight=gamma,
                bias=beta,
                use_welford=True,
            )

        try:
            per_frame = _best(fused_one, mesh_device)
            print(
                f"  {'fused GN, per frame':34s} {per_frame:8.2f} ms   x{T} = {per_frame * T:7.2f} ms"
                f"  (grid {grid})",
                flush=True,
            )
        except Exception as exc:
            print(f"  {'fused GN, per frame':34s} FAILED {str(exc)[:120]}", flush=True)

        # For reference: what the layout round trip alone costs, since two of the three
        # candidates pay it and it bounds how good any TILE-only norm can be.
        x_flat = ttnn.reshape(x_rm, (T, 1, H * W, C))
        tilize_ms = _best(lambda: ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT), mesh_device)
        x_tile = ttnn.to_layout(x_flat, ttnn.TILE_LAYOUT)
        untilize_ms = _best(lambda: ttnn.to_layout(x_tile, ttnn.ROW_MAJOR_LAYOUT), mesh_device)
        print(f"  {'(tilize + untilize floor)':34s} {tilize_ms + untilize_ms:8.2f} ms", flush=True)
        del x_rm, x_one, x_tile
