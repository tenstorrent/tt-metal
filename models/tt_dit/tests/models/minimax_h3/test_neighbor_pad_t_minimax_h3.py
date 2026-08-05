# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The T-axis halo exchange, isolated.

STATE.md amendment 107 measured T-parallel audio decode producing saturated garbage on every shard but
the first, at every factor, and showed it is *not* a halo-width problem: the error is uniform within each
shard rather than banded at the boundaries. That points at the halo exchange delivering wrong data
wholesale, and the audio decoder makes **381** halo calls per forward across many shapes -- far too
coarse a scope to debug in.

So this gates `_t_neighbor_pad` on its own, against a host-computed expectation, at one shape. If these
pass the bug is in how the decoder accumulates across calls; if they fail the wrapper or the op is wrong
and this is where to fix it.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.audio_ops import _partition_t, _t_neighbor_pad
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager

MESH = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="mesh4x8",
    )
]

# T must divide the shard factor. 256 is what 207 audio latents pad up to, and the factor-4 shard is 64.
BATCH, TOTAL_T, CHANNELS = 2, 256, 32


def _expected(host_BTC: torch.Tensor, *, factor: int, pad_left: int, pad_right: int, mode: str) -> list[torch.Tensor]:
    """Per-shard expected halo output, computed on host.

    Shard `s` owns rows `[s*L, (s+1)*L)` and wants `pad_left` rows before and `pad_right` after, taken
    from its neighbours. At the outer edges there is no neighbour, so the mode decides: `zeros` pads with
    zeros, `replicate` repeats the edge row.
    """
    length = host_BTC.shape[1] // factor
    out = []
    for shard in range(factor):
        lo, hi = shard * length, (shard + 1) * length
        left = host_BTC[:, max(0, lo - pad_left) : lo]
        if left.shape[1] < pad_left:
            missing = pad_left - left.shape[1]
            edge = (
                torch.zeros(host_BTC.shape[0], missing, host_BTC.shape[2])
                if mode == "zeros"
                else host_BTC[:, :1].expand(-1, missing, -1)
            )
            left = torch.cat([edge, left], dim=1)
        right = host_BTC[:, hi : hi + pad_right]
        if right.shape[1] < pad_right:
            missing = pad_right - right.shape[1]
            edge = (
                torch.zeros(host_BTC.shape[0], missing, host_BTC.shape[2])
                if mode == "zeros"
                else host_BTC[:, -1:].expand(-1, missing, -1)
            )
            right = torch.cat([right, edge], dim=1)
        out.append(torch.cat([left, host_BTC[:, lo:hi], right], dim=1))
    return out


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(("pad_left", "pad_right"), [(6, 0), (3, 3), (11, 0)], ids=["causal_6", "same_3", "causal_11"])
@pytest.mark.parametrize("mode", ["zeros", "replicate"], ids=["zeros", "replicate"])
def test_t_neighbor_pad_matches_host(mesh_device, pad_left, pad_right, mode):
    """One halo exchange, factor 4 on the 4-wide axis, against the host expectation.

    Gated per shard rather than on a whole-tensor metric, because amendment 107's whole point is that
    *which* shard is wrong is the diagnosis. A single PSNR over the concatenation cannot say that.
    """
    factor, mesh_axis = 4, 0
    parallel_config = ParallelFactor(factor=factor, mesh_axis=mesh_axis)
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    torch.manual_seed(0)
    # Row t carries the value t, so a wrong row is identifiable on sight rather than by distance.
    host = torch.arange(TOTAL_T, dtype=torch.float32).view(1, TOTAL_T, 1).expand(BATCH, TOTAL_T, CHANNELS).contiguous()

    # Mirror the production path exactly: the decoder uploads **replicated** and the vocoder shards with
    # `ttnn.mesh_partition(dim=1, cluster_axis=...)`, which splits along *one* mesh axis and leaves the
    # other replicated. `ShardTensorToMesh(dim=1)` would split 32 ways across the whole mesh instead --
    # the first version of this test did that and got a local T of 8 rather than 64, which is a different
    # question than the one being asked.
    replicated = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    sharded = _partition_t(replicated, parallel_config)
    assert sharded.shape[1] == TOTAL_T // factor, f"local T after partition is {sharded.shape[1]}"
    padded = _t_neighbor_pad(
        sharded,
        pad_left=pad_left,
        pad_right=pad_right,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        padding_mode=mode,
    )

    expected = _expected(host, factor=factor, pad_left=pad_left, pad_right=pad_right, mode=mode)
    shard_length = TOTAL_T // factor
    assert (
        padded.shape[1] == shard_length + pad_left + pad_right
    ), f"local T is {padded.shape[1]}, expected {shard_length + pad_left + pad_right}"

    # `mesh_partition(cluster_axis=0)` splits T along the 4-wide axis, so device `(r, c)` holds shard `r`
    # and every device in mesh row `r` holds the same shard. Shard `r` is therefore on device `r * cols`.
    replicas = ttnn.get_device_tensors(padded)
    failures = []
    for shard in range(factor):
        device_index = shard * tuple(mesh_device.shape)[1]
        actual = ttnn.to_torch(replicas[device_index]).float()
        want = expected[shard]
        if not torch.allclose(actual, want, atol=0.51):
            wrong_rows = (actual - want).abs().amax(dim=(0, 2)).gt(0.51).nonzero().flatten().tolist()
            failures.append((shard, wrong_rows[:12], float((actual - want).abs().max())))
            logger.error(
                f"shard {shard} (device {device_index}) differs at local rows {wrong_rows[:12]} "
                f"(max |err| {float((actual - want).abs().max()):.1f}); "
                f"row 0 wants {want[0, 0, 0]:.0f} got {actual[0, 0, 0]:.0f}, "
                f"last wants {want[0, -1, 0]:.0f} got {actual[0, -1, 0]:.0f}"
            )
        else:
            logger.info(f"shard {shard} (device {device_index}) OK")

    assert not failures, (
        f"halo exchange wrong on {len(failures)} of {factor} shards, pad=({pad_left},{pad_right}) "
        f"mode={mode}: {failures}"
    )
