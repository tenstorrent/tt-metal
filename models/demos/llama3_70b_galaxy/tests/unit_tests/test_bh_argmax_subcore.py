# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Repro for the BH force-argmax index displacement.

The Qwen accuracy run with the prefetcher (unfused-CCL) path shows the on-device force-argmax
returning true_index + 3*19456 whenever the max logit lives in vocab chunk 0. This test drives the
exact untilize + argmax (sub_core_grids-pinned) sequence tt_sampling uses, with a known max planted
at various positions.
"""

import pytest
import torch
from loguru import logger
import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
def test_bh_argmax_subcore(mesh_device):
    torch.manual_seed(1234)
    width = 155648  # padded vocab as gathered in the model
    batch = 32

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 7)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7)),
        ]
    )

    # Plant the max at known positions per row; positions span multiple vocab chunks.
    positions = [198, 257, 279, 11, 19456 + 5, 2 * 19456 + 100, 3 * 19456 + 7, 4 * 19456 + 3000]
    x = torch.rand(1, 1, batch, width) * 10.0 - 5.0
    expected = []
    for r in range(batch):
        pos = positions[r % len(positions)]
        x[0, 0, r, :] = torch.clamp(x[0, 0, r, :], max=20.0)
        x[0, 0, r, pos] = 50.0 + r  # unambiguous max
        expected.append(pos)

    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for grids, label in ((None, "default"), (sub_core_grids, "sub_core_grids")):
        x_unt = ttnn.untilize(tt_x, use_multicore=True, sub_core_grids=grids)
        tok = ttnn.argmax(x_unt, dim=-1, keepdim=False, sub_core_grids=grids)
        got = ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1).tolist()
        errs = [(r, expected[r], got[r]) for r in range(batch) if got[r] != expected[r]]
        logger.info(f"[{label}] mismatches: {len(errs)}")
        for r, exp, g in errs[:10]:
            logger.warning(f"[{label}] row {r}: expected {exp} got {g} (delta {g - exp}, chunks {(g - exp) / 19456})")
        assert not errs, f"argmax misattributed indices with {label}: {errs[:5]}"
