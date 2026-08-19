# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for Sampling2D on a Wormhole Galaxy."""

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.sampling.sampling_2d import Sampling2D


def _deallocate(tensor):
    if tensor is not None:
        tensor.deallocate(True)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
def test_sampling_2d_wh_galaxy_exact_padded_vocab_exclusion(mesh_device):
    vocab_size = 151936
    padded_vocab_size = 152064
    batch = 32
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    sub_core_grid_topk = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9))])
    sampler = Sampling2D(
        vocab_size,
        padded_vocab_size,
        mesh_device,
        sub_core_grids=sub_core_grids,
        sub_core_grid_topk=sub_core_grid_topk,
        start_core=ttnn.CoreCoord(1, 0),
    )
    logits = torch.full((1, 1, batch, padded_vocab_size), -20.0, dtype=torch.bfloat16)
    expected = torch.arange(batch, dtype=torch.int64) * 97
    logits[0, 0, torch.arange(batch), expected] = 10.0
    logits[..., vocab_size:] = 1000.0
    tt_logits = ttnn.from_torch(
        logits,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=(8, 4)),
    )

    try:
        for _ in range(2):
            output = sampler.decode_forward(
                tt_logits,
                top_k=32,
                top_p=1.0,
                temperature=0.0,
                seed=7,
                forced_argmax=True,
            )
            try:
                actual = to_torch_auto_compose(output).reshape(-1)[:batch].to(torch.int64)
                assert torch.equal(actual, expected)
                assert torch.all(actual < vocab_size)
            finally:
                _deallocate(output)
    finally:
        _deallocate(tt_logits)
        sampler.release()
