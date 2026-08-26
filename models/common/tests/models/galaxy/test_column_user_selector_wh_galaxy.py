# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware qualification for the Galaxy column user selector.

Decode logits leave ``LMHead2D`` with all 32 users present on every column,
while ``Sampling2D`` consumes one column's eight users. ``GalaxyColumnUserSelector``
bridges the two with a one-hot matmul whose selector rows differ per column.
That composition is the only unqualified step in the Milestone B device sampling
path, so it is worth qualifying on its own — a failure here is a placement
problem, whereas the same failure inside a 70B demo is a needle in a haystack.

Two tests, deliberately ordered:

1. the selector alone, on a tensor whose values name their user; and
2. the selector feeding ``Sampling2D``, which is exactly what
   ``<model>.sample_decode`` does.

**This file has never been executed.**

Run::

    pytest models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py -v
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.models.galaxy.collectives import GalaxyColumnUserSelector
from models.common.models.galaxy.recipes import sampling_core_grids
from models.common.modules.sampling.sampling_2d import Sampling2D
from models.common.tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_USERS_PER_COLUMN,
    deallocate,
)


def _stage_column_replicated(source: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Shard the width over mesh rows and replicate the users over columns.

    This is the placement ``LMHead2D`` decode output has after its column
    all-reduce: the vocabulary is row-sharded, the physical batch is everywhere.
    """

    return ttnn.from_torch(
        source,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=GALAXY_MESH_SHAPE),
    )


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_column_user_selector_gives_each_column_its_own_users(mesh_device: ttnn.MeshDevice):
    """Column ``c`` must receive exactly users ``8c .. 8c + 7``, in order."""

    width = 256
    source = torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.bfloat16).reshape(1, 1, -1, 1).repeat(1, 1, 1, width)
    selector = GalaxyColumnUserSelector(mesh_device)
    staged = selected = None
    try:
        staged = _stage_column_replicated(source, mesh_device)
        for _ in range(2):  # repeat invocation must reuse the cached selector
            selected = selector(staged)
            try:
                composed = to_torch_auto_compose(selected).float()
                message = f"expected the four column slices to compose back to 32 users, got {tuple(composed.shape)}"
                assert tuple(composed.shape[-2:]) == (GALAXY_PHYSICAL_BATCH, width), message
                users = composed.reshape(-1, width)[:GALAXY_PHYSICAL_BATCH, 0]
                message = f"column user order is wrong: {users.tolist()}"
                assert torch.equal(users, torch.arange(GALAXY_PHYSICAL_BATCH, dtype=users.dtype)), message
            finally:
                deallocate(selected)
                selected = None
    finally:
        deallocate(staged)
        selector.release()


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_column_user_selector_feeds_sampling_2d(mesh_device: ttnn.MeshDevice):
    """Selector plus ``Sampling2D`` reproduces a per-user argmax.

    The vocabulary is Llama-3.3-70B's, whose padded width equals its logical
    width, so the assertion is only about user placement.
    """

    vocab_size = padded_vocab_size = 128256
    logits = torch.full((1, 1, GALAXY_PHYSICAL_BATCH, padded_vocab_size), -20.0, dtype=torch.bfloat16)
    expected = torch.arange(GALAXY_PHYSICAL_BATCH, dtype=torch.int64) * 1013
    logits[0, 0, torch.arange(GALAXY_PHYSICAL_BATCH), expected] = 10.0

    sub_core_grids, topk_grid, start_core = sampling_core_grids()
    sampler = Sampling2D(
        vocab_size,
        padded_vocab_size,
        mesh_device,
        sub_core_grids=sub_core_grids,
        sub_core_grid_topk=topk_grid,
        start_core=start_core,
    )
    selector = GalaxyColumnUserSelector(mesh_device)
    staged = selected = output = None
    try:
        staged = _stage_column_replicated(logits, mesh_device)
        selected = selector(staged)
        assert tuple(selected.shape)[-2] == GALAXY_USERS_PER_COLUMN
        output = sampler.decode_forward(selected, top_k=32, top_p=1.0, temperature=0.0, forced_argmax=True)
        actual = to_torch_auto_compose(output).reshape(-1)[:GALAXY_PHYSICAL_BATCH].to(torch.int64)
        assert torch.equal(actual, expected), f"sampled {actual.tolist()}, expected {expected.tolist()}"
    finally:
        deallocate(output)
        deallocate(selected)
        deallocate(staged)
        selector.release()
        sampler.release()
