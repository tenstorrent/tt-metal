# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein on DP4 groups ((4, 2) submeshes of a Ring Galaxy, SP4 over axis 0, TP2 over axis 1), host only."""

from types import SimpleNamespace

import ttnn

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import (
    LINEAR1_N_CORES,
    Flux3KleinTransformerBlock,
)
from ....parallel.manager import CCLManager


def _ccl(mesh_shape: tuple[int, int]) -> CCLManager:
    manager = CCLManager.__new__(CCLManager)
    manager.mesh_device = SimpleNamespace(shape=mesh_shape)
    manager.topology = ttnn.Topology.Ring
    return manager


def _block(mesh_shape: tuple[int, int], tp_axis: int) -> Flux3KleinTransformerBlock:
    return Flux3KleinTransformerBlock(
        device=SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(12, LINEAR1_N_CORES - 1)),
        ccl_manager=_ccl(mesh_shape),
        parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(mesh_axis=tp_axis, factor=mesh_shape[tp_axis])),
        linear1=True,
    )


def test_linear1_rejects_a_two_chip_tp_axis(expect_error) -> None:
    with expect_error(ValueError, "linear1 needs a ring TP axis"):
        _block((4, 2), tp_axis=1)


def test_linear1_takes_a_ring_tp_axis(expect_error) -> None:
    # past the topology check: the stub grid, one row short of LINEAR1_N_CORES, stops it at the core-count check
    with expect_error(ValueError, "rows of cores"):
        _block((4, 8), tp_axis=0)
