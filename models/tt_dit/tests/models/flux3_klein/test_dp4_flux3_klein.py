# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein on DP4 groups ((4, 2) submeshes of a Ring Galaxy, SP4 over axis 0, TP2 over axis 1), host only."""

from types import SimpleNamespace

import ttnn

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import (
    LINEAR1_N_CORES,
    Flux3KleinTransformerBlock,
)
from ....models.transformers.flux3_klein.transformer_flux3_klein import DP4_MLP_ROWS, Flux3KleinSequenceLayout
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein.packing import TEXT_BUCKETS, TEXT_BUCKETS_BY_LAYOUT
from ....pipelines.flux3_klein.pipeline_flux3_klein import REFERENCE_TOKEN_BUCKETS, SIZES
from ....utils import matmul

SP = 4


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


def test_dp4_mlp_rows_are_the_served_layouts() -> None:
    rows = set()
    for size in SIZES:
        for ref in (0, *REFERENCE_TOKEN_BUCKETS):
            for text in TEXT_BUCKETS_BY_LAYOUT.get((size, ref), TEXT_BUCKETS):
                layout = Flux3KleinSequenceLayout(
                    SP, {"image": (size // 16) ** 2, "image_cond": ref, "text": text}, 1, ref
                )
                rows |= {sum(layout.rows.values()), layout.rows["image"]}
    assert tuple(sorted(m for m in rows if m >= 2048)) == DP4_MLP_ROWS


def test_dp4_mlp_rows_have_blockings() -> None:
    for m in DP4_MLP_ROWS:
        assert (m, 3072, 9216) in matmul.grid_12_9_configs, m
        if m not in (2048, 2304):
            assert (m, 3072, 4608) in matmul.grid_12_9_configs, m
