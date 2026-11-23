# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein on DP4 groups ((4, 2) submeshes of a Ring Galaxy, SP4 over axis 0, TP2 over axis 1), host only."""

from types import SimpleNamespace

import pytest

import ttnn

from ....layers.module import Module
from ....models.transformers.flux3_klein import transformer_flux3_klein
from ....models.transformers.flux3_klein.transformer_block_flux3_klein import (
    LINEAR1_N_CORES,
    Flux3KleinTransformerBlock,
)
from ....models.transformers.flux3_klein.transformer_flux3_klein import DP4_MLP_ROWS, Flux3KleinSequenceLayout
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein.packing import TEXT_BUCKETS, TEXT_BUCKETS_BY_LAYOUT
from ....pipelines.flux3_klein.pipeline_flux3_klein import (
    PAIR_FREEING_FROM_SIZE,
    REFERENCE_TOKEN_BUCKETS,
    SIZES,
    Flux3KleinPipeline,
)
from ....utils import matmul

SP = 4


def _ccl(mesh_shape: tuple[int, int]) -> CCLManager:
    manager = CCLManager.__new__(CCLManager)
    manager.mesh_device = SimpleNamespace(shape=mesh_shape)
    manager.topology = ttnn.Topology.Ring
    manager.axis_topologies = {}
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


@pytest.mark.parametrize(("mesh_shape", "tp_axis", "path"), [((4, 8), 0, "fused"), ((4, 2), 1, "addcmul")])
def test_proj_out_fuses_its_reduce_scatter_only_on_a_ring_tp_axis(monkeypatch, mesh_shape, tp_axis, path) -> None:
    used = []

    class ProjOut:
        def __call__(self, parts):
            return "out"

        def forward_fused_addcmul(self, *args, **kwargs):
            used.append("fused")

    block = SimpleNamespace(
        norm=lambda x, **kwargs: x,
        attn=SimpleNamespace(forward=lambda **kwargs: ("attn", None)),
        proj_mlp=lambda x, **kwargs: "mlp",
        proj_out=ProjOut(),
        _linear1=False,
        _ccl_manager=_ccl(mesh_shape),
        _parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(mesh_axis=tp_axis)),
        _mesh_device=SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(12, 10)),
    )
    monkeypatch.setattr(ttnn, "unsqueeze", lambda x, dim: x)
    monkeypatch.setattr(ttnn, "squeeze", lambda x, dim: x)
    monkeypatch.setattr(ttnn, "addcmul", lambda *args: used.append("addcmul"))
    x = SimpleNamespace(padded_shape=(1, 1024, 768), shape=(1, 1024, 768))
    Flux3KleinTransformerBlock.forward(block, x, rope=None, mod=("shift", "scale", "gate"), logical_length=1024)
    assert used == [path]


@pytest.mark.parametrize(("mesh_shape", "tp_axis", "ring"), [((4, 8), 0, True), ((4, 2), 1, False)])
def test_early_stacks_clone_only_on_a_ring_tp_axis(monkeypatch, mesh_shape, tp_axis, ring) -> None:
    class Block(Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()

        def forward(self, x, **kwargs):
            return x

    monkeypatch.setattr(transformer_flux3_klein, "Flux3KleinTransformerBlock", Block)
    monkeypatch.setattr(ttnn, "clone", lambda x: ("clone", x))
    stacks = transformer_flux3_klein.Flux3KleinEarlyStacks(
        device=None,
        ccl_manager=_ccl(mesh_shape),
        parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(mesh_axis=tp_axis)),
        is_fsdp=False,
        linear1=False,
    )
    assert stacks.forward("image", "x") == (("clone", "x") if ring else "x")


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


def test_dp4_mlp_rows_have_blockings(monkeypatch) -> None:
    grid = dict(matmul.grid_12_9_configs)
    monkeypatch.setattr(matmul, "agmm_shape_configs", {})
    monkeypatch.setattr(matmul, "matmul_shape_configs", {})
    try:
        transformer_flux3_klein.register_layout_shape_configs(4, 2)
        for m in DP4_MLP_ROWS:
            assert (m, 3072, 9216) in matmul.grid_12_9_configs, m
            if m not in (2048, 2304):
                assert (m, 3072, 4608) in matmul.grid_12_9_configs, m
    finally:
        matmul.grid_12_9_configs.clear()
        matmul.grid_12_9_configs.update(grid)
        matmul._clear_config_caches()


def test_claim_pairs_frees_only_pairs_other_largest_size_layouts_used(monkeypatch) -> None:
    monkeypatch.setattr(ttnn, "synchronize_device", lambda device: None)
    small, r0, r1 = ("distilled", 1024, 0, 1024), ("distilled", 4096, 0, 1024), ("distilled", 4096, 4096, 2048)
    ccl, freed = SimpleNamespace(live=set(), used=set()), []

    def take_used():
        used, ccl.used = ccl.used, set()
        return used

    def free(keys):
        freed.extend(keys)
        ccl.live.difference_update(keys)

    ccl.ping_pong_keys = lambda: set(ccl.live)
    ccl.take_used_ping_pong_keys = take_used
    ccl.free_ping_pong_buffers = free
    pipe = Flux3KleinPipeline.__new__(Flux3KleinPipeline)
    pipe._device, pipe._ccl_managers, pipe._pair_users, pipe._last_layout = None, (ccl,), {}, None

    def render(layout, *keys):
        pipe._claim_pairs(layout, free_others=layout[1] == 4096)
        ccl.live.update(keys)
        ccl.used.update(keys)

    render(small, "small ring")
    render(r0, "vae", "r0 ring")  # the smaller size's pair stays
    assert freed == []
    render(r1, "vae", "r1 ring")  # r1 has used nothing yet: every 4096 pair goes, this once
    render(r0, "vae", "r0 ring")
    render(r1, "vae", "r1 ring")  # from here on only the layout's own pairs churn; the shared pair stays
    assert sorted(freed) == sorted(["vae", "r0 ring", "r1 ring", "r0 ring"])
    assert ccl.live == {"small ring", "vae", "r1 ring"}


def test_pairs_are_freed_from_4096_on_dp4_and_from_2048_on_dp8() -> None:
    assert PAIR_FREEING_FROM_SIZE == {(4, 2): max(SIZES), (2, 2): 2048}
