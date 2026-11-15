# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the prefill CCL launch configs of tt/tp_common.py.

* The prefill all-gather matmuls (all_gather_matmul_prefill, all_gather_swiglu_prefill) take the tuned config of
  _AGMM_TUNED for the Qwen3.8-27B TP=8 shapes and row counts it lists, when the device grid holds it. Every tuned
  config keeps the default K blocking (agmm_k_block_size of K per device), the condition under which each output tile
  accumulates the same K blocks in the same order as with the default config.
* A 1xN prefill reduce-scatter (tp_reduce_scatter) runs 8 workers per link; a decode width keeps tt_all_reduce's
  workers, and the link count stays the TP axis's.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

# Per-device weight shapes of Qwen3.8-27B at TP=8: (K, N per device) and swiglu.
GATE_UP = ((5120, 4352), True)
GDN_IN = ((5120, 2060), False)
ATTN_QKV = ((5120, 2048), False)
GDN_OUT = ((6144, 640), False)


@pytest.fixture
def launches(monkeypatch):
    """Stand-ins for the AGMM config types and a recorder of all_gather_minimal_matmul_async launches."""
    calls = []
    monkeypatch.setattr(ttnn, "MinimalMatmulConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y))
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape, **kw: t)
    monkeypatch.setattr(
        ttnn.experimental,
        "all_gather_minimal_matmul_async",
        lambda **kw: calls.append(kw) or ["out"],
        raising=False,
    )
    return calls


def _ccl(grid_x, grid_y=10):
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid_x, y=grid_y))
    return SimpleNamespace(mesh_device=mesh, get_and_cycle_ag_semaphore_handles=lambda axis: "semaphores")


def _weight(shape):
    return SimpleNamespace(shape=shape)


def test_every_tuned_config_keeps_the_default_k_blocking(launches):
    """Each tuned entry keeps K_block = agmm_k_block_size(K per device), 2 links, subblock (1, 4), and workers that
    cover its sender axis; on a 12-wide grid every entry applies."""
    for (swiglu, k, n), rows_cfg in tpc._AGMM_TUNED.items():
        for rows, (gx, gy, mb, nb, transpose) in rows_cfg.items():
            got = tpc.agmm_tuned_config(rows, _weight((k, n)), k // 8, swiglu, _ccl(12).mesh_device)
            assert got is not None, (swiglu, k, n, rows)
            cfg, links, workers, tr = got
            assert cfg.K_block_size == tpc.agmm_k_block_size(k // 8)
            assert (cfg.M_block_size, cfg.N_block_size, cfg.subblock_h, cfg.subblock_w) == (mb, nb, 1, 4)
            assert cfg.compute_with_storage_grid_size == (gx, gy) and tr == transpose and links == 2
            assert links * workers >= (gx if transpose else gy)


@pytest.mark.parametrize(
    "weight, rows, grid_x, want",
    [
        # Qwen3.8-27B TP=8 on an 11-wide grid: only the 11-wide entries apply.
        (GATE_UP, 2048, 11, (11, 9, 8, 16, True, 6)),
        (GDN_OUT, 2048, 11, (11, 9, 4, 8, True, 6)),
        (GATE_UP, 1024, 11, None),
        (GDN_OUT, 1024, 11, None),
        # On a 12-wide grid the short buckets stream the weight through a 12-wide row of readers (not transposed).
        (GATE_UP, 1024, 12, (12, 9, 8, 16, False, 5)),
        (GDN_IN, 128, 12, (12, 9, 4, 8, False, 5)),
        (ATTN_QKV, 128, 12, (12, 9, 4, 8, False, 5)),
        # No entry: the default config.
        (GDN_IN, 2048, 12, None),
        (GATE_UP, 256, 12, None),
        (((5120, 4096), True), 2048, 12, None),
        # A decode width never takes a tuned config.
        (GATE_UP, 32, 12, None),
    ],
)
def test_the_tuned_config_of_a_shape_and_row_count(launches, weight, rows, grid_x, want):
    (k, n), swiglu = weight
    got = tpc.agmm_tuned_config(rows, _weight((k, n)), k // 8, swiglu, _ccl(grid_x).mesh_device)
    if want is None:
        assert got is None
        return
    cfg, links, workers, transpose = got
    assert (*cfg.compute_with_storage_grid_size, cfg.M_block_size, cfg.N_block_size, transpose, workers) == want


def test_a_non_transposed_entry_is_not_used_for_more_rows_than_n(launches, monkeypatch):
    """force_transpose=False only means "do not force": the op transposes a launch with M > N itself, so such an entry
    keeps the default config."""
    monkeypatch.setitem(tpc._AGMM_TUNED, (False, 5120, 640), {2048: (12, 9, 4, 8, False)})
    assert tpc.agmm_tuned_config(2048, _weight((5120, 640)), 640, False, _ccl(12).mesh_device) is None


@pytest.mark.parametrize("swiglu", [False, True])
def test_prefill_agmm_launches_the_tuned_config(launches, swiglu):
    """all_gather_matmul_prefill and all_gather_swiglu_prefill launch the tuned grid, blocks, workers and transpose;
    without an entry they launch the default (8-wide, 2 links, 4 workers, transposed); a config_override wins."""
    fn = tpc.all_gather_swiglu_prefill if swiglu else tpc.all_gather_matmul_prefill
    shape = GATE_UP[0] if swiglu else GDN_OUT[0]
    x = SimpleNamespace(shape=(1, 1, 2048, shape[0] // 8))
    fn(x, _weight(shape), _ccl(11), "compute", "ring")
    kw = launches[-1]
    assert kw["config"].compute_with_storage_grid_size == (11, 9) and kw["force_transpose"] is True
    assert (kw["num_links"], kw["num_workers_per_link"]) == (2, 6)
    assert kw["config"].K_block_size == tpc.agmm_k_block_size(shape[0] // 8)
    x = SimpleNamespace(shape=(1, 1, 512, shape[0] // 8))
    fn(x, _weight(shape), _ccl(11), "compute", "ring")
    kw = launches[-1]
    assert kw["config"].compute_with_storage_grid_size == (8, 9) and kw["force_transpose"] is True
    assert (kw["num_links"], kw["num_workers_per_link"]) == (2, 4)
    x = SimpleNamespace(shape=(1, 1, 2048, shape[0] // 8))
    fn(x, _weight(shape), _ccl(11), "compute", "ring", config_override={"grid_x": 10})
    kw = launches[-1]
    assert kw["config"].compute_with_storage_grid_size == (10, 9) and kw["force_transpose"] is True


# --------------------------------------------------------------------------------------- prefill reduce-scatter


@pytest.fixture
def reduce_scatters(monkeypatch):
    calls = []
    monkeypatch.setattr(tpc, "tt_all_reduce", lambda partial, mesh, tt_ccl, **kw: calls.append(kw) or "reduced")
    return calls


def _rs(partial_rows, hparams=None):
    mesh = SimpleNamespace(shape=(1, 8))
    tt_ccl = SimpleNamespace(get_num_links=lambda axis: {0: 1, 1: 2}[axis])
    args = SimpleNamespace(ccl_topology=lambda: "ring", prefill_rs_hparams=hparams)
    return tpc.tp_reduce_scatter(SimpleNamespace(shape=(1, 1, partial_rows, 5120)), mesh, tt_ccl, args)


def test_a_prefill_reduce_scatter_runs_8_workers_per_link(reduce_scatters):
    """A 1xN reduce-scatter of a prefill partial (MLP down, attention wo) runs 8 workers per link over the TP axis's
    links; a partial of one tile row keeps tt_all_reduce's workers; args.prefill_rs_hparams still wins."""
    assert _rs(2048) == "reduced"
    kw = reduce_scatters[-1]
    assert kw["num_workers_per_link"] == tpc.PREFILL_RS_WORKERS_PER_LINK == 8
    assert kw["num_reduce_scatter_links"] == 2 and kw["cluster_axis"] == 0
    _rs(128)
    assert reduce_scatters[-1]["num_workers_per_link"] == 8
    _rs(32)
    assert "num_workers_per_link" not in reduce_scatters[-1]
    _rs(2048, hparams=(4, 3, 2))
    kw = reduce_scatters[-1]
    assert (kw["chunks_per_sync"], kw["num_workers_per_link"], kw["num_buffers_per_channel"]) == (4, 3, 2)
