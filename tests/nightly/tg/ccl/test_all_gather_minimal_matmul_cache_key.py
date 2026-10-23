# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Program-cache key of all_gather_minimal_matmul_async across the MiniMax-H3 row counts: one program per
layer pin serves every per-chip length; width, dtype and (unpinned) orientation changes must still miss."""

import pytest
import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.tt_dit.models.transformers.minimax_h3.agmm_config import agmm_pins

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    _create_cluster_submesh,
    create_fabric_router_config,
    create_global_semaphores,
    make_ag_buffers,
    run_test_linear,
)
from tests.ttnn.utils_for_testing import assert_program_cache_flat

# Per-chip tile rows the H3 DiT serves on a 4x8 Galaxy: warm at the 5 s point, then the rest up to capacity.
H3_TILE_ROWS = [148, 74, 287, 427, 470]
CAPACITY_TILE_ROWS = 470

DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": create_fabric_router_config(4096),
    "trace_region_size": 90112,
}

# The three AGMM layers of MiniMax-H3 with TP=4 (N is per device); grid, blocking, workers and orientation
# come from the model's pin table, one pin per length class.
H3_LAYERS = {
    "to_qkv": dict(K=5376, N=5376),
    "to_out": dict(K=7168, N=1344, fuse_addcmul=True),
    "ff1": dict(K=5376, N=7168, fuse_swiglu=True),
}
# (layer, length class, persistent gather buffer); ff1 is the one layer whose pin differs between the classes.
H3_CASES = [
    pytest.param("to_qkv", "short", True, id="to_qkv"),
    pytest.param("to_out", "short", True, id="to_out_addcmul"),
    pytest.param("ff1", "short", True, id="ff1_swiglu_short"),
    pytest.param("ff1", "long", True, id="ff1_swiglu_long"),
    pytest.param("to_qkv", "short", False, id="to_qkv_barrier"),
]


def _layer_pin(mesh, layer, length_class="short"):
    cfg = H3_LAYERS[layer]
    return agmm_pins(cfg["K"], cfg["N"], full_grid=mesh.compute_with_storage_grid_size(), num_links=2)[length_class]


def _layer_cores(grid):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})


def _run_layer(mesh, layer, tile_rows, pin, pingpong=None, **overrides):
    """One AGMM call of `layer` at `tile_rows` per chip with the pin's program (overrides on top); `pingpong` =
    (semaphores, barrier, buffer) set. Host-tilized tensors keep the op the only device op in the window."""
    config = pin.config
    grid = config.compute_with_storage_grid_size
    cfg = {
        **H3_LAYERS[layer],
        "blocks": (config.M_block_size, config.K_block_size, config.N_block_size),
        "grid": (grid.x, grid.y),
        "workers": pin.num_workers_per_link,
        "transpose": pin.transpose_core_grid,
        **overrides,
    }
    m_block, k_block, n_block = cfg["blocks"]
    sems, barrier, buffer = pingpong if pingpong is not None else (None, None, None)
    results = run_test_linear(
        mesh,
        M=tile_rows * 32 * mesh.shape[1],
        K=cfg["K"],
        N=cfg["N"],
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=2,
        subblock_w=2,
        topology=ttnn.Topology.Ring,
        core_grid=ttnn.CoreCoord(*cfg["grid"]),
        num_workers_per_link=cfg["workers"],
        num_links=2,
        use_bias=True,
        dtype=cfg.get("dtype", ttnn.bfloat16),
        weight_dtype=cfg.get("weight_dtype"),
        fuse_addcmul=cfg.get("fuse_addcmul", False),
        addcmul_scalar=1.0,
        broadcast_gate=False,
        fuse_swiglu=cfg.get("fuse_swiglu", False),
        sp_axis=1,
        tp_axis=0,
        cluster_axis=0,
        num_buffers_per_channel=24,
        use_persistent_buffers=buffer is not None,
        force_transpose=False,
        transpose_core_grid=cfg["transpose"],
        ccl_semaphore_handles=[sems] if sems is not None else None,
        barrier_semaphore_handles=[barrier] if barrier is not None else None,
        persistent_output_buffers=[buffer] if buffer is not None else None,
        host_tilize=True,
    )
    for chunk in results[0]:
        for device_result in chunk:
            assert device_result["pcc"] > 0.999_500, device_result
            assert device_result["relative_rmse"] < 0.02, device_result


def _pingpong_sets(mesh, layer, persistent, pin):
    """Two A/B sets of (gather semaphores, barrier semaphore, capacity gather buffer), as the model alternates."""
    cfg = H3_LAYERS[layer]
    grid = pin.config.compute_with_storage_grid_size
    cores = _layer_cores((grid.x, grid.y))
    sems = [create_global_semaphores(mesh, mesh.get_num_devices(), cores, 0) for _ in range(2)]
    barriers = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
    # NaN beyond the live rows: a read past M at any length poisons the output.
    buffers = (
        make_ag_buffers(mesh, CAPACITY_TILE_ROWS * 32, cfg["K"], ttnn.bfloat16, 2, poison=True, host_tilize=True)
        if persistent
        else [None, None]
    )
    return [(sems[i], barriers[i], buffers[i]) for i in range(2)]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["fabric_ring"])
@pytest.mark.parametrize("layer, length_class, persistent", H3_CASES)
def test_h3_lengths_share_one_program(mesh_device, layer, length_class, persistent):
    """Warm at 148 tile rows, then every other served length hits the same entry, alternating A/B sets."""
    if is_wormhole_b0():
        pytest.skip("blackhole-only core grids")
    mesh = _create_cluster_submesh(mesh_device, 0)
    pin = _layer_pin(mesh, layer, length_class)
    sets = _pingpong_sets(mesh, layer, persistent, pin)

    def run(variant):
        index, tile_rows = variant
        _run_layer(mesh, layer, tile_rows, pin, pingpong=sets[index % 2])

    entries = assert_program_cache_flat(mesh, run, list(enumerate(H3_TILE_ROWS)))
    assert entries >= 1


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["fabric_ring"])
def test_width_and_dtype_still_miss(mesh_device):
    """A new length hits; a different weight dtype or output width must add an entry."""
    if is_wormhole_b0():
        pytest.skip("blackhole-only core grids")
    mesh = _create_cluster_submesh(mesh_device, 0)
    variants = [
        (148, {}),
        (74, {}),
        (148, dict(weight_dtype=ttnn.bfloat8_b)),
        (148, dict(N=5376 + 64)),
    ]

    pin = _layer_pin(mesh, "to_qkv")

    def run(variant):
        tile_rows, overrides = variant
        _run_layer(mesh, "to_qkv", tile_rows, pin, **overrides)

    assert_program_cache_flat(mesh, run, variants, miss_indices=(2, 3))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["fabric_ring"])
@pytest.mark.parametrize("transpose", [None, False, True], ids=["rule", "pin_false", "pin_true"])
def test_orientation_rule_vs_pin(mesh_device, transpose):
    """With the legacy `M > N` rule the entry forks once M crosses N (4736 -> 9184 rows at N=5376); a pinned
    orientation keeps one entry. The 10x9 grid is valid for both orientations with 5 workers per link."""
    if is_wormhole_b0():
        pytest.skip("blackhole-only core grids")
    mesh = _create_cluster_submesh(mesh_device, 0)

    pin = _layer_pin(mesh, "to_qkv")

    def run(tile_rows):
        _run_layer(mesh, "to_qkv", tile_rows, pin, grid=(10, 9), workers=5, transpose=transpose)

    assert_program_cache_flat(mesh, run, [148, 287], miss_indices=(1,) if transpose is None else ())


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["fabric_ring"])
def test_full_mesh_lengths_share_one_program(mesh_device):
    """The 32-program workload on the full 4x8 mesh (SP=8 on axis 1, TP ring on axis 0) stays flat too."""
    if is_wormhole_b0():
        pytest.skip("blackhole-only core grids")
    pin = _layer_pin(mesh_device, "to_out")
    sets = _pingpong_sets(mesh_device, "to_out", True, pin)

    def run(variant):
        index, tile_rows = variant
        _run_layer(mesh_device, "to_out", tile_rows, pin, pingpong=sets[index % 2])

    assert_program_cache_flat(mesh_device, run, list(enumerate([148, 74, 287])))
