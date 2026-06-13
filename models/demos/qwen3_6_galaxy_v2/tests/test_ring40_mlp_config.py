# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shape/consistency test for the ring-40 decode-MLP config set (Qwen3.6 BH_GLX).

Task 2 (TDD RED part 1). A pure-CPU ModelArgs build is INFEASIBLE here: the ring
program/memory configs are only populated inside ``_populate_program_configs``,
which is gated on ``mesh_device is not None`` and pulls
``compute_with_storage_grid_size`` / ``dram_grid_size`` from a live mesh. So this
test opens the 8x4 mesh and builds ``TtQwen36ModelArgs(dummy_weights=True)`` (no
weights, no model, no forward — fast + device-safe) purely to assert the ring-40
config SHAPES.

It must FAIL first with a KeyError (the ring-40 keys don't exist yet), then pass
once the configs are added.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_ring40_mlp_config.py -s -x
"""
from __future__ import annotations

import pytest

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

# Ring-40 verified math (see Task 2 background):
#   dim_per_tp = 1280 (=40 tiles), intermediate ring N = 2560 (vs ring-24's 3840).
#   FF1/FF3: M=32 K=1280 N=2560, num_blocks=40 -> grid (8,5), per_core_N=2.
#   FF2:     M=32 K=2560 N=1280, num_blocks=40 -> grid (8,5), per_core_N=1.
_DIM_PER_TP = 1280
_FF_RING40_N = 2560
_FF2_RING40_N = 1280


@pytest.fixture(scope="module")
def bh_glx_mesh():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)


def test_ring40_mlp_config_shapes(bh_glx_mesh):
    args = TtQwen36ModelArgs(bh_glx_mesh, dummy_weights=True)
    mc = args.model_config

    # ---- program configs ----
    ff13 = mc["FF1_3_RING40_PROGCFG"]
    ff2 = mc["FF2_RING40_PROGCFG"]

    # grid must be (x=8, y=5) = 40 cores for both stages
    assert (ff13.compute_with_storage_grid_size.x, ff13.compute_with_storage_grid_size.y) == (
        8,
        5,
    ), ff13.compute_with_storage_grid_size
    assert (ff2.compute_with_storage_grid_size.x, ff2.compute_with_storage_grid_size.y) == (
        8,
        5,
    ), ff2.compute_with_storage_grid_size

    # per_core_N: FF1/FF3 -> 2560/40/32 = 2 ; FF2 -> 1280/40/32 = 1
    assert ff13.per_core_N == 2, ff13.per_core_N
    assert ff2.per_core_N == 1, ff2.per_core_N
    # per_core_M = M/32 = 1 for both (M=32)
    assert ff13.per_core_M == 1, ff13.per_core_M
    assert ff2.per_core_M == 1, ff2.per_core_M
    # in0_block_w: FF1/FF3 K=1280 / 40 / 32 = 1 ; FF2 K=2560 / 40 / 32 = 2
    assert ff13.in0_block_w == 1, ff13.in0_block_w
    assert ff2.in0_block_w == 2, ff2.in0_block_w

    # ---- DRAM-sharded weight memcfgs: assert k/n shard spec ----
    w1w3 = mc["W1W3_RING40_MEMCFG"]
    w2 = mc["W2_RING40_MEMCFG"]
    dram_cores = args.dram_weight_grid.bounding_box().grid_size().x

    # W1W3: k = dim_per_tp = 1280; n = 2560 (padded to tile*dram_cores multiple)
    w1w3_shard = w1w3.shard_spec.shape
    assert w1w3_shard[0] == _DIM_PER_TP, w1w3_shard
    assert w1w3_shard[1] * dram_cores >= _FF_RING40_N, (w1w3_shard, dram_cores)
    assert w1w3_shard[1] * dram_cores % _FF_RING40_N == 0 or w1w3_shard[1] * dram_cores >= _FF_RING40_N

    # W2: k = NATIVE intermediate_dim_per_tp = 2176 (only N is padded, matching
    # the ring-24 W2_RING_MEMCFG native-K convention); n = dim_per_tp = 1280
    w2_shard = w2.shard_spec.shape
    assert w2_shard[0] == args.intermediate_dim_per_tp == 2176, (w2_shard, args.intermediate_dim_per_tp)
    assert w2_shard[1] * dram_cores >= _FF2_RING40_N, (w2_shard, dram_cores)

    # ---- sharded L1 memcfgs (40-core ring) ----
    ff12_in = mc["SHARDED_FF12_RING40_MEMCFG"]
    ff12_out = mc["SHARDED_FF12_OUT_RING40_MEMCFG"]
    ff2_in = mc["FF2_IN_RING40_MEMCFG"]
    rs_out = mc["REDUCE_SCATTER_OUT_RING40_MEMCFG"]

    # FF12 input shard: K=1280 over 40 ring cores -> width 32
    assert ff12_in.shard_spec.shape == [32, _DIM_PER_TP // 40], ff12_in.shard_spec.shape
    # FF12 out shard: N=2560 over 40 cores -> width 64
    assert ff12_out.shard_spec.shape == [32, _FF_RING40_N // 40], ff12_out.shard_spec.shape
    # FF2 input shard: 2560 over 40 cores -> width 64
    assert ff2_in.shard_spec.shape == [32, _FF_RING40_N // 40], ff2_in.shard_spec.shape
    # reduce-scatter out: FF2 N=1280 over 40 cores -> width 32
    assert rs_out.shard_spec.shape == [32, _FF2_RING40_N // 40], rs_out.shard_spec.shape

    # FF12 reduce-scatter out (NEW): w1/w3 partial sum N=2560 reduce-scattered
    # across cluster_axis=1 (4 cols) -> 640 wide, on a 20-core band, shard [32,32]
    # (20 x 32 = 640, tile-aligned).
    ff12_rs_out = mc["REDUCE_SCATTER_OUT_RING40_FF12_MEMCFG"]
    assert ff12_rs_out.shard_spec.shape == [32, 32], ff12_rs_out.shard_spec.shape
    assert ff12_rs_out.shard_spec.num_cores() == 20, ff12_rs_out.shard_spec.num_cores()
    assert ff12_rs_out.shard_spec.shape[1] * 20 == _FF_RING40_N // 4 == 640, ff12_rs_out.shard_spec.shape

    print(
        f"[ring40_cfg] OK grids ff13=({ff13.compute_with_storage_grid_size.x},{ff13.compute_with_storage_grid_size.y}) "
        f"ff2=({ff2.compute_with_storage_grid_size.x},{ff2.compute_with_storage_grid_size.y}) "
        f"per_core_N ff13={ff13.per_core_N} ff2={ff2.per_core_N} "
        f"W1W3 shard={w1w3_shard} W2 shard={w2_shard}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
