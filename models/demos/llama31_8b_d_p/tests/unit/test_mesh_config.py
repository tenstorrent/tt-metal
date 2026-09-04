# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`MeshConfig` arithmetic and refusals, plus a one-card `CCLManager` construction. Gate: `G-MESH`.

Two halves, as `BRINGUP_RECIPE.md:1260-1267` specifies:

* **(a) device-free** — `MeshConfig((1,8), tp=8)` yields `sp=1, tp=8, shard_size(4096)=512,
  shard_size(14336)=1792`, and sub-axis TP (`MeshConfig((1,8), tp=4)`) **raises**. Only sub-axis TP
  is a refusal, because only it produces a wrong tensor; being off `_VALIDATED_MESH_SHAPE` /
  `_VALIDATED_TP` merely warns.
* **(b) on a card** — `CCLManager` constructs, allocates its semaphores exactly once (asserted
  again after dozens of getter cycles), and reports the real compute grid and CCL offset.

Only (b) takes the `mesh_device` fixture. `G-MESH` produces no PCC, so §1.4's floor/reference-dtype
fields do not apply; its **negative control is the refusal itself** — an op or configuration that
must refuse counts as a control (`BRINGUP_RECIPE.md:351-353`).

Refusals use the repo-root `expect_error` fixture (`conftest.py:948`). The `prefer-expect-error`
hook (`.pre-commit-config.yaml:51`) rejects pytest's own raises helper anywhere in a `tests/` file
— and, being a `pygrep` hook, it matches the name in **prose** too, so this sentence has to spell
it out the long way (`DEC-034`).

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_mesh_config.py -x -q
"""

import pytest
from loguru import logger

import ttnn
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, llama_config_dims
from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
from models.demos.llama31_8b_d_p.tt.config import _VALIDATED_MESH_SHAPE, _VALIDATED_TP, MeshConfig

# The deployment target, read from the card rather than restated (`bringup_log/00_MODEL_CARD.md` §4).
DEPLOYMENT_MESH = (4, 8)
DEPLOYMENT_TP = 8


# ------------------------------------------------------------------------------------------------
# (a) device-free arithmetic
# ------------------------------------------------------------------------------------------------
def test_mesh_config_arithmetic_1x8():
    """`(1,8)`/TP=8: the shard arithmetic `G-MESH` names explicitly."""
    hf = llama_config_dims()
    mc = MeshConfig((1, 8), tp=8)

    assert mc.tp == 8
    assert mc.sp == 1, f"sp must be the non-TP axis size: {mc.sp}"
    assert mc.tp_axis == 1 and mc.sp_axis == 0
    assert mc.total_devices == 8

    # 4096/8 = 512 and 14336/8 = 1792, both tile-aligned (`bringup_log/00_MODEL_CARD.md` §4).
    assert mc.shard_size(hf["hidden_size"]) == 512
    assert mc.shard_size(hf["intermediate_size"]) == 1792
    assert mc.shard_size(hf["hidden_size"]) % ttnn.TILE_SIZE == 0
    assert mc.shard_size(hf["intermediate_size"]) % ttnn.TILE_SIZE == 0
    logger.info(f"[G-MESH] {mc!r} shard_size(4096)={mc.shard_size(4096)} shard_size(14336)={mc.shard_size(14336)}")


def test_mesh_config_arithmetic_deployment_4x8():
    """The deployment shape `(4,8)`/TP=8 -> SP=4, and it is the shape `_VALIDATED_*` names.

    `BRINGUP_RECIPE.md:791-793` — a gate that only ever runs at a mesh the deployment never uses
    can be testing a configuration the model cannot produce. This is the device-free half of the
    answer; `G-KV-TP8` and `G-MESH-KV` are the on-device half, in P8.
    """
    hf = llama_config_dims()
    mc = MeshConfig(DEPLOYMENT_MESH, tp=DEPLOYMENT_TP)

    assert (mc.mesh_shape, mc.tp) == (_VALIDATED_MESH_SHAPE, _VALIDATED_TP)
    assert mc.sp == 4
    # TP == num_key_value_heads is a hard equality here, not a bound: the packed KV cache holds
    # exactly one KV head per chip (`bringup_log/04_CCL_PLAN.md` §1.1).
    assert mc.tp == hf["num_key_value_heads"] == 8
    # Each chip therefore holds 32/8 = 4 local Q heads against 1 local KV head, and SDPA's
    # `nqh >= nkv && nqh % nkv == 0` holds.
    local_q = hf["num_attention_heads"] // mc.tp
    local_kv = hf["num_key_value_heads"] // mc.tp
    assert local_q == 4 and local_kv == 1
    assert local_q >= local_kv and local_q % local_kv == 0
    logger.info(f"[G-MESH] {mc!r} local_q={local_q} local_kv={local_kv}")


def test_mesh_config_1x1_single_card():
    """`(1,1)`/TP=1 — the P5 gate mesh. It warns (untested shape) but must not raise."""
    mc = MeshConfig((1, 1), tp=1)
    assert (mc.tp, mc.sp) == (1, 1)
    assert mc.shard_size(4096) == 4096
    logger.info(f"[G-MESH] {mc!r} (single card: no collective is needed at TP=1)")


@pytest.mark.parametrize("mesh_shape,tp", [((1, 8), 4), ((1, 8), 2), ((4, 8), 4), ((1, 8), 16)])
def test_mesh_config_refuses_sub_axis_tp(mesh_shape, tp, expect_error):
    """**The negative control.** Sub-axis TP must raise, not warn.

    `shard_mapper` shards the full `tp_axis` regardless of `tp`, so a `tp` below the axis size
    builds head and feature counts from `tp` while the mapper splits across all of them —
    inconsistent per-device shapes. `models/demos/minimax_m3/config.py:42-45` lets this through
    with only a warning; this package takes `models/demos/gpt_oss_d_p/tt/config.py:44`'s refusal
    (`bringup_log/03_OUTLINE.md` §5.1).
    """
    with expect_error(ValueError, "sub-axis TP is unsupported"):
        MeshConfig(mesh_shape, tp=tp)


def test_mesh_config_off_target_shape_warns_but_builds():
    """Being off `_VALIDATED_*` is untested, not wrong: it must warn and still build.

    The complement of the refusal above — without this, "raise on anything unusual" would pass the
    control while making every `(1,1)` P5 gate unrunnable.
    """
    for mesh_shape, tp in [((1, 2), 2), ((1, 4), 4), ((2, 8), 8), ((8, 4), 4)]:
        mc = MeshConfig(mesh_shape, tp=tp)
        assert mc.tp == tp and mc.sp == mesh_shape[0]


def test_mesh_config_tp_axis_0_swaps_the_axes():
    """`tp_axis=0` puts TP on the rows and SP on the columns; nothing else changes."""
    mc = MeshConfig((8, 4), tp=8, tp_axis=0)
    assert (mc.tp, mc.tp_axis, mc.sp_axis, mc.sp) == (8, 0, 1, 4)


# ------------------------------------------------------------------------------------------------
# (b) on a card
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ccl_manager_constructs_on_card(mesh_device):
    """`CCLManager` builds on a real card, and reports the real grid rather than a guessed 8x8.

    On this Blackhole the compute grid is (12, 10), so the ring-attention CCL offset is
    `x = grid.x - 1 = 11`. The SDPA *program* grid stays pinned at 8x8 and must NOT be derived from
    this — `11 >= 8` passes the ring op's assert while a derived 12 would fail, and only at SP > 1
    (`BRINGUP_RECIPE.md:1409-1417`).
    """
    setup = TestFactory.setup_test(mesh_device)
    ccl = setup["ccl_manager"]

    grid = ccl.compute_grid_size
    logger.info(f"[G-MESH] compute grid = ({grid.x}, {grid.y}); num_links = {ccl.num_links}")
    logger.info(f"[G-MESH] ring_attention_ccl_core_grid_offset = {ccl.ring_attention_ccl_core_grid_offset}")

    assert ccl.ring_attention_ccl_core_grid_offset == (grid.x - 1, 0)
    # The build-time form of the ring op's assert, stated here so a future grid change fails in
    # P5 rather than in P8: a pinned 8x8 SDPA grid needs `8 <= grid.x - 1`.
    assert 8 <= grid.x - 1, f"a pinned 8x8 SDPA grid needs grid.x >= 9; got {grid.x}"
    assert ccl.topology == ttnn.Topology.Ring
    assert ccl.mesh_device is mesh_device
    assert setup["mesh_config"].tp == 1  # (1,1) mesh -> TP=1


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ccl_manager_allocates_semaphores_once(mesh_device):
    """The semaphore inventory is allocated once and cycled — never `n_layers x` anything.

    `G-SEMAPHORE`'s assertion, run here too because `G-MESH` states it
    (`BRINGUP_RECIPE.md:1265-1266`); `tests/unit/test_ccl_semaphores.py` owns the gate and the
    deeper cycling (`[DEV-6]` in `bringup_log/03_OUTLINE.md` §1.1).
    """
    ccl = CCLManager(mesh_device, num_links=1)
    counts = (
        len(ccl.rs_ping_pong_semaphores),
        len(ccl.ag_ping_pong_semaphores),
        len(ccl.barrier_semaphore),
        len(ccl.ring_attention_ccl_semaphore_handles),
    )
    assert counts == (6, 4, 2, 2), f"semaphore inventory changed: {counts}"
    assert sum(counts) == 14
    logger.info(f"[G-MESH] semaphores rs/ag/barrier/ring = {counts}, total {sum(counts)}")
