# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gates `G-MESH` (semaphore half) and `G-SEMAPHORE` — `CCLManager` allocates its CCL resources
exactly once, and derives its core grid from the real device.

Why this is a gate and not an implementation detail: a `CCLManager` built per layer, or a semaphore
allocated per collective, is the single most common source of *nondeterministic* multi-device PCC
failures — it passes on one card, passes intermittently on a mesh, and looks like a model bug
(`bringup_log/04_CCL_PLAN.md` §6). The invariant is stated as four constants —
**6 / 4 / 2 / 2** (reduce-scatter / all-gather / barrier / ring-attention) — and this test asserts
they are still those numbers after the number of getter calls a 32-layer prefill makes, not
`32 x` them.

Also asserted here, because it is cheap and its failure mode is two phases away
(`BRINGUP_RECIPE.md` Appendix F.8, `DEC-012`):

* the CCL core grid is **derived** from `mesh_device.compute_with_storage_grid_size()` — (12, 10)
  on this Blackhole Galaxy — and the ring-attention offset is `(grid.x - 1, 0)`;
* that offset satisfies `ccl_core_grid_offset.x >= sdpa_grid.x` for the pinned 8x8 SDPA program
  grid, which is the assert at
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`.
  Deriving the SDPA grid instead would give `11 >= 12` and fail only at SP > 1, in P8.

P5 runs this on `(1,1)`; P8 adds the `(4,8)` target-mesh test at the bottom (`G-SEMAPHORE`),
where the semaphores are the ones a real collective consumes.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_ccl_semaphores.py -x -q
"""

from __future__ import annotations

import pytest
from loguru import logger

import ttnn
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, parametrize_galaxy_submeshes

# bringup_log/04_CCL_PLAN.md section 6: the four counts, with the constants they come from in
# models/demos/gpt_oss_d_p/tt/ccl.py (:65 3*2, :71 2*2, :77 2*1, :85 range(2)).
EXPECTED_RS_SEMAPHORES = 6
EXPECTED_AG_SEMAPHORES = 4
EXPECTED_BARRIER_SEMAPHORES = 2
EXPECTED_RING_ATTENTION_SEMAPHORES = 2

# The SDPA program grid is pinned, NOT derived (DEC-012 / Appendix F.8).
PINNED_SDPA_GRID_X = 8

# Llama-3.1-8B has 32 layers; scheme A issues one attention all-reduce + one MLP all-reduce per
# layer, and each all-reduce takes one RS + one AG + two barrier handles.
NUM_LAYERS = 32


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
def test_ccl_manager_constructs_and_allocates_once(mesh_device):
    """Build a `CCLManager`, then hammer the getters as a 32-layer prefill would. Counts must hold."""
    objs = TestFactory.setup_test(mesh_device)
    ccl = objs["ccl_manager"]

    counts = (
        len(ccl.rs_ping_pong_semaphores),
        len(ccl.ag_ping_pong_semaphores),
        len(ccl.barrier_semaphore),
        len(ccl.ring_attention_ccl_semaphore_handles),
    )
    logger.info(f"[G-MESH] CCLManager semaphore lists at construction (rs/ag/barrier/ring) = {counts}")
    assert counts == (
        EXPECTED_RS_SEMAPHORES,
        EXPECTED_AG_SEMAPHORES,
        EXPECTED_BARRIER_SEMAPHORES,
        EXPECTED_RING_ATTENTION_SEMAPHORES,
    )

    # Identity, not just length: re-allocating in a getter would keep the length and still be wrong.
    rs_ids = [id(s) for s in ccl.rs_ping_pong_semaphores]
    ag_ids = [id(s) for s in ccl.ag_ping_pong_semaphores]
    barrier_ids = [id(s) for s in ccl.barrier_semaphore]

    # 2 all-reduces per layer, each = 1 RS + 1 AG + 2 barriers.
    for _ in range(2 * NUM_LAYERS):
        rs = ccl.get_rs_ping_pong_semaphore()
        ag = ccl.get_ag_ping_pong_semaphore()
        ccl.get_barrier_semaphore()
        ccl.get_barrier_semaphore()
        assert len(rs) == 3, "the RS getter must hand out a 3-slice"
        assert len(ag) == 2, "the AG getter must hand out a 2-slice"

    counts_after = (
        len(ccl.rs_ping_pong_semaphores),
        len(ccl.ag_ping_pong_semaphores),
        len(ccl.barrier_semaphore),
        len(ccl.ring_attention_ccl_semaphore_handles),
    )
    logger.info(f"[G-MESH] after {2 * NUM_LAYERS} all-reduce-equivalents: lists = {counts_after} (unchanged)")
    assert counts_after == counts, f"semaphores were allocated per call: {counts} -> {counts_after}"
    assert [id(s) for s in ccl.rs_ping_pong_semaphores] == rs_ids
    assert [id(s) for s in ccl.ag_ping_pong_semaphores] == ag_ids
    assert [id(s) for s in ccl.barrier_semaphore] == barrier_ids


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
def test_ping_pong_indices_cycle_depth_two(mesh_device):
    """Two *consecutive* collectives never share a semaphore set; the cycle depth is exactly 2.

    Pinned deliberately: `G-RACE` in P8 validates that depth 2 is enough, and if it fails,
    deepening the barrier ring from 2 to 4 is the documented first move
    (`bringup_log/04_CCL_PLAN.md` §6 note 2). A test that did not pin the current depth would let
    that change land unnoticed.
    """
    ccl = TestFactory.setup_test(mesh_device)["ccl_manager"]

    assert (ccl.rs_ping_pong_idx, ccl.ag_ping_pong_idx, ccl.barrier_idx) == (0, 0, 0)

    first_rs = ccl.get_rs_ping_pong_semaphore()
    second_rs = ccl.get_rs_ping_pong_semaphore()
    third_rs = ccl.get_rs_ping_pong_semaphore()
    assert [id(s) for s in first_rs] != [id(s) for s in second_rs], "consecutive RS ops shared a semaphore set"
    assert [id(s) for s in first_rs] == [id(s) for s in third_rs], "RS ping-pong depth is not 2"

    first_barrier = ccl.get_barrier_semaphore()
    second_barrier = ccl.get_barrier_semaphore()
    third_barrier = ccl.get_barrier_semaphore()
    assert id(first_barrier) != id(second_barrier), "the RS and AG halves of one all-reduce shared a barrier"
    assert id(first_barrier) == id(third_barrier), "barrier ping-pong depth is not 2"
    logger.info("[G-MESH] ping-pong depth = 2 for rs / ag / barrier, as designed")


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
def test_ccl_core_grid_is_derived_from_the_device(mesh_device):
    """The CCL grid is the *device's* grid; the ring-attention offset is `(grid.x - 1, 0)`."""
    ccl = TestFactory.setup_test(mesh_device)["ccl_manager"]

    device_grid = mesh_device.compute_with_storage_grid_size()
    logger.info(
        f"[G-MESH] compute_with_storage_grid_size() = ({device_grid.x}, {device_grid.y}); "
        f"ring_attention_ccl_core_grid_offset = {ccl.ring_attention_ccl_core_grid_offset}"
    )

    assert (ccl.compute_grid_size.x, ccl.compute_grid_size.y) == (device_grid.x, device_grid.y)
    assert ccl.ring_attention_ccl_core_grid_offset == (device_grid.x - 1, 0)

    # The constraint that makes DEC-012 mandatory: ring_joint_sdpa asserts
    # ccl_core_grid_offset.x >= sdpa_grid.x. With the SDPA grid pinned at 8 this holds; deriving it
    # from a 12-wide device grid would give 11 >= 12 and fail at SP > 1 only.
    assert (
        ccl.ring_attention_ccl_core_grid_offset[0] >= PINNED_SDPA_GRID_X
    ), f"ring-attention CCL offset x={ccl.ring_attention_ccl_core_grid_offset[0]} < pinned SDPA grid x={PINNED_SDPA_GRID_X}"


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
def test_reset_global_semaphores_runs(mesh_device):
    """`reset_global_semaphores()` resets the RS/AG sets and leaves the counts alone.

    It deliberately does *not* touch the barrier or ring-attention semaphores — an upstream TODO
    that P7 owes a `DEC` on, because chunked prefill reuses one `CCLManager` across chunks
    (`bringup_log/04_CCL_PLAN.md` §6 note 3).
    """
    ccl = TestFactory.setup_test(mesh_device)["ccl_manager"]
    ccl.get_rs_ping_pong_semaphore()
    ccl.reset_global_semaphores()
    assert len(ccl.rs_ping_pong_semaphores) == EXPECTED_RS_SEMAPHORES
    assert len(ccl.ag_ping_pong_semaphores) == EXPECTED_AG_SEMAPHORES


# =====================================================================================
# G-SEMAPHORE (P8) — the same invariant, on the mesh the deployment actually uses
# =====================================================================================
@parametrize_galaxy_submeshes([(4, 8)])
def test_semaphore_counts_hold_on_the_target_mesh(mesh_device, device_params, submesh_shape):
    """`G-SEMAPHORE`: `CCLManager` allocates **6 / 4 / 2 / 2** on `(4,8)`, and a multi-layer model
    built on top of it does not multiply them.

    The `(1,1)` test above proves the getters do not allocate. This one proves the two things that
    `(1,1)` cannot: that the counts are the same on the mesh where the semaphores are actually used
    by a collective, and that **building layers does not create managers** — every `DecoderLayer`
    takes the manager it is handed (`tt/layer.py`), so a 32-layer model must leave the four lists at
    their constants. The recipe states the gate in exactly those terms
    (`BRINGUP_RECIPE.md:855-857`: "instantiate the model and check the manager's semaphore list
    lengths equal the constants, not `n_layers x` them").

    Two layers, random weights: the invariant is depth-independent (the layer loop is one expression)
    and 32 layers of real weights would spend two minutes proving nothing extra. The full-depth,
    real-weight version of this check is the `[prefill-pcc] semaphores` line the galaxy harness prints
    after a 32-layer run (`tests/galaxy_prefill_kv_pcc.py`).
    """
    import torch

    from models.demos.llama31_8b_d_p.tests.test_factory import llama_config_dims
    from models.demos.llama31_8b_d_p.tt.model import Model
    from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config

    objs = TestFactory.setup_submesh(mesh_device, submesh_shape)
    mesh, ccl = objs["mesh_device"], objs["ccl_manager"]
    counts_before = (
        len(ccl.rs_ping_pong_semaphores),
        len(ccl.ag_ping_pong_semaphores),
        len(ccl.barrier_semaphore),
        len(ccl.ring_attention_ccl_semaphore_handles),
    )
    expected = (
        EXPECTED_RS_SEMAPHORES,
        EXPECTED_AG_SEMAPHORES,
        EXPECTED_BARRIER_SEMAPHORES,
        EXPECTED_RING_ATTENTION_SEMAPHORES,
    )
    assert counts_before == expected, f"on {submesh_shape}: {counts_before} != {expected}"

    dims = llama_config_dims()
    hf_config = llama_hf_config(dims)
    generator = torch.Generator().manual_seed(0)
    num_layers = 2

    def rnd(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    hidden, inter = dims["hidden_size"], dims["intermediate_size"]
    n_heads, n_kv, head_dim = dims["num_attention_heads"], dims["num_key_value_heads"], dims["head_dim"]
    state = {
        "model.embed_tokens.weight": rnd(dims["vocab_size"], hidden),
        "model.norm.weight": torch.rand(hidden, generator=generator) + 0.5,
    }
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}."
        state.update(
            {
                prefix + "input_layernorm.weight": torch.rand(hidden, generator=generator) + 0.5,
                prefix + "post_attention_layernorm.weight": torch.rand(hidden, generator=generator) + 0.5,
                prefix + "self_attn.q_proj.weight": rnd(n_heads * head_dim, hidden),
                prefix + "self_attn.k_proj.weight": rnd(n_kv * head_dim, hidden),
                prefix + "self_attn.v_proj.weight": rnd(n_kv * head_dim, hidden),
                prefix + "self_attn.o_proj.weight": rnd(hidden, n_heads * head_dim),
                prefix + "mlp.gate_proj.weight": rnd(inter, hidden),
                prefix + "mlp.up_proj.weight": rnd(inter, hidden),
                prefix + "mlp.down_proj.weight": rnd(hidden, inter),
            }
        )
    model = Model(
        mesh,
        hf_config,
        state,
        mesh_config=objs["mesh_config"],
        ccl_manager=ccl,
        max_seq_len=512,
        num_layers=num_layers,
        with_lm_head=False,
    )
    counts_after = (
        len(ccl.rs_ping_pong_semaphores),
        len(ccl.ag_ping_pong_semaphores),
        len(ccl.barrier_semaphore),
        len(ccl.ring_attention_ccl_semaphore_handles),
    )
    assert counts_after == expected, f"building {num_layers} layers changed the counts: {counts_after}"
    # Identity, not just count: every layer must hold the SAME manager object.
    managers = {id(layer.self_attn.ccl_manager) for layer in model.layers} | {
        id(layer.mlp.ccl_manager) for layer in model.layers
    }
    assert managers == {id(ccl)}, f"{len(managers)} distinct CCLManagers across {num_layers} layers; expected 1"
    # Negative control: a SECOND manager on the same mesh must hand out DIFFERENT semaphore objects.
    # Without this, `managers == {id(ccl)}` above could pass for a trivial reason (e.g. if `id()`
    # collided, or if every CCLManager shared one global set), and the gate would be asserting
    # nothing about sharing.
    from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
    from models.demos.llama31_8b_d_p.utils.general_utils import get_default_num_links

    other = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Ring)
    shared = {id(s) for s in ccl.rs_ping_pong_semaphores} & {id(s) for s in other.rs_ping_pong_semaphores}
    assert not shared, (
        f"a second CCLManager handed out {len(shared)} of the first one's semaphore objects; the "
        f"'all layers share ONE manager' assertion above would then be vacuous"
    )

    logger.info(
        f"[G-SEMAPHORE] on {submesh_shape}: rs/ag/barrier/ring = {counts_after} before and after "
        f"building {num_layers} layers (2 collectives each); all layers share ONE CCLManager; CCL "
        f"grid {(ccl.compute_grid_size.x, ccl.compute_grid_size.y)}, ring-attention offset "
        f"{ccl.ring_attention_ccl_core_grid_offset}; negative control: a second manager shares NONE "
        f"of the first's semaphore objects"
    )
