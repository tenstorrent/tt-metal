# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`CCLManager` semaphore lifetime: allocated once, cycled per call. Gate: `G-SEMAPHORE`.

Created in P5.1 rather than P8 (`[DEV-6]`, `bringup_log/03_OUTLINE.md` §1.1) because `G-MESH`
already requires this assertion, and writing it twice would let the two copies disagree. This file
holds the **one-card half**: the inventory (6 RS + 4 AG + 2 barrier + 2 ring-attention = 14,
`bringup_log/04_CCL_PLAN.md` §3), the ping-pong cycling, and the depth. P8 adds the target-mesh
parametrisation and the after-a-real-harness-run assertion.

The failure this exists to catch is any count becoming `n_layers x` the constant — a `CCLManager`
built per layer instead of per model, which shows up as nondeterministic multi-device PCC rather
than as an error (`BRINGUP_RECIPE.md:900-902`). `G-SEMAPHORE` produces no PCC, so §1.4's
floor/reference-dtype fields do not apply; its **negative control** is
`test_semaphores_would_multiply_if_built_per_layer`, which constructs one manager per simulated
layer and asserts the count the correct code must NOT produce.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_ccl_semaphores.py -x -q
"""

import pytest
from loguru import logger

from models.demos.llama31_8b_d_p.tt.ccl import CCLManager

# `bringup_log/04_CCL_PLAN.md` §3, from `models/demos/gpt_oss_d_p/tt/ccl.py:65`, `:71`, `:77`, `:84`.
EXPECTED_RS = 6
EXPECTED_AG = 4
EXPECTED_BARRIER = 2
EXPECTED_RING_ATTENTION = 2
EXPECTED_TOTAL = 14
N_LAYERS = 32  # `bringup_log/00_MODEL_CARD.md` §2 — one manager serves all of them.


def _inventory(ccl):
    return {
        "rs": len(ccl.rs_ping_pong_semaphores),
        "ag": len(ccl.ag_ping_pong_semaphores),
        "barrier": len(ccl.barrier_semaphore),
        "ring_attention": len(ccl.ring_attention_ccl_semaphore_handles),
    }


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_semaphore_inventory_is_exact(mesh_device):
    """Exactly 14 global semaphores, in the four classes the plan names."""
    ccl = CCLManager(mesh_device, num_links=1)
    inv = _inventory(ccl)
    logger.info(f"[G-SEMAPHORE] inventory at construction: {inv} (total {sum(inv.values())})")

    assert inv["rs"] == EXPECTED_RS
    assert inv["ag"] == EXPECTED_AG
    assert inv["barrier"] == EXPECTED_BARRIER
    assert inv["ring_attention"] == EXPECTED_RING_ATTENTION
    assert sum(inv.values()) == EXPECTED_TOTAL


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_semaphores_survive_dozens_of_getter_cycles(mesh_device):
    """Handing semaphores out must not allocate. 4 collectives x 32 layers = 128 barrier cycles.

    That is the real per-forward load under residual scheme A (`DEC-025`): RS+AG for attention and
    RS+AG for the MLP, every layer.
    """
    ccl = CCLManager(mesh_device, num_links=1)
    before = _inventory(ccl)

    for _ in range(N_LAYERS):
        for _ in range(2):  # attention tail, then the MLP tail
            ccl.get_rs_ping_pong_semaphore()
            ccl.get_barrier_semaphore()
            ccl.get_ag_ping_pong_semaphore()
            ccl.get_barrier_semaphore()

    after = _inventory(ccl)
    logger.info(f"[G-SEMAPHORE] after {N_LAYERS} layers x 4 collectives: {after}")
    assert after == before, f"semaphores were allocated by a getter: {before} -> {after}"
    assert sum(after.values()) == EXPECTED_TOTAL


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ping_pong_cycles_and_is_two_deep(mesh_device):
    """Each getter alternates between two disjoint slices, and the barrier gap is exactly one op.

    `DEC-026` ships depth 2 knowingly: a reduce-scatter takes `barrier[0]`, the all-gather that
    follows takes `barrier[1]`, the next reduce-scatter takes `barrier[0]` again. `G-RACE` is the
    measurement that decides whether one op of separation is enough; deepening to 4 would blind it.
    """
    ccl = CCLManager(mesh_device, num_links=1)

    rs_a, rs_b, rs_c = (ccl.get_rs_ping_pong_semaphore() for _ in range(3))
    assert len(rs_a) == len(rs_b) == 3
    assert all(x is not y for x in rs_a for y in rs_b), "RS ping-pong handed out an overlapping slice"
    assert [id(s) for s in rs_c] == [id(s) for s in rs_a], "RS ping-pong is not 2-deep"

    ag_a, ag_b, ag_c = (ccl.get_ag_ping_pong_semaphore() for _ in range(3))
    assert len(ag_a) == len(ag_b) == 2
    assert all(x is not y for x in ag_a for y in ag_b), "AG ping-pong handed out an overlapping slice"
    assert [id(s) for s in ag_c] == [id(s) for s in ag_a], "AG ping-pong is not 2-deep"

    b0, b1, b2 = (ccl.get_barrier_semaphore() for _ in range(3))
    assert b0 is not b1, "the barrier ping-pong handed out the same semaphore twice in a row"
    assert b2 is b0, "the barrier ping-pong is not 2-deep"
    logger.info("[G-SEMAPHORE] rs/ag/barrier ping-pong all cycle with period 2 (DEC-026)")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_reset_skips_barrier_and_ring_attention(mesh_device):
    """`reset_global_semaphores` resets RS/AG and deliberately leaves barrier + ring attention.

    Asserted, not merely commented, because `DEC-026` ships the template's behaviour knowing its
    justification ("one-shot prefill never reuses a CCLManager") is false for chunked prefill. If a
    later phase changes it, this test is what says so.
    """
    ccl = CCLManager(mesh_device, num_links=1)
    barrier_ids = [id(s) for s in ccl.barrier_semaphore]
    ring_ids = [id(s) for s in ccl.ring_attention_ccl_semaphore_handles]

    ccl.get_barrier_semaphore()  # move the index off 0
    assert ccl.barrier_idx == 1
    ccl.reset_global_semaphores()

    # The handles are untouched objects, and the reset does not rewind the ping-pong index either.
    assert [id(s) for s in ccl.barrier_semaphore] == barrier_ids
    assert [id(s) for s in ccl.ring_attention_ccl_semaphore_handles] == ring_ids
    assert ccl.barrier_idx == 1, "reset_global_semaphores must not silently rewind the barrier index"
    assert _inventory(ccl) == {"rs": 6, "ag": 4, "barrier": 2, "ring_attention": 2}


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_semaphores_would_multiply_if_built_per_layer(mesh_device):
    """**The negative control.** A manager per layer is what the gate must be able to reject.

    Three managers stand in for three layers; the total semaphore count is `3 x 14`, which is the
    shape of the bug (`n_layers x` the constant). Only three are built, not 32, because each one
    allocates real device semaphores.
    """
    per_layer = [CCLManager(mesh_device, num_links=1) for _ in range(3)]
    total = sum(sum(_inventory(c).values()) for c in per_layer)
    logger.info(f"[G-SEMAPHORE] control: 3 managers -> {total} semaphores (correct code: {EXPECTED_TOTAL})")
    assert total == 3 * EXPECTED_TOTAL
    assert total != EXPECTED_TOTAL, "the control did not diverge from the correct inventory"
