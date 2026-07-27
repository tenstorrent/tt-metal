# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side tests for the GlobalCB prefetcher ring-config math.

No device needed. These exist because a ring config that does not divide both
matmul dimensions does not raise -- it DEADLOCKS on device, waiting for pages the
producer cannot send. That is an expensive failure mode to debug (it needs a Galaxy
reset), so the arithmetic is pinned here instead.

The specific regression these guard: an earlier revision of prefetcher_setup.py used
a 12-bank x 2-receiver contract, giving a 24-core ring. 24 divides neither of Flash's
o_proj dimensions (160 x 64 tiles), so it could only ever have hung. It also dropped
the K-divisibility assert, derived the program-config grid from the ring-core bounding
box rather than from num_cores, and sized the GlobalCB off DRAM banks instead of
receivers.
"""

from __future__ import annotations

import math

import pytest

from models.experimental.glm4_moe_lite.tt.prefetcher_setup import (
    NUM_GLOBAL_CB_RECEIVERS,
    TILE,
    global_cb_tiles_for,
    ring_feasibility,
)

# Flash MLA weight shapes (K, N) per device. Attention weights are replicated across
# the mesh (attn_row_mapper is None unless TP is on, and TP is off by default), so
# these are the full shapes each device sees.
W_O = (5120, 2048)  # o_proj          160 x  64 tiles  -- largest 2D decode weight
W_Q_B = (768, 5120)  # q_b_proj         24 x 160 tiles
W_Q_KV_A = (2048, 1344)  # fused q_a + kv_a  64 x  42 tiles
W_Q_A = (2048, 768)  # q_a alone        64 x  24 tiles
W_KV_A = (2048, 576)  # kv_a alone       64 x  18 tiles


def test_oproj_supports_the_16_core_ring() -> None:
    """The first-prototype target must admit exactly the ring we ship."""
    feasible = ring_feasibility(*W_O, max_cores=16)
    assert 16 in feasible, f"o_proj must support a 16-core ring; got {feasible}"
    assert feasible[0] == 16, "16 should be the widest feasible ring at max_cores=16"


def test_24_core_ring_is_infeasible_for_oproj() -> None:
    """The exact configuration that deadlocked. 160 x 64 tiles: 24 divides neither."""
    assert 24 not in ring_feasibility(*W_O, max_cores=24)
    k_tiles, n_tiles = W_O[0] // TILE, W_O[1] // TILE
    assert k_tiles % 24 != 0 and n_tiles % 24 != 0


@pytest.mark.parametrize(
    "name,shape,expected_widest",
    [
        ("w_o", W_O, 16),
        ("w_q_b", W_Q_B, 8),
        ("w_q_a", W_Q_A, 8),
        ("w_q_kv_a", W_Q_KV_A, None),  # gcd(64, 42) == 2 -> no usable ring
        ("w_kv_a", W_KV_A, None),  # gcd(64, 18) == 2 -> no usable ring
    ],
)
def test_flash_mla_weight_feasibility(name: str, shape: tuple[int, int], expected_widest: int | None) -> None:
    """Pin the per-weight verdict, so a shape or fusion change surfaces here.

    Note w_q_kv_a: enabling FUSE_QKV_A (a winning decode optimization, default on)
    concatenates q_a and kv_a into N=1344 = 42 tiles, which makes the fused weight
    ring-infeasible. Unfused, w_q_a alone would support an 8-core ring. That is a real
    tension between two optimizations, not an oversight -- w_q_a is 2.9 MB/layer
    against o_proj's 11.1 MB, so the fusion is worth more than prefetching it.
    """
    feasible = ring_feasibility(*shape, max_cores=16)
    if expected_widest is None:
        assert feasible == [] or max(feasible) < 8, f"{name}: expected no usable ring, got {feasible}"
    else:
        assert max(feasible) == expected_widest, f"{name}: expected widest {expected_widest}, got {feasible}"


def test_feasible_rings_divide_both_dimensions() -> None:
    """The core invariant: every returned ring size divides K_tiles and N_tiles."""
    for shape in (W_O, W_Q_B, W_Q_KV_A, W_Q_A, W_KV_A):
        k_tiles, n_tiles = shape[0] // TILE, shape[1] // TILE
        for n in ring_feasibility(*shape, max_cores=16):
            assert k_tiles % n == 0, f"{shape}: ring {n} does not divide K_tiles={k_tiles}"
            assert n_tiles % n == 0, f"{shape}: ring {n} does not divide N_tiles={n_tiles}"


def test_feasible_rings_factor_into_an_8_wide_grid() -> None:
    """make_ring_config builds the grid as (min(8, n), n // min(8, n)); n must factor."""
    for shape in (W_O, W_Q_B, W_Q_A):
        for n in ring_feasibility(*shape, max_cores=16):
            gx = min(8, n)
            assert gx > 0 and n % gx == 0, f"ring {n} does not factor into an 8-wide grid"
            assert gx * (n // gx) == n


def test_global_cb_sizing_is_per_receiver_not_per_bank() -> None:
    """640 tiles for o_proj: K_tiles * (N_tiles / ring) = 160 * (64 / 16).

    The superseded revision computed 10240 total tiles / 12 DRAM banks = 853 and
    rounded up to 900, over-allocating ~283 KB of L1 per core on a 1.5 MB budget --
    while the plan already flagged L1 pressure as a top risk.
    """
    assert global_cb_tiles_for(*W_O, 16) == 640
    stale_estimate = math.ceil((W_O[0] // TILE) * (W_O[1] // TILE) / 12)
    assert stale_estimate > global_cb_tiles_for(*W_O, 16)


def test_global_cb_sizing_for_the_next_increment() -> None:
    """w_q_b at its own feasible ring of 8: 24 * (160 / 8) = 480 tiles."""
    assert global_cb_tiles_for(*W_Q_B, 8) == 480


def test_receiver_contract_is_two_per_sender() -> None:
    """8 senders x 2 receivers = the 16-core ring the configs assume."""
    assert NUM_GLOBAL_CB_RECEIVERS == 2
    assert 8 * NUM_GLOBAL_CB_RECEIVERS == 16


def test_non_tile_aligned_shapes_are_rejected() -> None:
    assert ring_feasibility(5120, 2047) == []
    assert ring_feasibility(1000, 2048) == []
