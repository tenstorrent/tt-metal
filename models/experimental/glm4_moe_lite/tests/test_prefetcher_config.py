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


# --- DRAM shard layout for the prefetched weight -------------------------------
#
# _prefetch_dram_shard_weight_tt splits the weight's N across the PREFETCHER's bank
# set (8 cores), not the full 12-bank DRAM grid. The shard math is replicated here
# because getting it wrong yields a layout the ring matmul and dram_prefetcher
# disagree about, which is another deadlock rather than an error.

PREFETCH_DRAM_BANKS = 8


def _dram_shard_width(K: int, N: int, ring: int, banks: int = PREFETCH_DRAM_BANKS) -> tuple[int, int, int]:
    """Mirror of the padding + bank-split arithmetic. Returns (K_pad, N_pad, width)."""

    def round_up(x: int, mult: int) -> int:
        return ((x + mult - 1) // mult) * mult

    k_pad = round_up(math.ceil(K / ring), TILE) * ring
    n_pad = round_up(math.ceil(N / ring), TILE) * ring
    return k_pad, n_pad, n_pad // banks


def test_oproj_needs_no_padding_and_divides_the_banks() -> None:
    k_pad, n_pad, width = _dram_shard_width(*W_O, 16)
    assert (k_pad, n_pad) == W_O, "o_proj should need no padding at ring 16"
    assert width == 256, "2048 / 8 banks = 256 per bank"
    assert n_pad % PREFETCH_DRAM_BANKS == 0


def test_dram_shard_padding_rounds_up_to_ring_times_tile() -> None:
    """A shape needing padding pads N up to a multiple of ring*TILE, not just TILE."""
    k_pad, n_pad, _ = _dram_shard_width(5120, 2080, 16)
    assert n_pad == 2560, f"2080 -> ceil(2080/16)=130 -> round_up(130,32)=160 -> 160*16=2560, got {n_pad}"
    assert k_pad == 5120


def test_bank_count_is_not_the_full_dram_grid() -> None:
    """The regression this guards: sharding o_proj over all 12 DRAM banks.

    The bank count is half of the sender/receiver contract -- 12 banks x 2 receivers
    is the 24-core ring that deadlocks. It must stay 8.
    """
    assert PREFETCH_DRAM_BANKS == 8
    assert 24 not in ring_feasibility(*W_O, max_cores=24)
    # Fortunately o_proj's N does not divide the full 12-bank grid (2048 % 12 == 8), so
    # _prefetch_dram_shard_weight_tt's explicit bank-divisibility guard raises rather
    # than silently building a layout the ring matmul disagrees with.
    assert W_O[1] % 12 != 0
    assert W_O[1] % PREFETCH_DRAM_BANKS == 0


# ---------------------------------------------------------------------------
# Grid-derived layout: WH Galaxy (8x9) vs Blackhole Galaxy (12x10)
#
# The layout used to be hardcoded to WH Galaxy. It is now derived from the device grid,
# so these pin both arches: the WH cases must reproduce exactly what was validated there,
# and the BH cases must produce the wider ring that brings the GlobalCB under the L1
# budget. A fake device is enough -- only grid sizes are read.
# ---------------------------------------------------------------------------

BH_L1_SIZE = 1_572_864  # blackhole_140_arch.yaml worker_l1_size
WH_L1_SIZE = 1_499_136  # wormhole_b0_80_arch.yaml worker_l1_size
# SDPA decode circular buffers at the minimum legal k_chunk_size of 32, measured on WH.
SDPA_CB_BYTES_AT_K_CHUNK_32 = 1_033_568


class _FakeGrid:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class _FakeMeshDevice:
    """Just enough of MeshDevice for the layout math: two grid sizes."""

    def __init__(self, grid_x: int, grid_y: int, dram_banks: int) -> None:
        self._grid = _FakeGrid(grid_x, grid_y)
        self._dram = _FakeGrid(dram_banks, 1)

    def compute_with_storage_grid_size(self) -> _FakeGrid:
        return self._grid

    def dram_grid_size(self) -> _FakeGrid:
        return self._dram


WH_GALAXY = (8, 9, 12)  # 8x9 Tensix, 12 DRAM views
BH_GALAXY = (12, 10, 8)  # 12x10 Tensix (1x-harvested), 8 DRAM views -- measured on device


def test_wh_galaxy_layout_is_unchanged() -> None:
    """The derivation must reproduce the validated WH Galaxy layout exactly."""
    from models.experimental.glm4_moe_lite.tt.prefetcher_setup import get_glm_core_ranges

    senders, dram_cores, _, receivers, worker_crs, mapping = get_glm_core_ranges(
        _FakeMeshDevice(*WH_GALAXY), num_global_cb_receivers=2
    )
    assert [(c.x, c.y) for c in senders] == [(6, y) for y in range(8)], "senders in column 6"
    assert sorted({x for x, _ in receivers}) == [4, 5], "receivers in columns 4-5"
    assert len(receivers) == 16
    assert len(dram_cores) == 8, "ring is built on 8 banks even though WH exposes 12"
    assert worker_crs.num_cores() == 6 * 9, "worker rectangle is columns 0-5"
    assert len(mapping) == 8


def test_bh_galaxy_layout_hosts_a_32_core_ring() -> None:
    """Blackhole: 8 senders in column 10, 4 receivers each in columns 6-9."""
    from models.experimental.glm4_moe_lite.tt.prefetcher_setup import get_glm_core_ranges, ring_cores_for

    device = _FakeMeshDevice(*BH_GALAXY)
    ring, receivers_per_sender = ring_cores_for(device, *W_O)
    assert (ring, receivers_per_sender) == (32, 4)

    senders, dram_cores, _, receivers, worker_crs, mapping = get_glm_core_ranges(
        device, num_global_cb_receivers=receivers_per_sender
    )
    assert [(c.x, c.y) for c in senders] == [(10, y) for y in range(8)], "senders in column 10"
    assert sorted({x for x, _ in receivers}) == [6, 7, 8, 9], "receivers in columns 6-9"
    assert len(receivers) == 32
    assert len(dram_cores) == 8
    assert worker_crs.num_cores() == 10 * 10, "worker rectangle is columns 0-9"
    # Each sender's receivers must be contiguous along x, or the remote-CB core set stops
    # matching the GlobalCB receiver set and a hop core becomes necessary.
    for _, recv_crs in mapping:
        assert recv_crs.num_cores() == 4


def test_ring_32_is_the_widest_legal_ring_for_oproj() -> None:
    """gcd(160, 64) = 32, so no wider ring divides both of w_o's dimensions."""
    k_tiles, n_tiles = W_O[0] // TILE, W_O[1] // TILE
    assert math.gcd(k_tiles, n_tiles) == 32
    assert 64 not in ring_feasibility(*W_O, max_cores=64)
    assert 32 in ring_feasibility(*W_O, max_cores=32)


def test_doubling_the_ring_halves_the_global_cb() -> None:
    """The mechanism that unblocks the prefetcher: the per-receiver payload is
    K_tiles * (N_tiles / ring), so a wider ring shrinks it."""
    assert global_cb_tiles_for(*W_O, 16) == 640
    assert global_cb_tiles_for(*W_O, 32) == 320


def test_ring_32_brings_the_global_cb_plus_sdpa_under_l1() -> None:
    """The L1 budget that made the prefetcher look infeasible at ring=16.

    The ring width, not the architecture, is the lever. At ring=16 the GlobalCB plus
    SDPA's circular buffers overflow L1 on *both* arches; at ring=32 they fit on both.
    """
    bytes_at = lambda ring: global_cb_tiles_for(*W_O, ring) * 1088 + SDPA_CB_BYTES_AT_K_CHUNK_32  # noqa: E731

    assert bytes_at(16) > WH_L1_SIZE, "the measured WH verdict at ring=16"
    assert bytes_at(16) > BH_L1_SIZE, "ring=16 does not fit on BH either"
    assert bytes_at(32) <= BH_L1_SIZE, "ring=32 fits on Blackhole"
    assert BH_L1_SIZE - bytes_at(32) > 150_000, "with real headroom, not a rounding win"
    assert bytes_at(32) <= WH_L1_SIZE, "and it fits on Wormhole too -- see the note below"


def test_wh_galaxy_can_also_host_the_32_core_ring() -> None:
    """Correcting the record: WH was never geometrically stuck at ring=16.

    The original "the GlobalCB prefetcher and FlashMLA decode cannot coexist in L1"
    conclusion held ring=16 fixed, inherited from REAP's proven config, and searched for
    other knobs (k_chunk_size, the L1 activation flags, SDPA placement). Widening the ring
    was not among them -- but it is the one knob that moves the GlobalCB, since the
    per-receiver payload is K_tiles * (N_tiles / ring).

    On WH's 8x9 grid, 8 senders in column 6 leave columns 2-5 free for 4 receivers each,
    all inside the columns 0-5 worker rectangle. So ring=32 is reachable there as well, and
    the derivation returns it for both arches.

    This is arithmetic and geometry, not an on-device result: nobody has run the prefetcher
    at ring=32 on either arch yet. It says the L1 wall is not a dead end, not that the
    prefetcher works.
    """
    from models.experimental.glm4_moe_lite.tt.prefetcher_setup import get_glm_core_ranges, ring_cores_for

    device = _FakeMeshDevice(*WH_GALAXY)
    assert ring_cores_for(device, *W_O) == (32, 4)

    senders, _, _, receivers, worker_crs, _ = get_glm_core_ranges(device, num_global_cb_receivers=4)
    assert [(c.x, c.y) for c in senders] == [(6, y) for y in range(8)]
    assert sorted({x for x, _ in receivers}) == [2, 3, 4, 5]
    assert len(receivers) == 32
    # The invariant that makes origin-anchored matmul grids land inside the SubDevice.
    assert worker_crs.num_cores() == 6 * 9
    assert all(x < 6 for x, _ in receivers), "receivers must sit inside the worker rectangle"
