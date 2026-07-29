# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — the three remaining per-transaction levers (B8, B10, A3).

One lever landed and two are refuted; all three are tested, because a refuted
lever's *gate* is exactly as load-bearing as a shipped one — it is the thing that
keeps the next implementer from re-enabling a measured regression.

* **B8 `prefetch_blocks == 2`** (LANDED) — trid double-issue on the read path.
  Each chunk-block's 32 stick reads carry one of two NoC transaction ids and the
  barrier is `noc_async_read_barrier_with_trid` on the *previous* id, so the next
  block's reads are already in flight while the current one drains. Needs a third
  CB window (`cb_reserve_back` cannot hand out the next block's address before the
  current one is pushed), which is why the gate also checks L1.

* **B10 `vc_spread`** (REFUTED) — per-core static unicast VC. Measured a
  regression on every regime with real traffic: the write half is ~1.8-2.0x and
  the read half ~1.085x. Gate is identity-false.

* **A3 `bank_placement`** (REFUTED) — bank-adjacent work-unit -> core order.
  Measured neutral, and structurally there is nothing to exploit: a tilize block
  needs 32 CONSECUTIVE source pages, and interleaved round-robin puts page `p` in
  bank `p % NUM_DRAM_BANKS`, so every core touches every bank. Gate is
  identity-false.

tt-npe closes the story on why B10/A3 could not have paid: on `b_wide_short` the
**congestion impact is 0.4 %** (13 855 ideal vs 13 910 congested cycles) and the
**DRAM BW utilisation is 103 %** — the regime is at its achievable DRAM bound for
512 B partial-page reads, so the two congestion levers had at most 0.4 % to win.

Correctness is `torch.equal` throughout (tilize is value-preserving), and the
inputs are `arange` rather than `randn` on purpose: every element is unique, so a
block written into the wrong prefetch window cannot cancel out.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import (
    BANK_PLACEMENT_MIN_CORES,
    L1_CB_BUDGET_PREFETCH_BYTES,
    NUM_UNICAST_VCS,
    PREFETCH_DEPTH,
    PREFETCH_TRIDS,
    TRID_PREFETCH_MAX_CORES,
    TRID_PREFETCH_MAX_ROW_BYTES,
    TRID_PREFETCH_MIN_BLOCKS,
    VC_SPREAD_MIN_CORES,
    VC_SPREAD_READ,
    VC_SPREAD_WRITE,
    bank_placement_pays,
    build_plan,
    trid_prefetch_pays,
    vc_spread_pays,
)

# Shapes, with the plan each one lands on (grid 8x8, bf16):
#   B8_SINGLE_CORE  1 core,  16 blk, 1024 B  -> B8 via the CORE clause
#   B8_MULTI_CHUNK  1 core,   8 blk, 1024 B  -> B8, and the chunk x block flatten
#   B8_64B_MULTIBLK 64 cores, 4 blk,   64 B  -> B8 via the READ-SIZE clause
#   B8_128B_2BLK    64 cores, 2 blk,  128 B  -> B8 via the READ-SIZE clause
#   NO_B8_1BLK      64 cores, 1 blk,  512 B  -> B8 structurally off (no next block)
#   NO_B8_256B      64 cores, 2 blk,  256 B  -> B8 gated off (measured 1.004)
B8_SINGLE_CORE = (1, 1, 512, 512)
B8_MULTI_CHUNK = (1, 1, 128, 1024)
B8_64B_MULTIBLK = (1, 1, 8192, 32)
B8_128B_2BLK = (1, 1, 4096, 64)
NO_B8_1BLK = (1, 1, 32, 16384)
NO_B8_256B = (1, 1, 4096, 128)


def _plan(
    device,
    shape,
    *,
    dtype=ttnn.bfloat16,
    use_multicore=True,
    use_double_buffer=None,
    memory_config=None,
    out_memory_config=None,
):
    torch.manual_seed(0)
    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    out_mem = out_memory_config if out_memory_config is not None else mem
    tt_input = ttnn.from_torch(
        torch.randn(shape).bfloat16(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
    )
    tt_output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, out_mem)
    return build_plan(tt_input, tt_output, device, use_multicore=use_multicore, use_double_buffer=use_double_buffer)


def _roundtrip_exact(device, shape, *, memory_config=None, use_multicore=True, use_double_buffer=None, repeats=1):
    """tilize is value-preserving: assert bit-exactness, not a tolerance."""
    n = 1
    for d in shape:
        n *= d
    torch_input = torch.arange(n, dtype=torch.float32).reshape(shape).bfloat16()
    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
    )
    for _ in range(repeats):
        tt_output = tilize(tt_input, memory_config, use_multicore=use_multicore, use_double_buffer=use_double_buffer)
        assert tt_output.layout == ttnn.TILE_LAYOUT
        result = ttnn.to_torch(tt_output)
        assert torch.equal(result.float(), torch_input.float()), (
            f"not bit-exact on {shape}: "
            f"{(result.float() - torch_input.float()).abs().max()} max abs, "
            f"{(result.float() != torch_input.float()).sum()} mismatching elements"
        )


# ---------------------------------------------------------------------------
# B8 — the gate, pinned to the two device sweeps that set it
# ---------------------------------------------------------------------------


def test_b8_core_clause_pinned_to_its_sweep():
    """B8's core-count clause is the DRAM-saturation boundary, measured.

    `[1,1,4096,512]` (chunk 16 => 1024 B reads) with the core count FORCED through
    the planner's `core_cap` hook, so only `ncores` moves. 7 rounds x 10 launches,
    in-run A/B pairs, CV <= 0.9 %:

        cores | blk | no lever ns |    B8 ns | B8/none | GB/s (no lever)
        ------|-----|-------------|----------|---------|----------------
            1 | 128 |     218 946 |  190 302 | *0.869* |  38.3
            2 |  64 |     112 612 |   98 316 | *0.873* |  74.5
            4 |  32 |      61 084 |   52 915 | *0.866* | 137.3
            8 |  16 |      45 479 |   45 484 |   1.000 | 184.5  <- DRAM saturates
           16 |   8 |      45 193 |   43 649 |   0.966 | 185.6
           32 |   4 |      44 768 |   43 746 |   0.977 | 187.4
           64 |   2 |      44 433 |   44 107 |   0.993 | 188.8

    1-4 cores is a flat, reproducible -13 %; from 8 cores the wall clock stops
    moving with the core count at all, which is the saturation the mechanism
    predicts. The residual ~-3 % at 16/32 cores is NOT monotone with +0.0 % at 8
    and -0.7 % at 64, so it is scatter and the threshold stays at 4. Raising this
    constant without re-running `p_1024B_*c` / `x_1024B_*c_b8` spends L1 for
    nothing.
    """
    assert TRID_PREFETCH_MAX_CORES == 4
    fits = 3 * 16 * 4096  # chunk 16, bf16 in and out
    assert fits <= L1_CB_BUDGET_PREFETCH_BYTES
    for cores in (1, 2, 4):
        assert trid_prefetch_pays(cores, blocks_per_core=16, chunk_row_bytes=1024, prefetch_cb_bytes=fits)
    for cores in (8, 16, 32, 64):
        assert not trid_prefetch_pays(cores, blocks_per_core=16, chunk_row_bytes=1024, prefetch_cb_bytes=fits)


def test_b8_read_size_clause_pinned_to_its_sweep():
    """B8's read-size clause: the other way to be under DRAM saturation.

    Fixed 64 cores x 2 blocks/core (`[1,1,4096,W]`, W = 64/128/256/512 giving
    chunk 2/4/8/16, i.e. 128/256/512/1024 B reads), plus the 64 B / 4-block row:

        read B | no lever ns |   B8 ns | B8/none | GB/s (no lever)
        -------|-------------|---------|---------|----------------
          64 B |      13 967 |  11 258 | *0.806* |  75.1
         128 B |       9 572 |   7 861 | *0.821* | 109.5
         256 B |      13 536 |  13 709 |   1.013 | 154.9
         512 B |      23 391 |  22 753 |   0.972 | 179.3
        1024 B |      44 433 |  44 107 |   0.993 | 188.8

    At <= 128 B even the full grid is transaction-rate bound (75-110 GB/s, far
    under the ~190 GB/s achievable copy), so the per-block read drain is still on
    the critical path. From 256 B up it is not, and 256 B is measured *negative*,
    so the threshold cannot be moved up to 512 B on the strength of its 0.972
    alone — the sequence is not monotone.
    """
    assert TRID_PREFETCH_MAX_ROW_BYTES == 128
    small = 3 * 2 * 4096
    for row_bytes in (64, 128):
        assert trid_prefetch_pays(64, blocks_per_core=2, chunk_row_bytes=row_bytes, prefetch_cb_bytes=small)
    for row_bytes in (256, 512, 1024):
        assert not trid_prefetch_pays(64, blocks_per_core=2, chunk_row_bytes=row_bytes, prefetch_cb_bytes=small)


def test_b8_min_blocks_clause_is_structural():
    """One chunk-block per core has no next block to keep in flight.

    This is not a payoff question, so it must hold even inside the regimes where
    both measured clauses say yes.
    """
    assert TRID_PREFETCH_MIN_BLOCKS == 2
    assert not trid_prefetch_pays(1, blocks_per_core=1, chunk_row_bytes=64, prefetch_cb_bytes=1024)
    assert not trid_prefetch_pays(64, blocks_per_core=1, chunk_row_bytes=64, prefetch_cb_bytes=1024)
    assert trid_prefetch_pays(64, blocks_per_core=2, chunk_row_bytes=64, prefetch_cb_bytes=1024)


def test_b8_refuses_to_blow_the_l1_budget():
    """The third window must fit at the UNCHANGED chunk width.

    Refinement 1's rule: a lever may not move the transaction shape behind the
    caller's back. So when depth-3 does not fit, B8 declines rather than shrinking
    `chunk_wt` to make room.
    """
    assert not trid_prefetch_pays(
        1, blocks_per_core=16, chunk_row_bytes=1024, prefetch_cb_bytes=L1_CB_BUDGET_PREFETCH_BYTES + 1
    )
    assert trid_prefetch_pays(
        1, blocks_per_core=16, chunk_row_bytes=1024, prefetch_cb_bytes=L1_CB_BUDGET_PREFETCH_BYTES
    )


def test_b8_trids_are_distinct_and_not_the_firmware_default():
    """Two ids, neither of them 0.

    `noc_async_read_set_trid` writes the sticky NOC_PACKET_TAG register, and 0 is
    what the firmware leaves there. Using 0 as one of the pipeline's ids would make
    an untagged read from any other kernel indistinguishable from this block's.
    """
    assert len(set(PREFETCH_TRIDS)) == 2
    assert 0 not in PREFETCH_TRIDS
    assert all(0 < t <= 0xF for t in PREFETCH_TRIDS), "the trid field is 4 bits"


# ---------------------------------------------------------------------------
# B8 — the plan applies the gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,use_multicore,want_b8",
    [
        (B8_SINGLE_CORE, False, 2),  # 1 core, 16 blk, 1024 B -> core clause
        (B8_MULTI_CHUNK, False, 2),  # 1 core, 2 chunks x 4 rows -> core clause
        (B8_64B_MULTIBLK, True, 2),  # 64 cores, 4 blk, 64 B -> size clause
        (B8_128B_2BLK, True, 2),  # 64 cores, 2 blk, 128 B -> size clause
        (NO_B8_1BLK, True, 1),  # 1 blk/core -> structurally off
        (NO_B8_256B, True, 1),  # 256 B at 64 cores -> measured 1.013, off
    ],
    ids=["1core_1024B", "1core_multichunk", "64c_64B_4blk", "64c_128B_2blk", "64c_1blk", "64c_256B"],
)
def test_plan_applies_the_b8_gate(device, shape, use_multicore, want_b8):
    plan = _plan(device, shape, use_multicore=use_multicore)
    assert plan["prefetch_blocks"] == want_b8, (
        f"{shape}: ncores={plan['ncores']} blocks_per_core={plan['blocks_per_core']} "
        f"chunk_row_bytes={plan['chunk_row_bytes']} depth={plan['depth']} "
        f"got prefetch_blocks={plan['prefetch_blocks']}"
    )
    if want_b8 == 2:
        assert plan["depth"] == PREFETCH_DEPTH, "the prefetch needs its third window"
        assert plan["cb_bytes_per_core"] <= L1_CB_BUDGET_PREFETCH_BYTES


def test_b8_supersedes_b13_where_both_could_fire(device):
    """64 B / 4 blocks was B13's cell; B8 measured better, so B8 takes it.

    `x_tall_narrow_4blk_no_levers` 13 967 -> `x_tall_narrow_4blk_b13_only` 13 219
    (0.946) -> `x_tall_narrow_4blk_b8_forced` 11 258 (**0.806**). Both levers own
    the read command programming, so shipping both is not defined; the planner
    ships the measured winner.
    """
    plan = _plan(device, B8_64B_MULTIBLK)
    assert plan["prefetch_blocks"] == 2
    assert plan["stateful_read"] == 0, "B13 must yield to B8, not stack with it"
    assert plan["split_read"] == 0


def test_b8_never_fires_when_the_caller_pinned_the_depth(device):
    """`use_double_buffer=True/False` keep their documented meanings exactly.

    B8 wants a third window, which is neither "depth-2, +L1" nor "depth-1, minimal
    L1". Refinement 1's rule stands: only the *default* is gated, so a caller who
    pinned the depth gets the depth they asked for.
    """
    for request, want_depth in ((True, 2), (False, 1)):
        plan = _plan(device, B8_SINGLE_CORE, use_multicore=False, use_double_buffer=request)
        assert plan["depth"] == want_depth
        assert plan["prefetch_blocks"] == 1
    gated = _plan(device, B8_SINGLE_CORE, use_multicore=False)
    assert gated["prefetch_blocks"] == 2 and gated["depth"] == PREFETCH_DEPTH


def test_b8_and_c7_are_never_both_selected(device):
    """C7 hands BRISC the window NCRISC reserved and assumes exactly ONE live
    window at the CB base; B8 keeps two live windows at rotating addresses. Both
    at once would corrupt rather than fail loudly, so the exclusion is a
    static_assert in the reader AND a host invariant here."""
    for shape, mc in (
        (B8_64B_MULTIBLK, True),
        ((1, 1, 2048, 32), True),  # C7's own regime: 64 B, 1 block/core
        (B8_SINGLE_CORE, False),
        (NO_B8_1BLK, True),
    ):
        plan = _plan(device, shape, use_multicore=mc)
        assert not (plan["prefetch_blocks"] == 2 and plan["split_read"]), f"{shape} selected both B8 and C7"


def test_b8_off_on_the_zero_copy_path(device):
    """Path B moves no bytes over the NoC, so there is nothing to double-issue."""
    shape = (1, 1, 512, 64)
    cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (128, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    plan = _plan(device, shape, memory_config=cfg)
    assert plan["path"] == "alias"
    assert plan["prefetch_blocks"] == 1
    assert plan["vc_spread"] == 0
    assert plan["bank_placement"] == 0


def test_b8_off_on_a_multi_page_row_sharded_input(device):
    """`row_page_stride > 1` routes to the raw strided fallback, which the
    prefetch loop does not implement (it calls the one-page-per-row helper)."""
    shape = (1, 1, 2048, 512)
    cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            (256, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    # Sharded RM in -> DRAM-interleaved TILE out, so the generic path runs with the
    # sharded source's multi-page rows (the same-spec case would take the alias
    # path and have no reads at all).
    plan = _plan(device, shape, memory_config=cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert plan["path"] == "generic"
    assert plan["row_page_stride"] > 1
    assert plan["prefetch_blocks"] == 1


# ---------------------------------------------------------------------------
# B8 — bit-exactness of every new code path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,use_multicore",
    [
        (B8_SINGLE_CORE, False),
        (B8_MULTI_CHUNK, False),  # the chunk-outer x block-inner flatten
        (B8_64B_MULTIBLK, True),
        (B8_128B_2BLK, True),
        ((1, 1, 4096, 32), True),  # exactly 2 blocks/core: the shortest pipeline
        ((1, 1, 192, 96), False),  # awkward Wt=3 with 6 blocks on one core
        ((2, 3, 128, 64), False),  # rank-4 fold + multi-block
    ],
    ids=["1core_1024B", "multichunk", "64B_4blk", "128B_2blk", "2blk_min", "awkward_wt3", "rank4_fold"],
)
def test_b8_is_bit_exact(device, shape, use_multicore):
    """The prefetch computes CB window addresses itself, so a wrong window would
    silently write block i+1 over block i-1. arange input makes that visible."""
    plan = _plan(device, shape, use_multicore=use_multicore)
    assert plan["prefetch_blocks"] == 2, f"{shape} did not select B8 — this test would prove nothing"
    _roundtrip_exact(device, shape, use_multicore=use_multicore)


def test_b8_is_bit_exact_over_repeated_launches(device):
    """The trid tag and the CB base are re-derived per launch; a stale
    NOC_PACKET_TAG or a write pointer that did not start at the CB base would show
    up on the second call, not the first."""
    _roundtrip_exact(device, B8_MULTI_CHUNK, use_multicore=False, repeats=10)


def test_b8_is_bit_exact_on_l1_interleaved(device):
    """L1-interleaved source: 64 banks, a different accessor specialisation, and
    the reads are core-to-core rather than core-to-DRAM."""
    plan = _plan(device, B8_64B_MULTIBLK, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert plan["prefetch_blocks"] == 2
    _roundtrip_exact(device, B8_64B_MULTIBLK, memory_config=ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "b8_env,b10_env,a3_env",
    [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 2)],
    ids=["none", "b8", "b10", "a3", "all_forced"],
)
def test_bit_exact_for_every_refinement2_lever_combination(device, monkeypatch, b8_env, b10_env, a3_env):
    """All five combinations must be bit-exact, not only the shipped one.

    The bench measures the refuted levers through these same env switches, so a
    wrong result in a counterfactual row would otherwise be reported as a perf
    number rather than as a bug.
    """
    monkeypatch.setenv("TILIZE_LEVER_B8", str(b8_env))
    monkeypatch.setenv("TILIZE_LEVER_B10", str(b10_env))
    monkeypatch.setenv("TILIZE_LEVER_A3", str(a3_env))
    _roundtrip_exact(device, B8_MULTI_CHUNK, use_multicore=False)
    _roundtrip_exact(device, B8_64B_MULTIBLK)
    _roundtrip_exact(device, NO_B8_1BLK)


def test_b8_program_cache_hit(device):
    """The prefetch adds compile-time args only — no new runtime `Buffer*` arg —
    so a second identical call must still hit the program cache."""
    _roundtrip_exact(device, B8_MULTI_CHUNK, use_multicore=False)
    before = device.num_program_cache_entries()
    _roundtrip_exact(device, B8_MULTI_CHUNK, use_multicore=False)
    assert device.num_program_cache_entries() == before, "second identical call added a cache entry"


# ---------------------------------------------------------------------------
# B10 — refuted, and the gate that keeps it refuted
# ---------------------------------------------------------------------------


def test_b10_gate_is_off_everywhere_and_says_why():
    """Per-core static unicast VC is a measured regression. Do not re-enable.

    Ratio lever/none, 7 rounds x 10 launches (the write-half rows have CV 3.7-4.7 %
    because a saturated write VC is itself unstable — the effect is an order of
    magnitude larger than that):

        regime                  | reads only | writes only |   both
        ------------------------|------------|-------------|--------
        b_wide_short (64c)      |    1.105   |  **1.780**  | 1.991
        a_square     (64c)      |    1.083   |  **1.893**  | 2.142
        g_dram_to_sharded (64c) |     --     |     --      | 1.054
        d_tall_narrow (64c,1blk)|     --     |     --      | 1.006
        c_single_core (1c)      |     --     |     --      | 1.011

    Mechanism: the firmware picks ONE static VC deliberately (VC 1 for unicast,
    `dataflow_api_common.h:62`); rotating requests over VCs 0/2/3 SPLITS the per-VC
    buffering at the DRAM endpoint instead of pooling it, so each core's stream
    gets a fraction of the queue depth it had. tt-npe agrees before the fact: on
    `b_wide_short` the congestion impact is **0.4 %** (13 855 ideal vs 13 910
    congested cycles), so a congestion lever had at most 0.4 % available and
    ~78 % to lose.
    """
    assert VC_SPREAD_MIN_CORES > 64, "no current compute grid may enable B10"
    for cores in (1, 2, 4, 8, 16, 32, 64):
        assert not vc_spread_pays(cores)


def test_b10_bitmask_halves_are_independently_addressable():
    """The read half is a sticky NOC_CTRL program and the write half is a per-call
    field — two different mechanisms with two different measured costs, so the
    plan must be able to express either alone (that is how the ledger separated
    1.105 from 1.780)."""
    assert VC_SPREAD_READ == 1 and VC_SPREAD_WRITE == 2
    assert VC_SPREAD_READ & VC_SPREAD_WRITE == 0


def test_b10_vc_rotation_stays_inside_the_unicast_range(device, monkeypatch):
    """Unicast VCs are 0-3 (`noc_parameters.h`). A value outside that range would
    be silently masked into the wrong field of NOC_CTRL."""
    monkeypatch.setenv("TILIZE_LEVER_B10", "2")
    plan = _plan(device, NO_B8_1BLK)
    assert plan["vc_spread"] == (VC_SPREAD_READ | VC_SPREAD_WRITE)
    assert len(plan["read_vcs"]) == len(plan["work"])
    assert len(plan["write_vcs"]) == len(plan["work"])
    for vc in plan["read_vcs"] + plan["write_vcs"]:
        assert 0 <= vc < NUM_UNICAST_VCS
    # A core's read and write VCs must differ, or the two NoCs' streams collide on
    # the same static VC and the lever measures its own interference.
    assert all(r != w for r, w in zip(plan["read_vcs"], plan["write_vcs"]))


def test_b10_emits_a_default_vc_runtime_arg_even_when_off(device):
    """The runtime-arg layout must not depend on the lever, or turning it on
    would silently shift every later index."""
    plan = _plan(device, NO_B8_1BLK)
    assert plan["vc_spread"] == 0
    assert plan["read_vcs"] is None and plan["write_vcs"] is None


# ---------------------------------------------------------------------------
# A3 — refuted, and why it could not have worked
# ---------------------------------------------------------------------------


def test_a3_gate_is_off_everywhere_and_says_why():
    """Bank-adjacent work->core order: measured neutral, structurally inapplicable.

    Ratio lever/none: b_wide_short 1.017, a_square 1.003, d_tall_narrow 1.002 —
    at or inside the +-2 % noise floor, never a win.

    The structural reason is prior to the measurement, and it is the one to carry
    forward: a tilize block needs 32 CONSECUTIVE source pages, and an interleaved
    tensor puts page `p` in bank `p % NUM_DRAM_BANKS` (12 on WH B0). So EVERY core
    necessarily reads from all 12 banks and there is no core<->bank affinity for a
    placement to exploit — A3 can only change average hop count on a grid that the
    A0 split already fills. tt-npe puts the whole congestion term at 0.4 %.
    """
    assert BANK_PLACEMENT_MIN_CORES > 64, "no current compute grid may enable A3"
    for cores in (1, 4, 16, 64):
        assert not bank_placement_pays(cores)


def test_a3_forced_still_covers_the_tensor_exactly(device, monkeypatch):
    """A permutation must be a permutation: same core SET, same work units, each
    core exactly once. A duplicate would double-write a tile and drop another."""
    monkeypatch.setenv("TILIZE_LEVER_A3", "2")
    for shape in (NO_B8_1BLK, (1, 1, 2048, 2048), (1, 1, 96, 96)):
        plan = _plan(device, shape)
        assert plan["bank_placement"] == 1
        cores = [(int(c.x), int(c.y)) for c in plan["cores"]]
        assert len(cores) == len(set(cores)) == plan["ncores"], f"{shape}: A3 order is not a permutation"
        assert len(plan["work"]) == plan["ncores"]
        work_cores = [(int(u["core"].x), int(u["core"].y)) for u in plan["work"]]
        assert sorted(work_cores) == sorted(cores)
        # Every tile-row range x chunk range is covered exactly once.
        covered = sum(u["row_count"] * u["chunk_count"] for u in plan["work"])
        assert covered * plan["chunk_wt"] == plan["total_tiles"]


def test_a3_uses_the_same_core_set_as_the_default_on_a_full_grid(device, monkeypatch):
    """On the full grid A3 must change only the work->core MAPPING, so the CB and
    kernel placement (the `core_ranges`) stay byte-identical to the counterfactual
    and the bench row measures the permutation and nothing else."""
    baseline = _plan(device, NO_B8_1BLK)
    monkeypatch.setenv("TILIZE_LEVER_A3", "2")
    permuted = _plan(device, NO_B8_1BLK)
    assert baseline["ncores"] == permuted["ncores"] == 64
    assert {(int(c.x), int(c.y)) for c in baseline["cores"]} == {(int(c.x), int(c.y)) for c in permuted["cores"]}
    assert str(baseline["core_ranges"]) == str(permuted["core_ranges"])
    assert [int(c.x) for c in baseline["cores"]] != [int(c.x) for c in permuted["cores"]] or [
        int(c.y) for c in baseline["cores"]
    ] != [int(c.y) for c in permuted["cores"]], "A3 did not actually reorder anything"


# ---------------------------------------------------------------------------
# Cross-lever invariants
# ---------------------------------------------------------------------------


def test_refuted_levers_cost_nothing_when_off(device):
    """A refuted lever must leave the shipped plan untouched — no CB bytes, no
    extra semaphore, no depth change. This is what makes keeping the code (for
    re-measurability) free rather than a tax."""
    for shape, mc in ((NO_B8_1BLK, True), (B8_SINGLE_CORE, False), ((1, 1, 2048, 32), True)):
        plan = _plan(device, shape, use_multicore=mc)
        assert plan["vc_spread"] == 0
        assert plan["bank_placement"] == 0


def test_per_core_cb_bytes_stay_bounded_by_a_constant_in_w(device):
    """B8 raises the ceiling from 2 to 3 windows but the bound is still a constant
    in W — `PROPERTIES["bounded_cb"]`'s claim, re-checked with the new depth."""
    for w in (32, 256, 2048, 16384):
        plan = _plan(device, (1, 1, 4096, w))
        assert (
            plan["cb_bytes_per_core"] <= L1_CB_BUDGET_PREFETCH_BYTES
        ), f"W={w}: {plan['cb_bytes_per_core']} B/core with depth={plan['depth']} chunk={plan['chunk_wt']}"
        assert plan["chunk_wt"] <= tpd.WT_CHUNK_MAX
