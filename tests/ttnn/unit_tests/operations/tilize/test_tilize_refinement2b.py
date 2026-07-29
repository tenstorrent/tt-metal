# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2b — `b_wide_short`'s 64-way partial-page fan-in.

Two levers, one refuted and one shipped. Both are tested, because a refuted
lever's *gate* is exactly as load-bearing as a shipped one: it is the thing that
stops the next implementer re-enabling a measured 1.38x regression.

* **`fanin_mode` — whole-page staged read + L1 redistribution (REFUTED).**
  The entry's named algorithm: decouple "which bytes a core reads" from "which
  tiles a core owns" so the DRAM sees 32 whole-page reads instead of 2 048
  partial-page reads. Implemented in full (3 phases: one contiguous
  `piece_bytes` read per core -> an all-to-all posted-atomic ready handshake
  inside a 32-core group -> each core pulls its own slice out of every
  group-mate's staging buffer) and **bit-exact**, so the refutation is a *perf*
  verdict on a working implementation, not an implementation failure.

  It is refuted twice over, by the entry's own numbers:

  1. the **read-side ceiling probe** (`fanin_mode == 2`: phase 1 only, straight
     into the CB, no exchange) moves the same bytes with the same cores in ONE
     transaction instead of 32 -- and the one-sided DM ablation says the read leg
     costs **5 985 ns vs the baseline's 5 966 ns**, i.e. a 32x bigger transaction
     buys **zero** DRAM time. The entry's premise -- that a partial-page fan-in
     costs DRAM bandwidth -- is false on this hardware.
  2. the full algorithm measures **18 574 ns vs 13 461 ns = 1.380x SLOWER**: the
     L1 exchange leg is +4 676 ns and its 32-core barrier +1 217 ns of sync.

* **`stagger` — per-core transaction-order rotation (SHIPPED).**
  Discovered by this entry's decomposition. An interleaved tensor puts page `p`
  in bank `p % NUM_DRAM_BANKS`, and with `nt_h == 1` every core reads the SAME 32
  source pages in the SAME order, so at issue step `r` all 64 cores hit ONE bank
  while the other 11 idle. Rotating each core's issue order by its work-unit
  index spreads that. Pure index permutation -- same transactions, same count,
  same size, same L1 addresses, zero extra L1 or state.

  Measured **0.929 / 0.894 / 0.938** on the 16k / 8k / 32k wide-short members and
  0.991-1.019 (neutral) everywhere else. The two halves are **superadditive**
  (0.992 and 0.985 alone vs 0.929 together), which is why they ship as one gate.

tt-npe closes the "why is 1.14x unreachable" question: after the stagger,
`b_wide_short` runs at **116.5 % of the model's DRAM bandwidth** with a **0.2 %**
congestion term (12 714 golden cycles, down from 14 477 at 102.3 % / 0.7 %). It is
past its achievable DRAM bound, and there is no congestion left for a placement or
VC lever to recover.

Correctness is `torch.equal` throughout (tilize is value-preserving), on `arange`
inputs rather than `randn`: every element is unique, so a stick pulled from the
wrong group-mate -- or a rotation that permuted the DATA instead of the ISSUE
ORDER -- cannot cancel out.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import (
    CB_STAGE,
    FANIN_GROUP_ROWS,
    FANIN_MIN_READ_BYTES,
    L1_CB_BUDGET_FANIN_BYTES,
    SEM_FANIN_READY,
    STAGGER_MIN_CHUNK_WT,
    STAGGER_READ,
    STAGGER_WRITE,
    TILE_HW,
    build_plan,
    create_program_descriptor,
    fanin_pays,
    stagger_pays,
)

# Shapes, with the plan each one lands on (grid 8x8, bf16). All the wide-short
# members have `nt_h == 1` and therefore exactly ONE chunk-block per core, which is
# both the fan-in condition and the structural precondition of `fanin_mode`.
#   WS_4K   [1,1,32,4096]   64 cores, chunk  2, 128 B  -> stagger OFF (chunk clause)
#   WS_8K   [1,1,32,8192]   64 cores, chunk  4, 256 B  -> stagger ON, measured 0.894
#   WS_16K  [1,1,32,16384]  64 cores, chunk  8, 512 B  -> stagger ON, measured 0.934
#   WS_32K  [1,1,32,32768]  64 cores, chunk 16,1024 B  -> stagger ON, measured 0.938
#   WS_2ROW [1,1,64,16384]  64 cores, chunk 16,1024 B, nt_h=2 -> OFF (0.991)
#   SQUARE  [1,1,2048,2048] 64 cores, chunk 16, n_w = 1       -> OFF (1.006)
#   TALL    [1,1,2048,32]   64 cores, chunk  1, n_w = 1       -> OFF (1.005)
WS_4K = (1, 1, 32, 4096)
WS_8K = (1, 1, 32, 8192)
WS_16K = (1, 1, 32, 16384)
WS_32K = (1, 1, 32, 32768)
WS_2ROW = (1, 1, 64, 16384)
SQUARE = (1, 1, 2048, 2048)
TALL = (1, 1, 2048, 32)


def _levers(monkeypatch, **kwargs):
    """Set the env counterfactual switches the planner reads (default 1 == gated)."""
    for key in ("b13", "c7", "b8", "b10", "a3", "r2b", "stg"):
        monkeypatch.setenv(f"TILIZE_LEVER_{key.upper()}", str(kwargs.get(key, 1)))


def _plan(device, shape, *, dtype=ttnn.bfloat16, use_multicore=True, use_double_buffer=None, memory_config=None):
    torch.manual_seed(0)
    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch.randn(shape).bfloat16(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
    )
    tt_output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, mem)
    return build_plan(tt_input, tt_output, device, use_multicore=use_multicore, use_double_buffer=use_double_buffer)


def _roundtrip_exact(device, shape, *, memory_config=None, use_multicore=True, use_double_buffer=None, repeats=1):
    """tilize is value-preserving: assert bit-exactness, not a tolerance.

    `arange` (not `randn`) so every element is unique -- a stick pulled from the
    wrong group-mate, or a rotation that moved the data instead of the issue order,
    shows up instead of averaging out.
    """
    n = 1
    for d in shape:
        n *= d
    torch_input = (torch.arange(n, dtype=torch.float32) % 8192.0).reshape(shape).bfloat16()
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
# The REFUTED lever — `fanin_mode` (whole-page read + L1 redistribution)
# ---------------------------------------------------------------------------


def test_fanin_gate_is_off_everywhere_and_says_why():
    """The gate is identity-false, and the constant carries the measurement.

    Measured on `b_wide_short` (in-run A/B, 7 rounds x 10 launches, CV <= 1.6 %):

        off (32 x 512 B strided reads)        13 461 ns   1.000
        PROBE: 1 x 16 384 B read, no exchange 12 736 ns   0.946  <- the CEILING
        full 3-phase redistribution           18 574 ns   1.380

    and the one-sided DM ablation says the read leg is **5 966 -> 5 985 ns**, i.e.
    unchanged: a 32x bigger transaction buys zero DRAM time. Lowering
    ``FANIN_MIN_READ_BYTES`` to re-enable this fails here.
    """
    assert FANIN_MIN_READ_BYTES > 1 << 20, (
        "the fan-in redistribution is a MEASURED 1.380x regression on its own target "
        "regime, and its read-side probe shows the whole-page read buys zero DRAM "
        "time. Do not lower FANIN_MIN_READ_BYTES without re-running "
        "_bench_tilize.py's p_/x_wide_short_r2b_{off,probe,forced} triple."
    )
    # 1024 B is the widest read any reachable plan produces (WT_CHUNK_MAX=16, bf16).
    for chunk_row_bytes in (64, 128, 256, 512, 1024, 2048):
        for groups in (1, 2, 4):
            assert not fanin_pays(chunk_row_bytes, groups)


@pytest.mark.parametrize("shape", [WS_4K, WS_8K, WS_16K, SQUARE, TALL])
def test_plan_never_selects_the_fanin_path_by_default(device, shape):
    plan = _plan(device, shape)
    assert plan["fanin_mode"] == 0, f"{shape} selected the refuted fan-in path"
    assert plan["piece_bytes"] == 0
    assert plan["fanin_group_axes"] is None


def test_fanin_costs_nothing_when_off(device):
    """A refuted lever must not leave its staging CB or semaphore behind."""
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn(WS_16K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(WS_16K)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = build_plan(tt_input, tt_output, device)
    descriptor = create_program_descriptor(tt_input, tt_output, plan)
    assert len(descriptor.cbs) == 2, "the staging CB is allocated with the lever off"
    assert [f.buffer_index for cb in descriptor.cbs for f in cb.format_descriptors] != [CB_STAGE]
    assert len(descriptor.semaphores) == 0, "the group-ready semaphore is declared with the lever off"


@pytest.mark.parametrize("shape", [WS_4K, WS_8K, WS_16K])
def test_forced_fanin_is_structurally_wired(device, monkeypatch, shape):
    """Forced past its gate, the plan must be internally consistent.

    The refuted implementation is kept so the verdict stays re-measurable, so its
    structure is pinned: one group per source piece, a piece exactly
    ``FANIN_GROUP_ROWS`` chunks wide, a staging CB, and one semaphore.
    """
    _levers(monkeypatch, r2b=2)
    plan = _plan(device, shape)
    assert plan["fanin_mode"] == 1
    assert plan["fanin_group_rows"] == FANIN_GROUP_ROWS == TILE_HW
    assert plan["ncores"] % FANIN_GROUP_ROWS == 0
    assert plan["fanin_groups"] == plan["ncores"] // FANIN_GROUP_ROWS
    # The piece is what makes the read whole-page-shaped: 32 owners' slices wide.
    assert plan["piece_bytes"] == FANIN_GROUP_ROWS * plan["chunk_row_bytes"]
    # ... and `fanin_groups` pieces tile the source row exactly.
    assert plan["fanin_groups"] * plan["piece_bytes"] == plan["width"] * plan["elem_in"]
    # Bounded by a constant in W, which is what keeps PROPERTIES["bounded_cb"] true.
    assert plan["cb_bytes_per_core"] <= L1_CB_BUDGET_FANIN_BYTES
    assert len(plan["fanin_group_axes"]) == plan["fanin_groups"]
    for xs, ys in plan["fanin_group_axes"]:
        assert len(xs) * len(ys) == FANIN_GROUP_ROWS


def test_forced_fanin_declares_its_cb_and_semaphore(device, monkeypatch):
    _levers(monkeypatch, r2b=2)
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn(WS_16K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(WS_16K)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = build_plan(tt_input, tt_output, device)
    descriptor = create_program_descriptor(tt_input, tt_output, plan)
    indices = [f.buffer_index for cb in descriptor.cbs for f in cb.format_descriptors]
    assert CB_STAGE in indices, "the staging buffer must be a declared CB, not scratch L1"
    assert len(descriptor.semaphores) == 1
    assert descriptor.semaphores[0].id == SEM_FANIN_READY
    assert descriptor.semaphores[0].initial_value == 0, "the group counter must be re-zeroed every launch"


@pytest.mark.parametrize("shape", [WS_4K, WS_8K, WS_16K])
def test_forced_fanin_is_bit_exact(device, monkeypatch, shape):
    """The refutation is a PERF verdict on a WORKING implementation.

    The 32-way L1 gather has to reproduce the strided DRAM reader's stick order
    exactly (stick r at offset r * chunk_row_bytes, pulled from group-mate r). If it
    did not, the refutation would be indistinguishable from a bug.
    """
    _levers(monkeypatch, r2b=2)
    _roundtrip_exact(device, shape, repeats=3)


def test_forced_fanin_probe_is_off_by_default(device, monkeypatch):
    """`fanin_mode == 2` produces GARBAGE; it must never be reachable by default."""
    _levers(monkeypatch)  # all gated
    assert _plan(device, WS_16K)["fanin_mode"] == 0
    _levers(monkeypatch, r2b=3)
    assert _plan(device, WS_16K)["fanin_mode"] == 2, "the bench's read-ceiling probe is unreachable"


@pytest.mark.parametrize(
    "shape,use_multicore",
    [(SQUARE, True), (TALL, True), (WS_16K, False)],
)
def test_fanin_stays_off_where_it_is_structurally_impossible(device, monkeypatch, shape, use_multicore):
    """Even forced, the structural preconditions hold.

    `nt_h > 1` or more than one chunk-block per core means the single staging window
    would need per-block flow control, and one core has no group.
    """
    _levers(monkeypatch, r2b=2)
    plan = _plan(device, shape, use_multicore=use_multicore)
    assert plan["fanin_mode"] == 0


def test_fanin_and_the_other_read_levers_are_mutually_exclusive(device, monkeypatch):
    """The fan-in path REPLACES the stick reads, so B13/C7/B8 must yield.

    The reader carries a `static_assert` for this; the host must never build a plan
    that would trip it.
    """
    _levers(monkeypatch, r2b=2, b13=2, c7=2, b8=2)
    plan = _plan(device, WS_16K)
    if plan["fanin_mode"]:
        assert plan["stateful_read"] == 0
        assert plan["split_read"] == 0
        assert plan["prefetch_blocks"] == 1
        assert plan["stagger"] == 0


# ---------------------------------------------------------------------------
# The SHIPPED lever — `stagger` (per-core transaction-order rotation)
# ---------------------------------------------------------------------------


def test_stagger_gate_pinned_to_its_sweep():
    """The gate's three clauses, each pinned to the measurement that set it.

    In-run A/B, 7 rounds x 10 launches, CV <= 1.6 % (ratio = stagger / off):

        shape            | nt_h | n_w | chunk |    off ns |    stg ns | ratio
        -----------------|------|-----|-------|-----------|-----------|-------
        [1,1,32,4096]    |    1 |  64 |     2 |     4 989 |     4 972 | 0.997
        [1,1,32,8192]    |    1 |  64 |     4 |     8 046 |     7 194 | 0.894
        [1,1,32,16384]   |    1 |  64 |     8 |    13 433 |    12 543 | 0.934
        [1,1,32,32768]   |    1 |  64 |    16 |    25 394 |    23 820 | 0.938
        [1,1,64,16384]   |    2 |  32 |    16 |    24 669 |    24 447 | 0.991
        a_square         |   64 |   1 |    16 |    86 058 |    86 591 | 1.006
        d_tall_narrow    |   64 |   1 |     1 |     3 609 |     3 627 | 1.005
        g_dram_to_sharded|   64 |   1 |    16 |    19 049 |    19 402 | 1.019
        e_square_fp32    |   64 |   1 |     8 |   182 908 |   183 178 | 1.001
    """
    # nt_h == 1 with a wide enough chunk -> both halves.
    both = STAGGER_READ | STAGGER_WRITE
    assert stagger_pays(64, 1, 4) == both, "measured 0.894 at chunk 4"
    assert stagger_pays(64, 1, 8) == both, "measured 0.934 at chunk 8"
    assert stagger_pays(64, 1, 16) == both, "measured 0.938 at chunk 16"
    # chunk 2 is measured neutral (0.997) -> off.
    assert stagger_pays(64, 1, 2) == 0
    assert STAGGER_MIN_CHUNK_WT == 4
    # nt_h == 2 halves the clustering (only 32 of 64 cores share a page set) and the
    # win is gone (0.991) -> off.
    assert stagger_pays(64, 2, 16) == 0
    # n_w == 1 regimes reach here as nt_h >= grid_cores; nothing to de-cluster.
    assert stagger_pays(64, 64, 16) == 0
    # one core cannot cluster against itself.
    assert stagger_pays(1, 1, 16) == 0


def test_stagger_ships_both_halves_because_they_are_superadditive():
    """Neither half is worth much alone; together they are worth 7-11 %.

        shape            | chunk | read only | write only | BOTH
        -----------------|-------|-----------|------------|-------
        [1,1,32,16384]   |     8 |   0.992   |   0.985    | 0.929
        [1,1,32,8192]    |     4 |   0.993   |   0.924    | 0.897

    The instantaneous demand on a bank is read demand PLUS write demand, so
    spreading only one stream leaves the busiest bank -- which is what sets the
    time -- almost where it was. Returning a single half here would ship ~1 % of a
    7 % win, which is why the gate is all-or-nothing.
    """
    for chunk_wt in (4, 8, 16):
        got = stagger_pays(64, 1, chunk_wt)
        assert got == STAGGER_READ | STAGGER_WRITE, (
            "the halves are measured superadditive (0.992 / 0.985 alone vs 0.929 "
            "together); do not gate them independently"
        )


@pytest.mark.parametrize(
    "shape,want",
    [
        (WS_8K, STAGGER_READ | STAGGER_WRITE),
        (WS_16K, STAGGER_READ | STAGGER_WRITE),
        (WS_32K, STAGGER_READ | STAGGER_WRITE),
        (WS_4K, 0),
        (WS_2ROW, 0),
        (SQUARE, 0),
        (TALL, 0),
    ],
)
def test_plan_applies_the_stagger_gate(device, shape, want):
    plan = _plan(device, shape)
    assert plan["stagger"] == want, (
        f"{shape}: stagger={plan['stagger']} want={want} "
        f"(nt_h={plan['nt_h']}, n_w={plan['n_w']}, chunk_wt={plan['chunk_wt']})"
    )


def test_stagger_write_half_is_masked_when_there_is_one_page_per_block(device, monkeypatch):
    """With `chunk_wt == 1` there is nothing to rotate on the write side.

    Reporting it as on would make the bench column and the ledger lie about what the
    kernel does.
    """
    _levers(monkeypatch, stg=2, b13=0, c7=0, b8=0)
    plan = _plan(device, TALL)
    assert plan["chunk_wt"] == 1
    assert plan["stagger"] & STAGGER_WRITE == 0


def test_stagger_is_off_on_the_zero_copy_path(device, monkeypatch):
    """Path B has no NoC traffic at all, so there is no issue order to rotate."""
    _levers(monkeypatch, stg=2)
    shard = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (128, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    plan = _plan(device, (1, 1, 512, 64), memory_config=shard)
    assert plan["path"] == "alias"
    assert plan["stagger"] == 0


def test_stagger_never_coexists_with_a_lever_that_owns_the_row_loop(device, monkeypatch):
    """B8 / C7 / B13 each reshape the same 32 stick reads the rotation reorders.

    The reader carries a `static_assert`; this is the host-side guarantee that no
    plan can trip it. Every lever is forced at once, which is the adversarial case.
    """
    _levers(monkeypatch, stg=2, b13=2, c7=2, b8=2, r2b=2)
    for shape in (WS_4K, WS_8K, WS_16K, SQUARE, TALL, (1, 1, 8192, 32), (1, 1, 4096, 64)):
        plan = _plan(device, shape)
        if plan["stagger"]:
            assert plan["split_read"] == 0, shape
            assert plan["prefetch_blocks"] == 1, shape
            assert plan["stateful_read"] == 0, shape
            assert plan["fanin_mode"] == 0, shape


def test_stagger_costs_no_l1(device, monkeypatch):
    """A pure index permutation must not move the per-core CB footprint."""
    _levers(monkeypatch, stg=0)
    off = _plan(device, WS_16K)
    _levers(monkeypatch, stg=2)
    on = _plan(device, WS_16K)
    assert on["cb_bytes_per_core"] == off["cb_bytes_per_core"]
    assert on["chunk_wt"] == off["chunk_wt"], "the rotation must not change the transaction shape"
    assert on["depth"] == off["depth"]
    assert on["ncores"] == off["ncores"]
    assert on["blocks_per_core"] == off["blocks_per_core"]


@pytest.mark.parametrize("shape", [WS_4K, WS_8K, WS_16K, WS_2ROW, TALL, (1, 1, 96, 96), (2, 3, 128, 64)])
def test_stagger_is_bit_exact(device, monkeypatch, shape):
    """The rotation moves the ISSUE ORDER, never the (page, L1 address) pairing.

    An `arange` input makes every element unique, so a rotation that permuted the
    data would show as a mismatch rather than averaging out. Forced past the gate so
    the shapes the gate excludes are covered too.
    """
    _levers(monkeypatch, stg=2, b13=0, c7=0, b8=0)
    _roundtrip_exact(device, shape, repeats=2)


@pytest.mark.parametrize("rot_mod", [None, 12])
def test_stagger_is_bit_exact_for_every_rotation_modulus(device, monkeypatch, rot_mod):
    """The sweep hook that measured TILE_HW vs NUM_DRAM_BANKS must stay correct.

    Rejected on perf (12 673 vs 12 480 ns, +1.5 %), but the hook stays for
    re-measurement, so it has to keep producing the right answer.
    """
    _levers(monkeypatch, stg=2)
    monkeypatch.setattr(tpd, "STAGGER_MOD_OVERRIDE", rot_mod)
    _roundtrip_exact(device, WS_16K)


def test_stagger_is_bit_exact_on_l1_interleaved(device, monkeypatch):
    _levers(monkeypatch, stg=2, b13=0, c7=0, b8=0)
    _roundtrip_exact(device, WS_8K, memory_config=ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize("stg_env", [0, 1, 2, 3, 4])
def test_bit_exact_for_every_stagger_bitmask(device, monkeypatch, stg_env):
    """off / gated / both / read-only / write-only -- all four halves and the gate."""
    _levers(monkeypatch, stg=stg_env, b13=0, c7=0, b8=0)
    _roundtrip_exact(device, WS_16K)


def test_stagger_is_bit_exact_over_repeated_launches(device, monkeypatch):
    """Guards the cached-program path: runtime args are re-patched, CTs are not."""
    _levers(monkeypatch, stg=2)
    _roundtrip_exact(device, WS_16K, repeats=5)


def test_stagger_program_cache_hit(device, monkeypatch):
    _levers(monkeypatch, stg=2)
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn(WS_16K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tilize(tt_input)
    before = device.num_program_cache_entries()
    tilize(tt_input)
    assert device.num_program_cache_entries() == before, "the rotation broke program caching"


def test_stagger_rotations_cover_every_offset_exactly(device, monkeypatch):
    """The host's rotation is a PERMUTATION of the work units, not a subset.

    `row_rot = index % TILE_HW` and `col_rot = index % chunk_wt` must together
    de-cluster without dropping or duplicating a work unit -- and with 64 cores over
    32 rows each rotation value must be used exactly twice, which is what makes the
    starting banks uniform.
    """
    _levers(monkeypatch, stg=2)
    plan = _plan(device, WS_16K)
    assert plan["ncores"] == 64 and plan["chunk_wt"] == 8
    row_rots = [i % TILE_HW for i in range(plan["ncores"])]
    col_rots = [i % plan["chunk_wt"] for i in range(plan["ncores"])]
    assert sorted(set(row_rots)) == list(range(TILE_HW))
    assert all(row_rots.count(r) == plan["ncores"] // TILE_HW for r in set(row_rots))
    assert sorted(set(col_rots)) == list(range(plan["chunk_wt"]))
    # ... and the work units still tile the tensor exactly (the rotation is inside a
    # block, so it cannot change the cover -- assert it rather than assume it).
    covered = set()
    for unit in plan["work"]:
        for r in range(unit["row_start"], unit["row_start"] + unit["row_count"]):
            for c in range(unit["chunk_start"], unit["chunk_start"] + unit["chunk_count"]):
                assert (r, c) not in covered
                covered.add((r, c))
    assert len(covered) == plan["nt_h"] * (plan["wt"] // plan["chunk_wt"])


def test_a0_and_bounded_cb_survive_both_levers(device, monkeypatch):
    """Neither lever may change the active-core count or unbound the CB.

    A0 is the run's hard rule for the wide-short regime (it MUST fill the grid), and
    `PROPERTIES["bounded_cb"]` is a declared property; the fan-in path adds a staging
    window, so its footprint is checked against its own budget at four widths.
    """
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    for width in (4096, 8192, 16384, 32768):
        shape = (1, 1, 32, width)
        _levers(monkeypatch, stg=2)
        stg_plan = _plan(device, shape)
        _levers(monkeypatch, stg=0, r2b=2)
        fanin_plan = _plan(device, shape)
        for plan, budget in ((stg_plan, tpd.L1_CB_BUDGET_BYTES), (fanin_plan, L1_CB_BUDGET_FANIN_BYTES)):
            assert plan["ncores"] == min(grid_cores, plan["total_tiles"], tpd.A0_KNEE_CORES), shape
            assert plan["cb_bytes_per_core"] <= budget, (shape, plan["cb_bytes_per_core"])
