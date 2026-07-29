# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1c — the two sub-one-packet read-path levers.

* **B13 `stateful_read`** — `noc_async_read_set_state` / `noc_async_read_with_state`
  inside `dataflow_kernel_lib::read_stick_rows_for_tilize`.
  `set_state` pins the NoC *coordinate*, so the rows of a block are visited
  **bank-major**: for an interleaved tensor pages `p` and `p + num_banks` share a
  bank exactly one aligned page apart, so one armed command covers a whole group
  and its source address is a running increment (one accessor call per bank
  instead of one per row).

* **C7 `split_read`** — BRISC (the writer kernel) takes half of each block's 32
  stick reads. NCRISC stays the *only* producer of `cb_rm_input`; the reserved
  window is handed over with two monotonic local counting semaphores.

Both are **gated on the measured read-transaction size**, and they are **mutually
exclusive** — also measured. Outside the 64-128 B regime each costs real time (up
to +19.9 %), and inside it C7 wins at one block per core while B13 wins from two
blocks on, so every plan ships exactly one of the two. The gates below are
therefore load-bearing and every threshold in them is pinned to the number that
set it; the sweep lives in `changelog.md` § "Refinement 1c" and in the comment
above `STATEFUL_READ_MAX_ROW_BYTES` in the planner.

Correctness is `torch.equal` throughout (tilize is value-preserving), so every
new path is proven **bit-exact**, not merely fast. The B13 address identity is
additionally re-derived through the accessor by an `ASSERT` inside the helper, so
a `--dev` run of this file checks it per read on device.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import (
    SPLIT_READ_MAX_BLOCKS_PER_CORE,
    SPLIT_READ_MAX_ROW_BYTES,
    STATEFUL_READ_MAX_ROW_BYTES,
    build_plan,
    split_read_pays,
    stateful_read_pays,
)

# Read bytes per row = chunk_wt * 32 * elem_size, and the planner picks chunk_wt
# to fill the grid, so W selects the read size on an nt_h == 1 shape:
#   W=32   -> chunk 1 -> 64 B    (C7 at 1 block/core, B13 from 2 blocks on)
#   W=4096 -> chunk 2 -> 128 B   (B13)
#   W=8192 -> chunk 4 -> 256 B   (neither)
C7_REGIME = (1, 1, 2048, 32)  # nt_h=64, Wt=1, 1 block/core
B13_ONLY_MULTIBLOCK = (1, 1, 8192, 32)  # nt_h=256, 4 blocks/core
B13_ONLY_128B = (1, 1, 32, 4096)  # nt_h=1, chunk 2
NO_LEVERS_256B = (1, 1, 32, 8192)  # nt_h=1, chunk 4
NO_LEVERS_512B = (1, 1, 32, 16384)  # nt_h=1, chunk 8


def _plan(device, shape, *, dtype=ttnn.bfloat16, use_multicore=True, use_double_buffer=None, memory_config=None):
    torch.manual_seed(0)
    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
    )
    tt_output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, mem)
    return build_plan(tt_input, tt_output, device, use_multicore=use_multicore, use_double_buffer=use_double_buffer)


def _roundtrip_exact(device, shape, *, memory_config=None, use_multicore=True, use_double_buffer=None, seed=0):
    """tilize is value-preserving: assert bit-exactness, not a tolerance."""
    torch.manual_seed(seed)
    # arange, not randn: every element is unique, so a permuted / misplaced row
    # (exactly what a wrong bank-major address or a wrong split half produces)
    # cannot cancel out.
    n = 1
    for d in shape:
        n *= d
    torch_input = torch.arange(n, dtype=torch.float32).reshape(shape).bfloat16()
    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
    )
    tt_output = tilize(tt_input, memory_config, use_multicore=use_multicore, use_double_buffer=use_double_buffer)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    result = ttnn.to_torch(tt_output)
    assert torch.equal(result.float(), torch_input.float()), (
        f"not bit-exact on {shape}: "
        f"{(result.float() - torch_input.float()).abs().max()} max abs, "
        f"{(result.float() != torch_input.float()).sum()} mismatching elements"
    )


# ---------------------------------------------------------------------------
# The two gates, pinned to the measurements that set them
# ---------------------------------------------------------------------------


def test_b13_gate_pinned_to_its_sweep():
    """B13 pays at 64/128 B and loses from 256 B up — the turnover is measured.

    B13-alone/none ratio, 64 cores, 7 rounds x 10 launches (CV <= 2.1 %):
        64 B  0.978  ([1,1,2048,32], 1 blk)   0.957  ([1,1,8192,32], 4 blk)
       128 B  0.950  ([1,1,32,4096])
       256 B  1.023  ([1,1,32,8192])
       512 B  1.199  ([1,1,32,16384])   <- worst
      1024 B  1.057  ([1,1,2048,512] -> BLOCK sharded)

    `set_state` pins the NoC coordinate, so the lever REQUIRES bank-major issue
    order; past 128 B the DRAM-endpoint serialization of 2-3 consecutive
    same-bank reads costs more than the saved command programming buys. Raising
    this threshold without re-running the sweep re-introduces a +17 % regression
    on `b_wide_short`.
    """
    assert STATEFUL_READ_MAX_ROW_BYTES == 128
    assert stateful_read_pays(64)
    assert stateful_read_pays(128)
    assert not stateful_read_pays(256)
    assert not stateful_read_pays(512)
    assert not stateful_read_pays(1024)


def test_c7_gate_pinned_to_its_sweep():
    """C7 pays only at 64 B with one block per core.

    Three clauses, each measured:
      * depth == 1 is STRUCTURAL — BRISC writes into NCRISC's reserved window
        without touching the CB pointers, so the window must be the CB base.
      * blocks_per_core == 1: 0.956 at 1 block vs 1.145 at 4 blocks — the split
        spends the read/write overlap across the block boundary.
      * <= 64 B: 0.956 at 64 B, then 1.018 / 1.056 / 1.146 / 1.045 at
        128 / 256 / 512 / 1024 B.
    """
    assert SPLIT_READ_MAX_ROW_BYTES == 64
    assert SPLIT_READ_MAX_BLOCKS_PER_CORE == 1
    assert split_read_pays(depth=1, blocks_per_core=1, chunk_row_bytes=64)
    # depth-2 is refused structurally, whatever the other two say
    assert not split_read_pays(depth=2, blocks_per_core=1, chunk_row_bytes=64)
    # a second block costs more than the split buys
    assert not split_read_pays(depth=1, blocks_per_core=2, chunk_row_bytes=64)
    assert not split_read_pays(depth=1, blocks_per_core=4, chunk_row_bytes=64)
    # bigger reads are DRAM-service bound
    assert not split_read_pays(depth=1, blocks_per_core=1, chunk_row_bytes=128)
    assert not split_read_pays(depth=1, blocks_per_core=1, chunk_row_bytes=1024)


# ---------------------------------------------------------------------------
# The plan actually applies the gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,want_b13,want_c7",
    [
        (C7_REGIME, 0, 1),  # 64 B, 1 block/core -> C7 only (B13 does not pay on top)
        # 64 B, 4 blocks/core: was B13's cell until Refinement 2 measured lever B8
        # (trid double-issue) at 0.782 there vs B13's 0.925 on the same bench row
        # (`x_tall_narrow_4blk_no_levers` 14 319 -> `x_tall_narrow_4blk_b8_forced`
        # 11 197 vs `x_tall_narrow_4blk_b13_only` 13 220). B8 needs >= 2 blocks and
        # supersedes B13 wherever it fires, so BOTH R1c levers are off here now.
        (B13_ONLY_MULTIBLOCK, 0, 0),
        (B13_ONLY_128B, 1, 0),  # 128 B, 1 block/core -> C7 off, B8 needs 2 blocks
        (NO_LEVERS_256B, 0, 0),  # 256 B -> both off
        (NO_LEVERS_512B, 0, 0),  # 512 B -> both off
    ],
    ids=["64B_1blk", "64B_4blk", "128B", "256B", "512B"],
)
def test_plan_lever_selection_per_read_size(device, shape, want_b13, want_c7):
    plan = _plan(device, shape)
    assert (plan["stateful_read"], plan["split_read"]) == (want_b13, want_c7), (
        f"{shape}: chunk_row_bytes={plan['chunk_row_bytes']} "
        f"blocks_per_core={plan['blocks_per_core']} depth={plan['depth']} "
        f"got B13={plan['stateful_read']} C7={plan['split_read']}"
    )


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 2048, 32), (1, 1, 8192, 32), (1, 1, 32, 4096), (1, 1, 32, 8192), (1, 1, 2048, 2048)],
    ids=["64B_1blk", "64B_4blk", "128B", "256B", "1024B"],
)
def test_the_two_levers_are_mutually_exclusive(device, shape):
    """No shipped plan may carry both levers — measured, not stylistic.

    C7 already halves the reads each RISC-V issues, so B13's saved command
    programming is halved while its bank-major DRAM serialization is not. Three
    in-run A/B pairs on `[1,1,2048,32]`: C7 alone 3411.1 / 3404.1 / 3419.6 ns vs
    C7+B13 3462.4 / 3431.6 / 3434.2 ns (+0.9 % mean, never negative). A lever that
    does not move the number is a defect, so the planner ships exactly one.
    """
    plan = _plan(device, shape)
    assert not (plan["stateful_read"] and plan["split_read"]), (
        f"{shape} ships both levers (chunk_row_bytes={plan['chunk_row_bytes']}, "
        f"blocks_per_core={plan['blocks_per_core']})"
    )


def test_depth2_request_swaps_c7_for_b13(device):
    """`use_double_buffer=True` on the 64 B regime must turn C7 off, not corrupt.

    At depth 2 the reserved window alternates between two CB slots, and BRISC —
    which never touches the CB pointers — would keep writing the base slot. The
    gate refuses C7 structurally; B13 is independent of the CB depth, and with C7
    gone it is no longer suppressed by the mutual-exclusion clause, so the plan
    swaps one lever for the other.
    """
    plan = _plan(device, C7_REGIME, use_double_buffer=True)
    assert plan["depth"] == 2
    assert plan["split_read"] == 0
    assert plan["stateful_read"] == 1
    _roundtrip_exact(device, C7_REGIME, use_double_buffer=True)


def test_levers_off_on_the_zero_copy_path(device):
    """Path B has no NoC reads at all, so neither lever can apply."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    plan = _plan(device, (1, 1, 512, 64), memory_config=mem)
    assert plan["path"] == "alias"
    assert (plan["stateful_read"], plan["split_read"]) == (0, 0)


def test_levers_off_when_a_row_spans_several_source_pages(device):
    """Structural: both levers need one source page per logical row.

    A ROW_MAJOR-*sharded* input with shard_w < W puts several pages on one row
    (`row_page_stride > 1`), so the reader takes the raw strided fallback: the
    page index no longer advances by 1 per row, which is what B13's
    "p and p+num_banks share a bank" identity rests on.
    """
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (256, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(0)
    shape = (1, 1, 2048, 512)
    tt_input = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
    )
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = build_plan(tt_input, tt_output, device)
    assert plan["row_page_stride"] > 1
    assert (plan["stateful_read"], plan["split_read"]) == (0, 0)


# ---------------------------------------------------------------------------
# Bit-exactness of every new code path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 2048, 32),  # both levers, 1 block/core, 64 cores
        (1, 1, 64, 32),  # both levers, 2 cores
        (1, 1, 32, 32),  # both levers, 1 core, one single tile
        (1, 1, 96, 96),  # both levers, awkward 3x3 tile grid
        (1, 1, 8192, 32),  # B13 + 4 blocks/core
        (1, 1, 32, 4096),  # B13 at 128 B, nt_h == 1 (width split)
        (1, 1, 32, 8192),  # no levers (256 B) — the fallback stays correct
        (2, 3, 64, 64),  # rank-4 fold with both levers
    ],
    ids=["tall_narrow", "2core", "single_tile", "3x3", "4blk", "128B_widesplit", "256B", "rank4_fold"],
)
def test_bit_exact_on_the_gated_default(device, shape):
    _roundtrip_exact(device, shape)


def test_bit_exact_single_core_split_reader(device):
    """`use_multicore=False` still gets the split reader (1 core, 1 block)."""
    plan = _plan(device, (1, 1, 32, 32), use_multicore=False)
    assert plan["ncores"] == 1
    assert plan["split_read"] == 1
    _roundtrip_exact(device, (1, 1, 32, 32), use_multicore=False)


def test_bit_exact_with_l1_interleaved_input(device):
    """L1-interleaved: 64 banks, so the stateful path self-disables in-kernel.

    The helper's `period * 2 <= num_rows` guard is a *runtime* fallback to the
    plain per-row loop — the plan still offers B13 (the read is 128 B), so this is
    the test that the in-kernel guard produces correct data rather than a
    bank-major walk over a bank period it cannot amortize. Single-block shape so
    Refinement 2's B8 does not take the cell (see the note in
    `test_stateful_read_offered_to_every_interleaved_generic_plan`).
    """
    plan = _plan(device, B13_ONLY_128B, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert plan["stateful_read"] == 1
    _roundtrip_exact(device, B13_ONLY_128B, memory_config=ttnn.L1_MEMORY_CONFIG)
    # ...and the 64 B / 4-block shape, which now runs B8's prefetch instead, must
    # also be bit-exact on an L1-interleaved source.
    _roundtrip_exact(device, B13_ONLY_MULTIBLOCK, memory_config=ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "b13_env,c7_env,want",
    [(0, 0, (0, 0)), (1, 0, (1, 0)), (0, 1, (0, 1)), (2, 1, (1, 1))],
    ids=["none", "b13", "c7", "both_forced"],
)
def test_bit_exact_for_every_lever_combination(device, monkeypatch, b13_env, c7_env, want):
    """All four lever combinations must be bit-exact, not just the shipped one.

    The bench measures the same four via these env switches, so a wrong result in
    a counterfactual row would otherwise be reported as a perf number. `both` needs
    `b13=2` (force) because the shipped gate makes the two mutually exclusive —
    the combined code path still exists and still has to be correct.
    """
    monkeypatch.setenv("TILIZE_LEVER_B13", str(b13_env))
    monkeypatch.setenv("TILIZE_LEVER_C7", str(c7_env))
    plan = _plan(device, C7_REGIME)
    assert (plan["stateful_read"], plan["split_read"]) == want
    _roundtrip_exact(device, C7_REGIME)


def test_bit_exact_split_reader_past_its_gate(device, monkeypatch):
    """Force C7 onto multi-block / multi-chunk plans and check correctness.

    The gate keeps this out of production for *perf* reasons, but the code path is
    reachable (the bench forces it), and it is the only place that exercises the
    per-block sequence counter, the chunk-outer x block-inner handshake order and
    the depth-1 window being re-handed over N times. A mismatched sequence number
    between the two kernels would hang; a swapped iteration order would transpose
    blocks — both are caught here.
    """
    monkeypatch.setenv("TILIZE_LEVER_C7", "2")  # force past the payoff gate
    for shape in [(1, 1, 8192, 32), (1, 1, 512, 128), (1, 1, 32, 4096)]:
        plan = _plan(device, shape)
        assert plan["split_read"] == 1, f"{shape} did not take the forced split path"
        _roundtrip_exact(device, shape)
    assert plan["blocks_per_core"] > 1 or plan["chunk_wt"] > 1


def test_split_reader_is_bit_exact_over_repeated_launches(device):
    """The handshake uses monotonic per-launch counters, so it must be re-armed.

    Semaphore initial values are re-written by the dispatcher on every launch. If
    they were NOT, launch 2 would see a stale high `sem_reserve` and BRISC would
    read into a window NCRISC has not reserved yet — a race that only shows from
    the second launch on. Ten launches of the same program, each bit-exact.
    """
    plan = _plan(device, C7_REGIME)
    assert plan["split_read"] == 1
    for i in range(10):
        _roundtrip_exact(device, C7_REGIME, seed=i)


def test_program_cache_hits_with_the_split_reader(device):
    """The split path adds two semaphores and one writer runtime arg.

    Neither may break the program-cache key: same shape twice must reuse the
    entry (the semaphores are descriptor-level, the src address is a plain
    runtime arg that gets re-patched).
    """
    plan = _plan(device, C7_REGIME)
    assert plan["split_read"] == 1
    _roundtrip_exact(device, C7_REGIME)
    before = device.num_program_cache_entries()
    _roundtrip_exact(device, C7_REGIME, seed=7)
    assert device.num_program_cache_entries() == before, "second identical call added a cache entry"


def test_stateful_read_offered_to_every_interleaved_generic_plan(device):
    """B13's plan flag keys on the read size; the ACCESSOR kind is decided

    kernel-side by `if constexpr (DSpec::is_interleaved)`. Assert that split of
    responsibility — the host does not try to second-guess the accessor: a DRAM
    and an L1 interleaved input with the same read size both get the flag.

    Shape choice: 128 B with ONE block per core is the cell where B13 is the only
    lever left standing — C7 needs <= 64 B and Refinement 2's B8 needs >= 2 blocks,
    so neither can mask the result. (This used to use the 64 B / 4-block shape,
    which B8 now wins; see `test_plan_lever_selection_per_read_size`.)
    """
    for mem in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        plan = _plan(device, B13_ONLY_128B, memory_config=mem)
        assert plan["blocks_per_core"] == 1, "the shape must stay single-block or B8 supersedes B13"
        assert plan["split_read"] == 0
        assert plan["stateful_read"] == 1
        assert plan["chunk_row_bytes"] <= STATEFUL_READ_MAX_ROW_BYTES
