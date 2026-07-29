# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4b — resident (zero-copy) CBs: drop the per-block CB bookkeeping.

Refinement 4 left Path B as a compute-only program with `WaitMode::NoWait`, which
removed the per-block `cb_wait_front`. **Three** per-block CB calls survived, and on a
program with nothing on the other end of either CB all three are dead weight:

  * `cb_reserve_back(cb_out)` polls a free-space credit no consumer will ever decrement;
  * `cb_push_back(cb_out)` publishes a tiles-received credit nobody reads, and pays a
    `TTI_STALLWAIT(STALL_THCON, PACK)` to order that publish behind the packer;
  * `cb_pop_front(cb_in)` returns a free-space credit nobody reads, and pays a
    `TTI_STALLWAIT(STALL_THCON, UNPACK)` for the same reason.

What is load-bearing in those calls is NOT the credit but the **pointer walk** — it is
`fifo_rd_ptr` / `fifo_wr_ptr` that address block k. So this refinement keeps the
addressing and drops the credits, which the tilize LLK already supports directly:
`tilize_block` / `fast_tilize_block` take an `input_tile_index` / `output_tile_index`
resolved off the CB base, so block k is tile index `k*chunk_wt` on a CB whose pointers
never move.

**It ships as a HELPER change, not a raw-LLK block in this kernel** — which was the
condition the refinement set for taking it at all. `compute_kernel_lib::tilize` gained
`InputBufferMode` / `OutputBufferMode` (`Circular` | `Resident`), both defaulted to
`Circular` so every other call site in the tree is byte-identical, with the
"nothing is on the other end of this CB" contract written out in the header. The mode is
**per-CB**, which is what makes it a primitive rather than a Path-B special case:

    mask 3 -- Path B: both CBs aliased, compute-only program.   SHIPPED.
    mask 2 -- the `alias_out` crossover (R3b dropped the writer, the reader stays).
              Correct and bit-exact, but measured NEUTRAL, so gated off.

## What was measured, and why the parent's price was wrong

Refinement 4 priced the survivors at 40.5 ns/block from `sync_only(4 blk) -
sync_only(1 blk)`, and inferred they were *exposed* from `full - LLK` where
`LLK := full - no_compute`. **That identity is tautological** — it attributes every
nanosecond to either the LLK or the CB dance and therefore cannot detect that the two
OVERLAP. They do, almost completely: the CB calls sit on the same TRISCs as the LLK
stages they pipeline against (reserve/push on PACK, pop on UNPACK), so while TRISC0
pops block k the packer is still on block k-1.

In-run A/B against `rcb=0`, 15 rounds x 10 launches, CV <= 1.4 %:

    regime            blk  chk | shipped |  rcb=0 | ratio |    dns | dns/blk
    ------------------|--------|---------|--------|-------|--------|--------
    n_sharded_tiny      1    2 |   748.7 |  755.4 | 1.009 |   -6.7 |   -6.7
    f_sharded_small     4    2 |  1256.4 | 1272.5 | 1.013 |  -16.1 |   -4.0
    f_sharded_large     8    2 |  1952.2 | 1981.5 | 1.015 |  -29.3 |   -3.7
    n_sharded_deep     32    2 |  5563.0 | 5717.4 | 1.028 | -154.4 |   -4.8
    n_sharded_wide      1   64 |  4702.4 | 4715.4 | 1.003 |  -13.0 |  -13.0

=> **~4.6 ns/block, 12 % of what the isolation instrument reported.** Real and
proportional (the 32-block row is 1.028x at CV 0.3 %), but 1.3 % on `f_sharded_small`.

The structural proof that the lever removed **all** of the per-block traffic is the
ablation on the shipped plan: `sync_only` is now **flat in block count** (438.0 ns at 1
block, 438.0 at 32), where `rcb=0` gives 1486.5 at 32 blocks = 33.8 ns/block. So the
lever takes 100 % of the CB bookkeeping and only ~15 % of it was ever on the critical
path. `test_sync_only_is_flat_in_block_count_under_residency` is the on-device pin.

## The three `ttnn-static-analyzer` findings this file pins

* **F1** — the lever's one load-bearing clause ("no kernel is on the other end of this
  CB") is host-only and had no test; the descriptor comment even cited a test that did
  not exist. `test_resident_mask_is_set_exactly_when_the_counterpart_kernel_is_gone`.
* **F2** — `fast_tilize_block` scales the block index by the compile-time `TILE_R_DIM`
  (32) where the slow path reads the operand's real row dim, so a 16-row input tile
  would mis-stride every block after the first. Only reachable with a nonzero index,
  i.e. only under `Resident`, and NOT implied by `can_use_fast_tilize` (which checks
  32x32 on the OUTPUT only). Now a `static_assert` in the helper; the reachable half of
  the contract is pinned by `test_every_resident_cb_has_32x32_tiles`.
* **F3** — on tt-2xx `push_back`/`pop_front` round-robin the hardware tile counter, so
  `Resident` is only defined for `num_tcs_to_rr == 1`. An `ARCH_QUASAR` ASSERT plus a
  contract line; not testable on WH/BH.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import (
    build_plan,
    create_program_descriptor,
    resident_cb_pays,
)

_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
DRAM = ttnn.DRAM_MEMORY_CONFIG

# CT arg index of the resident-CB bitmask on the compute kernel (see
# `create_program_descriptor`). Named rather than inlined because three tests read it.
CT_RESIDENT_CB = 9
CT_NO_WAIT = 4
CT_SELF_ARM = 5
CT_PER_BLOCK_INIT = 8


def _crs(end_x, end_y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})


def _shard(scheme, grid, shape, orientation=_ROW):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shape, orientation))


@contextmanager
def _env(**kwargs):
    saved = {}
    try:
        for name, value in kwargs.items():
            saved[name] = os.environ.get(name)
            os.environ[name] = str(value)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _levers(**kwargs):
    return _env(**{f"TILIZE_LEVER_{k.upper()}": v for k, v in kwargs.items()})


_IS_INT = (ttnn.uint32, ttnn.int32, ttnn.uint16)


def _make(device, shape, cfg=DRAM, dtype=ttnn.bfloat16):
    """`arange`, not `randn`. A wrong resident tile index produces a PERMUTATION of the
    right bytes — the one failure mode a random input can cancel out."""
    n = 1
    for d in shape:
        n *= d
    if dtype in _IS_INT:
        torch_input = (torch.arange(n, dtype=torch.int64) % 4096).reshape(shape).to(torch.int32)
    elif dtype == ttnn.float32:
        torch_input = (torch.arange(n, dtype=torch.float32) % 65536).reshape(shape)
    else:
        torch_input = ((torch.arange(n, dtype=torch.float32) % 4096).reshape(shape)).bfloat16()
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg)
    return torch_input, tt_input


def _descriptor(device, shape, in_cfg, out_cfg, *, dtype=ttnn.bfloat16, out_dtype=None):
    _, tt_input = _make(device, shape, in_cfg, dtype)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), out_dtype or dtype, ttnn.TILE_LAYOUT, device, out_cfg
    )
    plan = build_plan(tt_input, tt_output, device)
    return plan, create_program_descriptor(tt_input, tt_output, plan), tt_input, tt_output


def _compute_ct(descriptor):
    compute = [k for k in descriptor.kernels if "tilize_compute" in k.kernel_source][0]
    return list(compute.compile_time_args)


def _kernel_names(descriptor):
    return [k.kernel_source.rsplit("/", 1)[-1] for k in descriptor.kernels]


def _exact(device, shape, in_cfg, out_cfg=None, *, dtype=ttnn.bfloat16, repeats=1):
    """tilize is value-preserving, so the oracle for a no-cast call is `torch.equal`."""
    out_cfg = in_cfg if out_cfg is None else out_cfg
    torch_input, tt_input = _make(device, shape, in_cfg, dtype)
    for launch in range(repeats):
        actual = ttnn.to_torch(tilize(tt_input, out_cfg))
        if dtype in _IS_INT:
            assert torch.equal(actual.to(torch.int32), torch_input.to(torch.int32)), f"launch {launch} not bit-exact"
        else:
            assert torch.equal(actual.float(), torch_input.float()), (
                f"launch {launch}: max_abs="
                f"{(actual.float() - torch_input.float()).abs().max().item()}, "
                f"{int((actual.float() != torch_input.float()).sum())} mismatching elements"
            )


# Path-B geometries the resident mode reshapes the CB protocol of. Chosen to cover the
# axes the resident tile index actually scales on -- `num_blocks` (1 / 2 / 4 / 8 / 32,
# the multiplier of the index) and `chunk_wt` (2 / 8 / 64, its step) -- across every
# shard scheme and both orientations.
PATH_B_CASES = [
    ("H 1 blk", (1, 1, 128, 64), _shard(_HEIGHT, _crs(3, 0), (32, 64))),
    ("H 4 blk", (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64))),
    ("H 32 blk", (1, 1, 4096, 64), _shard(_HEIGHT, _crs(3, 0), (1024, 64))),
    ("H col", (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64), _COL)),
    ("BLOCK 2x2", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64))),
    ("BLOCK col", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64), _COL)),
    ("BLOCK 8 blk", (1, 1, 2048, 512), _shard(_BLOCK, _crs(7, 7), (256, 64))),
    ("WIDTH", (1, 1, 64, 256), _shard(_WIDTH, _crs(3, 0), (64, 64))),
    ("wide chunk 8", (1, 1, 128, 256), _shard(_HEIGHT, _crs(1, 0), (64, 256))),
    ("wide chunk 64", (1, 1, 128, 2048), _shard(_HEIGHT, _crs(3, 0), (32, 2048))),
]

# The one-sided `alias_out` crossover: DRAM-interleaved RM in, sharded TILE out. The
# reader survives (it is the op's whole data movement) so only bit 1 may be set.
ALIAS_OUT_CASES = [
    ("small 4 blk", (1, 1, 512, 128), _shard(_BLOCK, _crs(1, 3), (128, 64))),
    ("large 8 blk", (1, 1, 2048, 512), _shard(_BLOCK, _crs(7, 7), (256, 64))),
    ("H out", (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64))),
]


# ---------------------------------------------------------------------------
# The load-bearing host clause (analyzer finding F1)
# ---------------------------------------------------------------------------


def test_resident_mask_is_set_exactly_when_the_counterpart_kernel_is_gone(device):
    """The invariant the compute kernel CANNOT assert, so the host must be pinned.

    `tilize_compute.cpp` has no compile-time arg for the reader's or the writer's
    existence, so no `static_assert` can catch a CB declared resident while its
    counterpart kernel is still launched. Both directions are silent-ish disasters:

      * output resident + writer launched  -> compute publishes NO credits and the
        writer blocks forever in `cb_wait_front(cb_tiled_output, shard_tiles)`
        (`tilize_writer.cpp`) => a deterministic BRISC hang;
      * input resident + reader launched   -> compute frees NO pages, so the reader's
        `cb_reserve_back` blocks, or (worse, if the CB is big enough not to block) the
        reader refills pages the unpacker has not read yet => silent corruption.

    `ttnn-static-analyzer` (F1) flagged that nothing pinned CT arg 9 at all and that
    `drop_writer` is decided by a *separate* lever chain (`nw`, `split_read`,
    `_zone_flag`, `drop_writer_pays`) from the one `no_wait` rests on. This is that
    guard: the mask bits must AGREE with the kernel list, on every path.
    """
    cases = [
        # name, levers, shape, in_cfg, out_cfg
        ("path B gated", {}, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), None),
        ("path B nd=0 (3 kernels)", {"nd": 0}, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), None),
        ("path B nd=3 (fold)", {"nd": 3}, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), None),
        ("path B rcb=0", {"rcb": 0}, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), None),
        ("alias_out gated", {}, (1, 1, 512, 128), DRAM, _shard(_BLOCK, _crs(1, 3), (128, 64))),
        ("alias_out rcb=2", {"rcb": 2}, (1, 1, 512, 128), DRAM, _shard(_BLOCK, _crs(1, 3), (128, 64))),
        # The writer is kept (`nw=0`), so bit 1 must NOT be set even when forced.
        ("alias_out nw=0 rcb=2", {"nw": 0, "rcb": 2}, (1, 1, 512, 128), DRAM, _shard(_BLOCK, _crs(1, 3), (128, 64))),
        ("alias_in", {}, (1, 1, 2048, 512), _shard(_BLOCK, _crs(7, 7), (256, 64)), DRAM),
        ("alias_in rcb=2", {"rcb": 2}, (1, 1, 2048, 512), _shard(_BLOCK, _crs(7, 7), (256, 64)), DRAM),
        ("interleaved rcb=2", {"rcb": 2}, (1, 1, 256, 256), DRAM, DRAM),
    ]
    for name, levers, shape, in_cfg, out_cfg in cases:
        with _levers(**levers):
            plan, descriptor, _, _ = _descriptor(device, shape, in_cfg, out_cfg or in_cfg)
            mask = plan["resident_cb"]
            names = _kernel_names(descriptor)
            reader_launched = "tilize_reader.cpp" in names
            writer_launched = "tilize_writer.cpp" in names

            # The mask must agree with the KERNEL LIST, not merely with the plan flags —
            # that is the only statement that actually rules the hang out.
            assert not (mask & 1) or not reader_launched, f"{name}: input CB resident but the reader is launched"
            assert not (mask & 2) or not writer_launched, f"{name}: output CB resident but the writer is launched"
            # A resident CB must also be an ALIASED one; a plain CB's pages are not the
            # tensor's, so addressing the whole run off its base would run off the end.
            assert not (mask & 1) or plan["alias_in"], f"{name}: input CB resident but not aliased"
            assert not (mask & 2) or plan["alias_out"], f"{name}: output CB resident but not aliased"
            # ... and the kernel's CT arg must carry exactly that mask.
            assert _compute_ct(descriptor)[CT_RESIDENT_CB] == mask, f"{name}: CT arg {CT_RESIDENT_CB} != {mask}"


def test_the_resident_mask_is_reachable_only_as_0_2_or_3(device):
    """Bit 0 without bit 2 is unreachable, and that is a property worth pinning.

    Bit 0 needs `drop_reader`, which `_plan_generic` hard-sets to 0 — so an
    input-resident CB only ever appears on Path B, where `drop_reader == drop_writer`.
    If a future refinement drops the reader on a one-sided path (an `alias_in`
    crossover with a compute-only read, say), mask 1 becomes reachable and
    `resident_cb_pays`' `mask == 3` clause would silently refuse it. Failing here is
    the signal to revisit the gate rather than to widen the mask blindly.
    """
    seen = set()
    for levers in ({}, {"rcb": 2}, {"rcb": 0}):
        with _levers(**levers):
            for _, shape, cfg in PATH_B_CASES[:4]:
                seen.add(_descriptor(device, shape, cfg, cfg)[0]["resident_cb"])
            for _, shape, out_cfg in ALIAS_OUT_CASES:
                seen.add(_descriptor(device, shape, DRAM, out_cfg)[0]["resident_cb"])
            seen.add(_descriptor(device, (1, 1, 256, 256), DRAM, DRAM)[0]["resident_cb"])
    assert seen <= {0, 2, 3}, f"unexpected resident_cb values: {sorted(seen)}"
    assert {0, 2, 3} <= seen, f"a reachable mask value was never produced: got {sorted(seen)}"


def test_resident_capacity_bound_holds_with_equality_on_every_geometry(device):
    """The helper ASSERTs `num_pages >= block_width_tiles * num_blocks` on a resident
    DFB, because a resident CB is never recycled — every block of the whole run is
    addressed off its base. That is a strictly stronger bound than the circular one
    (`>= block_width_tiles`), so it has to be checked host-side on every geometry and
    not only under a watcher build (the helper's ASSERTs compile out otherwise).

    It holds with EQUALITY on Path B (`shard_tiles == num_blocks * chunk_wt`), and on
    `alias_out` it reduces to the same identity via `shard_wt % chunk_wt == 0`.
    """
    for name, shape, cfg in PATH_B_CASES:
        plan, _, _, _ = _descriptor(device, shape, cfg, cfg)
        assert plan["resident_cb"] == 3, f"{name}: expected mask 3, got {plan['resident_cb']}"
        assert plan["shard_tiles"] == plan["num_blocks"] * plan["chunk_wt"], (
            f"{name}: resident bound violated -- CB has {plan['shard_tiles']} pages, the run "
            f"addresses {plan['num_blocks']} x {plan['chunk_wt']}"
        )
    for name, shape, out_cfg in ALIAS_OUT_CASES:
        with _levers(rcb=2):
            plan, _, _, _ = _descriptor(device, shape, DRAM, out_cfg)
        assert plan["resident_cb"] == 2, f"{name}: expected mask 2 when forced, got {plan['resident_cb']}"
        blocks = plan["blocks_per_core"]
        assert plan["shard_tiles"] == blocks * plan["chunk_wt"], (
            f"{name}: resident bound violated on alias_out -- output CB has {plan['shard_tiles']} "
            f"pages, the run addresses {blocks} x {plan['chunk_wt']}"
        )


def test_every_resident_cb_has_32x32_tiles(device):
    """Analyzer finding F2, the reachable half.

    `fast_tilize_block` converts the caller's tile-unit index into the unpacker's
    32-datum-unit index with the COMPILE-TIME constant `TILE_R_DIM` (32), where the
    slow path reads the operand's real row dim. So a 16-row input tile under
    `Resident` would stride every block after the first 2x too far while the pack side
    — which uses `fifo_page_size` — stayed correct: wrong tiles, no assert, no hang.
    The helper now `static_assert`s it; this pins the op side, i.e. that the tilize
    CBs really are full 32x32 tiles (the descriptor never sets tile dims, so the page
    size must be exactly one full tile).
    """
    for name, shape, cfg in PATH_B_CASES:
        plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        assert plan["resident_cb"] == 3, name
        tile_bytes = {0: plan["tile_in"], 16: plan["tile_out"]}
        for cb in descriptor.cbs:
            for fmt in cb.buffer_descriptors if hasattr(cb, "buffer_descriptors") else []:
                if fmt.buffer_index in tile_bytes:
                    assert fmt.page_size == tile_bytes[fmt.buffer_index], (
                        f"{name}: CB {fmt.buffer_index} page is {fmt.page_size} B, not one whole "
                        f"{tile_bytes[fmt.buffer_index]} B tile -- a partial tile would mis-stride "
                        f"the resident index on the fast path"
                    )
        # bf16 Path B takes fast tilize, and a 32x32 bf16 tile is exactly 2048 B.
        if plan["in_dtype"] == ttnn.bfloat16:
            assert plan["tile_in"] == 2048, f"{name}: tile_in={plan['tile_in']}"


# ---------------------------------------------------------------------------
# The kernel-visible pairings (static_asserts, checked by construction here)
# ---------------------------------------------------------------------------


def test_residency_never_coexists_with_the_arms_that_forbid_it(device):
    """The kernel `static_assert`s these four pairings, so a violation is a JIT failure
    rather than a wrong answer — but a JIT failure is only reachable if the host can
    produce the combination at all, and these are the arms that could:

      * `zones` (R3b's timeline) instruments the THREE-kernel program;
      * `per_block_init` (R4 lever 2's ceiling probe) issues one 1-block call per
        block, so a resident tile index would restart at 0 and rewrite block 0;
      * `self_arm` (the `zero_copy_fold` arm) drives BOTH CBs through the full protocol
        from the compute kernel itself;
      * a resident INPUT requires `no_wait` — a wait would block on a credit nobody posts.

    Asserted on the host derivation so the JIT never has to be the backstop.
    """
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    shape = (1, 1, 512, 64)

    with _levers(iu=1):
        plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        ct = _compute_ct(descriptor)
        assert ct[CT_PER_BLOCK_INIT] == 1, "the iu arm did not engage"
        assert plan["resident_cb"] == 0, "residency leaked into the per-block-init arm"
        assert ct[CT_RESIDENT_CB] == 0

    with _levers(nd=3):
        plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        ct = _compute_ct(descriptor)
        assert ct[CT_SELF_ARM] == 1, "the fold arm did not engage"
        assert plan["resident_cb"] == 0, "residency leaked into the zero_copy_fold arm"

    with _env(TILIZE_ZONES=1):
        plan, _, _, _ = _descriptor(device, shape, cfg, cfg)
        assert plan["resident_cb"] == 0, "residency leaked into the zones arm"

    # A resident input implies `no_wait`, on every plan the op can build.
    for levers in ({}, {"rcb": 2}):
        with _levers(**levers):
            for _, s, c in PATH_B_CASES[:4]:
                plan, descriptor, _, _ = _descriptor(device, s, c, c)
                ct = _compute_ct(descriptor)
                assert not (ct[CT_RESIDENT_CB] & 1) or ct[CT_NO_WAIT] == 1, "resident input without no_wait"


def test_the_shipped_gate_is_the_compute_only_program(device):
    """`resident_cb_pays` requires mask == 3, and the reason is measured, not stylistic.

    mask 3 means the program has NO dataflow kernel, hence no data movement, hence the
    compute thread is the bound by construction — the only condition under which
    shaving compute-side per-block overhead can move the number. On the one-sided
    `alias_out` crossover the lever measured 1.001 / 0.995 (neutral, sign-unstable),
    and R3b's per-RISC timeline says why: TRISC0 is blocked in `cb_wait_front` for
    90 % of that kernel, so there is microseconds of slack to hide three CB calls in.

    Pinned as a unit property of the gate function AND on real plans, so re-widening
    the gate to `alias_out` has to change this test and re-measure.
    """
    assert resident_cb_pays(3, 1) is True
    assert resident_cb_pays(3, 32) is True
    assert resident_cb_pays(2, 8) is False, "alias_out measured neutral; the gate must not fire"
    assert resident_cb_pays(1, 8) is False
    assert resident_cb_pays(0, 8) is False

    for name, shape, out_cfg in ALIAS_OUT_CASES:
        plan, _, _, _ = _descriptor(device, shape, DRAM, out_cfg)
        assert plan["path"] == "alias_out", f"{name}: got {plan['path']}"
        assert plan["drop_writer"] == 1, f"{name}: R3b's writer drop is the precondition"
        assert plan["resident_cb"] == 0, f"{name}: the gate must be off on the one-sided alias"


def test_the_interleaved_and_crossover_plans_are_untouched(device):
    """Structural non-regression: this refinement may not perturb any non-zero-copy
    plan. Nothing outside a zero-copy CB can be resident, so every other path must
    keep its exact core count, chunk width, depth and kernel list.
    """
    cases = [
        ("square", (1, 1, 256, 256), DRAM, DRAM, ["tilize_reader.cpp", "tilize_writer.cpp", "tilize_compute.cpp"]),
        ("wide short", (1, 1, 32, 4096), DRAM, DRAM, ["tilize_reader.cpp", "tilize_writer.cpp", "tilize_compute.cpp"]),
        (
            "alias_in",
            (1, 1, 2048, 512),
            _shard(_BLOCK, _crs(7, 7), (256, 64)),
            DRAM,
            ["tilize_reader.cpp", "tilize_writer.cpp", "tilize_compute.cpp"],
        ),
        (
            "alias_out",
            (1, 1, 2048, 512),
            DRAM,
            _shard(_BLOCK, _crs(7, 7), (256, 64)),
            ["tilize_reader.cpp", "tilize_compute.cpp"],
        ),
    ]
    for name, shape, in_cfg, out_cfg, want_kernels in cases:
        plan, descriptor, _, _ = _descriptor(device, shape, in_cfg, out_cfg)
        assert plan["resident_cb"] == 0, f"{name}: residency must not reach this plan"
        assert _kernel_names(descriptor) == want_kernels, f"{name}: {_kernel_names(descriptor)}"


# ---------------------------------------------------------------------------
# Numerics — a wrong resident index is a PERMUTATION, so every value must be unique
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,shape,cfg", PATH_B_CASES, ids=[c[0] for c in PATH_B_CASES])
def test_resident_path_b_is_bit_exact(device, name, shape, cfg):
    """The whole lever replaces pointer arithmetic with index arithmetic, so the
    failure mode is a block landing at the wrong tile — a permutation of the correct
    bytes. `arange` input + `torch.equal` is the only oracle that catches that; a PCC
    or an allclose against a symmetric input would not.

    4 repeat launches per geometry: with the pointers never moving, a stale
    `fifo_rd_ptr` / `fifo_wr_ptr` from the previous launch of a cached program would
    show up as wrong values on launch 2+, not launch 1.
    """
    _exact(device, shape, cfg, repeats=4)


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32, ttnn.uint16],
    ids=["bf16", "fp32", "uint32", "int32", "uint16"],
)
def test_resident_path_b_is_bit_exact_for_every_dtype(device, dtype):
    """Both LLK paths have to be right, and the dtype is what selects them: bf16 takes
    `fast_tilize_block` (whose index carries the `% full_dim` / `* TILE_R_DIM`
    transform), fp32-in-fp32-out takes the SLOW `tilize_block` via `Fp32Mode::Lossless`,
    and the integer dtypes take the slow path too (`has_supported_fast_tilize_format`
    admits only Float32 / Float16_b). The two resolve `input_tile_index` through
    different code, so a transform bug in either is a silent permutation.
    """
    _exact(device, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), dtype=dtype, repeats=2)


@pytest.mark.parametrize("name,shape,out_cfg", ALIAS_OUT_CASES, ids=[c[0] for c in ALIAS_OUT_CASES])
def test_resident_alias_out_is_bit_exact_when_forced(device, name, shape, out_cfg):
    """The mode is gated OFF on the one-sided crossover because it measured neutral —
    NOT because it is wrong there. That distinction is what makes the helper mode a
    reusable primitive rather than a Path-B special case, so it has to be tested: if a
    future op (or a future measurement on a compute-bound crossover) turns this on, the
    correctness evidence is already here.
    """
    with _levers(rcb=2):
        plan, _, _, _ = _descriptor(device, shape, DRAM, out_cfg)
        assert plan["resident_cb"] == 2, f"{name}: forcing did not engage"
        _exact(device, shape, DRAM, out_cfg, repeats=2)


def test_the_rcb_counterfactual_arm_is_still_bit_exact(device):
    """`rcb=0` is the permanent counterfactual (the Refinement-4 CB protocol). It has
    to keep working, or the bench pair that prices this lever stops being an A/B.
    """
    with _levers(rcb=0):
        for _, shape, cfg in PATH_B_CASES[:4]:
            plan, _, _, _ = _descriptor(device, shape, cfg, cfg)
            assert plan["resident_cb"] == 0
            _exact(device, shape, cfg, repeats=2)


def test_resident_program_cache_rebinding(device):
    """Two calls on DIFFERENT shard addresses must both be bit-exact from ONE cache
    entry. This is the property most at risk from residency: the CB pointers are never
    walked, so if the aliased base address were not re-patched per launch the second
    call would tilize the first call's shard. (`apply_descriptor_runtime_args`
    re-patches every CB descriptor carrying a `tensor`; the base-address runtime args
    on the compute kernel are the explicit witness.)
    """
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    shape = (1, 1, 512, 64)
    first_in, first_tt = _make(device, shape, cfg)
    second_torch = (first_in.float() * -1.0 - 7.0).bfloat16()
    second_tt = ttnn.from_torch(
        second_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg
    )
    assert first_tt.buffer_address() != second_tt.buffer_address(), "the probe needs two distinct shard addresses"

    first_out = tilize(first_tt, cfg)
    entries_after_first = device.num_program_cache_entries()
    second_out = tilize(second_tt, cfg)
    assert device.num_program_cache_entries() == entries_after_first, "the second call must hit the cache"

    assert torch.equal(ttnn.to_torch(second_out).float(), second_torch.float()), "second call not bit-exact"
    assert torch.equal(ttnn.to_torch(first_out).float(), first_in.float()), "the first result was disturbed"


# ---------------------------------------------------------------------------
# The measurement the entry turns on (on device, structural rather than a duration)
# ---------------------------------------------------------------------------


def test_sync_only_is_flat_in_block_count_under_residency(device):
    """The structural proof that the lever removes **all** per-block CB traffic.

    `sync_only` (`TILIZE_SKIP_DM=1` + `TILIZE_SKIP_COMPUTE=1`) keeps every CB call and
    loop trip count and drops both payloads, so its slope in `num_blocks` IS the
    per-block CB cost. Refinement 4 measured that slope at 40.5 ns/block and this pass
    reproduced 33.8-38.9. Under residency the slope must be **zero** — that is what
    "the per-block bookkeeping is gone" means, and unlike a duration ratio it is not a
    2 % effect fighting the noise floor.

    Asserted as a COUNT, not a time: the number of CB calls the ablation branch emits,
    read off the kernel's own `if constexpr` structure via the plan. A duration check
    lives in `_bench_tilize.py` (`n_sharded_deep` vs `p_sharded_deep_rcb_off`); this is
    the part that belongs in a correctness suite.
    """
    shallow = (1, 1, 128, 64), _shard(_HEIGHT, _crs(3, 0), (32, 64))
    deep = (1, 1, 4096, 64), _shard(_HEIGHT, _crs(3, 0), (1024, 64))

    for shape, cfg in (shallow, deep):
        plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        ct = _compute_ct(descriptor)
        mask = ct[CT_RESIDENT_CB]
        assert mask == 3, f"{shape}: expected mask 3, got {mask}"
        # With mask 3 the ablation branch emits NO CB call per block (no wait because
        # `no_wait`, no reserve/push because bit 1, no pop because bit 0), so the
        # per-block CB call count is 0 at BOTH depths -- a flat slope by construction.
        per_block_calls = (0 if ct[CT_NO_WAIT] else 1) + (0 if mask & 2 else 2) + (0 if mask & 1 else 1)
        assert per_block_calls == 0, f"{shape}: {per_block_calls} CB calls/block survive under mask 3"

    with _levers(rcb=0):
        for shape, cfg in (shallow, deep):
            ct = _compute_ct(_descriptor(device, shape, cfg, cfg)[1])
            assert ct[CT_RESIDENT_CB] == 0
            per_block_calls = (0 if ct[CT_NO_WAIT] else 1) + 2 + 1
            assert per_block_calls == 3, f"{shape}: the counterfactual must keep its 3 CB calls/block"

    # And the plans differ in EXACTLY that one way -- same cores, chunk, depth, blocks.
    for shape, cfg in (shallow, deep):
        shipped, _, _, _ = _descriptor(device, shape, cfg, cfg)
        with _levers(rcb=0):
            baseline, _, _, _ = _descriptor(device, shape, cfg, cfg)
        for key in ("ncores", "chunk_wt", "depth", "num_blocks", "shard_tiles", "cb_bytes_per_core", "fp32_lossless"):
            assert (
                shipped[key] == baseline[key]
            ), f"{shape}: the lever changed {key} ({shipped[key]} vs {baseline[key]})"


def test_residency_costs_no_l1(device):
    """The lever removes instructions, not buffers. Per-core CB bytes must be identical
    to the counterfactual on every geometry — a perf lever that quietly bought L1 would
    be a defect under this run's rules.
    """
    for name, shape, cfg in PATH_B_CASES:
        shipped, _, _, _ = _descriptor(device, shape, cfg, cfg)
        with _levers(rcb=0):
            baseline, _, _, _ = _descriptor(device, shape, cfg, cfg)
        assert shipped["cb_bytes_per_core"] == baseline["cb_bytes_per_core"], (
            f"{name}: residency changed per-core CB L1 "
            f"({shipped['cb_bytes_per_core']} vs {baseline['cb_bytes_per_core']})"
        )
        assert shipped["alias_cb_bytes"] == baseline["alias_cb_bytes"], name
