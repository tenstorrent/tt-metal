# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — the Path-B (same-spec sharded) compute + sync floor.

Path B is the zero-copy path: input and output shard specs are identical, so BOTH
CBs are aliased onto the resident L1 shards and there is **no NoC traffic at all**
(`no_dm == full` to within 0.4 %, re-confirmed this refinement). What is left is
therefore only compute and per-launch overhead — the Phase-0 ablation put
`f_sharded_small` at roughly 50 % tilize LLK / 50 % dispatch+CB-sync / 0 % DM — and
this refinement attacks both halves.

* **Lever 1 — the compute-only program** (SHIPPED, `TILIZE_LEVER_ND`). On Path B the
  reader's whole body is `cb_reserve_back(shard_tiles); cb_push_back(shard_tiles)`
  and the writer's is `cb_wait_front(shard_tiles); cb_pop_front(shard_tiles)`: two
  kernel launches that publish pages already sitting at the CB address. Both go, and
  the program ships **ONE kernel**. Measured, in-run A/B, 15 rounds x 10 launches:

      regime                        | 1 kernel | 3 kernels | fold | ratio (3k/1k)
      ------------------------------|----------|-----------|------|--------------
      n_sharded_tiny   (1 blk/core) |      757 |       895 |  868 | 1.182
      f_sharded_small  (4 blk/core) |     1277 |      1361 | 1370 | 1.066
      f_sharded_large  (8 blk/core) |     1979 |      2071 | 2070 | 1.046

  **The arm/drain is DELETED, not moved, and that distinction is the whole lever.**
  `ttnn/ttnn/operations/examples/zero_copy_fold` measures a compute-only program that
  FOLDS the arm/drain onto TRISC — on this very payload — at 0.74x-0.95x, i.e.
  *slower*, because the arm/drain used to overlap the tilize on NCRISC/BRISC. The
  `fold` column above is that variant reproduced here (`TILIZE_LEVER_ND=3`) and it
  reproduces the example's verdict: it is no better than the three-kernel program.
  Deleting the handshake is what pays; reducing the kernel count is not.

  Deleting it is sound only because of the exact-page argument (Refinement 3b's, now
  applied to both sides): `WaitMode::NoWait` drops the per-block `cb_wait_front`
  while the helper's `cb_pop_front` still walks `fifo_rd_ptr` across the shard, and
  the aliased output CB has exactly `shard_tiles` pages against exactly
  `shard_tiles` pushes so `cb_reserve_back` never blocks.

* **Lever 2 — `InitUninitMode` amortisation** (REFUTED, with a measured ceiling).
  Refinement 1's static-analysis pass already showed `InitAndUninit` sits OUTSIDE the
  `num_blocks` loop, so there is nothing to amortise inside one `tilize()` call. This
  refinement prices what the lever could ever be worth, by measuring the opposite:
  `TILIZE_LEVER_IU=1` issues one fully-inited call per block instead of one per
  kernel, which is bit-exact and therefore a measurement of the real kernel.

      f_sharded_small  4 blk: 1276 -> 1484  = +208 ns / 3 extra pairs = 69 ns/pair
      f_sharded_large  8 blk: 1982 -> 2477  = +495 ns / 7 extra pairs = 71 ns/pair

  So a config-burst pair costs **~70 ns**, the shipped kernel already issues exactly
  one, and `tilize_uninit` cannot be dropped (it would leak `tileize_mode=1` into the
  next program on the core). The lever's ceiling is 70 ns and it is **unreachable**.

* **Lever 3 — `Fp32Mode::Fast` on a narrowing fp32 cast** (SHIPPED,
  `TILIZE_LEVER_F32`). The kernel used to pick `Lossless` off the INPUT CB format
  alone, so `fp32 -> bf16` / `fp32 -> bf8b` paid the slow LLK path for a precision the
  narrower output cannot hold. Measured 1.256x / 1.429x on the compute-bound sharded
  cells and 1.003x (neutral) on the DM-bound interleaved one, with output
  **indistinguishable** from Lossless.

  It is a HOST+KERNEL pair: fast tilize on an fp32 input requires
  `unpack_to_dest_mode[cb_rm_input] = Default`, so `_compute_config` must flip with
  the kernel's `Fp32Mode`. And the legality clause is a WHITELIST of narrower FLOAT
  outputs — `ttnn-static-analyzer` caught the first draft's `out_dtype != float32`
  admitting `uint32`/`int32`/`uint16` outputs, which `can_use_fast_tilize` accepts
  and then silently corrupts (its pack stage steps DEST at bf16 stride). Both that
  hole and the force flag's bypass of it are pinned below.
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
    CB_RM_INPUT,
    _compute_config,
    build_plan,
    create_program_descriptor,
    drop_dataflow_pays,
    fp32_fast_legal,
    fp32_fast_pays,
)

_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
DRAM = ttnn.DRAM_MEMORY_CONFIG


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


# The Path-B geometries lever 1 changes the program shape of. Every scheme, both
# orientations, the 1-block-per-core corner (where the freed fixed cost is the
# biggest share) and the multi-chunk shard (where `chunk_wt > 1`, i.e. where the
# exact-page inequality has slack rather than equality on every block).
PATH_B_CASES = [
    ("H 4 blk", (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64))),
    ("H col", (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64), _COL)),
    ("H 1 blk", (1, 1, 128, 64), _shard(_HEIGHT, _crs(3, 0), (32, 64))),
    ("BLOCK 2x2", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64))),
    ("BLOCK col", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64), _COL)),
    ("WIDTH", (1, 1, 64, 256), _shard(_WIDTH, _crs(3, 0), (64, 64))),
    ("wide chunk", (1, 1, 128, 256), _shard(_BLOCK, _crs(1, 1), (64, 128))),
]

# Every (in, out) dtype pair lever 3 could touch, with the mode it MUST resolve to.
# `1` == Fp32Mode::Lossless (the Phase-0 behaviour), `0` == the fast path.
FP32_CASES = [
    ("fp32->bf16", ttnn.float32, ttnn.bfloat16, 0),
    ("fp32->bf8b", ttnn.float32, ttnn.bfloat8_b, 0),
    ("fp32->fp32", ttnn.float32, ttnn.float32, 1),
    # The analyzer's F1: `can_use_fast_tilize` accepts a 32-bit INTEGER output.
    ("fp32->uint32", ttnn.float32, ttnn.uint32, 1),
    ("fp32->int32", ttnn.float32, ttnn.int32, 1),
    ("fp32->uint16", ttnn.float32, ttnn.uint16, 1),
    # Non-fp32 inputs never selected Lossless in the first place.
    ("bf16->bf16", ttnn.bfloat16, ttnn.bfloat16, 1),
    ("bf16->bf8b", ttnn.bfloat16, ttnn.bfloat8_b, 1),
    ("bf16->fp32", ttnn.bfloat16, ttnn.float32, 1),
    ("uint32->uint32", ttnn.uint32, ttnn.uint32, 1),
    ("int32->int32", ttnn.int32, ttnn.int32, 1),
]

_TORCH_READBACK = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # bf8b reads back as bf16
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint16: torch.int32,
}
_IS_INT = (ttnn.uint32, ttnn.int32, ttnn.uint16)


def _make(device, shape, cfg=DRAM, dtype=ttnn.bfloat16):
    """`arange`, not `randn`: every element is unique, so a permutation — the failure
    mode a wrong read pointer produces — cannot cancel out."""
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


def _plan(device, shape, in_cfg, out_cfg, *, dtype=ttnn.bfloat16, out_dtype=None):
    _, tt_input = _make(device, shape, in_cfg, dtype)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), out_dtype or dtype, ttnn.TILE_LAYOUT, device, out_cfg
    )
    return build_plan(tt_input, tt_output, device)


def _exact(device, shape, cfg, *, dtype=ttnn.bfloat16, repeats=1):
    """tilize is value-preserving, so the oracle for a no-cast call is `torch.equal`."""
    torch_input, tt_input = _make(device, shape, cfg, dtype)
    for launch in range(repeats):
        tt_output = tilize(tt_input, cfg)
        actual = ttnn.to_torch(tt_output)
        if dtype in _IS_INT:
            assert torch.equal(actual.to(torch.int32), torch_input.to(torch.int32)), f"launch {launch} not bit-exact"
        else:
            assert torch.equal(actual.float(), torch_input.float()), (
                f"launch {launch}: max_abs="
                f"{(actual.float() - torch_input.float()).abs().max().item()}, "
                f"{int((actual.float() != torch_input.float()).sum())} mismatching elements"
            )


def _descriptor(device, shape, in_cfg, out_cfg, *, dtype=ttnn.bfloat16, out_dtype=None):
    _, tt_input = _make(device, shape, in_cfg, dtype)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), out_dtype or dtype, ttnn.TILE_LAYOUT, device, out_cfg
    )
    plan = build_plan(tt_input, tt_output, device)
    return plan, create_program_descriptor(tt_input, tt_output, plan), tt_input, tt_output


def _kernel_names(descriptor):
    return [k.kernel_source.rsplit("/", 1)[-1] for k in descriptor.kernels]


# ---------------------------------------------------------------------------
# Lever 1 — the compute-only Path-B program (SHIPPED)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,shape,cfg", PATH_B_CASES, ids=[c[0] for c in PATH_B_CASES])
def test_path_b_ships_a_compute_only_program(device, name, shape, cfg):
    """Structural, not a duration: the program must contain exactly ONE kernel.

    Asserted on the descriptor rather than measured, because a duration cannot
    distinguish "the writer was dropped" from "the writer was fast" — and because
    the saving is a fixed ~85-140 ns that is inside the noise on the larger shards.
    """
    plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
    assert plan["path"] == "alias", f"{name}: expected Path B, got {plan['path']}"
    assert plan["drop_reader"] == plan["drop_writer"] == 1, f"{name}: both dataflow kernels must go"
    assert plan["self_arm"] == 0, f"{name}: the shipped form DELETES the arm/drain, it does not fold it"
    assert _kernel_names(descriptor) == ["tilize_compute.cpp"], f"{name}: got {_kernel_names(descriptor)}"
    _exact(device, shape, cfg, dtype=ttnn.bfloat16)


@pytest.mark.parametrize("name,shape,cfg", PATH_B_CASES, ids=[c[0] for c in PATH_B_CASES])
def test_path_b_survives_repeat_launches(device, name, shape, cfg):
    """The cross-launch claim, and it needs a test because its failure mode is silent.

    With no writer nothing pops the output CB and with no reader nothing pushes the
    input CB, so launch N ends with `tiles_received != tiles_acked` on both. If that
    state leaked forward, launch N+1 would either hang in `cb_reserve_back` (output
    CB believed full) or read the wrong rows. It does not leak: BRISC's firmware
    calls `trigger_sync_register_init()` unconditionally every launch, which zeroes
    the stream scratch registers holding those counters — note this is a FIRMWARE
    guarantee that survives BRISC having no kernel, not the CB-interface reset
    Refinement 3b cited (that only zeroes the per-RISC local shadow).

    Regression observable: a hang or a wrong-values failure on launch 2+, so this
    test must keep running even though launch 1 already covers the arithmetic.
    """
    _exact(device, shape, cfg, dtype=ttnn.bfloat16, repeats=4)


def test_path_b_program_cache_rebinding(device):
    """Two calls, two different shard base addresses, one cache entry, both exact.

    With both CBs aliased there is no `Buffer*` runtime arg, so the concern is that a
    cached program keeps launch 1's shard address. It does not — `apply_descriptor_
    runtime_args` re-patches every CB descriptor carrying a `tensor` on every call —
    and the base-address runtime args this refinement moved onto the compute kernel
    are the explicit witness that it happened.
    """
    shape = (1, 1, 512, 64)
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    torch_input, tt_a = _make(device, shape, cfg)
    _, tt_b = _make(device, shape, cfg)
    assert tt_a.buffer_address() != tt_b.buffer_address(), "the two inputs must land at different addresses"

    ttnn.synchronize_device(device)
    out_a = tilize(tt_a, cfg)
    entries_after_first = device.num_program_cache_entries()
    out_b = tilize(tt_b, cfg)
    assert device.num_program_cache_entries() == entries_after_first, "the second call must HIT the cache"

    got_a = ttnn.to_torch(out_a)
    got_b = ttnn.to_torch(out_b)
    assert torch.equal(got_b.float(), torch_input.float()), "the cached re-launch is not bit-exact"
    assert torch.equal(got_a.float(), torch_input.float()), "the first result was disturbed by the second call"
    assert out_a.buffer_address() != out_b.buffer_address(), "the two outputs must be distinct buffers"


def test_the_base_addresses_ride_on_the_compute_kernel(device):
    """Both dataflow kernels carried a base address; with both gone, both args move."""
    shape = (1, 1, 512, 64)
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    plan, descriptor, tt_input, tt_output = _descriptor(device, shape, cfg, cfg)
    compute = descriptor.kernels[0]
    core = plan["cores"][0]
    args = list(compute.runtime_args[core.x][core.y])
    assert len(args) == 3, f"expected num_blocks + both base addresses, got {args}"
    assert args[0] == plan["num_blocks"]
    assert args[1] == tt_input.buffer_address()
    assert args[2] == tt_output.buffer_address()


def test_no_wait_is_set_exactly_when_the_reader_is_gone(device):
    """The invariant the compute kernel CANNOT assert, so the host must be pinned.

    `tilize_compute.cpp` has no compile-time arg for the reader's existence, so
    `static_assert` cannot catch `no_wait` without a dropped reader (a guaranteed
    hang: nothing would ever publish the input CB) or a surviving reader with
    `self_arm` (two producers on one CB). `ttnn-static-analyzer` flagged that the
    only guard is the host derivation — this is that guard.
    """
    cases = [
        ("path B, gated", {}, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), 1, 0),
        ("path B, nd=0", {"nd": 0}, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), 0, 0),
        ("path B, nd=3 fold", {"nd": 3}, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), 1, 1),
    ]
    for name, levers, shape, cfg, want_drop, want_self_arm in cases:
        with _levers(**levers):
            plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
            assert plan["drop_reader"] == want_drop, f"{name}: drop_reader"
            assert plan["self_arm"] == want_self_arm, f"{name}: self_arm"
            compute = [k for k in descriptor.kernels if "tilize_compute" in k.kernel_source][0]
            ct = list(compute.compile_time_args)
            # CT arg 4 is `no_wait`, 5 is `self_arm` (see create_program_descriptor).
            assert ct[4] == (1 if (want_drop and not want_self_arm) else 0), f"{name}: no_wait CT arg {ct}"
            assert ct[5] == want_self_arm, f"{name}: self_arm CT arg {ct}"
            assert not (ct[4] and ct[5]), f"{name}: no_wait and self_arm are exclusive"


def test_the_cb_page_count_equals_the_pushes(device):
    """The arithmetic the whole lever rests on, checked host-side on every geometry.

    `cb_reserve_back` on the output never blocks iff the aliased CB has at least as
    many pages as compute pushes; `cb_pop_front` on the input stays inside
    `fifo_limit` iff the pops sum to exactly the CB size. Both reduce to
    `shard_tiles == num_blocks * chunk_wt`, and both hold with EQUALITY (no margin),
    which is why a plan change that breaks it must fail here rather than on device.
    """
    for name, shape, cfg in PATH_B_CASES:
        plan = _plan(device, shape, cfg, cfg)
        assert plan["path"] == "alias", name
        assert plan["shard_tiles"] == plan["num_blocks"] * plan["chunk_wt"], (
            f"{name}: shard_tiles={plan['shard_tiles']} != num_blocks="
            f"{plan['num_blocks']} * chunk_wt={plan['chunk_wt']}"
        )
        # The shards must also tile the tensor exactly, or CB page k is not shard tile k.
        assert plan["shard_tiles"] * plan["ncores"] == plan["total_tiles"], f"{name}: shard cover"


def test_lever_1_is_path_b_only(device):
    """The reader is the whole data movement on every other path, so it must survive.

    `alias_out` keeps Refinement 3b's 2-kernel program (writer dropped, reader kept);
    `alias_in` and generic keep all three — `alias_in`'s OUTPUT CB is a plain CB that
    the writer still has to drain to DRAM, so only its READ side is zero-copy.
    """
    cases = [
        ("generic", (1, 1, 128, 128), DRAM, DRAM, 3),
        ("alias_in", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64)), DRAM, 3),
        ("alias_out", (1, 1, 256, 128), DRAM, _shard(_BLOCK, _crs(1, 1), (128, 64)), 2),
    ]
    for name, shape, in_cfg, out_cfg, want_kernels in cases:
        plan, descriptor, _, _ = _descriptor(device, shape, in_cfg, out_cfg)
        assert plan["path"] == name, f"{name}: got path {plan['path']}"
        assert plan["drop_reader"] == 0, f"{name}: the reader must survive"
        assert plan["self_arm"] == 0, f"{name}: nothing folds outside Path B"
        assert len(descriptor.kernels) == want_kernels, f"{name}: got {_kernel_names(descriptor)}"
        assert "tilize_reader.cpp" in _kernel_names(descriptor), f"{name}: {_kernel_names(descriptor)}"


def test_the_zone_variant_keeps_all_three_kernels(device):
    """Lever 1 defers to Refinement 3b's timeline: the zones instrument the reader's
    and writer's OWN stages, so dropping those kernels would delete the measurement.

    Plan-level only, deliberately not launched. `TILIZE_ZONES=1` has never been valid
    on Path B — the reader's `static_assert(!zones || (!alias_mode && ...))` predates
    this refinement, because the instrumented loop reproduces the GENERIC per-block
    read loop and an aliased read has no loop. So the pair (Path B, zones) is a JIT
    build failure both before and after lever 1; what this pins is only that lever 1
    does not silently take the zone variant's kernels away on the paths where it IS
    valid (the `alias_out` crossover, exercised by `test_tilize_refinement3b.py`).
    """
    shape = (1, 1, 512, 64)
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    with _env(TILIZE_ZONES=1):
        plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        assert plan["drop_reader"] == plan["drop_writer"] == 0, "lever 1 must yield to the zone variant"
        assert plan["self_arm"] == 0
        assert len(descriptor.kernels) == 3, _kernel_names(descriptor)


def test_the_three_kernel_counterfactual_is_still_bit_exact(device):
    """`TILIZE_LEVER_ND=0` is the Phase-0 program and the ledger's counterfactual
    row, so it has to keep working — including across repeat launches, where it is
    the arm that actually depends on the stream registers being re-zeroed."""
    shape = (1, 1, 512, 64)
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    with _levers(nd=0):
        plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        assert plan["drop_reader"] == plan["drop_writer"] == 0
        assert _kernel_names(descriptor) == ["tilize_reader.cpp", "tilize_writer.cpp", "tilize_compute.cpp"]
        _exact(device, shape, cfg, repeats=2)


@pytest.mark.parametrize("name,shape,cfg", PATH_B_CASES, ids=[c[0] for c in PATH_B_CASES])
def test_the_fold_counterfactual_is_still_bit_exact(device, name, shape, cfg):
    """`TILIZE_LEVER_ND=3` reproduces `examples/zero_copy_fold`'s compute_only variant:
    one kernel, but the arm/drain folded onto TRISC. It is the arm that proves the
    shipped lever's win comes from DELETING the handshake rather than from the kernel
    count, so it must stay measurable — and therefore correct.

    It is also the one branch where the compute kernel pushes its own input CB, i.e.
    where PACK is the producer and UNPACK the consumer of both CBs. That is sound
    (they are different threads with separate `cb_interface[]` arrays) but it is not
    obvious, so every geometry is checked rather than one.
    """
    with _levers(nd=3):
        plan, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        assert plan["self_arm"] == 1, name
        assert _kernel_names(descriptor) == ["tilize_compute.cpp"], name
        _exact(device, shape, cfg, repeats=2)


def test_drop_dataflow_gate_is_declared_on(device):
    """The gate itself, so a future edit that turns it off has to say so here."""
    assert drop_dataflow_pays(1) is True
    assert drop_dataflow_pays(8) is True
    assert drop_dataflow_pays(64) is True


# ---------------------------------------------------------------------------
# Lever 2 — InitUninitMode (REFUTED; the measurement arm must stay correct)
# ---------------------------------------------------------------------------


def test_the_per_block_init_arm_is_off_by_default(device):
    """It is a measurement, not a lever: `num_blocks` config-burst pairs instead of
    one is strictly worse (+208 ns at 4 blocks). It must never ship."""
    shape = (1, 1, 512, 64)
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    _, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
    compute = descriptor.kernels[0]
    # CT arg 8 is `per_block_init`.
    assert list(compute.compile_time_args)[8] == 0, "the per-block-init arm leaked into a shipped plan"


def test_the_per_block_init_arm_is_bit_exact(device):
    """Unlike the SKIP_* ablations this changes no CB count and drops no payload —
    every call is fully inited — so it measures the REAL kernel. That is only true
    while it stays correct, which is what this pins.
    """
    shape = (1, 1, 512, 64)
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64))
    with _levers(iu=1):
        _, descriptor, _, _ = _descriptor(device, shape, cfg, cfg)
        assert list(descriptor.kernels[0].compile_time_args)[8] == 1
        _exact(device, shape, cfg)
    # ... and on the interleaved path too, where num_blocks is larger.
    with _levers(iu=1):
        _exact(device, (1, 1, 256, 128), DRAM)


# ---------------------------------------------------------------------------
# Lever 3 — Fp32Mode::Fast on a narrowing fp32 cast (SHIPPED)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,in_dtype,out_dtype,want_lossless", FP32_CASES, ids=[c[0] for c in FP32_CASES])
def test_the_fp32_mode_gate_per_dtype_pair(device, name, in_dtype, out_dtype, want_lossless):
    """Exactly two cells may take the fast path: fp32 -> bf16 and fp32 -> bf8b.

    The rest of this table is the analyzer's F1 in test form. `can_use_fast_tilize`
    guards its output with `!is_fp32_output_format`, which refuses Float32 but
    ACCEPTS UInt32 / Int32 / UInt16 — all declared in `SUPPORTED["output_dtype"]`.
    Fast tilize's pack stage steps DEST at bf16 stride, so a 32-bit integer output
    would be silently corrupted, and no helper `static_assert` fires on that cell.
    Hence a whitelist (`_FP32_FAST_OUT`), not `out_dtype != float32`.
    """
    assert fp32_fast_legal(in_dtype, out_dtype) == (want_lossless == 0), f"{name}: legality clause"
    assert fp32_fast_pays(in_dtype, out_dtype) == (want_lossless == 0), f"{name}: gate"
    plan = _plan(device, (1, 1, 64, 128), DRAM, DRAM, dtype=in_dtype, out_dtype=out_dtype)
    assert plan["fp32_lossless"] == want_lossless, f"{name}: plan"


@pytest.mark.parametrize("name,in_dtype,out_dtype,want_lossless", FP32_CASES, ids=[c[0] for c in FP32_CASES])
def test_the_host_unpack_mode_pairs_with_the_kernel_mode(device, name, in_dtype, out_dtype, want_lossless):
    """The host half of the pair. `Fp32Mode::Fast` on an fp32 input REQUIRES
    `unpack_to_dest_mode[cb_rm_input] = Default`; `Lossless` REQUIRES
    `UnpackToDestFp32`. The helper static_asserts both directions, so a mismatch is a
    JIT build failure rather than a wrong number — but only for the two cells whose
    guard actually triggers, which is why this checks the config directly.
    """
    config = _compute_config(in_dtype, out_dtype, want_lossless)
    modes = list(config.unpack_to_dest_mode)
    if not modes:  # bf16 -> bf16: fp32_dest_acc_en is False, nothing to configure
        assert not config.fp32_dest_acc_en, f"{name}: no unpack modes but fp32 dest is on"
        return
    want = ttnn.UnpackToDestMode.Default if (want_lossless == 0) else ttnn.UnpackToDestMode.UnpackToDestFp32
    assert modes[CB_RM_INPUT] == want, f"{name}: unpack_to_dest_mode[{CB_RM_INPUT}]={modes[CB_RM_INPUT]}"


def test_the_force_flag_cannot_bypass_the_structural_clause(device):
    """The analyzer's F2. `TILIZE_LEVER_F32=2` forces past the PAYOFF gate; it must
    not force past LEGALITY. On `fp32 -> fp32` the bypass would land on the slow path
    (the output is fp32, so `use_fast` is false) with `Default` unpack, i.e. the slow
    path PLUS the fp32 -> tf32 unpack downgrade — neither fast nor lossless, and no
    static_assert fires on it. On `fp32 -> uint32` it would reach fast tilize.
    """
    with _levers(f32=2):
        for name, in_dtype, out_dtype, want_lossless in FP32_CASES:
            plan = _plan(device, (1, 1, 64, 128), DRAM, DRAM, dtype=in_dtype, out_dtype=out_dtype)
            assert plan["fp32_lossless"] == want_lossless, f"forced {name}: got {plan['fp32_lossless']}"


def test_fp32_to_fp32_stays_bit_exact(device):
    """The contract lever 3 must not touch, on both the interleaved and the Path-B
    (compute-only) program — the second is the one where BOTH refinement-4 levers
    are active at once."""
    _exact(device, (1, 1, 128, 256), DRAM, dtype=ttnn.float32)
    _exact(device, (1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64)), dtype=ttnn.float32)


@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8b"])
@pytest.mark.parametrize(
    "cfg_name",
    ["dram", "path_b"],
)
def test_fast_fp32_matches_lossless(device, out_dtype, cfg_name):
    """The fast path must be INDISTINGUISHABLE from the slow one on a narrowing cast.

    Compared against the Lossless arm element-wise rather than against a PCC bar,
    because a tolerance would hide a systematic 1-ULP shift. Fast truncates
    fp32 -> tf32 on the way into DEST, so the two can only differ where bits 11-23 of
    the mantissa would have changed a round-to-nearest decision at bit 8 — measured
    ZERO such elements on these inputs, and even in principle a 1-ULP tie-break that
    the oracle (PCC 0.999 for fp32 -> bf16, 0.99 into bf8b) does not care about.
    """
    shape = (1, 1, 512, 64)
    cfg = DRAM if cfg_name == "dram" else _shard(_HEIGHT, _crs(3, 0), (128, 64))
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    results = {}
    for arm, f32 in (("fast", 1), ("lossless", 0)):
        with _levers(f32=f32):
            tt_input = ttnn.from_torch(
                torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=cfg
            )
            probe = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), out_dtype, ttnn.TILE_LAYOUT, device, cfg)
            plan = build_plan(tt_input, probe, device)
            assert plan["fp32_lossless"] == (0 if arm == "fast" else 1), f"{arm}: gate did not flip"
            results[arm] = ttnn.to_torch(tilize(tt_input, cfg, dtype=out_dtype)).float()

    diff = (results["fast"] - results["lossless"]).abs()
    assert torch.equal(results["fast"], results["lossless"]), (
        f"fast differs from lossless: max_abs={diff.max().item():.4e}, "
        f"{int((diff != 0).sum())}/{diff.numel()} elements"
    )
    # ... and both still satisfy the op's own oracle against torch.
    expected = torch_input.to(_TORCH_READBACK[out_dtype]).float()
    for arm, got in results.items():
        pcc = torch.corrcoef(torch.stack([expected.flatten(), got.flatten()]))[0, 1].item()
        bar = 0.999 if out_dtype == ttnn.bfloat16 else 0.99
        assert pcc >= bar, f"{arm}: PCC {pcc} < {bar}"


def test_integer_and_bf16_inputs_are_byte_identical_to_before(device):
    """Lever 3 must be a no-op for every dtype whose config it does not change.

    Integer passthrough is the sensitive one: it keeps `UnpackToDestFp32` (its input
    format is UInt32/Int32, so `is_fp32_input_format` is false and the kernel never
    asks for Lossless), and a whitelist bug that dropped that mode would break
    bit-exactness here rather than in the fp32 tests.
    """
    for dtype in (ttnn.uint32, ttnn.int32, ttnn.bfloat16):
        _exact(device, (1, 1, 64, 128), DRAM, dtype=dtype)
    # bf16 -> fp32 widening must also stay exact (it never selected Lossless).
    torch_input, tt_input = _make(device, (1, 1, 64, 128), DRAM, ttnn.bfloat16)
    got = ttnn.to_torch(tilize(tt_input, DRAM, dtype=ttnn.float32))
    assert torch.equal(got.float(), torch_input.float()), "bf16 -> fp32 widening must stay exact"


def test_the_fp32_counterfactual_arm_is_still_bit_exact(device):
    """`TILIZE_LEVER_F32=0` is the Phase-0 Lossless-everywhere behaviour and the
    ledger's counterfactual row, so it must keep working."""
    with _levers(f32=0):
        plan = _plan(device, (1, 1, 512, 64), DRAM, DRAM, dtype=ttnn.float32, out_dtype=ttnn.bfloat16)
        assert plan["fp32_lossless"] == 1
        _exact(device, (1, 1, 128, 256), DRAM, dtype=ttnn.float32)


# ---------------------------------------------------------------------------
# Non-regression: the other paths' plans must be structurally untouched
# ---------------------------------------------------------------------------


def test_the_interleaved_and_crossover_plans_are_untouched(device):
    """Neither lever may perturb the DM-bound plans the prior refinements tuned.

    Lever 1 is Path-B-only and lever 3 only fires on an fp32 input, so every bf16
    interleaved / crossover plan must be structurally identical to Refinement 3b's:
    same core count, chunk width, depth and per-core CB bytes.
    """
    expected = {
        # shape                in_cfg  out_cfg  -> (path, ncores, chunk_wt, depth)
        "a_square": ((1, 1, 2048, 2048), DRAM, DRAM, ("generic", 64, 16, 1)),
        "d_tall_narrow": ((1, 1, 2048, 32), DRAM, DRAM, ("generic", 64, 1, 1)),
        # The `alias_out` crossover's chunk width is the SHARD's (64 cols = 2 tiles),
        # not the generic path's L1-budget choice, and Refinement 1's C16 gate gives
        # it depth 2 at 8 blocks/core.
        "g_dram_to_sharded": (
            (1, 1, 2048, 512),
            DRAM,
            _shard(_BLOCK, _crs(7, 7), (256, 64)),
            ("alias_out", 64, 2, 2),
        ),
    }
    for name, (shape, in_cfg, out_cfg, want) in expected.items():
        plan = _plan(device, shape, in_cfg, out_cfg)
        got = (plan["path"], plan["ncores"], plan["chunk_wt"], plan["depth"])
        assert got == want, f"{name}: plan moved, {got} != {want}"
        assert plan["drop_reader"] == 0 and plan["self_arm"] == 0, name
        assert plan["fp32_lossless"] == 1, f"{name}: a bf16 plan must not touch the fp32 mode"
