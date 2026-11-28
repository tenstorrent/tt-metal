# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.bringup.block_select_ttnn:block_keys``.

Every test calls the operation through the ``block_keys_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (block_keys, the README and the callable's docstring):
    block_keys(pos, ids, *, block, width, output_tensor=None, memory_config=None, max_grid=None)
    pos [.., R, KP] uint32 ROW_MAJOR DRAM interleaved (positions in block_gather's [.., R, K block] row; 0xFFFFFFFF:
    none), ids [.., R, K] uint32 ROW_MAJOR DRAM interleaved (block ids; >= width / block: none) ->
    [.., R, KP] uint32 ROW_MAJOR DRAM interleaved: ids[pos / block] block + pos % block, 0xFFFFFFFF where pos is none
    (>= K block, the golden reference's rule) or the block it names is.
    Refusals (the source asserts: AssertionError; a native port raises RuntimeError from TT_FATAL, so both types are
    accepted, each with a message naming the offending argument): block not a power of two in [1, 32]; pos / ids /
    output not uint32, not ROW_MAJOR or not DRAM interleaved (memory_config included); pos and ids with different
    leading dims; output_tensor of another dtype, layout or shape.
    output_tensor: written in place and returned (same buffer). max_grid: the cores' rectangle (0,0)..(max_grid - 1),
    clamped to the device grid, a CoreCoord or an (x, y) tuple. Precision: exact (index arithmetic).
    Program cache: one program per (shapes, block, width / block, grid); buffers, addresses and data never enter it
    (a hit on fresh buffers, a preallocated output or not). The source dispatches through ttnn.bringup.run_prepared, not
    ttnn.generic_op, so ``last_dispatches`` stays 0 on both routes and is not asserted.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys

import pytest
import ttnn
from loguru import logger

# --------------------------------------------------------------------------- route selection (written by the driver)

SOURCE_ENTRY = "ttnn.bringup.block_select_ttnn:block_keys"
MODE_ENV = "TT_PRE_MIGRATION_MODE"
ENTRY_ENV = "TT_PRE_MIGRATION_ENTRY"


def _resolve(spec):
    module_name, sep, name = spec.partition(":")
    if not sep or not module_name or not name:
        raise pytest.UsageError(f"{ENTRY_ENV} must be module.path:callable, got {spec!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, name, None)
    if fn is None or not callable(fn):
        raise pytest.UsageError(f"{spec} does not resolve to a callable")
    return module, fn


def _selection():
    mode = os.environ.get(MODE_ENV, "source") or "source"
    entry = os.environ.get(ENTRY_ENV, "")
    if mode == "source":
        if entry and entry != SOURCE_ENTRY:
            raise pytest.UsageError(f"source mode runs {SOURCE_ENTRY} only, but {ENTRY_ENV}={entry!r}")
        return mode, SOURCE_ENTRY
    if mode == "native":
        if not entry:
            raise pytest.UsageError(f"native mode requires {ENTRY_ENV}=module.path:callable")
        if entry == SOURCE_ENTRY:
            raise pytest.UsageError(f"native mode needs the op's registered entry, not the source path {SOURCE_ENTRY}")
        return mode, entry
    raise pytest.UsageError(f"{MODE_ENV} must be 'source' or 'native', got {mode!r}")


def _generic_op_owners():
    """Every (owner, attribute) through which the tested call path can reach ``ttnn.generic_op``."""
    original = ttnn.generic_op
    owners = [(ttnn, "generic_op")]
    for name, module in list(sys.modules.items()):
        if module is None or module is ttnn:
            continue
        if name.startswith("ttnn.bringup.block_select_ttnn") or name.startswith("ttnn._ttnn.operations.generic"):
            if getattr(module, "generic_op", None) is original:
                owners.append((module, "generic_op"))
    return original, owners


@contextlib.contextmanager
def _patched_generic_op(replacement):
    original, owners = _generic_op_owners()
    patched = []
    try:
        for owner, attr in owners:
            try:
                setattr(owner, attr, replacement(original))
            except (AttributeError, TypeError):
                continue
            patched.append((owner, attr))
        yield
    finally:
        for owner, attr in reversed(patched):
            setattr(owner, attr, original)


class _Entry:
    """The single call point for the operation under test. ``last_dispatches`` is how many
    ``ttnn.generic_op`` calls the last source call made (0 on the native route)."""

    def __init__(self, mode, spec, module, fn):
        self.mode = mode
        self.spec = spec
        self.module = module
        self.fn = fn
        self.calls = 0
        self.generic_dispatches = 0
        self.last_dispatches = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.mode == "native":
            return self._call_native(*args, **kwargs)
        return self._call_source(*args, **kwargs)

    def _call_source(self, *args, **kwargs):
        seen = []

        def spy(original):
            def forward(*a, **k):
                seen.append(1)
                return original(*a, **k)

            return forward

        with _patched_generic_op(spy):
            try:
                return self.fn(*args, **kwargs)
            finally:
                self.last_dispatches = len(seen)
                self.generic_dispatches += len(seen)

    def _call_native(self, *args, **kwargs):
        def forbid(_original):
            def raise_(*_a, **_k):
                raise AssertionError("native block_keys must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def block_keys_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"block_keys acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

pytestmark = pytest.mark.use_module_device

import torch  # noqa: E402

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
REFUSED = (AssertionError, RuntimeError)  # source: assert; a native port: TT_FATAL (RuntimeError)


def _run_and_check(bk, op, device, lead, K, KP, B, W, *, seed=0, pos_oob=False, **kwargs):
    """Upload fresh inputs, call the op, check metadata, values, untouched inputs; returns (out, pos_t, ids_t, want)."""
    pos, ids = bk.make_inputs(lead, K, KP, B, W, seed=seed, pos_oob=pos_oob)
    pt, it = bk.up(pos, device), bk.up(ids, device)
    out = op(pt, it, block=B, width=W, **kwargs)
    want = bk.ref_keys(pos, ids, B, W)
    bk.assert_dram_rm_u32(out, [*lead, KP])
    bk.assert_exact(bk.down(out), want, f"lead={lead} K={K} KP={KP} B={B} W={W} {kwargs}")
    bk.assert_exact(bk.down(pt), bk.u32(pos), "pos must be untouched")
    bk.assert_exact(bk.down(it), bk.u32(ids), "ids must be untouched")
    if "output_tensor" not in kwargs:
        assert bk.addr(out) not in (bk.addr(pt), bk.addr(it)), "a new output buffer is promised"
    return out, pt, it, want


# --------------------------------------------------------------------------- reference self-check (CPU only)


def test_reference_self_check(bk):
    """The vectorised golden reference agrees with an element-wise one on every special value."""
    for lead, K, KP, B, W in (((5,), 7, 19, 4, 64), ((2, 3), 16, 40, 1, 32), ((4,), 3, 9, 32, 128)):
        pos, ids = bk.make_inputs(lead, K, KP, B, W, seed=3, pos_oob=True)
        bk.assert_exact(bk.ref_keys(pos, ids, B, W), bk.ref_keys_loop(pos, ids, B, W), "reference")
    # hand-checked values: ids [2, SENT, 0], B 4, width 16 (4 blocks)
    pos = torch.tensor([[0, 3, 4, 7, 9, 11, 12, 0xFFFFFFFF]])
    ids = torch.tensor([[2, 0xFFFFFFFF, 0]])
    want = torch.tensor([[8, 11, 0xFFFFFFFF, 0xFFFFFFFF, 1, 3, 0xFFFFFFFF, 0xFFFFFFFF]])
    bk.assert_exact(bk.ref_keys(pos, ids, 4, 16), want, "hand values")


# --------------------------------------------------------------------------- values, outputs and planner branches

# (lead dims, K, KP, block, width, kwargs): one case per observable branch of the row split, the scratch sizes
# (4 KP and 4 K bytes, 64-byte aligned or not), the block shift and the rank.
VALUE_CASES = [
    pytest.param((1,), 1, 1, 1, 32, {}, id="min_rank2_r1_k1_kp1_b1"),
    pytest.param((7,), 17, 33, 4, 1024, {}, id="rank2_r7_fewer_rows_than_cores_unaligned_k_kp"),
    pytest.param((3, 5), 16, 16, 2, 512, {}, id="rank3_r15_aligned_k_kp_b2"),
    pytest.param((2, 3, 40), 64, 100, 16, 4096, {}, id="rank4_batch_r240_b16"),
    pytest.param((1, 1, 1000), 32, 64, 8, 2048, {}, id="r1000_uneven_split_b8"),
    pytest.param((1, 1, 8), 512, 2048, 32, 65536, {}, id="kp2048_k512_b32"),
    pytest.param((1, 1, 16), 4096, 16, 2, 16384, {}, id="k4096_kp16_b2"),
    pytest.param((1, 1, 6), 512, 512, 32, 1 << 20, {}, id="w1m_key_ids_above_16_bits"),
    pytest.param((1, 1, 50), 64, 96, 8, 4096, {"max_grid": (1, 1)}, id="grid1x1_tuple_r50"),
    pytest.param((1, 1, 37), 64, 96, 8, 4096, {"max_grid": ttnn.CoreCoord(2, 3)}, id="grid2x3_corecoord_r37"),
    pytest.param((1, 1, 64), 32, 48, 4, 1024, {"max_grid": (1000, 1000)}, id="grid_oversize_clamped"),
    pytest.param((1, 1, 24), 32, 48, 4, 1024, {"memory_config": DRAM}, id="memory_config_dram_explicit"),
]


@pytest.mark.parametrize(("lead", "K", "KP", "B", "W", "kwargs"), VALUE_CASES)
def test_values(device, block_keys_op, bk, lead, K, KP, B, W, kwargs):
    _run_and_check(bk, block_keys_op, device, lead, K, KP, B, W, seed=K * 7 + KP * 3 + B, **kwargs)


@pytest.mark.parametrize("multiple", [1, 2], ids=["rows_eq_grid", "rows_2x_grid"])
def test_rows_multiple_of_grid(device, block_keys_op, bk, multiple):
    """An even split: every core of the device grid gets the same number of rows."""
    g = device.compute_with_storage_grid_size()
    _run_and_check(bk, block_keys_op, device, (1, 1, g.x * g.y * multiple), 32, 32, 8, 2048, seed=multiple)


def test_positions_past_the_gathered_row(device, block_keys_op, bk):
    """Positions in [K block, 0xFFFFFFFF) are not none but name no gathered block: the golden reference gives none."""
    _run_and_check(bk, block_keys_op, device, (1, 1, 12), 16, 32, 8, 1024, seed=11, pos_oob=True)


def test_output_tensor_written_in_place(device, block_keys_op, bk):
    """A caller's output buffer (dirty) is written whole and returned; memory_config alongside it is accepted."""
    lead, K, KP, B, W = (2, 9), 24, 40, 4, 2048
    dirty = torch.full((*lead, KP), 0x5A5A5A5A, dtype=torch.int64)
    ot = bk.up(dirty, device)
    before = bk.addr(ot)
    out, pt, it, want = _run_and_check(
        bk, block_keys_op, device, lead, K, KP, B, W, seed=5, output_tensor=ot, memory_config=DRAM
    )
    assert bk.addr(out) == before, "the caller's output buffer must be the one written"
    bk.assert_exact(bk.down(ot), want, "caller's output_tensor")


def test_deterministic_repeat(device, block_keys_op, bk):
    lead, K, KP, B, W = (1, 1, 33), 128, 256, 16, 8192
    pos, ids = bk.make_inputs(lead, K, KP, B, W, seed=21)
    pt, it = bk.up(pos, device), bk.up(ids, device)
    first = bk.down(block_keys_op(pt, it, block=B, width=W))
    second = bk.down(block_keys_op(pt, it, block=B, width=W))
    bk.assert_exact(first, bk.ref_keys(pos, ids, B, W), "first")
    bk.assert_exact(second, first, "repeat")


def test_consumer_of_output(device, block_keys_op, bk):
    """Integration: the key ids gather the right scores from the full row (ttnn.to_torch + torch.gather consumer)."""
    lead, K, KP, B, W = (1, 1, 8), 32, 64, 8, 1024
    g = torch.Generator().manual_seed(9)
    scores = torch.randn(*lead, W, generator=g)
    ids = torch.stack([torch.randperm(W // B, generator=g)[:K] for _ in range(8)]).view(*lead, K)
    pos = torch.randint(0, K * B, (*lead, KP), generator=g)
    out = block_keys_op(bk.up(pos, device), bk.up(ids, device), block=B, width=W)
    keys = bk.down(out)
    # block_gather's row: the K blocks' B scores each, in ids' order; pos indexes it, keys index the scores
    row = torch.gather(scores, -1, (ids.unsqueeze(-1) * B + torch.arange(B)).reshape(*lead, K * B))
    assert torch.equal(torch.gather(scores, -1, keys), torch.gather(row, -1, pos))


# --------------------------------------------------------------------------- refusals


def _refusal_inputs(device, bk, case):
    lead, K, KP, B, W = (1, 1, 4), 16, 8, 4, 1024
    pos, ids = bk.make_inputs(lead, K, KP, B, W, seed=1)
    pt, it = bk.up(pos, device), bk.up(ids, device)
    kw = {"block": B, "width": W}
    if case.startswith("block_"):
        kw["block"] = int(case.split("_")[1])
    elif case == "pos_int32":
        pt = bk.up(pos, device, dtype=ttnn.int32)
    elif case == "pos_tile":
        pt = bk.up(pos, device, layout=ttnn.TILE_LAYOUT)
    elif case == "pos_l1":
        pt = bk.up(pos, device, memory_config=L1)
    elif case == "ids_int32":
        it = bk.up(ids, device, dtype=ttnn.int32)
    elif case == "ids_tile":
        it = bk.up(ids, device, layout=ttnn.TILE_LAYOUT)
    elif case == "ids_l1":
        it = bk.up(ids, device, memory_config=L1)
    elif case == "lead_mismatch":
        it = bk.up(ids[..., :3, :], device)
    elif case == "out_int32":
        kw["output_tensor"] = bk.up(torch.zeros(*lead, KP, dtype=torch.int64), device, dtype=ttnn.int32)
    elif case == "out_shape":
        kw["output_tensor"] = bk.up(torch.zeros(*lead, KP + 1, dtype=torch.int64), device)
    elif case == "out_tile":
        kw["output_tensor"] = bk.up(torch.zeros(*lead, KP, dtype=torch.int64), device, layout=ttnn.TILE_LAYOUT)
    elif case == "out_l1":
        kw["output_tensor"] = bk.up(torch.zeros(*lead, KP, dtype=torch.int64), device, memory_config=L1)
    elif case == "memory_config_l1":
        kw["memory_config"] = L1
    else:
        raise AssertionError(case)
    return pt, it, kw


REFUSALS = [
    ("block_3", "power of two"),
    ("block_0", "power of two"),
    ("block_24", "power of two"),
    ("block_64", "power of two"),
    ("pos_int32", "pos"),
    ("pos_tile", "pos"),
    ("pos_l1", "pos"),
    ("ids_int32", "ids"),
    ("ids_tile", "ids"),
    ("ids_l1", "ids"),
    ("lead_mismatch", r"(?i)shape"),
    ("out_int32", "output"),
    ("out_shape", "output"),
    ("out_tile", "output"),
    ("out_l1", "output"),
    ("memory_config_l1", "output"),
]


@pytest.mark.parametrize(("case", "message"), REFUSALS, ids=[c for c, _ in REFUSALS])
def test_refusal(device, block_keys_op, bk, expect_error, case, message):
    pt, it, kw = _refusal_inputs(device, bk, case)
    with expect_error(REFUSED, message):
        block_keys_op(pt, it, **kw)


# --------------------------------------------------------------------------- program cache sequences
#
# Program-defining (expected to miss): block (the position split and the key scale), K and KP (the row lengths and
# the scratch layout), the grid (max_grid). Possibly program-defining (a miss or a hit, both correct): the row count
# and width / block (the source compiles them in; a port may pass them at run time). Never program-defining (must
# hit): buffers and their addresses, the data, a caller's output buffer or not, memory_config=DRAM spelled out.


def _fresh(bk, device, lead, K, KP, B, W, seed):
    pos, ids = bk.make_inputs(lead, K, KP, B, W, seed=seed)
    return pos, ids, bk.up(pos, device), bk.up(ids, device)


def test_cache_miss_then_hits_on_fresh_buffers(device, block_keys_op, bk, fresh_cache):
    entries = fresh_cache(device)
    lead, K, KP, B, W = (1, 1, 45), 48, 72, 8, 4096
    live, outs = [], []
    for i in range(3):
        pos, ids, pt, it = _fresh(bk, device, lead, K, KP, B, W, seed=100 + i)
        live.append((pt, it))
        before = entries()
        out = block_keys_op(pt, it, block=B, width=W)
        delta = entries() - before
        assert delta == (1 if i == 0 else 0), f"call {i}: cache entries +{delta}"
        outs.append(out)
        bk.assert_exact(bk.down(out), bk.ref_keys(pos, ids, B, W), f"call {i}")
    addrs = [bk.addr(t) for pair in live for t in pair] + [bk.addr(o) for o in outs]
    assert len(set(addrs)) == len(addrs), f"live buffers must be distinct: {addrs}"
    # the first output is still correct (a later call did not write into it)
    bk.assert_exact(bk.down(outs[0]), bk.ref_keys(*bk.make_inputs(lead, K, KP, B, W, seed=100), B, W), "kept output")


def test_cache_output_tensor_transitions(device, block_keys_op, bk, fresh_cache):
    """present -> absent -> present -> absent: one program; each call writes the current output buffer."""
    entries = fresh_cache(device)
    lead, K, KP, B, W = (1, 1, 21), 40, 56, 4, 2048
    keep = []
    for i, present in enumerate((True, False, True, False)):
        pos, ids, pt, it = _fresh(bk, device, lead, K, KP, B, W, seed=200 + i)
        kw = {}
        if present:
            kw["output_tensor"] = bk.up(torch.full((*lead, KP), 7, dtype=torch.int64), device)
        keep.append((pt, it, kw.get("output_tensor")))
        before = entries()
        out = block_keys_op(pt, it, block=B, width=W, **kw)
        delta = entries() - before
        assert delta == (1 if i == 0 else 0), f"call {i} (output_tensor {present}): cache entries +{delta}"
        if present:
            assert bk.addr(out) == bk.addr(kw["output_tensor"])
        bk.assert_exact(bk.down(out), bk.ref_keys(pos, ids, B, W), f"call {i}")


# (field, A kwargs, B kwargs, B must miss): A -> B -> A, fresh buffers every call
A_CASE = {"lead": (1, 1, 30), "K": 32, "KP": 64, "B": 8, "W": 4096}
FIELD_CASES = [
    pytest.param("block", {"B": 16}, True, id="block_8_16_8"),
    pytest.param("KP", {"KP": 80}, True, id="kp_64_80_64"),
    pytest.param("K", {"K": 48}, True, id="k_32_48_32"),
    pytest.param("grid", {"max_grid": (2, 2)}, True, id="grid_full_2x2_full"),
    pytest.param("rows", {"lead": (1, 1, 31)}, None, id="rows_30_31_30"),
    pytest.param("width", {"W": 8192}, None, id="width_4096_8192_4096"),
]


@pytest.mark.parametrize(("field", "change", "must_miss"), FIELD_CASES)
def test_cache_program_field_round_trip(device, block_keys_op, bk, fresh_cache, field, change, must_miss):
    entries = fresh_cache(device)
    keep = []
    for i, c in enumerate(({}, change, {})):
        p = {**A_CASE, **c}
        mg = p.pop("max_grid", None)
        kw = {"max_grid": mg} if mg is not None else {}
        pos, ids, pt, it = _fresh(bk, device, p["lead"], p["K"], p["KP"], p["B"], p["W"], seed=300 + i)
        keep.append((pt, it))
        before = entries()
        out = block_keys_op(pt, it, block=p["B"], width=p["W"], **kw)
        delta = entries() - before
        if i == 0:
            assert delta == 1, f"A: cache entries +{delta}"
        elif i == 1:
            assert delta == 1 if must_miss else delta in (0, 1), f"B ({field}): cache entries +{delta}"
        else:
            assert delta == 0, f"back to A ({field}) must hit: cache entries +{delta}"
        keep.append(out)
        bk.assert_exact(bk.down(out), bk.ref_keys(pos, ids, p["B"], p["W"]), f"{field} call {i}")


def test_cache_valid_invalid_valid(device, block_keys_op, bk, fresh_cache, expect_error):
    """A warmed program does not let an invalid call through; the next valid call hits and is correct."""
    entries = fresh_cache(device)
    lead, K, KP, B, W = (1, 1, 26), 16, 24, 4, 1024
    pos, ids, pt, it = _fresh(bk, device, lead, K, KP, B, W, seed=400)
    bk.assert_exact(bk.down(block_keys_op(pt, it, block=B, width=W)), bk.ref_keys(pos, ids, B, W), "warm")
    warmed = entries()
    with expect_error(REFUSED, "power of two"):
        block_keys_op(pt, it, block=3, width=W)
    bad_out = bk.up(torch.zeros(*lead, KP, dtype=torch.int64), device, dtype=ttnn.int32)
    with expect_error(REFUSED, "output"):
        block_keys_op(pt, it, block=B, width=W, output_tensor=bad_out)
    with expect_error(REFUSED, "pos"):
        block_keys_op(bk.up(pos, device, memory_config=L1), it, block=B, width=W)
    assert entries() == warmed, "a refused call must not add a program"
    pos2, ids2, pt2, it2 = _fresh(bk, device, lead, K, KP, B, W, seed=401)
    before = entries()
    out = block_keys_op(pt2, it2, block=B, width=W)
    assert entries() == before, "the valid call after the refusals must hit"
    bk.assert_exact(bk.down(out), bk.ref_keys(pos2, ids2, B, W), "after refusals")
