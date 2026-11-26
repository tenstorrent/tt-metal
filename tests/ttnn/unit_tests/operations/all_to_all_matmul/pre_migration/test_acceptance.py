# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.operations.all_to_all_matmul:all_to_all_matmul``.

Every test calls the operation through the ``all_to_all_matmul_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (``all_to_all_matmul(input_tensor, weight, *, cluster_axis, topology=Ring, num_links=2,
memory_config=None, compute_kernel_config=None) -> ttnn.Tensor``):
    Within each group of G devices along ``cluster_axis``, device q returns rows [q*S/G, (q+1)*S/G) of every source
    p's (B, Hl, S, K) input, concatenated along the heads in source order (head p*Hl + j), each head multiplied by its
    own (K, N) slice of the replicated (1, G*Hl, K, N) weight: per-device output (B, G*Hl, S/G, N), bfloat16, TILE,
    interleaved in ``memory_config`` (default DRAM). ``compute_kernel_config`` (a ComputeConfigDescriptor) defaults to
    HiFi4 + fp32 DEST. Inputs are not modified; the output is a new allocation. One ``ttnn.generic_op`` per source call.
    Supported today: input bf16 TILE DRAM-interleaved; weight bf16 | bf8b TILE interleaved; fp32 DEST on | off;
    output DRAM | L1 interleaved; tile-aligned and non-aligned rows per device; cluster_axis 1, Ring, 2 links.
    Refusals: caller errors -> ValueError (rank, cluster_axis, group extent, weight dims / heads / K, S % G, K / N
    multiple of 32, sharded memory_config, num_links < 1 or more than the hop has, weight not TILE-interleaved, input
    not in DRAM); SUPPORTED misses -> NotImplementedError (UnsupportedAxisValue); planner capacity limits ->
    RuntimeError (weight row block > 256 KiB: regime R2; > 8 head slots per core; landed rows per core > 384 KiB:
    regime R3).
    Cache: one program per distinct configuration; a call with fresh buffers / new values of the same configuration
    is a hit. weight dtype, output buffer, DEST width, math fidelity and the shapes (alignment included) select a
    program.

Not reachable on a 1x4 snake ring (left to the golden / regression suites or untested): a Ring with no wrap link, a
fabric payload smaller than one tile, the header-pool and "route leaves through neither ring neighbour" planner
refusals, cluster_axis 0 with G > 1.
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

SOURCE_ENTRY = "ttnn.operations.all_to_all_matmul:all_to_all_matmul"
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
        if name.startswith("ttnn.operations.all_to_all_matmul") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native all_to_all_matmul must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def all_to_all_matmul_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"all_to_all_matmul acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch  # noqa: E402

from eval.golden_tests.all_to_all_matmul.conftest import focus_mesh_or_skip  # noqa: E402
from eval.golden_tests.all_to_all_matmul.helpers import (  # noqa: E402
    _tolerance_for,
    check_output,
    create_ttnn_input_tensor,
    output_shape,
    pytorch_all_to_all_matmul,
    read_back,
    replicate,
)
from eval.metrics import LOWER_FIDELITY_PCC  # noqa: E402

# mesh_device / _fabric_mesh_cache: the golden suite's focus ring (G = 4 along cluster_axis 1, FABRIC_2D_TORUS_XY),
# shared through the parent directory's conftest.py.

AXIS = 1
G = 4
BF16 = ttnn.bfloat16
BF8B = ttnn.bfloat8_b
DRAM = ttnn.BufferType.DRAM
L1 = ttnn.BufferType.L1
HIFI4 = ttnn.MathFidelity.HiFi4
HIFI2 = ttnn.MathFidelity.HiFi2

SMALL_X, SMALL_W = (1, 4, 128, 64), (1, 16, 64, 128)
KIMI_X, KIMI_W = (1, 24, 640, 128), (1, 96, 128, 512)


def _cfg(fp32_dest_acc_en=True, fidelity=HIFI4):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = fidelity
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


def _mc(buffer_type):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)


class _Case:
    """Host data (stacked per device), the device tensors and the reference of one call."""

    def __init__(self, mesh, x_shape, w_shape, *, seed, w_dtype=BF16, offset=0.0, x=None, w=None, x_mc=None):
        rows, cols = tuple(mesh.shape)
        assert (rows, cols)[AXIS] == G, f"focus ring has G={G}, mesh {(rows, cols)}"
        gen = torch.Generator().manual_seed(seed)
        if x is None:
            x = (torch.randn((rows, cols, *x_shape), generator=gen) + offset).to(torch.bfloat16)
        if w is None:
            w = (torch.randn(w_shape, generator=gen) / w_shape[-2] ** 0.5).to(torch.bfloat16)
        self.mesh, self.x_shape, self.w_shape = mesh, tuple(x.shape[2:]), tuple(w.shape)
        self.x = x
        self.tx = create_ttnn_input_tensor(x, mesh, dtype=BF16, layout=ttnn.TILE_LAYOUT, memory_config=x_mc)
        self.tw = create_ttnn_input_tensor(replicate(w, mesh), mesh, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)
        # The reference multiplies by the weight the devices hold (bfloat8_b quantized once, at creation).
        self.w = read_back(self.tw, (rows, cols))[0, 0] if w_dtype == BF8B else w
        self._expected = None

    @property
    def expected(self):
        if self._expected is None:
            self._expected = pytorch_all_to_all_matmul(self.x, self.w, AXIS)
        return self._expected

    def addresses(self):
        return {
            ttnn.get_device_tensors(self.tx)[0].buffer_address(),
            ttnn.get_device_tensors(self.tw)[0].buffer_address(),
        }


def _call(op, mesh, case, **kwargs):
    """Call the op once; return (output, program-cache entries it added). Counted around the call only."""
    before = mesh.num_program_cache_entries()
    out = op(case.tx, case.tw, cluster_axis=AXIS, **kwargs)
    delta = mesh.num_program_cache_entries() - before
    if op.mode == "source":
        assert op.last_dispatches == 1, f"source route made {op.last_dispatches} generic_op calls, expected 1"
    return out, delta


def _check(out, case, *, fp32_dest_acc_en=True, fidelity=HIFI4, out_buffer=DRAM):
    """Output metadata (shape, dtype, layout, memory config, one tensor per device) and values against the reference
    at the golden tolerances (a lower fidelity grades PCC at LOWER_FIDELITY_PCC, as the golden helpers do)."""
    assert isinstance(out, ttnn.Tensor), f"expected one ttnn.Tensor, got {type(out)}"
    mc = out.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    assert mc.buffer_type == out_buffer, mc
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.dtype == BF16
    shape = output_shape(case.x_shape, case.w_shape, G)
    assert tuple(out.shape) == shape, f"{tuple(out.shape)} vs {shape}"
    assert len(ttnn.get_device_tensors(out)) == case.mesh.get_num_devices()
    pcc, rms = _tolerance_for(BF16, fp32_dest_acc_en)
    if fidelity != HIFI4:
        pcc = LOWER_FIDELITY_PCC
    check_output(
        out,
        case.expected,
        shape=shape,
        dtype=BF16,
        expected_layout=ttnn.TILE_LAYOUT,
        group=G,
        tolerance=(pcc, rms),
        fp32_dest_acc_en=fp32_dest_acc_en,
    )


def _addr(t):
    return ttnn.get_device_tensors(t)[0].buffer_address()


def _bits(t):
    return t.contiguous().view(torch.int16)


# --------------------------------------------------------------------------- API / output contract


def test_defaults_output_contract_inputs_untouched(mesh_device, all_to_all_matmul_op):
    """Only cluster_axis given: DRAM-interleaved bf16 TILE output of (B, H, S/G, N) per device, a new allocation; the
    input and the weight are not modified."""
    case = _Case(mesh_device, SMALL_X, SMALL_W, seed=11)
    x_before = read_back(case.tx, tuple(mesh_device.shape))
    w_before = read_back(case.tw, tuple(mesh_device.shape))
    out, _ = _call(all_to_all_matmul_op, mesh_device, case)
    _check(out, case)
    assert _addr(out) not in case.addresses(), "output aliases an input buffer"
    assert torch.equal(_bits(read_back(case.tx, tuple(mesh_device.shape))), _bits(x_before)), "input was modified"
    assert torch.equal(_bits(read_back(case.tw, tuple(mesh_device.shape))), _bits(w_before)), "weight was modified"


def test_explicit_defaults_match_omitted(mesh_device, all_to_all_matmul_op):
    """absent -> present -> absent: spelling every default out (Ring, 2 links, DRAM interleaved, HiFi4 + fp32 DEST)
    gives the bit-identical result; returning to the omitted form reuses the first call's program."""
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    case = _Case(mesh_device, SMALL_X, SMALL_W, seed=12)
    first, d0 = _call(all_to_all_matmul_op, mesh_device, case)
    assert d0 == 1, f"first call on an empty cache added {d0} programs (one program per call)"
    _check(first, case)
    explicit, _ = _call(
        all_to_all_matmul_op,
        mesh_device,
        case,
        topology=ttnn.Topology.Ring,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_cfg(True, HIFI4),
    )
    _check(explicit, case)
    shape = tuple(mesh_device.shape)
    assert torch.equal(_bits(read_back(explicit, shape)), _bits(read_back(first, shape)))
    again, d2 = _call(all_to_all_matmul_op, mesh_device, case)
    assert d2 == 0, f"the omitted-defaults call again added {d2} programs (expected a hit)"
    _check(again, case)


# --------------------------------------------------------------------------- planner branches (miss, then a proven hit)

# (x_shape, w_shape, weight dtype). Every case: on an empty cache, a miss (one program), then a call with fresh
# buffers and new data that must hit and still match its own reference.
PLANNER_CASES = [
    pytest.param((1, 4, 128, 32), (1, 16, 32, 32), BF16, id="min_one_tile_per_head"),
    pytest.param((2, 8, 128, 128), (1, 32, 128, 512), BF16, id="multi_batch_shared_weight"),
    pytest.param((1, 4, 128, 128), (1, 16, 128, 4096), BF16, id="multi_n_block"),
    pytest.param(KIMI_X, KIMI_W, BF16, id="kimi_idle_cores_tail_split"),
    pytest.param((1, 32, 640, 128), (1, 128, 128, 512), BF16, id="deepseek_two_heads_per_core"),
    pytest.param((1, 16, 640, 192), (1, 64, 192, 512), BF16, id="glm_k192"),
    pytest.param((1, 192, 128, 32), (1, 768, 32, 32), BF16, id="max_head_slots_per_core"),
    pytest.param((1, 4, 128, 4096), (1, 16, 4096, 32), BF16, id="weight_block_at_limit_bf16_k4096"),
    pytest.param((1, 4, 128, 4128), (1, 16, 4128, 64), BF8B, id="bf8b_weight_k4128_past_bf16_limit"),
    pytest.param((1, 4, 4096, 128), (1, 16, 128, 64), BF16, id="long_rows_a_resident"),
    pytest.param((1, 24, 160, 128), (1, 96, 128, 512), BF16, id="non_aligned_40_rows"),
    pytest.param((1, 8, 96, 128), (1, 32, 128, 512), BF16, id="non_aligned_24_rows"),
    pytest.param((1, 8, 48, 64), (1, 32, 64, 128), BF16, id="non_aligned_input_1p5_tiles"),
    pytest.param((2, 4, 200, 64), (1, 16, 64, 128), BF16, id="non_aligned_multi_batch_50_rows"),
]


@pytest.mark.parametrize("x_shape,w_shape,w_dtype", PLANNER_CASES)
def test_planner_branch_miss_then_hit(mesh_device, all_to_all_matmul_op, x_shape, w_shape, w_dtype):
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    a = _Case(mesh_device, x_shape, w_shape, seed=21, w_dtype=w_dtype)
    out_a, d_a = _call(all_to_all_matmul_op, mesh_device, a)
    assert d_a == 1, f"miss added {d_a} programs, expected 1"
    _check(out_a, a)
    b = _Case(mesh_device, x_shape, w_shape, seed=22, w_dtype=w_dtype, offset=0.5)
    assert not (a.addresses() & b.addresses()), "fresh inputs reuse a live buffer address"
    out_b, d_b = _call(all_to_all_matmul_op, mesh_device, b)
    assert d_b == 0, f"same configuration with fresh buffers added {d_b} programs (expected a hit)"
    assert _addr(out_b) != _addr(out_a)
    _check(out_b, b)
    _check(out_a, a)  # the earlier output is still intact


# --------------------------------------------------------------------------- cache sequences


def test_fresh_buffers_sequence_and_determinism(mesh_device, all_to_all_matmul_op):
    """A -> new A -> new A -> A again (Kimi shape, all tensors kept alive): one miss then hits; every output follows
    its own current inputs (different data and weight values each time, no stale rows or arrival / fence state), and
    the same host data through new buffers is bit-identical to the first result."""
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    cases, outs, deltas = [], [], []
    for seed, offset in ((31, 0.0), (32, 2.0), (33, -1.0)):
        case = _Case(mesh_device, KIMI_X, KIMI_W, seed=seed, offset=offset)
        for prev in cases:
            assert not (prev.addresses() & case.addresses()), "fresh inputs reuse a live buffer address"
        out, d = _call(all_to_all_matmul_op, mesh_device, case)
        cases.append(case)
        outs.append(out)
        deltas.append(d)
        _check(out, case)
    assert deltas == [1, 0, 0], f"program-cache deltas {deltas}, expected a miss then hits"
    assert len({_addr(o) for o in outs}) == 3
    repeat = _Case(mesh_device, KIMI_X, KIMI_W, seed=31, x=cases[0].x, w=cases[0].w)
    out, d = _call(all_to_all_matmul_op, mesh_device, repeat)
    assert d == 0
    shape = tuple(mesh_device.shape)
    assert torch.equal(_bits(read_back(out, shape)), _bits(read_back(outs[0], shape))), "repeat is not bit-identical"


# field -> (A kwargs, B kwargs). Each field changes what the program is built from (CB data formats, the output
# buffer the writer addresses, the DEST width / math fidelity compiled into compute, the tile counts): B must miss,
# the return to A must hit, then B again must hit.
_ROUND_TRIPS = {
    "weight_dtype": (dict(w_dtype=BF16), dict(w_dtype=BF8B)),
    "output_buffer": (dict(out_buffer=DRAM), dict(out_buffer=L1)),
    "fp32_dest_acc_en": (dict(fp32=True), dict(fp32=False)),
    "math_fidelity": (dict(fidelity=HIFI4), dict(fidelity=HIFI2)),
    "rows_per_device": (dict(x_shape=(1, 4, 128, 64)), dict(x_shape=(1, 4, 256, 64))),
    "alignment": (dict(x_shape=(1, 4, 128, 64)), dict(x_shape=(1, 4, 96, 64))),
    "n_width": (dict(w_shape=(1, 16, 64, 128)), dict(w_shape=(1, 16, 64, 256))),
}


@pytest.mark.parametrize("field", list(_ROUND_TRIPS))
def test_program_field_round_trip(mesh_device, all_to_all_matmul_op, field):
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    base = dict(x_shape=SMALL_X, w_shape=SMALL_W, w_dtype=BF16, out_buffer=DRAM, fp32=True, fidelity=HIFI4)
    a_kw, b_kw = _ROUND_TRIPS[field]
    expected_deltas = [1, 1, 0, 0]
    deltas = []
    for step, (kw, seed) in enumerate(((a_kw, 41), (b_kw, 42), (a_kw, 43), (b_kw, 44))):
        p = {**base, **kw}
        case = _Case(mesh_device, p["x_shape"], p["w_shape"], seed=seed, w_dtype=p["w_dtype"])
        out, d = _call(
            all_to_all_matmul_op,
            mesh_device,
            case,
            memory_config=_mc(p["out_buffer"]),
            compute_kernel_config=_cfg(p["fp32"], p["fidelity"]),
        )
        deltas.append(d)
        _check(out, case, fp32_dest_acc_en=p["fp32"], fidelity=p["fidelity"], out_buffer=p["out_buffer"])
    assert deltas == expected_deltas, f"{field}: program-cache deltas {deltas}, expected {expected_deltas}"


@pytest.mark.parametrize("optional", ["memory_config", "compute_kernel_config"])
def test_optional_absent_present_absent(mesh_device, all_to_all_matmul_op, optional):
    """Optional argument omitted -> set to a non-default value -> omitted: each output follows the current setting;
    the non-default value is its own program and the return to the omitted form reuses the first."""
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    present = {
        "memory_config": dict(memory_config=_mc(L1)),
        "compute_kernel_config": dict(compute_kernel_config=_cfg(False, HIFI2)),
    }
    check_present = {
        "memory_config": dict(out_buffer=L1),
        "compute_kernel_config": dict(fp32_dest_acc_en=False, fidelity=HIFI2),
    }
    deltas = []
    for kw, chk, seed in (({}, {}, 51), (present[optional], check_present[optional], 52), ({}, {}, 53)):
        case = _Case(mesh_device, SMALL_X, SMALL_W, seed=seed)
        out, d = _call(all_to_all_matmul_op, mesh_device, case, **kw)
        deltas.append(d)
        _check(out, case, **chk)
    assert deltas == [1, 1, 0], f"program-cache deltas {deltas}, expected [1, 1, 0]"


def test_valid_invalid_valid(mesh_device, all_to_all_matmul_op, expect_error):
    """A warmed cache still rejects an invalid call (weight K != input K) without adding a program; the next valid
    call hits and is correct."""
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    a = _Case(mesh_device, SMALL_X, SMALL_W, seed=61)
    out, d = _call(all_to_all_matmul_op, mesh_device, a)
    assert d == 1
    _check(out, a)
    bad = _Case(mesh_device, SMALL_X, (1, 16, 128, 128), seed=62)
    before = mesh_device.num_program_cache_entries()
    with expect_error(ValueError, "weight K"):
        all_to_all_matmul_op(bad.tx, bad.tw, cluster_axis=AXIS)
    assert mesh_device.num_program_cache_entries() == before, "a rejected call added a program"
    b = _Case(mesh_device, SMALL_X, SMALL_W, seed=63)
    out, d = _call(all_to_all_matmul_op, mesh_device, b)
    assert d == 0
    _check(out, b)


# --------------------------------------------------------------------------- exact data movement


@pytest.mark.parametrize(
    "B,Hl,S,K",
    [(1, 8, 256, 64), (1, 8, 160, 64), (2, 4, 48, 32)],
    ids=["aligned_64_rows", "non_aligned_40_rows", "non_aligned_b2_12_rows"],
)
def test_row_and_head_provenance_bit_exact(mesh_device, all_to_all_matmul_op, B, Hl, S, K):
    """Identity weight per head, closed-form input (even columns: source row index + 1; odd columns: a unique negative
    id per source chip / head / batch; all exact in bf16): every output row equals its source row bit for bit, so a
    row from the wrong quarter, a head from the wrong source or index, or a re-tile shift error is caught exactly."""
    H = G * Hl
    rows, cols = tuple(mesh_device.shape)
    x = torch.zeros((rows, cols, B, Hl, S, K), dtype=torch.float32)
    x[..., 0::2] = torch.arange(S, dtype=torch.float32).view(S, 1) + 1
    for c in range(cols):
        for j in range(Hl):
            for b in range(B):
                x[:, c, b, j, :, 1::2] = -(1 + c * Hl * B + j * B + b)
    x = x.to(torch.bfloat16)
    w = torch.eye(K, dtype=torch.float32).expand(1, H, K, K).contiguous().to(torch.bfloat16)
    case = _Case(mesh_device, (B, Hl, S, K), (1, H, K, K), seed=0, x=x, w=w)
    out, _ = _call(all_to_all_matmul_op, mesh_device, case)
    got = read_back(out, (rows, cols))
    want = case.expected.to(torch.bfloat16)
    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    bad = (_bits(got) != _bits(want)).any(dim=-1)
    where = [
        f"dev{c} b{b} head{h}: rows {torch.nonzero(bad[0, c, b, h]).flatten().tolist()[:8]}"
        for c in range(cols)
        for b in range(B)
        for h in range(H)
        if bad[0, c, b, h].any()
    ]
    assert not where, "rows not moved bit-exactly:\n" + "\n".join(where[:12])


# --------------------------------------------------------------------------- refusals


def _refusal_case(mesh, name):
    """(input tensor, weight tensor, kwargs) for one refusal."""
    x, w, kw, w_dtype, x_mc = SMALL_X, SMALL_W, {}, BF16, None
    rows, cols = tuple(mesh.shape)
    if name == "rank3_input":
        x = (4, 128, 64)
    elif name == "rank3_weight":
        w = (16, 64, 128)
    elif name == "cluster_axis_2":
        kw = dict(cluster_axis=2)
    elif name == "cluster_axis_extent_1":
        kw = dict(cluster_axis=0)
    elif name == "weight_leading_dim_2":
        w = (2, 16, 64, 128)
    elif name == "weight_heads_not_g_times_hl":
        w = (1, 8, 64, 128)
    elif name == "weight_k_mismatch":
        w = (1, 16, 128, 128)
    elif name == "s_not_divisible_by_g":
        x = (1, 4, 34, 64)
    elif name == "k_not_multiple_of_32":
        x, w = (1, 4, 128, 48), (1, 16, 48, 128)
    elif name == "n_not_multiple_of_32":
        w = (1, 16, 64, 48)
    elif name == "sharded_memory_config":
        kw = dict(memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, L1))
    elif name == "num_links_0":
        kw = dict(num_links=0)
    elif name == "num_links_beyond_hop":
        kw = dict(num_links=64)
    elif name == "input_in_l1":
        x_mc = ttnn.L1_MEMORY_CONFIG
    elif name == "planner_weight_block_r2":
        x, w = (1, 4, 128, 4128), (1, 16, 4128, 32)
    elif name == "planner_head_slots_per_core":
        x, w = (1, 256, 128, 32), (1, 1024, 32, 32)
    elif name == "planner_a_resident_r3":
        x, w = (1, 4, 16384, 128), (1, 16, 128, 32)
    gen = torch.Generator().manual_seed(7)
    hx = torch.randn((rows, cols, *x), generator=gen).to(torch.bfloat16)
    hw = torch.randn(w, generator=gen).to(torch.bfloat16)
    x_dtype, x_layout, w_layout = BF16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT
    if name == "input_float32":
        x_dtype = ttnn.float32
    elif name == "weight_float32":
        w_dtype = ttnn.float32
    elif name == "input_row_major":
        x_layout = ttnn.ROW_MAJOR_LAYOUT
    elif name == "weight_row_major":
        w_layout = ttnn.ROW_MAJOR_LAYOUT
    elif name == "topology_linear":
        kw = dict(topology=ttnn.Topology.Linear)
    elif name == "num_links_1":
        kw = dict(num_links=1)
    tx = create_ttnn_input_tensor(hx, mesh, dtype=x_dtype, layout=x_layout, memory_config=x_mc)
    tw = create_ttnn_input_tensor(replicate(hw, mesh), mesh, dtype=w_dtype, layout=w_layout)
    return tx, tw, {"cluster_axis": AXIS, **kw}


# name -> (exception type, a key fragment of the diagnostic)
REFUSALS = {
    "rank3_input": (ValueError, "rank 4"),
    "rank3_weight": (ValueError, "rank 4"),
    "cluster_axis_2": (ValueError, "cluster_axis"),
    "cluster_axis_extent_1": (ValueError, "nothing to exchange"),
    "weight_leading_dim_2": (ValueError, "leading dim"),
    "weight_heads_not_g_times_hl": (ValueError, "heads"),
    "weight_k_mismatch": (ValueError, "weight K"),
    "s_not_divisible_by_g": (ValueError, "divisible"),
    "k_not_multiple_of_32": (ValueError, "multiples of 32"),
    "n_not_multiple_of_32": (ValueError, "multiples of 32"),
    "sharded_memory_config": (ValueError, "interleaved"),
    "num_links_0": (ValueError, "num_links"),
    "num_links_beyond_hop": (ValueError, "usable link"),
    "weight_row_major": (ValueError, "TILE"),
    "input_in_l1": (ValueError, "DRAM"),
    "input_float32": (NotImplementedError, "dtype"),
    "weight_float32": (NotImplementedError, "weight_dtype"),
    "input_row_major": (NotImplementedError, "layout"),
    "topology_linear": (NotImplementedError, "topology"),
    "num_links_1": (NotImplementedError, "num_links"),
    "planner_weight_block_r2": (RuntimeError, "R2"),
    "planner_head_slots_per_core": (RuntimeError, "head slots per core"),
    "planner_a_resident_r3": (RuntimeError, "R3"),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_refusal(mesh_device, all_to_all_matmul_op, expect_error, name):
    if name == "cluster_axis_extent_1" and tuple(mesh_device.shape)[0] != 1:
        pytest.skip(f"mesh {tuple(mesh_device.shape)} has more than one row: cluster_axis 0 is not a 1-device group")
    exc, fragment = REFUSALS[name]
    tx, tw, kw = _refusal_case(mesh_device, name)
    before = mesh_device.num_program_cache_entries()
    with expect_error(exc, fragment):
        all_to_all_matmul_op(tx, tw, **kw)
    assert mesh_device.num_program_cache_entries() == before, "a refused call added a program"


# --------------------------------------------------------------------------- fabric packet size (keep last: reopens)


@pytest.mark.parametrize("payload_bytes", [2 * 2048 + 64, 7232], ids=["two_tiles_4160B", "kimi_production_7232B"])
def test_fabric_packet_size(_fabric_mesh_cache, all_to_all_matmul_op, payload_bytes):
    """The Kimi call on a ring opened with a smaller router payload than the arch maximum (two bf16 tiles; Kimi K3.1's
    production 7232 B, three tiles): packets are sized from the active fabric config, and the result is unchanged."""
    mesh = _fabric_mesh_cache.get(focus_mesh_or_skip(), payload_bytes=payload_bytes)
    case = _Case(mesh, KIMI_X, KIMI_W, seed=71)
    out, _ = _call(all_to_all_matmul_op, mesh, case)
    _check(out, case)
