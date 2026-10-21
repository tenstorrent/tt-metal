# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.operations.matmul_reduce_scatter:matmul_reduce_scatter``.

Every test calls the operation through the ``matmul_reduce_scatter_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Contract (``matmul_reduce_scatter(input_tensor, weight, *, cluster_axis, scatter_dim=-2, topology=Linear,
num_links=None, compute_kernel_config=None, memory_config=None)``):
    Every device of a group along ``cluster_axis`` (G >= 2 devices) holds A (..., M, K) (rank 2-4, leading dims 1) and
    W (K, N). Each device returns one ttnn.Tensor: block p (its group position) of sum_g A[g] @ W[g], split along
    ``scatter_dim`` (-2: M, -1: N) into G tile-aligned blocks. Output: A's rank, shape [..., M/G, N] or [..., M, N/G],
    bfloat16, TILE, ``memory_config`` (interleaved; default DRAM). Inputs are not modified.
    SUPPORTED: dtype bf16 | bf8b, weight dtype bf16 | bf8b | bf4b, TILE, M/K/N multiples of 32, cluster_axis 0 | 1,
    scatter_dim -1 | -2 (or the positive index of those dims), topology Linear | Ring, num_links 1 | 2 (None: every
    usable link of the worst hop). A Ring of 2 devices runs as the line. compute_kernel_config: math fidelity is
    honoured; fp32 DEST accumulation is a floor (always on).
    Refusals: ``UnsupportedAxisValue`` (a NotImplementedError) for a value outside SUPPORTED; ``ValueError`` for
    caller errors (rank / leading dims, K mismatch, non-DRAM-interleaved inputs, an axis of < 2 devices, a scattered
    extent that does not split into G tile blocks, more links than a hop has, a Ring under a fabric config that does
    not wrap the axis); ``NotImplementedError`` for a sharded output memory_config.
    Program variants (the source's compile-time args): shapes, dtypes, scatter_dim, cluster_axis, topology (line /
    balanced ring), the link count, math fidelity, the output memory config, and the planner's regime (resident /
    stream cross-block operand) and output chunking, both derived from shapes. Buffer addresses (inputs, output, the
    per-mesh staging tensor, mailbox and global semaphores) are runtime args: fresh buffers hit the program cache.

Meshes (conftest.py): ``mrs_ring_mesh`` (a ring of >= 3 devices, FABRIC_2D_TORUS_XY), ``mrs_line_mesh`` (the system
mesh, FABRIC_2D) and ``mrs_line4_mesh`` (the ring's shape under FABRIC_2D). Tests are grouped by mesh.

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``native`` (default, the op is ported) | ``source``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode defaults it to the op's registered name (``ttnn:matmul_reduce_scatter``),
                            requires a different entry *string* from the source path and a registered C++ op
                            (``is_cpp_operation``).
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

SOURCE_ENTRY = "ttnn.operations.matmul_reduce_scatter:matmul_reduce_scatter"
NATIVE_ENTRY = "ttnn:matmul_reduce_scatter"  # the registered C++ op the public path aliases since the port
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
    mode = os.environ.get(MODE_ENV, "native") or "native"
    entry = os.environ.get(ENTRY_ENV, "") or (NATIVE_ENTRY if mode == "native" else "")
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
        if name.startswith("ttnn.operations.matmul_reduce_scatter") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native matmul_reduce_scatter must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def matmul_reduce_scatter_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"matmul_reduce_scatter acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import torch
from ttnn.operations._op_contract import UnsupportedAxisValue

BF16, BF8, BF4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
RING, LINE = ttnn.Topology.Ring, ttnn.Topology.Linear
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
HIFI2 = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
HIFI2_NO_FP32 = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
HIFI4 = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
# (PCC, relative RMS) keyed by the weight dtype, the narrowest input: the golden suite's helpers.TOLERANCES
TOLERANCES = {BF16: (0.999, 0.02), BF8: (0.998, 0.03), BF4: (0.99, 0.08)}

# per-device (A, W) shapes
SMALL = ((1, 1, 256, 512), (512, 1024))
RANK2 = ((256, 384), (384, 512))
RANK3 = ((1, 256, 256), (256, 768))
EXACT = ((1, 1, 256, 64), (64, 512))
# Kimi K2.7 o_proj, the golden perf focus: A stays resident across blocks (scatter -1)
FOCUS = ((1, 1, 640, 2048), (2048, 7168))
# MiMo-V2 o_proj, bf16 W, scatter -2, a 2-device group: W cannot stay resident, so it streams
MIMO = ((1, 1, 2048, 2048), (2048, 4096))
# a per-chip block of 296 x 40 tiles (scatter -1 over 4 devices): ~115 tiles per core, whose fp32 hold slot plus
# staging slots alone miss L1 under either regime, so the planner splits the output into chunks automatically
CHUNKED = ((1, 1, 9472, 256), (256, 5120))


# --------------------------------------------------------------------------- helpers


def _group_axis(mesh):
    """The group axis of a ring placement: its longer mesh axis."""
    rows, cols = tuple(mesh.shape)
    return 1 if cols >= rows else 0


def _group_size(mesh, axis):
    return tuple(mesh.shape)[axis]


def _max_links(mesh, axis, ring):
    """Fewest fabric links over every hop (both ways) of every group along `axis`: what num_links=None resolves to."""
    rows, cols = tuple(mesh.shape)
    node = lambda rc: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*rc))  # noqa: E731
    groups = (
        [[(r, c) for r in range(rows)] for c in range(cols)]
        if axis == 0
        else [[(r, c) for c in range(cols)] for r in range(rows)]
    )
    best = None
    for grp in groups:
        hops = list(zip(grp[:-1], grp[1:])) + ([(grp[-1], grp[0])] if ring else [])
        for s, d in hops:
            for x, y in ((s, d), (d, s)):
                n = len(ttnn.get_forwarding_link_indices(node(x), node(y)))
                best = n if best is None else min(best, n)
    return best or 0


def _stacked(mesh, shape, seed, scale=1.0):
    """Distinct random data per device as a stacked (R, C, *shape) bf16 tensor; [r, c] is what device (r, c) holds."""
    gen = torch.Generator().manual_seed(seed)
    rows, cols = tuple(mesh.shape)
    return (torch.randn((rows, cols, *shape), generator=gen) * scale).to(torch.bfloat16)


def _quantize(stacked, dtype):
    """The values as the device holds them (block-float rounding), so the reference scores the op, not the format."""
    if dtype in (BF8, BF4):
        host = ttnn.from_torch(stacked.reshape(-1, stacked.shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(host).reshape(stacked.shape).to(torch.bfloat16)
    return stacked


def _to_mesh(stacked, mesh, dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM):
    rows = stacked.shape[0]
    whole = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        whole,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )


def _make(mesh, shapes, seed, dtype=BF16, weight_dtype=BF16):
    """(host A, host W, device A, device W): unit-scale partials (W scaled by 1/sqrt(K)), in DRAM."""
    a_shape, w_shape = shapes
    a = _quantize(_stacked(mesh, a_shape, seed), dtype)
    w = _quantize(_stacked(mesh, w_shape, seed + 1, scale=w_shape[0] ** -0.5), weight_dtype)
    return a, w, _to_mesh(a, mesh, dtype), _to_mesh(w, mesh, weight_dtype)


def _canonical(scatter_dim, rank):
    return scatter_dim - rank if scatter_dim >= 0 else scatter_dim


def _reference(a, w, axis, scatter_dim):
    """Pure torch, fp32: per-device partials A[r, c] @ W[r, c], summed over the group along `axis`; the device at
    group position p keeps block p of the sum along `scatter_dim`. Returns [[expected of device (r, c)]]."""
    rows, cols = a.shape[0], a.shape[1]
    wb = w.float().reshape(rows, cols, *([1] * (a.dim() - 4)), *w.shape[2:])
    partial = torch.matmul(a.float(), wb)
    total = partial.sum(dim=axis, keepdim=True).expand_as(partial)
    g = (rows, cols)[axis]
    return [[torch.chunk(total[r, c], g, dim=scatter_dim)[(r, c)[axis]] for c in range(cols)] for r in range(rows)]


def _out_shape(shapes, scatter_dim, g):
    a_shape, w_shape = shapes
    out = [*a_shape[:-1], w_shape[-1]]
    out[scatter_dim] //= g
    return out


def _pcc_rms(got, ref):
    a, e = got.flatten(), ref.flatten()
    ac, ec = a - a.mean(), e - e.mean()
    den = ac.norm() * ec.norm()
    pcc = float((ac * ec).sum() / den) if den > 1e-30 else 1.0
    return pcc, float((a - e).pow(2).mean().sqrt() / max(float(e.pow(2).mean().sqrt()), 1e-30))


def _host(t):
    return [ttnn.to_torch(part) for part in ttnn.get_device_tensors(t)]


def _check(mesh, out, expected, weight_dtype, out_shape, memory_config=DRAM):
    """Metadata of the returned tensor, then every device against its own expected block (the worst device decides)."""
    assert isinstance(out, ttnn.Tensor), f"one ttnn.Tensor is returned, got {type(out)}"
    assert list(out.shape) == list(out_shape)
    assert out.dtype == BF16, "the output is bfloat16 whatever the input dtypes"
    assert out.layout == ttnn.TILE_LAYOUT
    mc = out.memory_config()
    assert mc.buffer_type == memory_config.buffer_type, mc
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    rows, cols = tuple(mesh.shape)
    parts = _host(out)
    assert len(parts) == rows * cols, "one output per device of the mesh"
    pcc_min, rms_max = TOLERANCES[weight_dtype]
    for idx, got in enumerate(parts):
        r, c = divmod(idx, cols)
        ref = expected[r][c].double()
        got = got.double()
        assert list(got.shape) == list(ref.shape), f"device ({r},{c}): {list(got.shape)} vs {list(ref.shape)}"
        assert torch.isfinite(got).all(), f"device ({r},{c}): non-finite output"
        pcc, rms = _pcc_rms(got, ref)
        assert pcc >= pcc_min and rms <= rms_max, f"device ({r},{c}): pcc {pcc:.6f} rel_rms {rms:.5f}"
    return parts


def _assert_unchanged(t, stacked):
    """The op does not write its inputs: every device still holds exactly what it was given."""
    cols = stacked.shape[1]
    for idx, part in enumerate(ttnn.get_device_tensors(t)):
        r, c = divmod(idx, cols)
        assert torch.equal(ttnn.to_torch(part).float(), stacked[r, c].float()), f"input changed on device ({r},{c})"


def _fresh_cache(mesh):
    ttnn.synchronize_device(mesh)
    mesh.enable_program_cache()
    mesh.clear_program_cache()
    assert mesh.num_program_cache_entries() == 0


def _call(op, mesh, *args, **kwargs):
    """(output, program-cache entries the call added), counted around the operation only."""
    before = mesh.num_program_cache_entries()
    out = op(*args, **kwargs)
    added = mesh.num_program_cache_entries() - before
    if op.mode == "source":
        assert op.last_dispatches == 1, "the source op runs as one ttnn.generic_op per call"
    return out, added


def _run(op, mesh, axis, shapes, call, *, seed=0, dtype=BF16, weight_dtype=BF16):
    """Make inputs, call the op with `call` (+ cluster_axis), check the output and the inputs. (output, added)."""
    a, w, ta, tw = _make(mesh, shapes, seed, dtype, weight_dtype)
    out, added = _call(op, mesh, ta, tw, cluster_axis=axis, **call)
    sd = _canonical(call.get("scatter_dim", -2), len(shapes[0]))
    expected = _reference(a, w, axis, sd)
    mem = call.get("memory_config") or DRAM
    _check(mesh, out, expected, weight_dtype, _out_shape(shapes, sd, _group_size(mesh, axis)), mem)
    _assert_unchanged(ta, a)
    _assert_unchanged(tw, w)
    return out, added


def _needs_links(mesh, axis, ring, n):
    if _max_links(mesh, axis, ring) < n:
        pytest.skip(f"a hop of this cluster has fewer than {n} fabric links")


# --------------------------------------------------------------------------- ring placement (>= 3 devices)


def _case(case_id, shapes, call, dtype=BF16, weight_dtype=BF16):
    return pytest.param(shapes, call, dtype, weight_dtype, id=case_id)


# One representative call per program variant the planner builds from these inputs.
RING_CASES = [
    _case("defaults_linear_s-2", SMALL, {}),  # scatter_dim=-2, Linear, num_links=None, HiFi2, DRAM output
    _case("ring_s-1", SMALL, {"scatter_dim": -1, "topology": RING, "num_links": 1}),
    _case("ring_s-2", SMALL, {"scatter_dim": -2, "topology": RING, "num_links": 1}),
    _case("linear_s-1", SMALL, {"scatter_dim": -1, "topology": LINE, "num_links": 1}),
    _case("ring_links2_s-2", SMALL, {"scatter_dim": -2, "topology": RING, "num_links": 2}),
    _case("ring_rank2_s-1", RANK2, {"scatter_dim": -1, "topology": RING, "num_links": 1}),
    _case("ring_rank3_s-2", RANK3, {"scatter_dim": -2, "topology": RING, "num_links": 1}),
    _case("ring_positive_scatter_dim_3", SMALL, {"scatter_dim": 3, "topology": RING, "num_links": 1}),
    _case("ring_bf8b_bf8b_s-1", SMALL, {"scatter_dim": -1, "topology": RING, "num_links": 1}, BF8, BF8),
    _case("ring_bf4b_weight_s-2", SMALL, {"scatter_dim": -2, "topology": RING, "num_links": 1}, BF16, BF4),
    _case(
        "ring_hifi4_s-1", SMALL, {"scatter_dim": -1, "topology": RING, "num_links": 1, "compute_kernel_config": HIFI4}
    ),
    _case("ring_l1_output_s-2", SMALL, {"scatter_dim": -2, "topology": RING, "num_links": 1, "memory_config": L1}),
    _case("ring_focus_resident_links2", FOCUS, {"scatter_dim": -1, "topology": RING, "num_links": 2}, BF16, BF8),
    _case("ring_auto_chunked_s-1", CHUNKED, {"scatter_dim": -1, "topology": RING}),
]


@pytest.mark.parametrize(("shapes", "call", "dtype", "weight_dtype"), RING_CASES)
def test_ring_placement(mrs_ring_mesh, matmul_reduce_scatter_op, shapes, call, dtype, weight_dtype):
    axis = _group_axis(mrs_ring_mesh)
    if call.get("num_links"):
        _needs_links(mrs_ring_mesh, axis, call.get("topology") == RING, call["num_links"])
    _run(matmul_reduce_scatter_op, mrs_ring_mesh, axis, shapes, call, dtype=dtype, weight_dtype=weight_dtype)


@pytest.mark.parametrize(("topology", "scatter_dim"), [(RING, -1), (LINE, -2)], ids=["ring_s-1", "linear_s-2"])
def test_every_partial_exactly_once(mrs_ring_mesh, matmul_reduce_scatter_op, topology, scatter_dim):
    """A = ones on the first 32 K columns, W = the device's rank (r * C + c + 1) on those rows: every partial is
    exactly 32 * rank and every relayed / final sum an exact bf16 integer, so a dropped, doubled or foreign partial
    (or a block landing on the wrong device) is a wrong integer."""
    mesh, axis = mrs_ring_mesh, _group_axis(mrs_ring_mesh)
    (a_shape, w_shape), (rows, cols) = EXACT, tuple(mesh.shape)
    a = torch.zeros((rows, cols, *a_shape), dtype=torch.bfloat16)
    a[..., :32] = 1
    w = torch.zeros((rows, cols, *w_shape), dtype=torch.float32)
    w[:, :, :32, :] = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols, 1, 1)
    w = w.to(torch.bfloat16)
    out, _ = _call(
        matmul_reduce_scatter_op,
        mesh,
        _to_mesh(a, mesh, BF16),
        _to_mesh(w, mesh, BF16),
        cluster_axis=axis,
        scatter_dim=scatter_dim,
        topology=topology,
        num_links=1,
    )
    expected = _reference(a, w, axis, scatter_dim)
    assert list(out.shape) == _out_shape(EXACT, scatter_dim, _group_size(mesh, axis))
    for idx, got in enumerate(_host(out)):
        r, c = divmod(idx, cols)
        bad = (got.float() != expected[r][c]).sum().item()
        assert bad == 0, f"device ({r},{c}): {bad}/{got.numel()} elements differ"


@pytest.mark.parametrize(
    ("shapes", "call"),
    [
        pytest.param(SMALL, {"scatter_dim": -1, "topology": RING, "num_links": 1}, id="ring_s-1"),
        pytest.param(CHUNKED, {"scatter_dim": -1, "topology": RING}, id="ring_auto_chunked_s-1"),
    ],
)
def test_cache_hits_on_fresh_buffers(mrs_ring_mesh, matmul_reduce_scatter_op, shapes, call):
    """A -> new A -> new A: one miss, then hits on fresh live inputs and outputs with different data. Every result
    follows its own inputs, and the earlier outputs (kept alive) are not written again."""
    mesh, axis = mrs_ring_mesh, _group_axis(mrs_ring_mesh)
    _fresh_cache(mesh)
    kept = []
    for i, seed in enumerate((11, 12, 13)):
        a, w, ta, tw = _make(mesh, shapes, seed)
        out, added = _call(matmul_reduce_scatter_op, mesh, ta, tw, cluster_axis=axis, **call)
        assert added >= 1 if i == 0 else added == 0, f"call {i}: {added} new program-cache entries"
        sd = call["scatter_dim"]
        parts = _check(mesh, out, _reference(a, w, axis, sd), BF16, _out_shape(shapes, sd, _group_size(mesh, axis)))
        _assert_unchanged(ta, a)
        _assert_unchanged(tw, w)
        kept.append((ta, tw, out, parts))
    for role, name in enumerate(("input_tensor", "weight", "output")):
        addresses = {k[role].buffer_address() for k in kept}
        assert len(addresses) == len(kept), f"the live {name} buffers must differ: {addresses}"
    for i, (_, _, out, parts) in enumerate(kept):
        assert all(torch.equal(x, y) for x, y in zip(_host(out), parts)), f"output of call {i} changed later"


def test_cache_shape_change_and_return(mrs_ring_mesh, matmul_reduce_scatter_op):
    """SMALL -> CHUNKED -> SMALL: a new program for the chunked plan (and a much larger per-mesh staging buffer), then
    a hit on the first program whose staging buffer was reallocated in between."""
    mesh, axis = mrs_ring_mesh, _group_axis(mrs_ring_mesh)
    call = {"scatter_dim": -1, "topology": RING}
    _fresh_cache(mesh)
    for i, (shapes, miss) in enumerate(((SMALL, True), (CHUNKED, True), (SMALL, False))):
        _, added = _run(matmul_reduce_scatter_op, mesh, axis, shapes, call, seed=20 + i)
        assert added >= 1 if miss else added == 0, f"call {i}: {added} new program-cache entries"


def _option_sequences():
    """(id, base call, [(override, expect a new program)]): A -> B -> A over one option."""
    base = {"scatter_dim": -1, "topology": RING, "num_links": 1}
    return [
        pytest.param(
            base,
            [({}, True), ({"compute_kernel_config": HIFI4}, True), ({"compute_kernel_config": HIFI2}, False)],
            id="math_fidelity_default_hifi4_hifi2",
        ),
        pytest.param(
            {**base, "scatter_dim": -2},
            [({}, True), ({"memory_config": L1}, True), ({"memory_config": DRAM}, False)],
            id="memory_config_none_l1_dram",
        ),
        pytest.param(
            base,
            [({}, True), ({"topology": LINE}, True), ({}, False)],
            id="topology_ring_linear_ring",
        ),
        pytest.param(
            base,
            [({}, True), ({"scatter_dim": -2}, True), ({"scatter_dim": 3}, False)],
            id="scatter_dim_-1_-2_3",
        ),
        pytest.param(
            base,
            [({}, True), ({"num_links": 2}, True), ({}, False), ({"num_links": None}, False)],
            id="num_links_1_2_1_none",
        ),
    ]


@pytest.mark.parametrize(("base", "steps"), _option_sequences())
def test_cache_option_change_and_return(mrs_ring_mesh, matmul_reduce_scatter_op, base, steps):
    """Each option that is a compile-time property of the program misses when it changes and hits when it returns;
    every call is correct on its own fresh inputs. scatter_dim=3 is the rank-4 index of -1, and num_links=None
    resolves to every usable link (2 here), so both reuse the program of their explicit equivalent."""
    mesh, axis = mrs_ring_mesh, _group_axis(mrs_ring_mesh)
    if any("num_links" in o for o, _ in steps):
        _needs_links(mesh, axis, True, 2)
        if _max_links(mesh, axis, True) != 2:
            pytest.skip("num_links=None resolves to 2 links only where the worst hop has exactly 2")
    _fresh_cache(mesh)
    for i, (override, miss) in enumerate(steps):
        _, added = _run(matmul_reduce_scatter_op, mesh, axis, SMALL, {**base, **override}, seed=30 + i)
        assert added >= 1 if miss else added == 0, f"step {i} {override}: {added} new program-cache entries"


def test_fp32_dest_accumulation_is_a_floor(mrs_ring_mesh, matmul_reduce_scatter_op):
    """fp32_dest_acc_en=False does not lower the op's precision: it accumulates in fp32 DEST either way, so both
    calls give the same bits (and both match the reference)."""
    mesh, axis = mrs_ring_mesh, _group_axis(mrs_ring_mesh)
    a, w, ta, tw = _make(mesh, SMALL, 40)
    expected = _reference(a, w, axis, -1)
    results = []
    for cfg in (HIFI2, HIFI2_NO_FP32):
        out, _ = _call(
            matmul_reduce_scatter_op,
            mesh,
            ta,
            tw,
            cluster_axis=axis,
            scatter_dim=-1,
            topology=RING,
            num_links=1,
            compute_kernel_config=cfg,
        )
        results.append(_check(mesh, out, expected, BF16, _out_shape(SMALL, -1, _group_size(mesh, axis))))
    assert all(torch.equal(x, y) for x, y in zip(*results))


def test_refusals_after_a_warm_call(mrs_ring_mesh, matmul_reduce_scatter_op, expect_error):
    """valid -> every refusal -> valid. A warmed op still refuses each unsupported value or caller error with its
    type, adds no program for it, and the next valid call hits and is correct."""
    mesh, axis = mrs_ring_mesh, _group_axis(mrs_ring_mesh)
    op = matmul_reduce_scatter_op
    other_axis = 1 - axis
    if _group_size(mesh, other_axis) != 1:
        pytest.skip("the 'axis of one device' refusal needs a 1 x N ring placement")
    a, w, ta, tw = _make(mesh, SMALL, 50)
    good = {"cluster_axis": axis, "scatter_dim": -1, "topology": RING, "num_links": 1}

    def mesh_tensor(shape, dtype=BF16, layout=ttnn.TILE_LAYOUT, memory_config=DRAM):
        return _to_mesh(_stacked(mesh, shape, 51), mesh, dtype, layout, memory_config)

    a_f32 = mesh_tensor(SMALL[0], dtype=ttnn.float32)
    w_f32 = mesh_tensor(SMALL[1], dtype=ttnn.float32)
    a_row_major = mesh_tensor(SMALL[0], layout=ttnn.ROW_MAJOR_LAYOUT)
    a_m_unaligned = mesh_tensor((1, 1, 144, 512))
    a_l1 = mesh_tensor(SMALL[0], memory_config=L1)
    w_l1 = mesh_tensor(SMALL[1], memory_config=L1)
    a_batch2 = mesh_tensor((2, 1, 256, 512))
    w_k_mismatch = mesh_tensor((384, 1024))
    w_rank3 = mesh_tensor((1, 512, 1024))
    w_n_unsplit = mesh_tensor((512, 1024 + 32))  # N / 32 tiles = 33, not a multiple of the 4-device group
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    refusals = [
        # (inputs, call overrides, exception, message)
        ((a_f32, tw), {}, UnsupportedAxisValue, "dtype"),
        ((ta, w_f32), {}, UnsupportedAxisValue, "weight_dtype"),
        ((a_row_major, tw), {}, UnsupportedAxisValue, "layout"),
        ((a_m_unaligned, tw), {}, UnsupportedAxisValue, "alignment"),
        ((ta, tw), {"cluster_axis": 2}, UnsupportedAxisValue, "cluster_axis"),
        ((ta, tw), {"scatter_dim": 0}, UnsupportedAxisValue, "scatter_dim"),
        ((ta, tw), {"scatter_dim": 1}, UnsupportedAxisValue, "scatter_dim"),
        ((ta, tw), {"topology": ttnn.Topology.Mesh}, UnsupportedAxisValue, "topology"),
        ((ta, tw), {"num_links": 0}, UnsupportedAxisValue, "num_links"),
        ((ta, tw), {"num_links": 3}, UnsupportedAxisValue, "num_links"),
        ((a_batch2, tw), {}, ValueError, "leading dims"),
        ((ta, w_rank3), {}, ValueError, r"\(K, N\)"),
        ((ta, w_k_mismatch), {}, ValueError, "K mismatch"),
        ((a_l1, tw), {}, ValueError, "DRAM interleaved"),
        ((ta, w_l1), {}, ValueError, "DRAM interleaved"),
        ((ta, w_n_unsplit), {}, ValueError, "does not split"),
        ((ta, tw), {"cluster_axis": other_axis}, ValueError, "need >= 2"),
        ((ta, tw), {"memory_config": sharded}, NotImplementedError, "sharded"),
    ]
    _fresh_cache(mesh)
    out, added = _call(op, mesh, ta, tw, **good)
    assert added >= 1
    shape = _out_shape(SMALL, -1, _group_size(mesh, axis))
    _check(mesh, out, _reference(a, w, axis, -1), BF16, shape)
    warm = mesh.num_program_cache_entries()
    for args, override, error, message in refusals:
        with expect_error(error, message):
            op(*args, **{**good, **override})
        assert mesh.num_program_cache_entries() == warm, f"a refused call ({override or message}) added a program"
    a2, w2, ta2, tw2 = _make(mesh, SMALL, 52)
    out, added = _call(op, mesh, ta2, tw2, **good)
    assert added == 0, "the valid call after the refusals must hit"
    _check(mesh, out, _reference(a2, w2, axis, -1), BF16, shape)


# --------------------------------------------------------------------------- line placement (the system mesh)


@pytest.mark.parametrize(("axis", "scatter_dim"), [(0, -2), (1, -1)], ids=["axis0_s-2", "axis1_s-1"])
def test_line_mesh_axes(mrs_line_mesh, matmul_reduce_scatter_op, axis, scatter_dim):
    """Every group along `axis` reduces on its own (2 x 2: two groups of 2 per axis)."""
    if _group_size(mrs_line_mesh, axis) < 2:
        pytest.skip(f"mesh axis {axis} has one device")
    call = {"scatter_dim": scatter_dim, "topology": LINE, "num_links": 1}
    _run(matmul_reduce_scatter_op, mrs_line_mesh, axis, SMALL, call, seed=60)


def test_line_stream_regime(mrs_line_mesh, matmul_reduce_scatter_op):
    """MiMo-V2 o_proj with a bf16 W over a 2-device group: the resident cross-block operand misses L1, so the plan
    streams both operands every block."""
    axis = 1 if _group_size(mrs_line_mesh, 1) >= 2 else 0
    if _group_size(mrs_line_mesh, axis) != 2:
        pytest.skip("the stream case is sized for a 2-device group")
    call = {"scatter_dim": -2, "topology": LINE, "num_links": 1}
    _run(matmul_reduce_scatter_op, mrs_line_mesh, axis, MIMO, call, seed=61)


def test_ring_of_two_devices_is_the_line(mrs_line_mesh, matmul_reduce_scatter_op):
    """A Ring over a 2-device group is the 2-device line: accepted under a fabric config that does not wrap, and the
    same bits as Linear."""
    axes = [a for a in (0, 1) if _group_size(mrs_line_mesh, a) == 2]
    if not axes:
        pytest.skip("no mesh axis has exactly 2 devices")
    mesh, axis = mrs_line_mesh, axes[-1]
    a, w, ta, tw = _make(mesh, SMALL, 62)
    expected = _reference(a, w, axis, -1)
    shape = _out_shape(SMALL, -1, 2)
    results = []
    for topology in (LINE, RING):
        out, _ = _call(
            matmul_reduce_scatter_op, mesh, ta, tw, cluster_axis=axis, scatter_dim=-1, topology=topology, num_links=1
        )
        results.append(_check(mesh, out, expected, BF16, shape))
    assert all(torch.equal(x, y) for x, y in zip(*results))


# --------------------------------------------------------------------------- the ring's devices under FABRIC_2D


def test_ring_needs_a_wrapping_fabric_config(mrs_line4_mesh, matmul_reduce_scatter_op, expect_error):
    """A Ring of >= 3 devices needs a fabric config that wraps the axis; FABRIC_2D does not."""
    mesh, axis = mrs_line4_mesh, _group_axis(mrs_line4_mesh)
    _, _, ta, tw = _make(mesh, SMALL, 70)
    with expect_error(ValueError, "fabric config"):
        matmul_reduce_scatter_op(ta, tw, cluster_axis=axis, scatter_dim=-1, topology=RING, num_links=1)
