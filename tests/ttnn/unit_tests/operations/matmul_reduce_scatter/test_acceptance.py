# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.operations.matmul_reduce_scatter:matmul_reduce_scatter``.

Every test calls the operation through the ``matmul_reduce_scatter_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract (``ttnn/ttnn/operations/matmul_reduce_scatter``):
    matmul_reduce_scatter(input_tensor, weight, *, cluster_axis, scatter_dim=-2, topology=Linear, num_links=None,
                          compute_kernel_config=None, memory_config=None) -> ttnn.Tensor
    Every device of a group (the devices along ``cluster_axis``) holds A (..., M, K) and W (K, N); the op returns, on
    each device, block p (its group position) of the group SUM of A @ W, split along ``scatter_dim`` (-1: N, -2: M;
    a non-negative dim counts from the front). One ``ttnn.generic_op`` per call (one mesh workload).
    Output: A's leading dims, [..., M, N] with the scattered dim / G; bfloat16, TILE, ``memory_config`` (interleaved)
    or DRAM interleaved; a new buffer (never an input's). Inputs are read only.
    Support (validate, UnsupportedAxisValue, a NotImplementedError): A dtype bf16 | bf8b; W dtype bf16 | bf8b | bf4b;
    TILE layout; M, K, N tile aligned; cluster_axis 0 | 1; scatter_dim -1 | -2; topology Linear | Ring; num_links 1 | 2
    (None: every link a hop has). A sharded output memory_config: NotImplementedError.
    Caller errors (ValueError): A rank 2-4 with leading dims 1; W rank 2; K mismatch; A / W not DRAM interleaved;
    fewer than 2 devices on the axis; a scattered extent that does not split into G tile blocks; a Ring of >= 3
    devices without a fabric config that wraps the axis; more links than a hop has.
    Planner branches reached here: schedule line (G = 2, and G = 4 with relays) / ring_balanced (G = 4); a 2-device
    Ring is the line; cross-block operand resident (A for scatter -1, W for scatter -2, mirrored halves on the ring)
    or streamed (bf16 W, scatter -2); unchunked or the automatic output-chunk split with per-chunk credits; 1, 2 or
    all links; bf16 / bf8b / bf4b weights and bf8b activations; HiFi4 and no-fp32-DEST compute configs; an L1 output.
    ring_unbalanced (an odd group of >= 3) needs a mesh this suite's machine does not have.
    Program cache (generic_op hashes kernels, compile-time args, core ranges and CBs; runtime arg VALUES, including
    every buffer address, are not keyed): fresh inputs / outputs hit; shape, scatter_dim, dtypes, topology, num_links,
    math fidelity and the output memory config each build a program; an explicit default compute config /
    memory_config / num_links equal to the hop's links hit the program the defaults built.
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

from dataclasses import dataclass, field

import torch

from ttnn.operations._op_contract import UnsupportedAxisValue

# Meshes come from conftest.py: mrs_line_mesh (system mesh, FABRIC_2D), mrs_ring_mesh / mrs_ring_axis (a real ring,
# FABRIC_2D_TORUS_XY: on a QuietBox 2 the 4-chip snake ring opened as 1 x 4) and mrs_unwrapped_ring_mesh.

BF16, BF8, BF4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
LINEAR, RING = ttnn.Topology.Linear, ttnn.Topology.Ring
HIFI4 = (ttnn.MathFidelity.HiFi4, True)
HIFI2_NO_FP32 = (ttnn.MathFidelity.HiFi2, False)
L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
# (min PCC, max rel-RMS) per weight dtype (the narrowest input): the golden suite's helpers.TOLERANCES
TOL = {BF16: (0.999, 0.02), BF8: (0.998, 0.03), BF4: (0.99, 0.08)}

# Per-device (A, W) shapes as a function of the group size g.
SHAPES = {
    "one_tile_block": lambda g: ((1, 1, 32 * g, 64), (64, 32 * g)),  # every scattered block is one tile
    "multi_tile": lambda g: ((1, 1, 256, 512), (512, 1024)),
    "rank2": lambda g: ((256, 384), (384, 512)),
    "rank3": lambda g: ((1, 512, 256), (256, 768)),
    "kimi": lambda g: ((1, 1, 640, 2048), (2048, 7168)),  # Kimi K2.7 o_proj (the PERF FOCUS): A resident
    "mimo": lambda g: ((1, 1, 2048, 2048), (2048, 4096)),  # MiMo-V2 o_proj: W resident with bf8b, streamed with bf16
    # Wan2.2 FF2 720p's output (9472 x 5120) with a short K: the per-core output block alone (>= 11840 / 117 tiles of
    # fp32 hold + bf16 staging) misses L1 unchunked, so the planner takes the automatic output-chunk split
    "wide_out_short_k": lambda g: ((1, 1, 9472, 256), (256, 5120)),
}


@dataclass(frozen=True)
class Case:
    id: str
    shapes: str
    scatter_dim: int
    axis: int | None = None  # None: the ring placement's axis
    topology: object = LINEAR
    num_links: int | None = 1
    dtype: object = BF16
    weight_dtype: object = BF16
    compute: tuple | None = None  # (math fidelity, fp32_dest_acc_en)
    memory_config: object = None
    note: str = field(default="", compare=False)

    def kwargs(self, axis):
        kw = dict(cluster_axis=axis, scatter_dim=self.scatter_dim, topology=self.topology, num_links=self.num_links)
        if self.compute is not None:
            kw["compute_kernel_config"] = ttnn.ComputeConfigDescriptor(
                math_fidelity=self.compute[0], fp32_dest_acc_en=self.compute[1]
            )
        if self.memory_config is not None:
            kw["memory_config"] = self.memory_config
        return kw


# One representative call per planner branch (see the module docstring); the golden suite sweeps shapes and dtypes.
LINE_CASES = [
    Case("one_tile_block_sd-1_ax1", "one_tile_block", -1, axis=1, note="smallest work: one tile per block"),
    Case("multi_tile_sd-2_ax0_w_bf8b_all_links", "multi_tile", -2, axis=0, weight_dtype=BF8, num_links=None),
    Case("rank2_act_bf8b_w_bf4b", "rank2", -1, axis=1, dtype=BF8, weight_dtype=BF4),
    Case(
        "rank3_pos_sd_hifi4_l1_out",
        "rank3",
        1,
        axis=0,
        compute=HIFI4,
        memory_config=L1_INTERLEAVED,
        note="a positive scatter_dim (rank 3: 1 is -2), HiFi4 and an L1 interleaved output",
    ),
    Case("ring_topology_on_2_devices", "multi_tile", -1, axis=1, topology=RING, note="a 2-device Ring is the line"),
    Case("mimo_w_bf16_stream", "mimo", -2, axis=1, note="bf16 W does not stay resident: the stream regime"),
    Case(
        "kimi_w_bf8b_hifi2_no_fp32_all_links",
        "kimi",
        -1,
        axis=0,
        weight_dtype=BF8,
        num_links=None,
        compute=HIFI2_NO_FP32,
        note="A resident; the production 16-bit compute config",
    ),
]

RING_CASES = [
    Case("ring_one_tile_block_sd-2", "one_tile_block", -2, topology=RING),
    Case(
        "ring_kimi_focus",
        "kimi",
        -1,
        topology=RING,
        num_links=2,
        weight_dtype=BF8,
        compute=HIFI2_NO_FP32,
        note="the PERF FOCUS contract: balanced ring, 2 links, A resident with mirrored halves",
    ),
    Case("ring_mimo_w_bf8b_resident", "mimo", -2, topology=RING, num_links=2, weight_dtype=BF8),
    Case(
        "ring_mesh_linear_all_links",
        "multi_tile",
        -1,
        topology=LINEAR,
        num_links=None,
        note="Linear over the ring placement: the line schedule with interior relays",
    ),
    Case("ring_auto_chunks", "wide_out_short_k", -1, topology=RING, num_links=2, note="automatic output chunks"),
    Case(
        "ring_auto_chunks_sd-2",
        "wide_out_short_k",
        -2,
        topology=RING,
        num_links=2,
        note="automatic output chunks walked n-outer (scatter -2: the W axis stays resident across m chunks)",
    ),
]


# --------------------------------------------------------------------------- helpers


def _group(mesh, axis):
    return tuple(mesh.shape)[axis]


def _stacked(mesh, shape, seed, scale=1.0):
    """Distinct randn data per device: a stacked (R, C, *shape) tensor; stacked[r, c] is device (r, c)'s."""
    gen = torch.Generator().manual_seed(seed)
    rows, cols = tuple(mesh.shape)
    return (torch.randn((rows, cols, *shape), generator=gen, dtype=torch.float32) * scale).to(torch.bfloat16)


def _quantize(stacked, dtype):
    """The values as the device holds them (block-float rounding, on the host), so the reference scores the op."""
    if dtype in (BF8, BF4):
        flat = ttnn.from_torch(stacked.reshape(-1, stacked.shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(flat).reshape(stacked.shape).to(torch.bfloat16)
    return stacked


def _to_mesh(stacked, mesh, dtype, *, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    rows, cols = stacked.shape[0], stacked.shape[1]
    whole = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        whole,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )


def _per_device(t):
    return [ttnn.to_torch(part) for part in ttnn.get_device_tensors(t)]


def _reference(a, w, axis, scatter_dim):
    """Independent fp32 torch reference, {(r, c): device (r, c)'s output}: each device's A @ W, summed over its group
    (the mesh column for axis 0, the row for axis 1); group position p keeps block p of the scattered dim."""
    rows, cols = a.shape[0], a.shape[1]
    partial = {(r, c): a[r, c].float() @ w[r, c].float() for r in range(rows) for c in range(cols)}
    g = (rows, cols)[axis]
    out = {}
    for r in range(rows):
        for c in range(cols):
            members = [(i, c) for i in range(rows)] if axis == 0 else [(r, j) for j in range(cols)]
            total = sum(partial[m] for m in members)
            out[(r, c)] = torch.chunk(total, g, dim=scatter_dim)[(r, c)[axis]]
    return out


def _canonical(scatter_dim, rank):
    return scatter_dim - rank if scatter_dim >= 0 else scatter_dim


def _out_shape(a_shape, w_shape, scatter_dim, g):
    out = [*a_shape[:-1], w_shape[-1]]
    out[scatter_dim] //= g
    return out


def _pcc_rms(got, ref):
    a, e = got.flatten(), ref.flatten()
    ac, ec = a - a.mean(), e - e.mean()
    den = ac.norm() * ec.norm()
    pcc = float((ac * ec).sum() / den) if den > 1e-30 else 1.0
    rms = float((a - e).pow(2).mean().sqrt() / max(float(e.pow(2).mean().sqrt()), 1e-30))
    return pcc, rms


@dataclass
class Call:
    """One prepared call: host data, device inputs, the reference and the expected output metadata."""

    a: torch.Tensor
    w: torch.Tensor
    ta: object
    tw: object
    kwargs: dict
    expected: dict
    out_shape: list
    weight_dtype: object
    buffer_type: object


def _prepare(mesh, case, axis, seed=0):
    g = _group(mesh, axis)
    a_shape, w_shape = SHAPES[case.shapes](g)
    sd = _canonical(case.scatter_dim, len(a_shape))
    a = _quantize(_stacked(mesh, a_shape, seed), case.dtype)
    w = _quantize(_stacked(mesh, w_shape, seed + 1, scale=w_shape[0] ** -0.5), case.weight_dtype)
    mc = case.memory_config
    return Call(
        a=a,
        w=w,
        ta=_to_mesh(a, mesh, case.dtype),
        tw=_to_mesh(w, mesh, case.weight_dtype),
        kwargs=case.kwargs(axis),
        expected=_reference(a, w, axis, sd),
        out_shape=_out_shape(a_shape, w_shape, sd, g),
        weight_dtype=case.weight_dtype,
        buffer_type=ttnn.BufferType.DRAM if mc is None else mc.buffer_type,
    )


def _call(op, mesh, *args, **kwargs):
    """(output, program-cache entries added), counted around the selected operation only (after setup, before
    readback)."""
    before = mesh.num_program_cache_entries()
    out = op(*args, **kwargs)
    return out, mesh.num_program_cache_entries() - before


def _assert_dispatched(op):
    # source: exactly one ttnn.generic_op per call (one heterogeneous mesh workload); native: none (guarded)
    assert op.last_dispatches == (1 if op.mode == "source" else 0), op.last_dispatches


def _check(mesh, out, call):
    """The returned tensor's metadata and every device's values against the independent reference; returns the
    per-device host outputs."""
    assert isinstance(out, ttnn.Tensor)
    assert list(out.shape) == call.out_shape
    assert out.dtype == BF16, "the output is bfloat16 whatever the input dtypes"
    assert out.layout == ttnn.TILE_LAYOUT
    mc = out.memory_config()
    assert mc.buffer_type == call.buffer_type and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    rows, cols = tuple(mesh.shape)
    host = _per_device(out)
    assert len(host) == rows * cols
    pcc_min, rms_max = TOL[call.weight_dtype]
    for idx, got in enumerate(host):
        r, c = divmod(idx, cols)
        ref = call.expected[(r, c)].double()
        got = got.double()
        assert got.shape == ref.shape, (idx, got.shape, ref.shape)
        assert torch.isfinite(got).all(), f"device ({r},{c}): non-finite output"
        pcc, rms = _pcc_rms(got, ref)
        assert pcc >= pcc_min and rms <= rms_max, f"device ({r},{c}): pcc {pcc:.6f} rel_rms {rms:.6f}"
    return host


def _check_inputs_untouched(call):
    for t, src in ((call.ta, call.a), (call.tw, call.w)):
        cols = src.shape[1]
        for idx, got in enumerate(_per_device(t)):
            assert torch.equal(got.to(torch.bfloat16), src[idx // cols, idx % cols]), "an input was modified"


def _addresses_distinct(out, *inputs):
    for t in inputs:
        if t.memory_config().buffer_type == out.memory_config().buffer_type:
            assert out.buffer_address() != t.buffer_address(), "the output must not reuse an input's buffer"


def _run_case(op, mesh, case, axis):
    g = _group(mesh, axis)
    if g < 2:
        pytest.skip(f"cluster_axis {axis} has {g} device(s) on this cluster")
    call = _prepare(mesh, case, axis)
    out, _ = _call(op, mesh, call.ta, call.tw, **call.kwargs)
    _assert_dispatched(op)
    _check(mesh, out, call)
    _addresses_distinct(out, call.ta, call.tw)
    if call.a[0, 0].numel() <= 1 << 20:
        _check_inputs_untouched(call)


def _line_axis(mesh):
    axes = [a for a in (1, 0) if _group(mesh, a) >= 2]
    if not axes:
        pytest.skip("no mesh axis has >= 2 devices")
    return axes[0]


def _hop_links(mesh, axis, ring=False):
    """Fewest usable fabric links over every hop (both directions) of every group along `axis`."""
    rows, cols = tuple(mesh.shape)
    node = lambda rc: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*rc))  # noqa: E731
    groups = (
        [[(r, c) for r in range(rows)] for c in range(cols)]
        if axis == 0
        else [[(r, c) for c in range(cols)] for r in range(rows)]
    )
    counts = [
        len(ttnn.get_forwarding_link_indices(node(s), node(d)))
        for grp in groups
        for a, b in list(zip(grp[:-1], grp[1:])) + ([(grp[-1], grp[0])] if ring else [])
        for s, d in ((a, b), (b, a))
    ]
    return min(counts)


def _same_bits(first, again, what):
    for d, (x, y) in enumerate(zip(first, again)):
        assert torch.equal(x, y), f"device {d}: {what}"


# --------------------------------------------------------------------------- supported calls, one per planner branch


@pytest.mark.parametrize("case", LINE_CASES, ids=[c.id for c in LINE_CASES])
def test_line(mrs_line_mesh, matmul_reduce_scatter_op, case):
    _run_case(matmul_reduce_scatter_op, mrs_line_mesh, case, case.axis)


@pytest.mark.parametrize("case", RING_CASES, ids=[c.id for c in RING_CASES])
def test_ring(mrs_ring_mesh, mrs_ring_axis, matmul_reduce_scatter_op, case):
    _run_case(matmul_reduce_scatter_op, mrs_ring_mesh, case, mrs_ring_axis)


def _exact_pattern(mesh, a_shape, w_shape):
    """A = 1 on the first K tile, 0 elsewhere; W = the device's rank (1 + row-major index) on that tile's rows, 0
    elsewhere. Every partial element is exactly 32 * rank and every sum an exact bf16 integer, so a dropped, doubled,
    foreign or misplaced partial shows as a wrong integer."""
    rows, cols = tuple(mesh.shape)
    a = torch.zeros((rows, cols, *a_shape), dtype=torch.bfloat16)
    a[..., :32] = 1
    ranks = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols, 1, 1)
    w = torch.zeros((rows, cols, *w_shape), dtype=torch.float32)
    w[:, :, :32, :] = ranks
    return a, w.to(torch.bfloat16)


def _run_exact(op, mesh, axis, scatter_dim, topology):
    a_shape, w_shape = (1, 1, 256, 64), (64, 512)
    a, w = _exact_pattern(mesh, a_shape, w_shape)
    expected = _reference(a, w, axis, scatter_dim)
    out, _ = _call(
        op,
        mesh,
        _to_mesh(a, mesh, BF16),
        _to_mesh(w, mesh, BF16),
        cluster_axis=axis,
        scatter_dim=scatter_dim,
        topology=topology,
        num_links=1,
    )
    _assert_dispatched(op)
    assert list(out.shape) == _out_shape(a_shape, w_shape, scatter_dim, _group(mesh, axis))
    cols = tuple(mesh.shape)[1]
    for idx, got in enumerate(_per_device(out)):
        ref = expected[divmod(idx, cols)]
        bad = int((got.float() != ref).sum())
        assert bad == 0, f"axis {axis} device {idx}: {bad}/{got.numel()} elements differ"


@pytest.mark.parametrize("scatter_dim", [-2, -1])
@pytest.mark.parametrize("axis", [0, 1])
def test_line_every_partial_exactly_once(mrs_line_mesh, matmul_reduce_scatter_op, axis, scatter_dim):
    if _group(mrs_line_mesh, axis) < 2:
        pytest.skip(f"cluster_axis {axis} has a single device on this cluster")
    _run_exact(matmul_reduce_scatter_op, mrs_line_mesh, axis, scatter_dim, LINEAR)


@pytest.mark.parametrize("scatter_dim", [-2, -1])
def test_ring_every_partial_exactly_once(mrs_ring_mesh, mrs_ring_axis, matmul_reduce_scatter_op, scatter_dim):
    _run_exact(matmul_reduce_scatter_op, mrs_ring_mesh, mrs_ring_axis, scatter_dim, RING)


def test_fp32_dest_flag_is_a_floor(mrs_line_mesh, matmul_reduce_scatter_op):
    """The op accumulates in fp32 DEST whatever the caller asks (the cross-device adds need it): HiFi2 without fp32 DEST
    gives the same bits as the default config (HiFi2, fp32 DEST) on the same inputs."""
    op, mesh = matmul_reduce_scatter_op, mrs_line_mesh
    axis = _line_axis(mesh)
    call = _prepare(mesh, Case("A", "multi_tile", -1, axis=axis), axis, seed=5)
    out, _ = _call(op, mesh, call.ta, call.tw, **call.kwargs)
    default = _check(mesh, out, call)
    no_fp32 = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    out, _ = _call(op, mesh, call.ta, call.tw, **call.kwargs, compute_kernel_config=no_fp32)
    _assert_dispatched(op)
    _same_bits(default, _check(mesh, out, call), "fp32_dest_acc_en=False changed the result")


# --------------------------------------------------------------------------- program cache sequences


def _step(op, mesh, call, misses, **override):
    """Run one prepared call with optional kwarg overrides; assert the cache delta and the output."""
    out, added = _call(op, mesh, call.ta, call.tw, **{**call.kwargs, **override})
    _assert_dispatched(op)
    assert added == misses, f"expected {misses} new program(s), got {added} ({override})"
    return out, _check(mesh, out, call)


def test_cache_line_sequence(mrs_line_mesh, matmul_reduce_scatter_op, expect_error):
    """A miss; proven hits on fresh live buffers; optional arguments given their default values (hits); num_links None
    against the hop's link count; program-defining transitions (fidelity, output memory config, scatter_dim, a larger
    shape and weight format that resize the per-mesh staging buffer); back to A on its first inputs (a hit,
    bit-identical: no stale counters or scratch); then valid -> invalid -> valid."""
    op, mesh = matmul_reduce_scatter_op, mrs_line_mesh
    axis = _line_axis(mesh)
    base = Case("A", "multi_tile", -1)
    mesh.clear_program_cache()
    first = _prepare(mesh, base, axis, seed=10)
    out1, host1 = _step(op, mesh, first, 1)  # a miss: the program cache is on
    second = _prepare(mesh, base, axis, seed=20)
    out2, _ = _step(op, mesh, second, 0)  # a hit on fresh inputs and a fresh output
    assert second.ta.buffer_address() != first.ta.buffer_address()
    assert second.tw.buffer_address() != first.tw.buffer_address()
    assert out2.buffer_address() != out1.buffer_address()
    third = _prepare(mesh, base, axis, seed=30)
    out3, _ = _step(op, mesh, third, 0)  # again, every earlier tensor still alive
    assert len({out1.buffer_address(), out2.buffer_address(), out3.buffer_address()}) == 3

    # optional arguments: absent -> present with their default values -> absent, all one program
    default_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
    _step(op, mesh, second, 0, compute_kernel_config=default_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _step(op, mesh, third, 0)
    # num_links None is every link a hop has: one program with the explicit count
    links = _hop_links(mesh, axis)
    _step(op, mesh, _prepare(mesh, base, axis, seed=35), 0 if links == 1 else 1, num_links=None)
    _step(op, mesh, _prepare(mesh, base, axis, seed=36), 0, num_links=links)

    hifi4 = _prepare(mesh, Case("A_hifi4", "multi_tile", -1, compute=HIFI4), axis, seed=40)
    _step(op, mesh, hifi4, 1)  # math fidelity is compiled in
    l1_out = _prepare(mesh, Case("A_l1", "multi_tile", -1, memory_config=L1_INTERLEAVED), axis, seed=50)
    _step(op, mesh, l1_out, 1)  # the output accessor's compile-time args
    rows = _prepare(mesh, Case("B", "multi_tile", -2), axis, seed=60)
    _step(op, mesh, rows, 1)  # scatter_dim: another block geometry
    big = _prepare(mesh, Case("C", "kimi", -1, weight_dtype=BF8), axis, seed=70)
    _step(op, mesh, big, 1)  # another shape and weight format (the staging buffer is resized)

    _, host_again = _step(op, mesh, first, 0)  # back to A on its first inputs: a hit
    _same_bits(host1, host_again, "a repeated call on the same inputs gave different bits")

    bad_w = _to_mesh(_stacked(mesh, (96, 1024), seed=80), mesh, BF16)
    before = mesh.num_program_cache_entries()
    with expect_error(ValueError, "K mismatch"):
        op(first.ta, bad_w, **first.kwargs)
    assert op.last_dispatches == 0
    assert mesh.num_program_cache_entries() == before
    _step(op, mesh, _prepare(mesh, base, axis, seed=90), 0)  # the warmed program still serves the next valid call


def test_cache_cluster_axis_round_trip(mrs_line_mesh, matmul_reduce_scatter_op):
    """axis A -> axis B -> axis A on one cache (both axes of a 2D mesh): B gets its own groups and peers (at most one
    new program; whether it shares A's depends on the per-device programs), A then hits and stays correct on fresh
    inputs."""
    op, mesh = matmul_reduce_scatter_op, mrs_line_mesh
    axes = [a for a in (0, 1) if _group(mesh, a) >= 2]
    if len(axes) < 2:
        pytest.skip("needs two mesh axes with >= 2 devices")
    base = Case("A", "multi_tile", -2)
    mesh.clear_program_cache()
    _step(op, mesh, _prepare(mesh, base, axes[1], seed=1), 1)
    other = _prepare(mesh, base, axes[0], seed=2)
    out, added = _call(op, mesh, other.ta, other.tw, **other.kwargs)
    _assert_dispatched(op)
    assert added in (0, 1), added
    _check(mesh, out, other)
    _step(op, mesh, _prepare(mesh, base, axes[1], seed=3), 0)
    _step(op, mesh, _prepare(mesh, base, axes[0], seed=4), 0)


def test_cache_ring_sequence(mrs_ring_mesh, mrs_ring_axis, matmul_reduce_scatter_op):
    """Balanced ring: a miss and a hit on fresh buffers; topology (Linear on the same mesh) and num_links (the port
    core count) each define a program; returning to the first configuration hits, bit-identical on its first
    inputs."""
    op, mesh, axis = matmul_reduce_scatter_op, mrs_ring_mesh, mrs_ring_axis
    base = Case("R", "multi_tile", -1, topology=RING, num_links=1)
    mesh.clear_program_cache()
    first = _prepare(mesh, base, axis, seed=11)
    _, host1 = _step(op, mesh, first, 1)
    second = _prepare(mesh, base, axis, seed=21)
    _step(op, mesh, second, 0)
    assert second.ta.buffer_address() != first.ta.buffer_address()
    _step(op, mesh, _prepare(mesh, Case("R_linear", "multi_tile", -1, topology=LINEAR), axis, seed=31), 1)
    if _hop_links(mesh, axis, ring=True) >= 2:
        _step(
            op, mesh, _prepare(mesh, Case("R_links2", "multi_tile", -1, topology=RING, num_links=2), axis, seed=41), 1
        )
    _, host_again = _step(op, mesh, first, 0)
    _same_bits(host1, host_again, "a repeated call on the same inputs gave different bits")
    _step(op, mesh, _prepare(mesh, base, axis, seed=51), 0)


def test_cache_ring_auto_chunks_rearm(mrs_ring_mesh, mrs_ring_axis, matmul_reduce_scatter_op):
    """The automatic chunk split (more chunks than zone sets: downstream credits gate the sends) across cache hits,
    with an unchunked program in between: every counter (arrivals, credits, ready) is re-armed per call, so the
    results are correct each time and bit-identical on repeated inputs."""
    op, mesh, axis = matmul_reduce_scatter_op, mrs_ring_mesh, mrs_ring_axis
    chunked = Case("chunks", "wide_out_short_k", -1, topology=RING, num_links=2)
    mesh.clear_program_cache()
    first = _prepare(mesh, chunked, axis, seed=12)
    _, host1 = _step(op, mesh, first, 1)
    _step(op, mesh, _prepare(mesh, Case("small", "one_tile_block", -1, topology=RING, num_links=2), axis, seed=22), 1)
    _step(op, mesh, _prepare(mesh, chunked, axis, seed=32), 0)
    _, host_again = _step(op, mesh, first, 0)
    _same_bits(host1, host_again, "a repeated chunked call on the same inputs gave different bits")


# --------------------------------------------------------------------------- refusals


def _refusal_inputs(mesh, g, name):
    """(A, W, kwargs overrides) of one refused call; without overrides it is a valid scatter -1 Linear call."""
    a_shape, w_shape = (1, 1, 64, 64), (64, 32 * g)
    a_dtype, w_dtype, a_layout, w_layout = BF16, BF16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT
    a_mem, w_mem = ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    kw = {}
    if name == "act_float32":
        a_dtype = ttnn.float32
    elif name == "weight_float32":
        w_dtype = ttnn.float32
    elif name == "row_major":
        a_layout = w_layout = ttnn.ROW_MAJOR_LAYOUT
    elif name == "non_tile_aligned":
        a_shape = (1, 1, 48, 64)
    elif name == "cluster_axis_2":
        kw["cluster_axis"] = 2
    elif name == "scatter_dim_0":
        kw["scatter_dim"] = 0
    elif name == "topology_mesh":
        kw["topology"] = ttnn.Topology.Mesh
    elif name == "num_links_0":
        kw["num_links"] = 0
    elif name == "num_links_3":
        kw["num_links"] = 3
    elif name == "k_mismatch":
        w_shape = (96, 32 * g)
    elif name == "extent_not_split":
        w_shape = (64, 32 * g + 32)
    elif name == "leading_dim_not_1":
        a_shape = (2, 1, 64, 64)
    elif name == "weight_rank3":
        w_shape = (1, 64, 32 * g)
    elif name == "input_in_l1":
        a_mem = L1_INTERLEAVED
    elif name == "weight_in_l1":
        w_mem = L1_INTERLEAVED
    elif name == "sharded_output":
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        kw["memory_config"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (64, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
    elif name != "valid":
        raise AssertionError(name)
    a = _to_mesh(_stacked(mesh, a_shape, 1), mesh, a_dtype, layout=a_layout, memory_config=a_mem)
    w = _to_mesh(_stacked(mesh, w_shape, 2), mesh, w_dtype, layout=w_layout, memory_config=w_mem)
    return a, w, kw


REFUSALS = [
    # support refusals: validate() against SUPPORTED
    ("act_float32", UnsupportedAxisValue, "dtype"),
    ("weight_float32", UnsupportedAxisValue, "weight_dtype"),
    ("row_major", UnsupportedAxisValue, "layout"),
    ("non_tile_aligned", UnsupportedAxisValue, "alignment"),
    ("cluster_axis_2", UnsupportedAxisValue, "cluster_axis"),
    ("scatter_dim_0", UnsupportedAxisValue, "scatter_dim"),
    ("topology_mesh", UnsupportedAxisValue, "topology"),
    ("num_links_0", UnsupportedAxisValue, "num_links"),
    ("num_links_3", UnsupportedAxisValue, "num_links"),
    # no placement axis for a sharded output
    ("sharded_output", NotImplementedError, "sharded"),
    # caller errors
    ("k_mismatch", ValueError, "K mismatch"),
    ("extent_not_split", ValueError, "split"),
    ("leading_dim_not_1", ValueError, "leading dims"),
    ("weight_rank3", ValueError, "W must be"),
    ("input_in_l1", ValueError, "DRAM interleaved"),
    ("weight_in_l1", ValueError, "DRAM interleaved"),
]


@pytest.mark.parametrize("name, error, message", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusal(mrs_line_mesh, matmul_reduce_scatter_op, expect_error, name, error, message):
    """Each refusal raises its type before any dispatch and leaves the program cache as it was."""
    op, mesh = matmul_reduce_scatter_op, mrs_line_mesh
    axis = _line_axis(mesh)
    a, w, kw = _refusal_inputs(mesh, _group(mesh, axis), name)
    kwargs = {"cluster_axis": axis, "scatter_dim": -1, "topology": LINEAR, "num_links": 1, **kw}
    before = mesh.num_program_cache_entries()
    with expect_error(error, message):
        op(a, w, **kwargs)
    assert op.last_dispatches == 0
    assert mesh.num_program_cache_entries() == before


def test_refusal_more_links_than_a_hop_has(mrs_line_mesh, matmul_reduce_scatter_op, expect_error):
    """One link more than the fewest a hop has: a caller error (ValueError) while that count is a supported value
    (<= 2), else the support refusal."""
    op, mesh = matmul_reduce_scatter_op, mrs_line_mesh
    axis = _line_axis(mesh)
    n = _hop_links(mesh, axis) + 1
    a, w, _ = _refusal_inputs(mesh, _group(mesh, axis), "valid")
    error = ValueError if n <= 2 else UnsupportedAxisValue
    with expect_error(error, "num_links"):
        op(a, w, cluster_axis=axis, scatter_dim=-1, num_links=n)
    assert op.last_dispatches == 0


def test_refusal_single_device_axis(mrs_ring_mesh, mrs_ring_axis, matmul_reduce_scatter_op, expect_error):
    """A cluster axis with one device has nothing to reduce (the 1 x N ring placement's axis 0)."""
    op, mesh = matmul_reduce_scatter_op, mrs_ring_mesh
    single = [a for a in (0, 1) if _group(mesh, a) == 1]
    if not single:
        pytest.skip("both mesh axes have >= 2 devices on this placement")
    a, w, _ = _refusal_inputs(mesh, _group(mesh, mrs_ring_axis), "valid")
    with expect_error(ValueError, "need >= 2"):
        op(a, w, cluster_axis=single[0], scatter_dim=-1, num_links=1)
    assert op.last_dispatches == 0


def test_refusal_ring_without_wrapping_fabric(
    mrs_unwrapped_ring_mesh, mrs_ring_axis, matmul_reduce_scatter_op, expect_error
):
    """A Ring of >= 3 devices needs a fabric config that wraps its axis: the ring placement opened under FABRIC_2D
    refuses it, and the same mesh still runs the Linear call after the refusal."""
    op, mesh = matmul_reduce_scatter_op, mrs_unwrapped_ring_mesh
    g = _group(mesh, mrs_ring_axis)
    if g < 3:
        pytest.skip(f"a {g}-device Ring is the line")
    a, w, _ = _refusal_inputs(mesh, g, "valid")
    with expect_error(ValueError, "wrap"):
        op(a, w, cluster_axis=mrs_ring_axis, scatter_dim=-1, topology=RING, num_links=1)
    assert op.last_dispatches == 0
    call = _prepare(mesh, Case("line", "one_tile_block", -1), mrs_ring_axis, seed=3)
    out, _ = _call(op, mesh, call.ta, call.tw, **call.kwargs)
    _assert_dispatched(op)
    _check(mesh, out, call)
