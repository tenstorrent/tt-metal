# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-migration acceptance suite for ``ttnn.operations.all_gather_rms_norm:all_gather_rms_norm``.

Every test calls the operation through the ``all_gather_rms_norm_op`` fixture. The same file runs before the port
(``source``: the Python ``ttnn.generic_op`` route) and after it (``native``: the registered C++ op that the
public path then aliases).

Route selection (test-only, read at test setup):
    TT_PRE_MIGRATION_MODE   ``source`` (default) | ``native``
    TT_PRE_MIGRATION_ENTRY  ``module.path:callable``. Source mode accepts only the declared source entry.
                            Native mode requires it, requires a different entry *string* (the op's
                            registered name) and a registered C++ op (``is_cpp_operation``).

Contract under test (the public callable ``ttnn.operations.all_gather_rms_norm.all_gather_rms_norm``):

    all_gather_rms_norm(input_tensor, *, cluster_axis, gamma=None, beta=None, norm_type="rms_norm", epsilon=1e-6,
                        topology=Topology.Linear, num_links=None, compute_kernel_config=None, memory_config=None,
                        output_tensor=None) -> ttnn.Tensor

  * Every device of a ``cluster_axis`` group holds a (..., M, W_local) TILE slice; every member returns the full
    (..., M, G * W_local) row normalized over the gathered W: RMSNorm x / sqrt(mean(x^2) + eps) * gamma, or
    LayerNorm (x - mean) / sqrt(var + eps) * gamma + beta. Output dtype / layout = the input's. Slices are gathered
    in group-position order. Reference: the golden suite's pure-torch ``pytorch_all_gather_rms_norm``; gate: its
    ``TOLERANCES`` via ``check_output`` (every device, every group-position slice).
  * gamma / beta: the full-W weight replicated on every device, TILE (1, 1, 1, W) or ROW_MAJOR sticks
    (1, 1, ceil(W/32), 32), float32 or bfloat16, interleaved; beta only with gamma, layer_norm, gamma's format.
  * Defaults: no gamma, rms_norm, eps 1e-6, Linear, all usable links (capped at 2), HiFi4 + fp32 DEST, DRAM output.
  * Placement: input DRAM interleaved (output DRAM or L1 interleaved), or L1 HEIGHT / WIDTH sharded (output an L1
    shard spec of the gathered shape). ``output_tensor`` is written in place and returned.
  * Support (``SUPPORTED`` / ``EXCLUSIONS``): dtype float32 / bfloat16 / bfloat8_b, rank 2-4, cluster_axis 0 / 1,
    Linear / Ring, num_links 1 / 2; float32 needs fp32 DEST; bfloat8_b needs W_local % 32 == 0.
  * Refusals: argument errors raise ValueError; support refusals raise NotImplementedError (the registry's
    UnsupportedAxisValue / ExcludedCell both subclass it; asserted on the base class); a sharded input whose shard
    grid leaves no port cores raises RuntimeError.
  * Source dispatch: exactly one ``ttnn.generic_op`` per call; one program-cache entry per distinct program.
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

SOURCE_ENTRY = "ttnn.operations.all_gather_rms_norm:all_gather_rms_norm"
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
        if name.startswith("ttnn.operations.all_gather_rms_norm") or name.startswith("ttnn._ttnn.operations.generic"):
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
                raise AssertionError("native all_gather_rms_norm must not fall back to ttnn.generic_op")

            return raise_

        self.last_dispatches = 0
        with _patched_generic_op(forbid):
            return self.fn(*args, **kwargs)


_SESSION = {"calls": 0}


@pytest.fixture
def all_gather_rms_norm_op():
    mode, spec = _selection()
    module, fn = _resolve(spec)
    if mode == "native" and not getattr(fn, "is_cpp_operation", False):
        raise pytest.UsageError(f"native entry {spec} is not a registered C++ operation (is_cpp_operation unset)")
    entry = _Entry(mode, spec, module, fn)
    yield entry
    _SESSION["calls"] += entry.calls
    logger.info(
        f"all_gather_rms_norm acceptance: mode={mode} entry={spec} module={getattr(module, '__file__', module.__name__)} "
        f"calls={entry.calls} generic_dispatches={entry.generic_dispatches} session_calls={_SESSION['calls']}"
    )


# --------------------------------------------------------------------------- cases (written by the test writer)

import math  # noqa: E402

import torch  # noqa: E402

from eval.golden_tests.all_gather_rms_norm.helpers import (  # noqa: E402
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gamma_sticks,
    gathered_shape,
    random_gamma,
    replicate,
)
from tests.ttnn.unit_tests.operations.all_gather_rms_norm.pre_migration.agrn_cases import (  # noqa: E402
    BF8,
    BF16,
    EPS,
    F32,
    HS,
    IL,
    LINEAR,
    MESH_1x4,
    MESH_4x1,
    RING,
    RM,
    TILE,
    WS,
    addresses,
    assert_input_untouched,
    build,
    compute_config,
    entries,
    garbage_output,
    per_device_torch,
)


def _one_dispatch(op):
    """The source plans in Python and dispatches exactly one ttnn.generic_op per call (the native route none)."""
    if op.mode == "source":
        assert op.last_dispatches == 1, f"source call made {op.last_dispatches} generic_op dispatches"


def _delta(mesh, fn):
    """Program-cache entries the selected call adds (measured around the call only, after setup, before readback)."""
    before = entries(mesh)
    out = fn()
    return out, entries(mesh) - before


def _fresh_cache(mesh):
    mesh.clear_program_cache()
    mesh.enable_program_cache()
    assert entries(mesh) == 0


# ======================================================================================================== API


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_defaults_minimal_call(mesh_device, all_gather_rms_norm_op):
    """Only input + cluster_axis: no gamma, rms_norm, eps 1e-6, Linear, all usable links, HiFi4 + fp32 DEST, DRAM
    interleaved output of the gathered shape. Spelling every default out gives the same program (a cache hit) and a
    bit-identical result."""
    mesh = mesh_device
    b = build(mesh, (1, 1, 64, 128), topology=LINEAR, num_links=None, pass_compute_config=False, seed=3)
    _fresh_cache(mesh)
    out, d0 = _delta(mesh, lambda: all_gather_rms_norm_op(b.tx, cluster_axis=1))
    _one_dispatch(all_gather_rms_norm_op)
    assert d0 == 1, f"first call must add one program, added {d0}"
    b.check(out)
    assert list(out.shape) == [1, 1, 64, 512] and out.dtype == BF16 and out.layout == TILE
    assert_input_untouched(b)

    explicit, d1 = _delta(
        mesh,
        lambda: all_gather_rms_norm_op(
            b.tx,
            cluster_axis=1,
            gamma=None,
            beta=None,
            norm_type="rms_norm",
            epsilon=1e-6,
            topology=LINEAR,
            num_links=None,
            compute_kernel_config=None,
            memory_config=None,
            output_tensor=None,
        ),
    )
    assert d1 == 0, "explicit defaults are the same program"
    b.check(explicit)
    assert addresses(explicit) != addresses(out), "an op-allocated output is a new buffer per call"
    for a, e in zip(per_device_torch(out), per_device_torch(explicit)):
        assert torch.equal(a, e), "explicit defaults must reproduce the minimal call bit for bit"


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_keyword_only_arguments(mesh_device, all_gather_rms_norm_op, expect_error):
    """Everything after the input is keyword-only."""
    b = build(mesh_device, (1, 1, 32, 64), topology=LINEAR, num_links=1)
    with expect_error(TypeError, "argument"):
        all_gather_rms_norm_op(b.tx, 1)


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_integer_valued_epsilon_and_num_links_default(mesh_device, all_gather_rms_norm_op):
    """epsilon accepts any finite non-negative real (0 and an int included); num_links=None = all usable links."""
    for eps in (0, 0.0):
        b = build(mesh_device, (1, 1, 32, 256), epsilon=eps, num_links=None, seed=5)
        out = b.call(all_gather_rms_norm_op)
        b.check(out)


# ======================================================================================================== planner
# One representative case per program branch reached through the public inputs (regimes named after the op's own
# pinned tests: interleaved k > 1, one tile per slice, single tile, W / H tails, ranks, HEIGHT / WIDTH sharded inputs
# and their shard geometries, L1 interleaved output, 16-bit DEST, block-float input, both cluster axes).

_A = dict(gamma_layout=RM, topology=RING, num_links=2)
BRANCHES = [
    # (id, mesh, per-device shape, build kwargs)
    ("interleaved_k12_rms_gRM_ring_L2", MESH_1x4, (1, 1, 640, 384), _A),
    ("one_tile_slice_ln_beta", MESH_1x4, (1, 1, 640, 32), dict(_A, norm_type="layer_norm", with_beta=True)),
    ("single_tile_nogamma_linear_L1", MESH_1x4, (1, 1, 32, 32), dict(topology=LINEAR, num_links=1)),
    ("wide_decode_gTILE_ring_L1", MESH_1x4, (1, 1, 32, 1792), dict(gamma_layout=TILE, topology=RING, num_links=1)),
    (
        "rank4_batched_ln_beta",
        MESH_1x4,
        (2, 4, 64, 128),
        dict(_A, gamma_layout=TILE, norm_type="layer_norm", with_beta=True),
    ),
    ("rank2_ln_nogamma_linear", MESH_1x4, (640, 384), dict(norm_type="layer_norm", topology=LINEAR, num_links=1)),
    ("w_and_h_tail_rms", MESH_1x4, (1, 1, 17, 50), dict(_A, num_links=1)),
    (
        "w_tail_ln_fp32_linear",
        MESH_1x4,
        (1, 1, 32, 50),
        dict(dtype=F32, gamma_layout=TILE, norm_type="layer_norm", topology=LINEAR, num_links=2),
    ),
    ("h_tail_aligned_w", MESH_1x4, (1, 1, 100, 512), dict(_A, gamma_layout=TILE)),
    (
        "dest16_ln_w_tail",
        MESH_1x4,
        (1, 1, 64, 1000),
        dict(_A, norm_type="layer_norm", with_beta=True, fp32_dest_acc_en=False),
    ),
    (
        "bf8b_ln_dest16",
        MESH_1x4,
        (1, 1, 640, 384),
        dict(_A, dtype=BF8, norm_type="layer_norm", with_beta=True, fp32_dest_acc_en=False),
    ),
    ("height_sharded_row_shards", MESH_1x4, (1, 1, 640, 384), dict(_A, memory_layout=HS)),
    (
        "height_sharded_w_tail_ln",
        MESH_1x4,
        (1, 1, 32, 50),
        dict(_A, memory_layout=HS, norm_type="layer_norm", with_beta=True),
    ),
    (
        "height_sharded_fp32_l1_fit",
        MESH_1x4,
        (1, 1, 640, 1536),
        dict(_A, dtype=F32, gamma_dtype=BF16, memory_layout=HS, _l1_tight=True),
    ),
    ("width_sharded_k12", MESH_1x4, (1, 1, 640, 384), dict(_A, memory_layout=WS)),
    (
        "width_sharded_tail_member_ln",
        MESH_1x4,
        (2, 1, 128, 100),
        dict(_A, memory_layout=WS, norm_type="layer_norm", with_beta=True),
    ),
    ("width_sharded_ragged_R1", MESH_1x4, (1, 1, 32, 8192), dict(_A, memory_layout=WS)),
    (
        "width_sharded_precombine_R64",
        MESH_1x4,
        (1, 1, 2048, 384),
        dict(_A, memory_layout=WS, norm_type="layer_norm", with_beta=True),
    ),
    ("l1_interleaved_out", MESH_1x4, (1, 1, 640, 384), dict(_A, output_placement="l1_interleaved")),
    (
        "l1_interleaved_out_w_tail_ln",
        MESH_1x4,
        (1, 1, 17, 50),
        dict(_A, output_placement="l1_interleaved", norm_type="layer_norm", with_beta=True),
    ),
    ("axis0_ring_L2", MESH_4x1, (1, 1, 640, 384), dict(_A, cluster_axis=0)),
    (
        "axis0_ln_beta_linear_L1",
        MESH_4x1,
        (1, 1, 64, 128),
        dict(gamma_layout=TILE, norm_type="layer_norm", with_beta=True, cluster_axis=0, topology=LINEAR, num_links=1),
    ),
]


@pytest.mark.parametrize(
    "mesh_device,shape,kw",
    [pytest.param(m.values[0], s, k, id=i) for i, m, s, k in BRANCHES],
    indirect=["mesh_device"],
)
def test_planner_branch(mesh_device, shape, kw, all_gather_rms_norm_op):
    """Each branch: correct on every device / slice, output placement as requested, input untouched, and a repeat
    call with fresh data on fresh buffers (a cache hit) is correct too."""
    kw = dict(kw)
    l1_tight = kw.pop("_l1_tight", False)
    b = build(mesh_device, shape, seed=11, **kw)
    out, d0 = _delta(mesh_device, lambda: b.call(all_gather_rms_norm_op))
    _one_dispatch(all_gather_rms_norm_op)
    b.check(out)
    assert_input_untouched(b)
    ttnn.deallocate(out)  # the largest L1 outputs cannot coexist twice; the input stays alive (fresh addresses)
    if l1_tight:  # the op only fits next to ONE input shard set of this size: release it too
        ttnn.deallocate(b.tx)
    b2 = build(mesh_device, shape, seed=12, **kw)
    if not l1_tight:
        assert addresses(b2.tx) != addresses(b.tx)
    out2, d1 = _delta(mesh_device, lambda: b2.call(all_gather_rms_norm_op))
    b2.check(out2)
    if kw.get("memory_layout", IL) == IL and kw.get("output_placement") is None:
        assert d1 == 0, f"the same configuration on fresh buffers must hit the program cache (added {d1})"
    else:
        # L1-resident input / output: the source sizes its staging buffers from the L1 left free next to the live L1
        # tensors, and the first call's input is still alive here, so a second program is allowed (never more).
        assert d1 in (0, 1), f"added {d1} programs"


@pytest.mark.parametrize(
    "mesh_device,cluster_axis",
    [pytest.param((1, 4), 1, id="1x4-axis1"), pytest.param((4, 1), 0, id="4x1-axis0")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
def test_gather_order_closed_form(mesh_device, cluster_axis, norm_type, all_gather_rms_norm_op):
    """Device (r, c) holds the constant k = r * C + c + 1 (fp32). Every member's slice p is exactly
    k_p / sqrt(mean(k^2) + eps) (RMS) or (k_p - mean) / sqrt(var + eps) (LN), statistics over the whole group: a
    dropped, doubled, foreign or misplaced slice, or statistics over W_local, misses by far more than fp32 noise."""
    mesh = mesh_device
    rows, cols = tuple(mesh.shape)
    shape = (1, 1, 64, 128)
    ranks = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)
    x = ranks.reshape(rows, cols, 1, 1, 1, 1).expand(rows, cols, *shape).contiguous()
    b = build(mesh, shape, cluster_axis=cluster_axis, dtype=F32, norm_type=norm_type, topology=LINEAR, num_links=1, x=x)
    out = b.call(all_gather_rms_norm_op)
    b.check(out)
    w = shape[-1]
    for idx, got in enumerate(per_device_torch(out)):
        r, c = divmod(idx, cols)
        group = (
            [ranks[p, c].item() for p in range(rows)]
            if cluster_axis == 0
            else [ranks[r, p].item() for p in range(cols)]
        )
        mean = sum(group) / len(group)
        for p, k in enumerate(group):
            if norm_type == "rms_norm":
                want = k / math.sqrt(sum(v * v for v in group) / len(group) + EPS)
            else:
                want = (k - mean) / math.sqrt(sum((v - mean) ** 2 for v in group) / len(group) + EPS)
            err = (got[..., p * w : (p + 1) * w].float() - want).abs().max().item()
            assert err <= 1e-3 * max(abs(want), 1.0), f"device ({r},{c}) slice {p}: max err {err}, want {want}"


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_layer_norm_large_mean_fp32(mesh_device, all_gather_rms_norm_op):
    """LayerNorm, fp32 end to end, row mean ~50 and spread ~1: the variance must not cancel."""
    b = build(
        mesh_device,
        (1, 1, 128, 384),
        dtype=F32,
        gamma_layout=TILE,
        with_beta=True,
        norm_type="layer_norm",
        topology=LINEAR,
        num_links=1,
        offset=50.0,
        seed=21,
    )
    b.check(b.call(all_gather_rms_norm_op))


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_gamma_formats_agree(mesh_device, all_gather_rms_norm_op):
    """A position-dependent gamma (a ramp over the FULL row) in TILE and in ROW_MAJOR sticks: each output position
    takes its own global gamma, and both formats give the reference."""
    shape = (1, 1, 64, 96)
    W = 4 * shape[-1]
    ramp = torch.arange(1, W + 1, dtype=torch.float32) / W
    for layout in (TILE, RM):
        b = build(
            mesh_device,
            shape,
            dtype=F32,
            gamma_layout=layout,
            gamma_dtype=F32,
            gamma=ramp,
            topology=LINEAR,
            num_links=1,
            seed=31,
        )
        b.check(b.call(all_gather_rms_norm_op))


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_l1_output_feeds_matmul(mesh_device, all_gather_rms_norm_op):
    """Integration: the L1 interleaved output (the MLA q_a norm's real call) is a usable input of the next op."""
    b = build(mesh_device, (1, 1, 640, 384), output_placement="l1_interleaved", seed=41, **_A)
    out = b.call(all_gather_rms_norm_op)
    b.check(out)
    torch.manual_seed(42)
    w = torch.randn((1536, 64), dtype=torch.float32).to(torch.bfloat16)
    tw = ttnn.from_torch(
        w.reshape(1, 1, 1536, 64),
        dtype=BF16,
        layout=TILE,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    y = ttnn.matmul(out, tw)
    cols = tuple(mesh_device.shape)[1]
    for idx, got in enumerate(per_device_torch(y)):
        r, c = divmod(idx, cols)
        want = b.expected[r, c].float() @ w.float()
        pcc = torch.corrcoef(torch.stack([got.float().flatten(), want.flatten()]))[0, 1].item()
        assert pcc >= 0.99, f"device ({r},{c}): matmul of the op's output has PCC {pcc}"


# ======================================================================================================== memory


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
@pytest.mark.parametrize("placement", ["dram", "l1"])
def test_preallocated_output_written_in_place(mesh_device, placement, all_gather_rms_norm_op):
    """output_tensor (garbage-filled) is written in place and returned, on every device; a second call with new data
    into the same buffer overwrites it completely; the input is untouched."""
    mc = ttnn.DRAM_MEMORY_CONFIG if placement == "dram" else ttnn.L1_MEMORY_CONFIG
    shape = (1, 1, 640, 384)
    buf = garbage_output(mesh_device, gathered_shape(shape, 4), memory_config=mc)
    buf_addr = addresses(buf)
    for seed in (51, 52):
        b = build(mesh_device, shape, seed=seed, **_A)
        out = b.call(all_gather_rms_norm_op, output_tensor=buf)
        _one_dispatch(all_gather_rms_norm_op)
        assert addresses(out) == buf_addr, "the op must return the provided output_tensor"
        b.out_buffer_type = mc.buffer_type
        b.check(out)
        b.check(buf)
        assert_input_untouched(b)


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
@pytest.mark.parametrize("memory_layout", [HS, WS], ids=["height", "width"])
def test_preallocated_sharded_output(mesh_device, memory_layout, all_gather_rms_norm_op):
    """A sharded input with a caller-provided L1 shard output of the gathered shape: written in place, returned."""
    shape = (1, 1, 640, 384)
    b = build(mesh_device, shape, memory_layout=memory_layout, seed=61, **_A)
    out_mc = b.kwargs.pop("memory_config")
    buf = garbage_output(mesh_device, b.out_shape, memory_config=out_mc)
    out = b.call(all_gather_rms_norm_op, output_tensor=buf)
    assert addresses(out) == addresses(buf)
    b.check(out)
    assert_input_untouched(b)


# ======================================================================================================== cache

_CACHE_SHAPE = (1, 1, 640, 384)


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_cache_fresh_buffers_and_data(mesh_device, all_gather_rms_norm_op):
    """A -> new A -> new A: one miss, then hits on fresh live input / gamma / output buffers and different data; every
    output follows its own call's inputs; earlier outputs stay intact."""
    mesh = mesh_device
    _fresh_cache(mesh)
    built, outs, deltas = [], [], []
    for seed in (71, 72, 73):
        b = build(mesh, _CACHE_SHAPE, seed=seed, **_A)
        built.append(b)
        out, d = _delta(mesh, lambda: b.call(all_gather_rms_norm_op))
        _one_dispatch(all_gather_rms_norm_op)
        outs.append(out)
        deltas.append(d)
        b.check(out)
    assert deltas == [1, 0, 0], f"program-cache deltas {deltas}"
    in_addr = [tuple(addresses(b.tx)) for b in built]
    g_addr = [tuple(addresses(b.kwargs["gamma"])) for b in built]
    out_addr = [tuple(addresses(o)) for o in outs]
    for name, addr in (("input", in_addr), ("gamma", g_addr), ("output", out_addr)):
        assert len(set(addr)) == 3, f"{name} buffers must be distinct live allocations: {addr}"
    for b, o in zip(built, outs):
        b.check(o)


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_cache_fresh_preallocated_outputs(mesh_device, all_gather_rms_norm_op):
    """Hits that write a different caller buffer each time: call k writes only buffer k (no write to a stale
    address of an earlier call)."""
    mesh = mesh_device
    _fresh_cache(mesh)
    bufs = [garbage_output(mesh, gathered_shape(_CACHE_SHAPE, 4), fill=7.0 + i) for i in range(3)]
    assert len({tuple(addresses(x)) for x in bufs}) == 3
    built, deltas = [], []
    for i, seed in enumerate((81, 82, 83)):
        b = build(mesh, _CACHE_SHAPE, seed=seed, **_A)
        built.append(b)
        out, d = _delta(mesh, lambda: b.call(all_gather_rms_norm_op, output_tensor=bufs[i]))
        deltas.append(d)
        assert addresses(out) == addresses(bufs[i])
    assert deltas == [1, 0, 0], f"program-cache deltas {deltas}"
    for b, buf in zip(built, bufs):
        b.check(buf)


# A -> B -> A -> B: B changes one program-defining field. Expected: B is its own program (one new entry) for fields
# that change kernels, circular-buffer formats or the core / link plan; returning to A and repeating B hit.
_FIELD_CHANGES = [
    ("norm_type", dict(norm_type="layer_norm"), True),
    ("gamma_absent", dict(gamma_layout=None), True),
    ("gamma_layout_tile", dict(gamma_layout=TILE), True),
    ("gamma_dtype_fp32", dict(gamma_dtype=F32), True),
    ("beta_present", dict(norm_type="layer_norm", with_beta=True), True),
    ("dtype_fp32", dict(dtype=F32), True),
    ("fp32_dest_off", dict(fp32_dest_acc_en=False), True),
    ("topology_linear", dict(topology=LINEAR), True),
    ("num_links_1", dict(num_links=1), True),
    ("shape_decode", dict(shape=(1, 1, 32, 384)), True),
    ("input_height_sharded", dict(memory_layout=HS), True),
    # where the output pages land: the source specializes it; a port may pass it as runtime state
    ("output_l1_interleaved", dict(output_placement="l1_interleaved"), False),
]


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
@pytest.mark.parametrize("change,must_miss", [pytest.param(c, m, id=i) for i, c, m in _FIELD_CHANGES])
def test_cache_program_field_transition(mesh_device, change, must_miss, all_gather_rms_norm_op):
    mesh = mesh_device
    a_kw = dict(_A)
    b_kw = {**a_kw, **change}
    b_shape = b_kw.pop("shape", _CACHE_SHAPE)
    _fresh_cache(mesh)
    deltas = []
    for i, (shape, kw) in enumerate([(_CACHE_SHAPE, a_kw), (b_shape, b_kw), (_CACHE_SHAPE, a_kw), (b_shape, b_kw)]):
        b = build(mesh, shape, seed=90 + i, **kw)
        out, d = _delta(mesh, lambda: b.call(all_gather_rms_norm_op))
        deltas.append(d)
        b.check(out)
    assert deltas[0] == 1, f"first call must miss: {deltas}"
    if must_miss:
        assert deltas[1] == 1, f"B must be its own program: {deltas}"
    else:
        assert deltas[1] in (0, 1), f"B adds at most one program: {deltas}"
    assert deltas[2:] == [0, 0], f"returning to A and repeating B must hit: {deltas}"


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_cache_epsilon_values(mesh_device, all_gather_rms_norm_op):
    """eps a -> b -> a on small inputs (mean square ~0.01) where eps visibly changes the result: the output follows
    each eps (a stale eps fails the reference). b may or may not be a new program; returning to a hits."""
    mesh = mesh_device
    _fresh_cache(mesh)
    deltas = []
    for i, eps in enumerate((1e-6, 0.25, 1e-6, 0.25)):
        b = build(mesh, _CACHE_SHAPE, epsilon=eps, scale=0.1, seed=100 + i, **_A)
        out, d = _delta(mesh, lambda: b.call(all_gather_rms_norm_op))
        deltas.append(d)
        b.check(out)
    assert deltas[0] == 1 and deltas[1] in (0, 1) and deltas[2:] == [0, 0], f"deltas {deltas}"


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_cache_output_alias_transition(mesh_device, all_gather_rms_norm_op):
    """op-allocated -> caller buffer -> op-allocated -> caller buffer: each call writes its own current output; once
    both forms are seen, both hit."""
    mesh = mesh_device
    _fresh_cache(mesh)
    deltas, keep = [], []
    for i, prealloc in enumerate((False, True, False, True)):
        b = build(mesh, _CACHE_SHAPE, seed=110 + i, **_A)
        extra = {}
        if prealloc:
            extra["output_tensor"] = garbage_output(mesh, b.out_shape)
        out, d = _delta(mesh, lambda: b.call(all_gather_rms_norm_op, **extra))
        deltas.append(d)
        if prealloc:
            assert addresses(out) == addresses(extra["output_tensor"])
        b.check(out)
        keep.append((b, out))
    assert deltas[0] == 1 and deltas[1] in (0, 1) and deltas[2:] == [0, 0], f"deltas {deltas}"
    for b, out in keep:
        b.check(out)


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_cache_valid_invalid_valid(mesh_device, all_gather_rms_norm_op, expect_error):
    """A warmed program does not let invalid calls through, the refusals add no program, and the next valid call hits
    and is correct."""
    mesh = mesh_device
    _fresh_cache(mesh)
    b = build(mesh, _CACHE_SHAPE, seed=121, **_A)
    out, d = _delta(mesh, lambda: b.call(all_gather_rms_norm_op))
    assert d == 1
    b.check(out)
    warmed = entries(mesh)  # after readback, which may add its own entries
    with expect_error(ValueError, "beta"):
        b.call(all_gather_rms_norm_op, beta=b.kwargs["gamma"])  # beta with rms_norm
    with expect_error(NotImplementedError, "num_links"):
        b.call(all_gather_rms_norm_op, num_links=3)
    with expect_error(ValueError, "epsilon"):
        b.call(all_gather_rms_norm_op, epsilon=-1.0)
    assert entries(mesh) == warmed, "refused calls must not add programs"
    b2 = build(mesh, _CACHE_SHAPE, seed=122, **_A)
    out, d = _delta(mesh, lambda: b2.call(all_gather_rms_norm_op))
    assert d == 0
    b2.check(out)


# ======================================================================================================== refusals

_X_SHAPE = (1, 1, 64, 128)  # per device; W = 4 * 128 = 512 on the 1x4 group


def _x(mesh, shape=_X_SHAPE, dtype=BF16, layout=TILE, memory_config=None, torch_dtype=torch.bfloat16):
    torch.manual_seed(0)
    rows, cols = tuple(mesh.shape)
    if torch_dtype == torch.int32:
        data = torch.randint(-8, 8, (rows, cols, *shape), dtype=torch.int32)
    else:
        data = torch.randn((rows, cols, *shape), dtype=torch.float32).to(torch_dtype)
    return create_ttnn_input_tensor(data, mesh, dtype=dtype, layout=layout, memory_config=memory_config)


def _w(mesh, W=512, dtype=BF16, layout=TILE, memory_config=None):
    t = random_gamma(W, torch.float32 if dtype == F32 else torch.bfloat16)
    if memory_config is None:
        return create_ttnn_gamma(t, mesh, dtype=dtype, layout=layout)
    return create_ttnn_input_tensor(
        replicate(t.reshape(1, 1, 1, W), mesh), mesh, dtype=dtype, layout=layout, memory_config=memory_config
    )


def _core_grid(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _rank1(m):
    t = ttnn.from_torch(
        torch.randn(128).to(torch.bfloat16),
        dtype=BF16,
        layout=TILE,
        device=m,
        mesh_mapper=ttnn.ReplicateTensorToMesh(m),
    )
    return (t,), {"cluster_axis": 1}


def _full_grid_sharded(m):
    """A HEIGHT_SHARDED input whose shard grid is the whole worker grid: no cores are left for the fabric ports."""
    g = m.compute_with_storage_grid_size()
    n = g.x * g.y
    grid = _core_grid(0, 0, g.x - 1, g.y - 1)
    in_mc = ttnn.MemoryConfig(HS, ttnn.BufferType.L1, ttnn.ShardSpec(grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR))
    out_mc = ttnn.MemoryConfig(HS, ttnn.BufferType.L1, ttnn.ShardSpec(grid, (32, 256), ttnn.ShardOrientation.ROW_MAJOR))
    return (_x(m, shape=(1, 1, 32 * n, 64), memory_config=in_mc),), {"cluster_axis": 1, "memory_config": out_mc}


def _block_sharded_out(m):
    spec = ttnn.ShardSpec(_core_grid(0, 0, 1, 1), (32, 256), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)
    return (_x(m),), {"cluster_axis": 1, "memory_config": mc}


def _sharded_gamma(m):
    spec = ttnn.ShardSpec(_core_grid(0, 0, 0, 0), (32, 512), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(HS, ttnn.BufferType.L1, spec)
    return (_x(m),), {"cluster_axis": 1, "gamma": _w(m, memory_config=mc)}


_K = {"cluster_axis": 1}
REFUSALS = [
    # (id, exception, message regex, builder(mesh) -> (args, kwargs))
    ("rank1", ValueError, "rank", _rank1),
    ("unknown_norm_type", ValueError, "norm_type", lambda m: ((_x(m),), {**_K, "norm_type": "group_norm"})),
    (
        "beta_without_gamma",
        ValueError,
        "beta requires gamma",
        lambda m: ((_x(m),), {**_K, "norm_type": "layer_norm", "beta": _w(m)}),
    ),
    ("beta_with_rms_norm", ValueError, "beta", lambda m: ((_x(m),), {**_K, "gamma": _w(m), "beta": _w(m)})),
    ("cluster_axis_2", ValueError, "cluster_axis", lambda m: ((_x(m),), {"cluster_axis": 2})),
    ("cluster_axis_negative", ValueError, "cluster_axis", lambda m: ((_x(m),), {"cluster_axis": -1})),
    ("cluster_axis_size_1", ValueError, "has size 1", lambda m: ((_x(m),), {"cluster_axis": 0})),
    ("epsilon_negative", ValueError, "epsilon", lambda m: ((_x(m),), {**_K, "epsilon": -1e-3})),
    ("epsilon_inf", ValueError, "epsilon", lambda m: ((_x(m),), {**_K, "epsilon": float("inf")})),
    ("epsilon_nan", ValueError, "epsilon", lambda m: ((_x(m),), {**_K, "epsilon": float("nan")})),
    ("gamma_tile_wrong_width", ValueError, "gamma", lambda m: ((_x(m),), {**_K, "gamma": _w(m, W=480)})),
    ("gamma_tile_local_width", ValueError, "gamma", lambda m: ((_x(m),), {**_K, "gamma": _w(m, W=128)})),
    ("gamma_sticks_wrong_count", ValueError, "gamma", lambda m: ((_x(m),), {**_K, "gamma": _w(m, W=480, layout=RM)})),
    (
        "beta_wrong_width",
        ValueError,
        "beta",
        lambda m: ((_x(m),), {**_K, "norm_type": "layer_norm", "gamma": _w(m), "beta": _w(m, W=480)}),
    ),
    (
        "beta_dtype_differs",
        ValueError,
        "beta",
        lambda m: ((_x(m),), {**_K, "norm_type": "layer_norm", "gamma": _w(m), "beta": _w(m, dtype=F32)}),
    ),
    (
        "beta_layout_differs",
        ValueError,
        "beta",
        lambda m: ((_x(m),), {**_K, "norm_type": "layer_norm", "gamma": _w(m), "beta": _w(m, layout=RM)}),
    ),
    (
        "output_tensor_wrong_shape",
        ValueError,
        "output_tensor",
        lambda m: ((_x(m),), {**_K, "output_tensor": garbage_output(m, _X_SHAPE)}),
    ),
    (
        "output_tensor_wrong_dtype",
        ValueError,
        "output_tensor",
        lambda m: ((_x(m),), {**_K, "output_tensor": garbage_output(m, gathered_shape(_X_SHAPE, 4), dtype=F32)}),
    ),
    (
        "input_int32",
        NotImplementedError,
        "dtype",
        lambda m: ((_x(m, dtype=ttnn.int32, torch_dtype=torch.int32),), dict(_K)),
    ),
    ("input_row_major", NotImplementedError, "layout", lambda m: ((_x(m, layout=RM),), dict(_K))),
    ("input_rank5", NotImplementedError, "rank", lambda m: ((_x(m, shape=(1, 1, 1, 64, 128)),), dict(_K))),
    ("gamma_bfloat8_b", NotImplementedError, "gamma_dtype", lambda m: ((_x(m),), {**_K, "gamma": _w(m, dtype=BF8)})),
    ("num_links_3", NotImplementedError, "num_links", lambda m: ((_x(m),), {**_K, "num_links": 3})),
    ("num_links_0", NotImplementedError, "num_links", lambda m: ((_x(m),), {**_K, "num_links": 0})),
    ("topology_mesh", NotImplementedError, "topology", lambda m: ((_x(m),), {**_K, "topology": ttnn.Topology.Mesh})),
    (
        "fp32_without_fp32_dest",
        NotImplementedError,
        "fp32_dest_acc_en",
        lambda m: (
            (_x(m, dtype=F32, torch_dtype=torch.float32),),
            {**_K, "compute_kernel_config": compute_config(False)},
        ),
    ),
    (
        "bf8b_w_non_aligned",
        NotImplementedError,
        "w_non_aligned",
        lambda m: ((_x(m, shape=(1, 1, 64, 50), dtype=BF8),), dict(_K)),
    ),
    (
        "l1_interleaved_input",
        NotImplementedError,
        "DRAM",
        lambda m: ((_x(m, memory_config=ttnn.L1_MEMORY_CONFIG),), dict(_K)),
    ),
    ("block_sharded_output", NotImplementedError, "not supported", _block_sharded_out),
    (
        "sharded_output_without_spec",
        NotImplementedError,
        "shard spec",
        lambda m: ((_x(m),), {**_K, "memory_config": ttnn.MemoryConfig(HS, ttnn.BufferType.L1)}),
    ),
    ("gamma_sharded", NotImplementedError, "interleaved", _sharded_gamma),
    ("sharded_input_no_port_cores", RuntimeError, "port cores", _full_grid_sharded),
]


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
@pytest.mark.parametrize("exc,message,make", [pytest.param(e, msg, f, id=i) for i, e, msg, f in REFUSALS])
def test_refusal(mesh_device, exc, message, make, all_gather_rms_norm_op, expect_error):
    args, kwargs = make(mesh_device)
    before = entries(mesh_device)
    with expect_error(exc, message):
        all_gather_rms_norm_op(*args, **kwargs)
    assert entries(mesh_device) == before, "a refused call must not build a program"


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_refusal_then_valid_bf8b_aligned(mesh_device, all_gather_rms_norm_op):
    """Just inside the bfloat8_b limit: W_local % 32 == 0 with H non-aligned is supported (not excluded)."""
    b = build(mesh_device, (1, 1, 17, 64), dtype=BF8, gamma_layout=TILE, topology=LINEAR, num_links=1, seed=131)
    b.check(b.call(all_gather_rms_norm_op))


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_gamma_sticks_padded_tail(mesh_device, all_gather_rms_norm_op):
    """ROW_MAJOR sticks of a non-aligned W (4 * 50 = 200 -> 7 sticks, tail zero-padded) are accepted and the padding
    does not leak into the logical output."""
    b = build(mesh_device, (1, 1, 32, 50), gamma_layout=RM, topology=LINEAR, num_links=1, seed=141)
    assert list(b.kwargs["gamma"].shape) == list(gamma_sticks(torch.zeros(200)).shape)
    b.check(b.call(all_gather_rms_norm_op))


@pytest.mark.parametrize("mesh_device", [MESH_1x4], indirect=True)
def test_largest_fp32_layer_norm_l1_output(mesh_device, all_gather_rms_norm_op):
    """The largest golden shape, fp32 LayerNorm (the largest circular-buffer set), output in L1 interleaved: the
    source fits its buffers next to the 40 MB L1 output and produces the reference (no CB clash, no refusal)."""
    b = build(
        mesh_device,
        (1, 1, 2048, 1280),
        dtype=F32,
        norm_type="layer_norm",
        topology=LINEAR,
        num_links=2,
        output_placement="l1_interleaved",
        seed=151,
    )
    out = b.call(all_gather_rms_norm_op)
    _one_dispatch(all_gather_rms_norm_op)
    b.check(out)


def test_mesh_reopen_between_calls(_agrn_mesh_cache, all_gather_rms_norm_op):
    """Close and reopen the mesh (fabric included) between calls: nothing kept from the first mesh's lifetime may be
    reused against the new one."""
    for i in range(2):
        mesh = _agrn_mesh_cache.get((1, 4))
        b = build(mesh, (1, 1, 64, 128), seed=161 + i, **_A)
        b.check(b.call(all_gather_rms_norm_op))
        b = None
        _agrn_mesh_cache.close()


@pytest.mark.parametrize("mesh_device", [MESH_4x1], indirect=True)
def test_refusal_l1_fit_wide_height_shards(mesh_device, all_gather_rms_norm_op, expect_error):
    """bfloat8_b LayerNorm of 128 x 8192 per device on HEIGHT shards along axis 0 (full 32768-wide gathered rows per
    core): the circular buffers cannot fit next to the input / output shards, so the call is refused up front as
    unsupported (NotImplementedError naming the fit), not left to clash or hang. Nothing is dispatched."""
    b = build(
        mesh_device,
        (128, 8192),
        dtype=BF8,
        gamma_layout=RM,
        with_beta=True,
        norm_type="layer_norm",
        memory_layout=HS,
        cluster_axis=0,
        topology=RING,
        num_links=2,
        seed=171,
    )
    before = entries(mesh_device)
    with expect_error(NotImplementedError, "do not fit"):
        b.call(all_gather_rms_norm_op)
    assert entries(mesh_device) == before
