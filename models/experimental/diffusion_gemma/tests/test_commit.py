# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the commit-time KV write: mode selection, on-device equivalence, and tracing.

``tt/commit_batched.py`` appends a 256-token canvas into the frozen contiguous KV cache at
absolute positions ``[start_pos, start_pos+256)``. Two mechanisms exist
(``_write_canvas_kv_contiguous``), selected by ``DG_COMMIT_KV_WRITE`` (#47557):

* ``"position"`` — the device-proven reference: 256 x (slice + reshard +
  ``paged_update_cache``) per K/V, ~1536 dispatches per layer;
* ``"fill"`` — the default: ONE ``ttnn.fill_cache`` per K/V at the tile-aligned
  ``update_idx=start_pos``, 2 dispatches per layer.

Because ``start_pos`` and ``canvas_len`` are both multiples of 32, the write span is
tile-aligned and FILL is a pure tile copy, so the two mechanisms must agree
**bit-for-bit** — not within a PCC. The device section asserts exactly that, over the WHOLE
cache tensor (so a disturbed frozen prefix ``[0, start_pos)`` or tail
``[start_pos+256, max_seq)`` fails too), against a torch oracle, and it pins the fallback
behaviour for geometries ``fill`` cannot serve. Checkpoint-free (raw tensors + the write
helper only) — runs in seconds.

The last section covers the canvas-tail workspace primitive (#51080 roadmap item 4), which
removes the per-step prefix copy: instead of ``concat([prefix_kv, canvas_kv])`` (which
re-copies the whole p_max prefix every step), size the KV cache as ``p_max + canvas_len``
and write the canvas K/V into the TAIL with ``ttnn.fill_cache(update_idx=p_max)``, so SDPA
reads one already-contiguous tensor. That rests on one assumption that must be proven
before any model wiring, because it is the part that can fail catastrophically rather than
merely slowly:

    Can a captured Metal trace contain ``fill_cache`` writing into a persistent tensor AND a
    subsequent read of that same tensor, such that every replay observes the freshly written
    tail rather than a stale one?

A trace bakes buffer addresses, so the write and the read must agree across replays. If they
do not, the failure is silent wrong output, not a crash.

Device tests run with DG_RUN_DEVICE=1 and a mesh; skipped otherwise:
  DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_commit.py

Two operational notes on the traced tests, both learned by getting them wrong:

* A trace cannot load new kernel binaries, so every program inside a capture must ALREADY be
  in the program cache. Each test therefore runs its ops once eagerly before
  ``begin_trace_capture``; without that warm-up the capture dies with "Cannot load new
  binaries during trace capture".
* If a capture aborts, the trace must still be ended and released, or every later op in the
  process fails with "Reads/Writes are not supported during trace capture" and the real
  failure is buried under the cascade. Hence the ``_capture`` guard below. Relatedly: do NOT
  SIGKILL a process that is mid-capture — it keeps the ``CHIP_IN_USE`` lock and its TLB
  windows, and the next run fails to even open the device (``tt_tlb_alloc failed ... error
  code -12``). Let the holder exit; the driver releases on exit and no ``tt-smi -r`` is
  needed.
"""

import importlib
import os
from contextlib import contextmanager

import pytest
import torch

import ttnn
import models.experimental.diffusion_gemma.tt.commit_batched as commit_batched
from models.experimental.diffusion_gemma.tt.commit_batched import (
    _fill_write_unsupported_reason,
    _read_cache_kv,
    _write_canvas_kv_contiguous,
)

_needs_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)
_needs_mesh = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="device test; set DG_RUN_DEVICE=1",
)


# --- KV-write mode selection --------------------------------------------------


def _resolve(monkeypatch, *, kv_write=None):
    """Resolve the default mode under a given environment."""
    if kv_write is None:
        monkeypatch.delenv("DG_COMMIT_KV_WRITE", raising=False)
    else:
        monkeypatch.setenv("DG_COMMIT_KV_WRITE", kv_write)
    return commit_batched._default_kv_write_mode()


class _FakeTensor:
    """Stand-in for a ttnn canvas tensor: only ``shape`` is read before the guard fires."""

    shape = (1, 2, 32, 256)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(None, "fill", id="unset-defaults-to-one-op-fill"),
        pytest.param("fill", "fill", id="explicit-fill"),
        pytest.param("position", "position", id="explicit-position"),
        pytest.param(" FILL ", "fill", id="space-and-case-insensitive"),
        pytest.param("Position", "position", id="case-insensitive"),
    ],
)
def test_kv_write_mode_resolves(monkeypatch, value, expected):
    assert _resolve(monkeypatch, kv_write=value) == expected


def test_paged_mode_is_gone(monkeypatch, expect_error):
    """The 1-block-paged batched write was racy by construction and was removed.

    Also the loud-rejection contract for any unrecognised ``DG_COMMIT_KV_WRITE`` value.
    """
    assert "paged" not in commit_batched._KV_WRITE_MODES
    with expect_error(ValueError, match="DG_COMMIT_KV_WRITE"):
        _resolve(monkeypatch, kv_write="paged")


def test_module_default_is_fill_in_a_clean_env(monkeypatch):
    """Re-importing with no knobs set leaves the shipped default at ``fill``."""
    monkeypatch.delenv("DG_COMMIT_KV_WRITE", raising=False)
    reloaded = importlib.reload(commit_batched)
    try:
        assert reloaded._DEFAULT_KV_WRITE_MODE == "fill"
    finally:
        importlib.reload(commit_batched)


def test_unknown_write_mode_argument_fails_loudly(expect_error):
    with expect_error(ValueError, match="write_mode must be one of"):
        commit_batched._write_canvas_kv_contiguous(
            None,
            None,
            _FakeTensor(),
            _FakeTensor(),
            start_pos=0,
            canvas_len=32,
            mesh_device=None,
            write_mode="turbo",
        )


# --- one device session for the whole module ----------------------------------

TRACE_REGION = 64 << 20


# One device open/teardown for the whole module: repeated per-test CreateDevice on QB2 can
# hang an active-erisc core (see test_attention.py).
#
# It must also be exactly ONE session. Two co-resident module-scoped sessions (a single
# device plus a mesh over the same chips) silently break the traced tests: the second open
# does not re-create an already-active device, so chip 0 keeps the trace_region_size of
# whichever session opened it first, and both sessions then try to close it. So the mesh is
# opened once, with the trace region the traced tests need, and the single-chip tests take a
# (1,1) submesh of it rather than opening a device of their own.
@pytest.fixture(scope="module")
def mesh():
    shape = os.environ.get("MESH_DEVICE", "P150x4")
    rows, cols = (1, 4) if shape == "P150x4" else (int(shape.split("x")[0]), int(shape.split("x")[1]))
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), trace_region_size=TRACE_REGION)
    try:
        yield device
    finally:
        ttnn.close_mesh_device(device)


@pytest.fixture(scope="module")
def device(mesh):
    """Single-chip view of the module's mesh, for the tests that drive one chip directly."""
    if mesh.get_num_devices() == 1:
        return mesh
    return mesh.create_submesh(ttnn.MeshShape(1, 1))


# --- contiguous KV write: fill vs per-position (device) -----------------------


def _to_dev(device, host, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        host,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _write(device, *, mode, cache_host, canvas_k_host, canvas_v_host, start_pos, canvas_len, canvas_dtype):
    """Run one write mode into a fresh copy of ``cache_host``; return (k, v) host caches."""
    k_cache = _to_dev(device, cache_host)
    v_cache = _to_dev(device, cache_host)
    canvas_k = _to_dev(device, canvas_k_host, dtype=canvas_dtype)
    canvas_v = _to_dev(device, canvas_v_host, dtype=canvas_dtype)
    _write_canvas_kv_contiguous(
        k_cache,
        v_cache,
        canvas_k,
        canvas_v,
        start_pos=start_pos,
        canvas_len=canvas_len,
        mesh_device=device,
        write_mode=mode,
    )
    out = (ttnn.to_torch(k_cache), ttnn.to_torch(v_cache))
    for t in (k_cache, v_cache, canvas_k, canvas_v):
        t.deallocate(True)
    return out


# (nkv, head_dim, max_seq, start_pos, canvas_len). The 26B-A4B contiguous commit runs
# nkv_local in {1, 2, 8} (kv_replicated / TP-split sliding-vs-full), head_dim in
# {256, 512}, canvas_len 256, start_pos a multiple of 32.
GEOMETRIES = [
    (1, 256, 1024, 512, 256),
    (2, 256, 1024, 512, 256),  # 26B-A4B sliding layers on a 1x4 mesh
    (1, 512, 1024, 512, 256),  # 26B-A4B full-attention layers (kv_replicated, head_dim 512)
    (8, 256, 1024, 512, 256),
    (2, 512, 1024, 256, 256),
    (2, 256, 1024, 0, 256),  # first block: writes at offset 0
    (2, 256, 1024, 768, 256),  # last block: write ends exactly at max_seq
    (2, 256, 1024, 32, 32),  # single-tile canvas
    (8, 256, 2048, 1024, 256),
]


@_needs_device
@pytest.mark.parametrize("nkv,head_dim,max_seq,start_pos,canvas_len", GEOMETRIES)
def test_fill_write_is_bit_identical_to_per_position(device, nkv, head_dim, max_seq, start_pos, canvas_len):
    torch.manual_seed(0)
    # A non-zero cache so a disturbed prefix/tail cannot hide behind zeros.
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    canvas_k_host = torch.randn(1, nkv, canvas_len, head_dim).bfloat16()
    canvas_v_host = torch.randn(1, nkv, canvas_len, head_dim).bfloat16()
    kw = dict(
        cache_host=cache_host,
        canvas_k_host=canvas_k_host,
        canvas_v_host=canvas_v_host,
        start_pos=start_pos,
        canvas_len=canvas_len,
        canvas_dtype=ttnn.bfloat16,
    )

    assert (
        _fill_write_unsupported_reason(
            _to_dev(device, cache_host),
            _to_dev(device, cache_host),
            _to_dev(device, canvas_k_host),
            _to_dev(device, canvas_v_host),
            start_pos=start_pos,
            canvas_len=canvas_len,
            mesh_device=device,
        )
        is None
    ), "fill must be supported at the DiffusionGemma commit geometry"

    ref_k, ref_v = _write(device, mode="position", **kw)
    fill_k, fill_v = _write(device, mode="fill", **kw)

    # 1. the two mechanisms agree over the WHOLE cache, exactly.
    assert torch.equal(ref_k, fill_k), f"K differs, max_abs={(ref_k - fill_k).abs().max()}"
    assert torch.equal(ref_v, fill_v), f"V differs, max_abs={(ref_v - fill_v).abs().max()}"

    # 2. and both equal the torch oracle: canvas in the span, cache elsewhere.
    for name, got, canvas in (("K", fill_k, canvas_k_host), ("V", fill_v, canvas_v_host)):
        want = cache_host.clone()
        want[:, :, start_pos : start_pos + canvas_len, :] = canvas
        assert torch.equal(got, want), f"{name} != oracle, {int((got != want).sum())} elements wrong"


@_needs_device
def test_consecutive_blocks_preserve_each_other(device):
    """Two blocks in a row, as the generation loop does it.

    Block 2's write must leave block 1's committed K/V intact — the failure mode a
    single whole-canvas write could introduce (wrong offset, or a whole-slot fill) that
    a single-block test cannot see.
    """
    nkv, head_dim, max_seq, canvas_len = 2, 256, 1024, 256
    torch.manual_seed(3)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    blocks = [
        (256, torch.randn(1, nkv, canvas_len, head_dim).bfloat16()),
        (512, torch.randn(1, nkv, canvas_len, head_dim).bfloat16()),
    ]

    k_cache = _to_dev(device, cache_host)
    v_cache = _to_dev(device, cache_host)
    want = cache_host.clone()
    for start_pos, canvas_host in blocks:
        canvas = _to_dev(device, canvas_host)
        _write_canvas_kv_contiguous(
            k_cache,
            v_cache,
            canvas,
            canvas,
            start_pos=start_pos,
            canvas_len=canvas_len,
            mesh_device=device,
            write_mode="fill",
        )
        canvas.deallocate(True)
        want[:, :, start_pos : start_pos + canvas_len, :] = canvas_host

    for name, cache in (("K", k_cache), ("V", v_cache)):
        got = ttnn.to_torch(cache)
        assert torch.equal(got, want), f"{name} != oracle after 2 blocks, {int((got != want).sum())} wrong"


@_needs_device
def test_fill_falls_back_and_stays_correct_on_dtype_mismatch(device):
    """FILL refuses to convert dtypes; the guard must catch it and still write correctly."""
    nkv, head_dim, max_seq, start_pos, canvas_len = 2, 256, 1024, 512, 256
    torch.manual_seed(1)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    canvas_k_host = torch.randn(1, nkv, canvas_len, head_dim)
    canvas_v_host = torch.randn(1, nkv, canvas_len, head_dim)
    kw = dict(
        cache_host=cache_host,
        canvas_k_host=canvas_k_host,
        canvas_v_host=canvas_v_host,
        start_pos=start_pos,
        canvas_len=canvas_len,
        canvas_dtype=ttnn.float32,  # cache is bfloat16 -> FILL's dtype equality fails
    )

    reason = _fill_write_unsupported_reason(
        _to_dev(device, cache_host),
        _to_dev(device, cache_host),
        _to_dev(device, canvas_k_host, dtype=ttnn.float32),
        _to_dev(device, canvas_v_host, dtype=ttnn.float32),
        start_pos=start_pos,
        canvas_len=canvas_len,
        mesh_device=device,
    )
    assert reason is not None and "dtype" in reason

    ref_k, ref_v = _write(device, mode="position", **kw)
    fell_back_k, fell_back_v = _write(device, mode="fill", **kw)
    assert torch.equal(ref_k, fell_back_k)
    assert torch.equal(ref_v, fell_back_v)


@_needs_device
def test_cache_read_at_max_seq_does_not_alias_the_cache(device):
    """The commit reads back ``[0, start_pos+C)`` and deallocates the result.

    When the committed block ends exactly at ``max_seq`` that read is a FULL-span slice,
    which ttnn short-circuits to an alias of the input — so deallocating it would free the
    KV cache itself. ``_read_cache_kv`` must hand back a distinct buffer.
    """
    nkv, head_dim, max_seq = 2, 256, 1024
    torch.manual_seed(4)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    k_cache = _to_dev(device, cache_host)
    v_cache = _to_dev(device, cache_host)

    full_k, full_v = _read_cache_kv((k_cache, v_cache), end_pos=max_seq)
    assert full_k is not k_cache and full_v is not v_cache
    assert torch.equal(ttnn.to_torch(full_k), cache_host)
    full_k.deallocate(True)
    full_v.deallocate(True)

    # The cache must have survived the caller's deallocate and still be readable.
    assert torch.equal(ttnn.to_torch(k_cache), cache_host)
    assert torch.equal(ttnn.to_torch(v_cache), cache_host)


@_needs_device
def test_fill_guard_rejects_head_boundary_spill(device):
    """The one FILL hazard the op does not self-check, and the fallback that covers it.

    ``fill_cache``'s program factory splits (nkv * C/32) tile-rows over the core grid and
    each core writes its rows contiguously from a single ``cache_start_id``, assuming no
    core's range crosses a kv-head boundary. Once the rows exceed the core count (and the
    input does not span the whole cache) that assumption breaks and the op silently writes
    rows to the wrong head — device-confirmed here: with nkv=8, C=1024, max_seq=2048 a raw
    ``ttnn.fill_cache`` corrupts the cache, so the guard must reject it and fall back.
    """
    nkv, head_dim, max_seq, start_pos, canvas_len = 8, 256, 2048, 512, 1024
    grid = device.compute_with_storage_grid_size()
    rows = nkv * (canvas_len // ttnn.TILE_SIZE)
    assert rows > grid.x * grid.y, f"geometry no longer spills on this grid ({rows} rows, {grid.x * grid.y} cores)"

    torch.manual_seed(2)
    cache_host = torch.randn(1, nkv, max_seq, head_dim).bfloat16()
    canvas_host = torch.randn(1, nkv, canvas_len, head_dim).bfloat16()
    want = cache_host.clone()
    want[:, :, start_pos : start_pos + canvas_len, :] = canvas_host

    reason = _fill_write_unsupported_reason(
        _to_dev(device, cache_host),
        _to_dev(device, cache_host),
        _to_dev(device, canvas_host),
        _to_dev(device, canvas_host),
        start_pos=start_pos,
        canvas_len=canvas_len,
        mesh_device=device,
    )
    assert reason is not None and "spill" in reason

    # The raw op really does corrupt this geometry (guard is not superstition).
    raw_cache = _to_dev(device, cache_host)
    ttnn.fill_cache(raw_cache, _to_dev(device, canvas_host), 0, update_idx=start_pos)
    assert not torch.equal(ttnn.to_torch(raw_cache), want), (
        "ttnn.fill_cache no longer spills across kv-head boundaries here — the factory may "
        "have been fixed; re-check the guard in _fill_write_unsupported_reason"
    )
    raw_cache.deallocate(True)

    # ...and the guarded write is correct anyway, via the per-position fallback.
    got_k, got_v = _write(
        device,
        mode="fill",
        cache_host=cache_host,
        canvas_k_host=canvas_host,
        canvas_v_host=canvas_host,
        start_pos=start_pos,
        canvas_len=canvas_len,
        canvas_dtype=ttnn.bfloat16,
    )
    assert torch.equal(got_k, want)
    assert torch.equal(got_v, want)


# --- canvas-tail workspace inside a trace (device) ----------------------------

P_MAX = 128  # stands in for the served fixed span
CANVAS = 32  # tile-aligned canvas
NKV = 2
HEAD_DIM = 32


@contextmanager
def _capture(mesh_device, *, cq_id=0):
    """Capture a trace, releasing it if the body raises.

    Without this a failing op leaves the device stuck in capture state and every later test
    dies with "Reads/Writes are not supported during trace capture" — a cascade that hides the
    real failure. Mirrors tt/traced_denoise.py's ``_trace_capture_guard``.
    """
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=cq_id)
    try:
        yield trace_id
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)
            ttnn.release_trace(mesh_device, trace_id)
        except BaseException:
            pass
        raise
    else:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)


def _replicate(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None


def _to_mesh_replicated(t, mesh_device):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate(mesh_device),
    )


def _first_shard(tt, mesh_device):
    out = ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[:1] if out.shape[0] > 1 else out


def _make_workspace(mesh_device, prefix):
    """Cache sized [1, NKV, P_MAX + CANVAS, HEAD_DIM] with the prefix pre-written."""
    host = torch.zeros(1, NKV, P_MAX + CANVAS, HEAD_DIM, dtype=torch.bfloat16)
    host[:, :, :P_MAX, :] = prefix
    return _to_mesh_replicated(host, mesh_device)


@_needs_mesh
def test_fill_cache_writes_only_the_tail_eager(mesh):
    """Baseline: fill_cache(update_idx=P_MAX) must touch the tail and leave the prefix intact."""
    prefix = torch.randn(1, NKV, P_MAX, HEAD_DIM).to(torch.bfloat16)
    canvas = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)

    cache = _make_workspace(mesh, prefix)
    tt_canvas = _to_mesh_replicated(canvas, mesh)
    ttnn.fill_cache(cache, tt_canvas, 0, update_idx=P_MAX)

    got = _first_shard(cache, mesh).to(torch.float32)
    assert torch.equal(got[:, :, :P_MAX, :], prefix.to(torch.float32)), "prefix must be untouched"
    assert torch.equal(got[:, :, P_MAX:, :], canvas.to(torch.float32)), "tail must hold the canvas"


@_needs_mesh
def test_traced_fill_then_read_sees_each_replays_own_write(mesh):
    """The load-bearing one: a replayed trace must read the tail IT just wrote, not a stale one.

    The trace contains fill_cache(tail) followed by a read of the whole span. Between replays we
    only change the CONTENTS of the persistent canvas input buffer (never its address), which is
    exactly how the denoise loop would refresh the canvas per step.
    """
    prefix = torch.randn(1, NKV, P_MAX, HEAD_DIM).to(torch.bfloat16)
    cache = _make_workspace(mesh, prefix)
    addr_before = cache.buffer_address()

    canvas_a = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_b = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    assert not torch.equal(canvas_a, canvas_b)

    # Persistent input buffer allocated BEFORE capture (session-8 rule: anything a replay reads
    # must not come from trace scratch).
    canvas_buf = _to_mesh_replicated(canvas_a, mesh)

    # WARM UP first: a trace cannot load new kernel binaries, so every program it contains must
    # already be in the program cache ("Cannot load new binaries during trace capture").
    ttnn.fill_cache(cache, canvas_buf, 0, update_idx=P_MAX)
    ttnn.clone(cache).deallocate(True)

    with _capture(mesh, cq_id=0) as tid:
        ttnn.fill_cache(cache, canvas_buf, 0, update_idx=P_MAX)
        out_buf = ttnn.clone(cache)

    results = {}
    for tag, canvas in (("a", canvas_a), ("b", canvas_b)):
        fresh = _to_mesh_replicated(canvas, mesh)
        ttnn.copy(fresh, canvas_buf)
        fresh.deallocate(True)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        results[tag] = _first_shard(out_buf, mesh).to(torch.float32).clone()

    ttnn.release_trace(mesh, tid)

    for tag, canvas in (("a", canvas_a), ("b", canvas_b)):
        got = results[tag]
        assert torch.equal(got[:, :, :P_MAX, :], prefix.to(torch.float32)), f"replay {tag} corrupted the prefix"
        assert torch.equal(got[:, :, P_MAX:, :], canvas.to(torch.float32)), (
            f"replay {tag} read a STALE tail -- the traced fill and the traced read disagree, "
            "which would silently corrupt denoise attention"
        )
    assert not torch.equal(results["a"], results["b"]), "replays must differ; otherwise the read is frozen"
    assert cache.buffer_address() == addr_before, "cache buffer moved; a trace bakes this address"


@_needs_mesh
def test_traced_fill_then_sdpa_reads_the_fresh_tail(mesh):
    """Same contract, but the consumer is the real one: SDPA over [prefix ; canvas]."""
    torch.manual_seed(0)
    prefix = torch.randn(1, NKV, P_MAX, HEAD_DIM).to(torch.bfloat16)
    cache_k = _make_workspace(mesh, prefix)
    cache_v = _make_workspace(mesh, prefix)

    q = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    tt_q = _to_mesh_replicated(q, mesh)

    canvas_a = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_b = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_buf_k = _to_mesh_replicated(canvas_a, mesh)
    canvas_buf_v = _to_mesh_replicated(canvas_a, mesh)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
        q_chunk_size=CANVAS,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    def _fill_and_attend():
        ttnn.fill_cache(cache_k, canvas_buf_k, 0, update_idx=P_MAX)
        ttnn.fill_cache(cache_v, canvas_buf_v, 0, update_idx=P_MAX)
        return ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            cache_k,
            cache_v,
            is_causal=False,
            scale=1.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
        )

    # Warm the program cache before capture (see the note in the previous test).
    _fill_and_attend().deallocate(True)

    with _capture(mesh, cq_id=0) as tid:
        sdpa_out = _fill_and_attend()

    outs = {}
    for tag, canvas in (("a", canvas_a), ("b", canvas_b)):
        for buf in (canvas_buf_k, canvas_buf_v):
            fresh = _to_mesh_replicated(canvas, mesh)
            ttnn.copy(fresh, buf)
            fresh.deallocate(True)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        outs[tag] = _first_shard(sdpa_out, mesh).to(torch.float32).clone()

    ttnn.release_trace(mesh, tid)

    # Reference: SDPA over the span the replay should have seen.
    for tag, canvas in (("a", canvas_a), ("b", canvas_b)):
        k_ref = torch.cat([prefix, canvas], dim=2).to(torch.float32)
        ref = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float32), k_ref, k_ref, scale=1.0)
        got = outs[tag]
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        assert pcc > 0.99, f"replay {tag}: SDPA over the workspace disagrees with the reference (pcc={pcc:.5f})"
    assert not torch.equal(outs["a"], outs["b"]), "SDPA output frozen across replays -> stale tail"
