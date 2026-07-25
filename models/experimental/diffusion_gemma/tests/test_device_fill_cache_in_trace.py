# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test for the canvas-tail workspace primitive (#51080 roadmap item 4).

Item 4 removes the per-step prefix copy entirely: instead of
``concat([prefix_kv, canvas_kv])`` (which re-copies the whole p_max prefix every step), size the
KV cache as ``p_max + canvas_len`` and write the canvas K/V into the TAIL with
``ttnn.fill_cache(update_idx=p_max)``, so SDPA reads one already-contiguous tensor.

That rests on one assumption that must be proven before any model wiring, because it is the
part that can fail catastrophically rather than merely slowly:

    Can a captured Metal trace contain ``fill_cache`` writing into a persistent tensor AND a
    subsequent read of that same tensor, such that every replay observes the freshly written
    tail rather than a stale one?

A trace bakes buffer addresses, so the write and the read must agree across replays. If they do
not, the failure is silent wrong output, not a crash. This test pins it in isolation:

1. eager reference: fill the tail, read the whole span, compare to torch;
2. capture a trace doing fill+read, then replay it TWICE with DIFFERENT tail contents and check
   each replay observes its own write (the stale-read failure mode);
3. assert the cache buffer address is unchanged across replays (the baked-address assumption).

Run with DG_RUN_DEVICE=1 and a mesh; skipped otherwise.

Two operational notes, both learned by getting them wrong here:

* A trace cannot load new kernel binaries, so every program inside a capture must ALREADY be in
  the program cache. Each test therefore runs its ops once eagerly before ``begin_trace_capture``;
  without that warm-up the capture dies with "Cannot load new binaries during trace capture".
* If a capture aborts, the trace must still be ended and released, or every later op in the
  process fails with "Reads/Writes are not supported during trace capture" and the real failure
  is buried under the cascade. Hence the ``_capture`` guard below. Relatedly: do NOT SIGKILL a
  process that is mid-capture — it keeps the ``CHIP_IN_USE`` lock and its TLB windows, and the
  next run fails to even open the device (``tt_tlb_alloc failed ... error code -12``). Let the
  holder exit; the driver releases on exit and no ``tt-smi -r`` is needed.
"""

import os
from contextlib import contextmanager

import pytest
import torch

import ttnn


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


pytestmark = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="device test; set DG_RUN_DEVICE=1",
)

P_MAX = 128  # stands in for the served fixed span
CANVAS = 32  # tile-aligned canvas
NKV = 2
HEAD_DIM = 32
TRACE_REGION = 64 << 20


@pytest.fixture(scope="module")
def mesh():
    shape = os.environ.get("MESH_DEVICE", "P150x4")
    rows, cols = (1, 4) if shape == "P150x4" else (int(shape.split("x")[0]), int(shape.split("x")[1]))
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), trace_region_size=TRACE_REGION)
    try:
        yield device
    finally:
        ttnn.close_mesh_device(device)


def _replicate(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None


def _to_device(t, mesh_device):
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
    return _to_device(host, mesh_device)


def test_fill_cache_writes_only_the_tail_eager(mesh):
    """Baseline: fill_cache(update_idx=P_MAX) must touch the tail and leave the prefix intact."""
    prefix = torch.randn(1, NKV, P_MAX, HEAD_DIM).to(torch.bfloat16)
    canvas = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)

    cache = _make_workspace(mesh, prefix)
    tt_canvas = _to_device(canvas, mesh)
    ttnn.fill_cache(cache, tt_canvas, 0, update_idx=P_MAX)

    got = _first_shard(cache, mesh).to(torch.float32)
    assert torch.equal(got[:, :, :P_MAX, :], prefix.to(torch.float32)), "prefix must be untouched"
    assert torch.equal(got[:, :, P_MAX:, :], canvas.to(torch.float32)), "tail must hold the canvas"


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
    canvas_buf = _to_device(canvas_a, mesh)

    # WARM UP first: a trace cannot load new kernel binaries, so every program it contains must
    # already be in the program cache ("Cannot load new binaries during trace capture").
    ttnn.fill_cache(cache, canvas_buf, 0, update_idx=P_MAX)
    ttnn.clone(cache).deallocate(True)

    with _capture(mesh, cq_id=0) as tid:
        ttnn.fill_cache(cache, canvas_buf, 0, update_idx=P_MAX)
        out_buf = ttnn.clone(cache)

    results = {}
    for tag, canvas in (("a", canvas_a), ("b", canvas_b)):
        fresh = _to_device(canvas, mesh)
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


def test_traced_fill_then_sdpa_reads_the_fresh_tail(mesh):
    """Same contract, but the consumer is the real one: SDPA over [prefix ; canvas]."""
    torch.manual_seed(0)
    prefix = torch.randn(1, NKV, P_MAX, HEAD_DIM).to(torch.bfloat16)
    cache_k = _make_workspace(mesh, prefix)
    cache_v = _make_workspace(mesh, prefix)

    q = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    tt_q = _to_device(q, mesh)

    canvas_a = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_b = torch.randn(1, NKV, CANVAS, HEAD_DIM).to(torch.bfloat16)
    canvas_buf_k = _to_device(canvas_a, mesh)
    canvas_buf_v = _to_device(canvas_a, mesh)

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
            fresh = _to_device(canvas, mesh)
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
