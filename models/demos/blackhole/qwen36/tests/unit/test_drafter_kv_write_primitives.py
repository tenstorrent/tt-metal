# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which in-place KV writes does the hardware actually accept? This decides the drafter trace design.

Tracing the drafter (lever A) needs its KV history to live in a PERSISTENT, FIXED-SHAPE buffer:
a trace bakes buffer addresses, so the current "concat a new tensor each step and rebind" scheme
cannot be captured. The open question is how to get new rows INTO such a buffer, and the answer
constrains everything downstream:

* If TILE interleaved slice_write works at an arbitrary row offset, the buffer can stay in the
  layout SDPA wants and the append is exact -- no gaps, capacity == real context.
* If it needs 32-row alignment, each step's commit (1..16 rows) must be padded to a 32-row page and
  the wasted rows masked out, which costs capacity and needs the bidirectional layer masked too.
* If TILE interleaved is unsupported entirely (the op ships RM-interleaved, RM-sharded and
  TILE-sharded program factories, but no TILE-interleaved one), the buffer must be ROW_MAJOR and
  converted per step, or sharded.

Source reading says the 32-row rule is real but binds only the TILE path
(slice_write_tiled_sharded_input_program_factory.cpp: "output start for the second last dimension
to be a multiple of tile height"). Reading is not measurement, and this is load-bearing enough to
measure. Each case below is a claim about the primitive, not about the drafter.

Separately and independently: slice_start/slice_end are OPERATION ATTRIBUTES, so whatever this
finds, a write offset that varies per step bakes at capture and cannot vary during replay. That is
why the probe also checks the fixed-offset shift, which is the trace-safe alternative.

Run::

    MESH_DEVICE=T3K pytest -svq \\
      models/demos/blackhole/qwen36/tests/unit/test_drafter_kv_write_primitives.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn

NKV, HD, CAP = 8, 128, 256


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _dev(t, device, layout):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _host(t, device):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:1]


def _try_slice_write(device, layout, offset, rows):
    """Attempt an in-place write of `rows` rows at `offset`. Returns (ok, detail)."""
    buf_t = torch.zeros(1, NKV, CAP, HD, dtype=torch.bfloat16)
    new_t = torch.arange(rows * NKV * HD, dtype=torch.float32).reshape(1, NKV, rows, HD).to(torch.bfloat16)
    buf = _dev(buf_t, device, layout)
    new = _dev(new_t, device, layout)
    try:
        ttnn.experimental.slice_write(new, buf, (0, 0, offset, 0), (1, NKV, offset + rows, HD), (1, 1, 1, 1))
    except Exception as e:  # noqa: BLE001 -- the whole point is to learn WHICH ones raise
        return False, type(e).__name__ + ": " + str(e).split("\n")[0][:160]
    got = _host(buf, device)
    want = buf_t.clone()
    want[:, :, offset : offset + rows, :] = new_t
    if not torch.equal(got.to(torch.bfloat16), want):
        wrote = (got.abs().sum(dim=(0, 1, 3)) > 0).nonzero().flatten().tolist()
        return False, f"wrote wrong rows; nonzero rows {wrote[:8]}{'...' if len(wrote) > 8 else ''}"
    return True, "exact"


@pytest.mark.parametrize(
    "layout,offset,rows",
    [
        (ttnn.ROW_MAJOR_LAYOUT, 7, 16),  # arbitrary offset, RM  -- the permissive case
        (ttnn.ROW_MAJOR_LAYOUT, 32, 16),  # aligned offset, RM
        (ttnn.TILE_LAYOUT, 32, 32),  # aligned offset + aligned rows, TILE
        (ttnn.TILE_LAYOUT, 32, 16),  # aligned offset, PART-tile rows
        (ttnn.TILE_LAYOUT, 7, 16),  # arbitrary offset, TILE -- expected to be rejected
    ],
    ids=["rm_off7", "rm_off32", "tile_off32_rows32", "tile_off32_rows16", "tile_off7"],
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_slice_write_support_matrix(mesh_device, layout, offset, rows, reset_seeds, ensure_gc):
    """Record, per (layout, offset, rows), whether an in-place write lands exactly.

    This test does NOT assert success -- it is a capability probe, and a rejection is a result, not
    a failure. It asserts only that the op either works exactly or refuses cleanly: a write that
    "succeeds" while landing on the wrong rows is the one outcome that would silently corrupt a
    drafter's KV history, so that is what fails here.
    """
    ok, detail = _try_slice_write(mesh_device, layout, offset, rows)
    name = "ROW_MAJOR" if layout == ttnn.ROW_MAJOR_LAYOUT else "TILE"
    logger.info(f"slice_write {name:9} offset={offset:3d} rows={rows:3d}: {'OK  ' if ok else 'NO  '} {detail}")
    assert ok or not detail.startswith("wrote wrong rows"), (
        f"slice_write({name}, offset={offset}, rows={rows}) reported success but wrote the wrong "
        f"rows -- silent KV corruption: {detail}"
    )


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_fixed_offset_shift_is_exact(mesh_device, reset_seeds, ensure_gc):
    """The trace-safe alternative: shift the whole buffer by a FIXED stride, write at a fixed tail.

    Every offset here is a compile-time constant, so this form survives a capture no matter what
    the matrix above says. The cost is that the stride must be the WORST case (block_size), so a
    step accepting fewer rows leaves a masked gap. This checks the shift itself is exact.
    """
    stride = 32
    buf_t = torch.arange(NKV * CAP * HD, dtype=torch.float32).reshape(1, NKV, CAP, HD).to(torch.bfloat16)
    new_t = torch.full((1, NKV, stride, HD), -1.0, dtype=torch.bfloat16)
    buf = _dev(buf_t, mesh_device, ttnn.TILE_LAYOUT)
    new = _dev(new_t, mesh_device, ttnn.TILE_LAYOUT)

    kept = ttnn.slice(buf, (0, 0, stride, 0), (1, NKV, CAP, HD), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shifted = ttnn.concat([kept, new], dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    want = torch.cat([buf_t[:, :, stride:, :], new_t], dim=-2)
    got = _host(shifted, mesh_device).to(torch.bfloat16)
    assert got.shape == want.shape, f"shift changed the shape: {tuple(got.shape)} vs {tuple(want.shape)}"
    assert torch.equal(got, want), "fixed-stride shift did not preserve the retained rows exactly"
    logger.info(f"fixed-stride shift by {stride}: exact, shape {tuple(got.shape)} preserved")


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_sub_block_write_into_a_fused_layer_buffer(mesh_device, reset_seeds, ensure_gc):
    """Can one tensor hold ALL layers' K, written a layer at a time? That is 20 ops -> 4.

    The drafter's KV commit is the largest phase of a traced step (17.9 ms, 37 %, 20 dispatches:
    5 layers x {K,V} x {slice, slice_write}) and it is the one phase a capture cannot hold, because
    its row offset advances with the accept count. Fusing the per-layer buffers into ONE
    ``[L, nkv, rows, hd]`` tensor collapses it to a single slice plus a single slice_write per
    K/V -- but only if slice_write will write a one-layer sub-block at ``(i, 0, 0, 0)``.

    Two earlier assumptions about this op were wrong (a part-tile destination segfaulted; a
    full-width ttnn.slice aliased its input), so this asks the hardware rather than assuming.
    The inner rows here are 16 -- a PART TILE -- which is exactly the shape that crashed before,
    now as a sub-range of a taller fused buffer.
    """
    L, ROWS = 5, 16
    fused_t = torch.zeros(L, NKV, ROWS, HD, dtype=torch.bfloat16)
    layer_t = torch.arange(NKV * ROWS * HD, dtype=torch.float32).reshape(1, NKV, ROWS, HD).to(torch.bfloat16)

    for layout, name in ((ttnn.TILE_LAYOUT, "TILE"), (ttnn.ROW_MAJOR_LAYOUT, "ROW_MAJOR")):
        fused = _dev(fused_t, mesh_device, layout)
        layer = _dev(layer_t, mesh_device, layout)
        target_layer = 3
        try:
            ttnn.experimental.slice_write(
                layer, fused, (target_layer, 0, 0, 0), (target_layer + 1, NKV, ROWS, HD), (1, 1, 1, 1)
            )
        except Exception as e:  # noqa: BLE001 -- a rejection is a result
            logger.info(f"{name:9} sub-block write REJECTED: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
            continue
        got = ttnn.to_torch(fused, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:L]
        want = fused_t.clone()
        want[target_layer] = layer_t[0]
        ok = torch.equal(got.to(torch.bfloat16), want)
        touched = (got.abs().sum(dim=(1, 2, 3)) > 0).nonzero().flatten().tolist()
        logger.info(
            f"{name:9} sub-block write into layer {target_layer}: {'EXACT' if ok else 'WRONG'}, layers touched {touched}"
        )
        assert ok, (
            f"{name} slice_write into a fused [L, nkv, rows, hd] buffer wrote the wrong data "
            f"(layers touched {touched}, expected [{target_layer}]) -- fusing the drafter's KV "
            "commit this way would silently corrupt another layer's history"
        )


@pytest.mark.parametrize("offset,rows", [(0, 16), (7, 16), (7, 3), (32, 16)], ids=lambda v: str(v))
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_partial_row_write_across_all_layers(mesh_device, offset, rows, reset_seeds, ensure_gc):
    """The COMMIT's actual shape: write a partial row range into EVERY layer at once.

    The probe above validated a different write -- one layer, full row extent, contiguous -- and
    fusing the drafter's KV commit across layers was built on generalizing from it. That was wrong:
    with the fused layout the staged-vs-unstaged gate went from pcc 1.000000 to 0.8177, starting at
    the first commit that wrote a full 16-row block.

    This is the write that actually matters. For a ``[L, nkv, C, hd]`` destination, writing rows
    ``[offset, offset+rows)`` for all L layers is STRIDED -- each layer's slab lives C rows apart --
    where the single-layer case was contiguous. Whether slice_write handles that is the question,
    and the per-layer `touched` check is what distinguishes "wrote the wrong rows" from "wrote one
    layer's data into another layer's history", which is the failure that would silently poison the
    drafter's context.
    """
    L, CAP = 5, 128
    dst_t = torch.zeros(L, NKV, CAP, HD, dtype=torch.bfloat16)
    src_t = (torch.arange(L * NKV * rows * HD, dtype=torch.float32) + 1).reshape(L, NKV, rows, HD).to(torch.bfloat16)

    dst = _dev(dst_t, mesh_device, ttnn.TILE_LAYOUT)
    src = _dev(src_t, mesh_device, ttnn.TILE_LAYOUT)
    try:
        ttnn.experimental.slice_write(src, dst, (0, 0, offset, 0), (L, NKV, offset + rows, HD), (1, 1, 1, 1))
    except Exception as e:  # noqa: BLE001
        logger.info(f"offset={offset:3d} rows={rows:3d}: REJECTED {type(e).__name__}: {str(e).splitlines()[0][:110]}")
        pytest.skip("slice_write rejected the fused partial-row write")

    got = ttnn.to_torch(dst, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:L].to(torch.bfloat16)
    want = dst_t.clone()
    want[:, :, offset : offset + rows, :] = src_t
    ok = torch.equal(got, want)
    per_layer = [bool(torch.equal(got[i], want[i])) for i in range(L)]
    logger.info(f"offset={offset:3d} rows={rows:3d}: {'EXACT' if ok else 'WRONG'}  per-layer {per_layer}")
    assert ok, (
        f"fused partial-row slice_write (offset={offset}, rows={rows}) wrote the wrong data; "
        f"per-layer correctness {per_layer}. A [L, nkv, C, hd] destination needs a STRIDED write "
        "per layer, and if the op does not do that, the drafter's KV commit cannot be fused "
        "across layers"
    )
