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
