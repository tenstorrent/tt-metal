# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro of the `slice_rm` sub-tile boundary SEGFAULT on Quasar (FAST dispatch).

Origin: the llama32_1b e2e on the 3 MB-L1 Quasar variant, run under FAST dispatch, segfaults during
decode compile inside the SDPA-decode-split workaround (`test_llama_e2e.py::_split_decode`). That
workaround runs SDPA once per kv-head (nkv=8, qpk = nq/nkv = 32/8 = 4) and reassembles the output by
slicing 4 rows out of each `[1,1,32,64]` ROW-MAJOR SDPA output and concatenating them:

    ttnn.slice(outs[h], [0, 0, h*4, 0], [1, 1, (h+1)*4, 64])     # h = 0 .. 7

In the failing run (`~/trinity261007b_e2e.txt`), the first SEVEN of these slices succeed:

    h=1 rows 4:8  OK    h=4 rows 16:20 OK
    h=2 rows 8:12 OK    h=5 rows 20:24 OK
    h=3 rows 12:16 OK   h=6 rows 24:28 OK

and the EIGHTH -- `h=7`, rows **28:32**, whose `slice_end[2] == 32 == the tensor height** -- hard
SEGFAULTS (a C++ crash: no device error, no `Not done phys cores`, no TT_FATAL). Because it is a
process-level segfault, the Python `try/except` host-stitch fallback in `_split_decode` never runs.

This is NOT the slow-dispatch tile-counter residue hang, and NOT 3 MB-specific (the op is ~1 KB). In
the SLOW-dispatch run of the same build, these exact slices all ran fine (decode compile completed).
So the trigger appears to be: a **sub-tile (< one 32-row tile) ROW-MAJOR slice on the tiled height
dim whose end touches the tensor boundary, under FAST dispatch.**

This file isolates that. Each case is a separate parametrized test so a segfault in one does not hide
the others ACROSS separate invocations. A segfault kills the pytest process -> that death IS the repro
signal (the case cannot "fail" an assert; it crashes). Run a single case at a time with `-k`.

Run under FAST dispatch (the default -- do NOT set TT_METAL_SLOW_DISPATCH_MODE; slow dispatch does NOT
reproduce):

    MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_slice_rm_boundary.py

    # the exact e2e crash (expected to SEGFAULT the process):
    MESH_DEVICE=N150 pytest ...::test_slice_rm_rowblock -k "h7_rows28_32"

    # the interior neighbor (expected to PASS):
    MESH_DEVICE=N150 pytest ...::test_slice_rm_rowblock -k "h6_rows24_28"
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

# The SDPA decode output per kv-head: [1, 1, num_heads=32, head_dim=64], ROW-MAJOR, DRAM, bf16.
B0, B1, H, W = 1, 1, 32, 64
QPK = 4  # q-heads per kv-head (nkv = 32 / 4 = 8); each reassembly slice is QPK rows tall.


@pytest.fixture()
def qsr_device():
    """Open a single-device mesh, honoring LLAMA_QSR_WORKER_L1_SIZE for parity with the e2e (the bug is
    not L1-size-dependent, but keep the device config identical so nothing else changes)."""
    wl1 = os.environ.get("LLAMA_QSR_WORKER_L1_SIZE", "").strip()
    kwargs = {}
    if wl1:
        kwargs["worker_l1_size"] = int(wl1, 0)
    try:
        num_pcie = ttnn.get_num_pcie_devices()
    except Exception as e:  # pragma: no cover - environment probe
        pytest.skip(f"cannot query TT devices: {e}")
    if isinstance(num_pcie, int) and num_pcie == 0:
        pytest.skip("no TT devices detected")
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **kwargs)
    try:
        yield dev
    finally:
        ttnn.close_mesh_device(dev)


def _upload(t, dev, layout):
    """Upload a torch tensor to DRAM in the requested layout (ROW_MAJOR mirrors the SDPA output)."""
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
    )


def _readback(x):
    """to_torch of a (replicated) single-device mesh tensor -> plain torch."""
    try:
        return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(x.device(), dim=0))[:B0]
    except Exception:
        return ttnn.to_torch(x)


def _check_slice(dev, src_h, start_row, end_row, layout):
    """Slice rows [start_row:end_row] of a [B0,B1,src_h,W] tensor and verify against torch.
    A SEGFAULT here crashes the process (the repro); a clean return + matching values is a PASS."""
    torch.manual_seed(0)
    t = torch.randn(B0, B1, src_h, W, dtype=torch.float32)
    x = _upload(t, dev, layout)
    logger.info(
        f"[slice-repro] src=[{B0},{B1},{src_h},{W}] layout={layout} slice rows [{start_row}:{end_row}] "
        f"(end==height: {end_row == src_h})"
    )
    y = ttnn.slice(x, [0, 0, start_row, 0], [B0, B1, end_row, W], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = _readback(y).to(torch.float32)
    ref = t[:, :, start_row:end_row, :]
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    # bf16 round-trip; exact slice so values should match closely.
    assert torch.allclose(out, ref.to(torch.bfloat16).to(torch.float32), atol=1e-2), "sliced values mismatch"
    logger.info(f"[slice-repro] rows [{start_row}:{end_row}] PASSED")


# ---- Case 1: the exact e2e reassembly slices, one per kv-head (h=7 end==height is the crash) ----
_ROWBLOCKS = [(f"h{h}_rows{h * QPK}_{(h + 1) * QPK}", h * QPK, (h + 1) * QPK) for h in range(H // QPK)]


@pytest.mark.parametrize("name,start_row,end_row", _ROWBLOCKS, ids=[c[0] for c in _ROWBLOCKS])
def test_slice_rm_rowblock(qsr_device, name, start_row, end_row):
    """The 8 per-head reassembly slices from _split_decode on a [1,1,32,64] ROW-MAJOR tensor.
    Expected: h0..h6 PASS; h7 (rows 28:32, end==32==height) SEGFAULTS the process under fast dispatch."""
    _check_slice(qsr_device, H, start_row, end_row, ttnn.ROW_MAJOR_LAYOUT)


# ---- Case 2: isolate "end == height" vs "offset 28" by growing the source height past 32 ----
# If [28:32] of a height-32 tensor crashes but [28:32] of a height-64 tensor (same offset, NOT the
# boundary) passes, the trigger is end==height, not the row offset.
@pytest.mark.parametrize(
    "name,src_h,start_row,end_row",
    [
        ("h32_rows28_32_BOUNDARY", 32, 28, 32),  # end == height  -> expected crash
        ("h64_rows28_32_interior", 64, 28, 32),  # same offset, interior of a taller tensor -> expected pass
        ("h64_rows60_64_BOUNDARY", 64, 60, 64),  # boundary again at a different height -> expected crash
    ],
    ids=["h32_rows28_32_BOUNDARY", "h64_rows28_32_interior", "h64_rows60_64_BOUNDARY"],
)
def test_slice_rm_boundary_vs_interior(qsr_device, name, src_h, start_row, end_row):
    """Does the crash depend on slice_end touching the tensor boundary, or on the row offset itself?"""
    _check_slice(qsr_device, src_h, start_row, end_row, ttnn.ROW_MAJOR_LAYOUT)


# ---- Case 3: layout sensitivity -- does TILE layout avoid the crash the RM boundary slice hits? ----
@pytest.mark.parametrize("layout_name", ["row_major", "tile"])
def test_slice_boundary_layout(qsr_device, layout_name):
    """Boundary slice rows [28:32] of [1,1,32,64] in ROW_MAJOR vs TILE. RM is the e2e path (expected
    crash); TILE tells us whether a tiled slice sidesteps it (a candidate device-only workaround)."""
    layout = ttnn.ROW_MAJOR_LAYOUT if layout_name == "row_major" else ttnn.TILE_LAYOUT
    _check_slice(qsr_device, H, 28, 32, layout)
