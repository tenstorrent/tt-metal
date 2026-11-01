# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Milestone 1 diagnosis: dump one core's inputs so the tile pairing can be checked by hand.

The kernel runs to completion (no hang) but produces wrong values, and I have ruled out:
  - the diagonal product maths       (torch check: (state @ diag) row0 == elementwise product, exact)
  - diagonal page indexing           (all page ids in range, K*c_t = 1280 pages)
  - the reference conv/shift logic   (torch check: row0..row2 <- state1..3, row3 <- qkv, exact)

So the remaining suspect is which TILE is paired with which: the reader packs a head's channels as
q..k..v contiguously (per_core tiles), while everything downstream indexes by full-row offsets
(q_col / k_col / v_col against c_t). This dumps, for ONE value head, the exact tiles the reader
fetched and the exact tiles torch says it should have fetched, so the two can be compared tile by
tile instead of by end-to-end error.

Run: MESH_DEVICE=P150 pytest <this> -s
"""
import os
import sys

import pytest
import torch

import ttnn

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "ops"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("conv_silu_op", os.path.join(_here, "ops", "conv_silu", "op.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
conv_silu, geometry, tap_diagonals = _mod.conv_silu, _mod.geometry, _mod.tap_diagonals

KEY_DIM, VALUE_DIM, NV, GQA, K = 2048, 6144, 48, 3, 4
TILE = 32


def _dev(mesh, t):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_dump_head0(mesh_device):
    g = geometry(KEY_DIM, VALUE_DIM, NV, GQA, K)
    print(f"geometry {g}")
    torch.manual_seed(0)
    C = g["C"]
    qkv = torch.randn(1, 1, C, dtype=torch.bfloat16)
    state = torch.randn(K, 1, C, dtype=torch.bfloat16)

    conv, _ = conv_silu(
        _dev(mesh_device, qkv), _dev(mesh_device, state), _dev(mesh_device, tap_diagonals(torch.randn(K, 1, C, dtype=torch.bfloat16))),
        KEY_DIM, VALUE_DIM, NV, GQA,
    )
    got = ttnn.to_torch(conv).float()

    # What head 0's channels are, per the full-row layout the reader claims to use.
    kd_t, v_t, nk = g["kd_t"], g["v_t"], g["nk"]
    hv = 0
    qh = hv // GQA
    q_slice = slice(qh * kd_t * TILE, (qh + 1) * kd_t * TILE)
    k_slice = slice((nk + qh) * kd_t * TILE, (nk + qh + 1) * kd_t * TILE)
    v_slice = slice(2 * nk * kd_t * TILE + hv * v_t * TILE, 2 * nk * kd_t * TILE + (hv + 1) * v_t * TILE)

    print(f"head 0 q_slice {q_slice.start}:{q_slice.stop}  k_slice {k_slice.start}:{k_slice.stop}  v_slice {v_slice.start}:{v_slice.stop}")
    for name, sl in (("q", q_slice), ("k", k_slice), ("v", v_slice)):
        got_vals = got[0, 0, sl]
        # the same channels straight out of the raw qkv (identity check: if the conv taps were all 1
        # and state zero, conv would equal silu(qkv)); here we only check the RANGES overlap sanely
        raw = qkv[0, 0, sl].float()
        print(f"  {name}: got max {got_vals.abs().max():.4f} mean {got_vals.mean():.4f} | raw max {raw.abs().max():.4f}")

    # Which output columns are non-zero at all? Any column outside the union of the head slices being
    # non-zero means the writer wrote outside its own range.
    nz = (got[0, 0, :].abs() > 1e-6).nonzero().flatten()
    print(f"non-zero output columns: {nz.numel()} of {C}, first {nz[:6].tolist()}, last {nz[-3:].tolist()}")

    # Per-head coverage: a 48-head run should cover 48 * (2*kd_t + v_t) * 32 = 48*12*32 = 18432
    # channel writes, but q/k are SHARED by gqa heads, so distinct columns = 320 tiles' worth of
    # channels only if every head's write lands in a distinct place. Count them.
    covered = set()
    for h in range(NV):
        qh2 = h // GQA
        covered |= set(range(qh2 * kd_t * TILE, (qh2 + 1) * kd_t * TILE))
        covered |= set(range((nk + qh2) * kd_t * TILE, (nk + qh2 + 1) * kd_t * TILE))
        covered |= set(range(2 * nk * kd_t * TILE + h * v_t * TILE, 2 * nk * kd_t * TILE + (h + 1) * v_t * TILE))
    print(f"distinct columns the 48 heads should cover: {len(covered)} of {C}")
