# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Equivalence test for the fused-op-backed ``_apply_rope`` in attention.py.

Covers the real decode call-site shapes: ``q`` (cos/sin broadcast across the H head-rows),
``kv`` (single row), the compressor (per-window cos/sin rows), and the ``d == rope_dim`` case.
Compares the on-device ``_apply_rope`` against a torch float32 reference of the same math.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.attention import _apply_rope, _interleaved_rotate_matrix


PCC_THRESHOLD = 0.999


def _torch_reference(x, cos, sin, rot, rope_dim):
    d = x.shape[-1]
    if d == rope_dim:
        nope, rope = None, x
    else:
        nope = x[..., : d - rope_dim]
        rope = x[..., d - rope_dim :]
    rotated = rope * cos + (rope @ rot) * sin
    if nope is None:
        return rotated
    return torch.cat([nope, rotated], dim=-1)


# (D, Rd, rows, cos_rows) -- cos_rows == 1 exercises the head/broadcast path.
@pytest.mark.parametrize(
    "D, Rd, rows, cos_rows",
    (
        (512, 64, 64, 1),  # q: [1,1,H=64,Dh], cos broadcast across heads
        (512, 64, 1, 1),  # kv: single row
        (512, 64, 5, 5),  # compressor: n_win rows, per-window cos/sin
        (64, 64, 64, 1),  # d == rope_dim (no nope), head-broadcast
    ),
)
def test_apply_rope_fused(device, reset_seeds, D, Rd, rows, cos_rows):
    x = torch.randn(1, 1, rows, D, dtype=torch.float32)
    cos = torch.randn(1, 1, cos_rows, Rd, dtype=torch.float32)
    sin = torch.randn(1, 1, cos_rows, Rd, dtype=torch.float32)
    rot = _interleaved_rotate_matrix(Rd)

    # ``_apply_rope`` accepts either a per-row cos/sin table or a single row that the fused op
    # broadcasts across all input rows on device. Feed the raw (possibly single-row) table and
    # broadcast only for the torch reference.
    cos_ref = cos.expand(1, 1, rows, Rd).contiguous()
    sin_ref = sin.expand(1, 1, rows, Rd).contiguous()
    ref = _torch_reference(x, cos_ref, sin_ref, rot, Rd)

    def to_tt(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    rot_tt = ttnn.from_torch(rot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def to_tt_dram(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    out_tt = _apply_rope(to_tt(x), to_tt_dram(cos), to_tt_dram(sin), rot_tt, Rd)
    got = ttnn.to_torch(out_tt).reshape(ref.shape).float()

    passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
    logger.info(f"[apply_rope D={D} Rd={Rd} rows={rows} cos_rows={cos_rows}] {comp_allclose(ref, got)}")
    logger.info(f"[apply_rope D={D} Rd={Rd} rows={rows} cos_rows={cos_rows}] PCC: {pcc_message}")
    assert (
        passing
    ), f"_apply_rope PCC < {PCC_THRESHOLD} (D={D}, Rd={Rd}, rows={rows}, cos_rows={cos_rows}): {pcc_message}"
