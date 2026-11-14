# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The speculative decoder's dense token pick (SpeculativeDecoder._argmax_last) without the pad to 32 rows.

The pick untilizes the logits and runs the multicore ROW_MAJOR argmax, which splits the vocab dim across cores. For
every row count 1..32 at the real vocab width it must return the same index as the padded path
(GEMMA4_ARGMAX_PAD32=1) and as a host argmax. Single chip, no checkpoint."""

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
VOCAB = 262144


def _pick(mesh_device, monkeypatch, logits, rows, padded):
    """Indices from the production token pick, unpadded (PAD32 unset) or padded (PAD32=1)."""
    if padded:
        monkeypatch.setenv("GEMMA4_ARGMAX_PAD32", "1")
    else:
        monkeypatch.delenv("GEMMA4_ARGMAX_PAD32", raising=False)
    out = SpeculativeDecoder.__new__(SpeculativeDecoder)._argmax_last(logits, rows)
    ttnn.synchronize_device(mesh_device)
    assert tuple(out.shape) == (1, 1, rows), f"rows={rows}: index shape {tuple(out.shape)}"
    idx = ttnn.to_torch(out).reshape(-1).to(torch.int64)
    out.deallocate(True)
    return idx


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_unpadded_token_pick_matches_padded(mesh_device, device_params, dtype, monkeypatch):
    planted_fail, natural_defect = [], []
    for rows in range(1, 33):
        g = torch.Generator().manual_seed(1000 * rows + (0 if dtype == ttnn.bfloat16 else 1))
        # Planted: a strictly unique max per row at a row-dependent position (first, last, middle, scattered),
        # so a work split that misses or misplaces part of a row fails positionally.
        t = -torch.rand(1, 1, rows, VOCAB, generator=g) * 0.5 - 0.5
        for r in range(rows):
            t[0, 0, r, (0, VOCAB - 1, VOCAB // 2, (r * 104729 + 13) % VOCAB)[r % 4]] = 1.0
        if dtype == ttnn.bfloat16:
            t = t.bfloat16().float()
        gold = t[0, 0].argmax(dim=-1)
        src = ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        for padded in (False, True):
            bad = int((_pick(mesh_device, monkeypatch, src, rows, padded) != gold).sum())
            if bad:
                planted_fail.append((rows, "padded" if padded else "unpadded", bad))
        src.deallocate(True)
        # Natural data, where bf16 can tie: both paths must agree, or point at equal values.
        t2 = torch.randn(1, 1, rows, VOCAB, generator=g)
        if dtype == ttnn.bfloat16:
            t2 = t2.bfloat16().float()
        src2 = ttnn.from_torch(t2, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        fast = _pick(mesh_device, monkeypatch, src2, rows, False)
        padv = _pick(mesh_device, monkeypatch, src2, rows, True)
        src2.deallocate(True)
        for r in torch.nonzero(fast != padv).flatten().tolist():
            if float(t2[0, 0, r, fast[r]]) != float(t2[0, 0, r, padv[r]]):
                natural_defect.append((rows, r, int(fast[r]), int(padv[r])))
    assert not planted_fail, f"wrong index on planted data (rows, path, wrong rows): {planted_fail}"
    assert not natural_defect, f"paths disagree on non-tied values (rows, row, unpadded, padded): {natural_defect}"


@pytest.mark.parametrize("rows", [1, 4], ids=["drafter", "verify"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_token_pick_of_l1_logits(mesh_device, device_params, rows, monkeypatch):
    """Logits in L1 (GEMMA4_ASSISTANT_LOGITS_L1) give the same index as the host argmax."""
    g = torch.Generator().manual_seed(11 + rows)
    t = -torch.rand(1, 1, rows, VOCAB, generator=g) * 0.5 - 0.5
    for r in range(rows):
        t[0, 0, r, (r * 104729 + 13) % VOCAB] = 1.0
    t = t.bfloat16().float()
    src = ttnn.from_torch(
        t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    got = _pick(mesh_device, monkeypatch, src, rows, False)
    src.deallocate(True)
    assert torch.equal(got, t[0, 0].argmax(dim=-1))
