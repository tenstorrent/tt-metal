# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the llama32_1b lm_head concat L1-overflow OOM on the 2-compute-node emulator.

The lm_head splits the vocab projection and concatenates the pieces into the full-vocab logits
(lm_head_1d.py:154):

    output = ttnn.concat(outputs, dim=-1, memory_config=cfg.output_memcfg)   # output_memcfg = L1_MEMORY_CONFIG

The output is [1, 1, 32, VOCAB=128256] bf16 (~8 MB). L1_MEMORY_CONFIG is L1 INTERLEAVED, so on the 2-node
emulator it spreads across 2 banks = ~4 MB/bank > the 3.88 MB bank size -> Out of Memory (bank_manager.cpp).
The full-vocab logits cannot live in L1 on 2 cores; they must go to DRAM.

The e2e fix (_install_quasar_concat_l1_overflow_to_dram) routes an oversized L1 concat output to DRAM
interleaved. This file isolates BOTH:
  * test_lm_head_concat_l1_overflow: the raw L1 concat -> OOM on a grid where it doesn't fit (skips where it
    does, e.g. a >= ~34-core grid whose per-bank share is under budget), documenting the limit.
  * test_lm_head_concat_dram: the fix -- the same concat to DRAM -> fits ANY grid, PCC-checked vs a torch
    concat reference.

Inputs are built bf16 via row-major upload + quasar.tilize (from_torch(TILE) hangs on the sim). VOCAB is the
real 128256 (needed to actually overflow L1); split into a few wide tile-aligned pieces to limit tilize calls.

Run (Quasar sim, 2-node emulator, SLOW dispatch = the true emulator grid):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_lm_head_concat.py
"""

import pytest
import torch
from loguru import logger

import ttnn

VOCAB = 128256  # llama-3.2-1B vocab; the concat output width (must be this big to overflow L1 on 2 cores)
ROWS = 32  # batch padded to one tile
N_SPLITS = 4  # few wide splits (VOCAB/4 = 32064, tile-aligned) to keep tilize cheap on the sim
SPLIT_W = VOCAB // N_SPLITS
_L1_BANK_BUDGET = 3_800_000  # under the ~3.88 MB L1 bank size


def _tile_bf16_dram(t_bf16, mesh_device):
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_splits(mesh_device):
    assert SPLIT_W % 32 == 0, f"split width {SPLIT_W} must be tile-aligned"
    torch.manual_seed(0)
    parts = [torch.randn(1, 1, ROWS, SPLIT_W, dtype=torch.bfloat16) for _ in range(N_SPLITS)]
    tt = [_tile_bf16_dram(p, mesh_device) for p in parts]
    return parts, tt


@pytest.mark.timeout(3600)
def test_lm_head_concat_l1_overflow(mesh_device, expect_error):
    """Raw L1 concat of the full-vocab logits -> OOM when the [1,1,32,VOCAB] output exceeds the per-bank L1
    budget (the 2-node emulator: ~4 MB/bank > 3.88 MB). On a grid with enough banks that it fits, skip. This is
    the FATAL the e2e hit at the lm_head before the DRAM coercion."""
    dev = mesh_device.compute_with_storage_grid_size()
    ncores = max(int(dev.x) * int(dev.y), 1)
    per_bank = (ROWS * VOCAB * 2) / ncores  # bf16 interleaved across `ncores` banks
    _, tt = _build_splits(mesh_device)
    if per_bank <= _L1_BANK_BUDGET:
        pytest.skip(
            f"L1 concat output fits ({per_bank:.0f} B/bank <= {_L1_BANK_BUDGET}) on {ncores} cores; "
            f"L1 overflow only reproduces on a smaller grid"
        )
    logger.info(f"[lm-head-concat] expecting L1 OOM: {per_bank:.0f} B/bank > {_L1_BANK_BUDGET} on {ncores} cores")
    with expect_error(RuntimeError, "Not enough space|Out of Memory"):
        out = ttnn.concat(tt, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh_device)
        _ = ttnn.to_torch(out)


@pytest.mark.timeout(3600)
def test_lm_head_concat_dram(mesh_device):
    """The fix: the same full-vocab concat to DRAM interleaved fits ANY grid (incl. the 2-node emulator).
    Validates shape, finiteness, and value (PCC vs a torch concat reference)."""
    parts, tt = _build_splits(mesh_device)
    out = ttnn.concat(tt, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)
    ref = torch.cat(parts, dim=-1)
    pcc = _pcc(o, ref)
    logger.info(f"[lm-head-concat] DRAM out {tuple(o.shape)} finite={torch.isfinite(o).all().item()} PCC={pcc:.5f}")
    assert tuple(o.shape) == (1, 1, ROWS, VOCAB), f"unexpected shape {tuple(o.shape)}"
    assert torch.isfinite(o).all(), "DRAM lm_head concat produced non-finite output"
    assert pcc > 0.99, f"DRAM lm_head concat value mismatch vs torch reference: PCC={pcc}"
