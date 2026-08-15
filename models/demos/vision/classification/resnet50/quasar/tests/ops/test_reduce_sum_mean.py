# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal, standalone isolation test for the generic Quasar reduction (ttnn.sum / ttnn.mean),
decoupled from the avg_pool2d pipeline (no tilize_with_val_padding / halo / untilize around it).

This directly exercises the FPU GAPOOL reduce that avg_pool2d's global fast path uses internally:
reducing dim=2 (H) of a [1, 1, H, C] tile tensor is the same ReduceOpDim::H / REDUCE_COL GAPOOL.

  - ttnn.sum  -> reduce with scaler = 1.0 (bf16-exact) -> should be exact.
  - ttnn.mean -> reduce with scaler = 1/H (fractional)  -> exercises the scaler path.

Known Quasar bug (documented, LLK/HW ticket): the Quasar GAPOOL SUM/AVG reduce applies a FIXED
~1.1504x multiplicative gain that WH/BH do not (independent of the scaler value and of fidelity).
So on the Quasar emulator we EXPECT `mean` (and any scaled reduce) to fail PCC by ~1.15x until the
GAPOOL fix lands; `sum` isolates whether the base (scaler=1.0) reduce is clean. The per-run
[REDUCE-DIAG] line prints dev/golden so the gain is visible directly.

Run (emulator / WH, slow dispatch + forced JIT):
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_reduce_sum_mean.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99

# (H, C, id) -- all tile-aligned. H is the reduced dim; C is kept.
_SHAPES = [
    (32, 64, "H32_C64"),
    (64, 64, "H64_C64"),  # 2 H-tiles (the avgpool 49->64 case shape)
    (96, 128, "H96_C128"),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("H,C,sid", _SHAPES, ids=[s[2] for s in _SHAPES])
@pytest.mark.parametrize("op", ["sum", "mean"], ids=["sum", "mean"])
def test_quasar_reduce_h(mesh_device, H, C, sid, op):
    device = mesh_device
    torch.manual_seed(0)

    x = torch.rand((1, 1, H, C), dtype=torch.float32)
    if op == "sum":
        golden = torch.sum(x, dim=2, keepdim=True)  # [1, 1, 1, C]
    else:
        golden = torch.mean(x, dim=2, keepdim=True)  # [1, 1, 1, C]

    xt = ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    if op == "sum":
        out = ttnn.sum(xt, dim=2, keepdim=True)
    else:
        out = ttnn.mean(xt, dim=2, keepdim=True)
    ttnn.synchronize_device(device)

    tt = ttnn.to_torch(ttnn.from_device(out)).float().reshape(golden.shape)

    g = golden.reshape(-1)
    d = tt.reshape(-1)
    # ratio dev/golden: exposes a fixed multiplicative gain (e.g. the ~1.1504x Quasar GAPOOL bug).
    ratio = (d / g.clamp_min(1e-6)).mean().item()
    print(
        f"[REDUCE-DIAG {op} {sid}] out={tuple(tt.shape)} "
        f"golden[min,max]=[{float(g.min()):.4f},{float(g.max()):.4f}] "
        f"dev[min,max]=[{float(d.min()):.4f},{float(d.max()):.4f}] dev/golden={ratio:.4f}"
    )

    assert_with_pcc(golden, tt, pcc=PCC)


# Also cover the W-dim reduce (dim=3) so both COL and ROW reduce paths are exercised.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("H,C,sid", _SHAPES, ids=[s[2] for s in _SHAPES])
@pytest.mark.parametrize("op", ["sum", "mean"], ids=["sum", "mean"])
def test_quasar_reduce_w(mesh_device, H, C, sid, op):
    device = mesh_device
    torch.manual_seed(0)

    # reduce the last (C) dim here; keep H.
    x = torch.rand((1, 1, H, C), dtype=torch.float32)
    if op == "sum":
        golden = torch.sum(x, dim=3, keepdim=True)  # [1, 1, H, 1]
    else:
        golden = torch.mean(x, dim=3, keepdim=True)

    xt = ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    if op == "sum":
        out = ttnn.sum(xt, dim=3, keepdim=True)
    else:
        out = ttnn.mean(xt, dim=3, keepdim=True)
    ttnn.synchronize_device(device)

    tt = ttnn.to_torch(ttnn.from_device(out)).float().reshape(golden.shape)
    g = golden.reshape(-1)
    d = tt.reshape(-1)
    ratio = (d / g.clamp_min(1e-6)).mean().item()
    print(
        f"[REDUCE-DIAG {op}-W {sid}] out={tuple(tt.shape)} "
        f"golden[min,max]=[{float(g.min()):.4f},{float(g.max()):.4f}] "
        f"dev[min,max]=[{float(d.min()):.4f},{float(d.max()):.4f}] dev/golden={ratio:.4f}"
    )
    assert_with_pcc(golden, tt, pcc=PCC)
