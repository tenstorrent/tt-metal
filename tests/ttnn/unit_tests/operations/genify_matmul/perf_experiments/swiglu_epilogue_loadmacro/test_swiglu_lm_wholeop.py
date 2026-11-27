# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op PERF FOCUS 1 with the lm2 swiglu element, without touching the real op: the program descriptor's
module-level KERNEL_DIR is monkeypatched to perf_experiments/swiglu_epilogue_loadmacro/op_kernels (a copy of
the op's kernels whose only change is SwiGluMul -> SwiGluLM<3> + PackTile from its out slot).

    SWG_WHOLE=op,lm2 scripts/run_safe_pytest.sh --run-all --profile <this file>
"""
import math
import os
from pathlib import Path

import pytest
import torch
import ttnn
import ttnn.operations.genify_matmul.genify_matmul_program_descriptor as gpd
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

ROOT = Path(__file__).resolve().parents[7]
LM_DIR = ROOT / "ttnn/ttnn/operations/genify_matmul/perf_experiments/swiglu_epilogue_loadmacro/op_kernels"
ORDER = os.environ.get("SWG_WHOLE", "op,lm2").split(",")


@pytest.mark.parametrize("idx", range(len(ORDER)))
def test_wholeop_focus1(device, idx, monkeypatch):
    variant = ORDER[idx]
    if variant in ("lm2", "lm3"):
        monkeypatch.setattr(gpd, "KERNEL_DIR", LM_DIR)
    elif variant in ("nw_base", "nw_lm3"):  # diagnostic: output DRAM writes stubbed in the kernel copy
        monkeypatch.setattr(gpd, "KERNEL_DIR", LM_DIR.parent / f"op_kernels_{variant}")
    m, k, n = 2048, 5120, 8704
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, activation="swiglu")
    ttnn.synchronize_device(device)
    Bq = ttnn.to_torch(tb).float()
    Aq = ttnn.to_torch(ta).float()
    y = Aq.double() @ Bq.double()
    g = y.reshape(1, m, n // 64, 2, 32)
    ref = (torch.nn.functional.silu(g[..., 0, :]) * g[..., 1, :]).reshape(1, m, n // 2)
    got = ttnn.to_torch(out).double()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    print(f"SWG_WHOLE {idx} {variant} pcc {pcc:.6f} rel_rms {rms:.5f}")
    if not variant.startswith("nw_"):
        assert pcc > 0.9995
