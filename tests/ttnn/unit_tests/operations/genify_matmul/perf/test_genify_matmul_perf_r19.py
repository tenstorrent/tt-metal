# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 19: Panoptic-DeepLab 1x1 convs read in place (interleaved A in DRAM or L1, L1 interleaved output) at
PDL's golden config: bfp8 x bfp8 -> bfp8, bfp8 bias, LoFi, fp32 DEST. The ten `placement=interleaved` PDL
LOOSE_CASES of feature_spec.py.

Skipped unless GENIFY_MATMUL_PERF=1. GENIFY_MATMUL_KNOBS="NAME=value,..." overrides program-descriptor knobs
(python literals); GENIFY_MATMUL_ONLY="id1,id2" selects cases by exact id; GENIFY_MATMUL_PLAN=1 prints the
chosen split; GENIFY_MATMUL_CHECK=1 also checks the output against torch (PCC).
"""
import math
import os
import pytest
import torch
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_genify_perf_r15", Path(__file__).with_name("test_genify_matmul_perf_r15.py")
)
_r15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r15)
_apply_knobs = _r15._apply_knobs

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf guard: set GENIFY_MATMUL_PERF=1"),
]

# (id, m, k, n, A dtype, epilogue, A in L1, achievable_ns, roofline_ns)
R19_CASES = [
    ("res3_conv1", 32768, 256, 128, "bfloat8_b", "bias_relu", False, 9360, 19400),
    ("res4_conv1", 8192, 512, 256, "bfloat8_b", "bias_relu", False, 16560, 10000),
    ("res5_conv1", 2048, 1024, 512, "bfloat8_b", "bias_relu", False, 10610, 6100),
    ("res5_short", 2048, 1024, 2048, "bfloat8_b", "bias", False, 72350, 20200),
    ("dec_res2", 32768, 256, 32, "bfloat8_b", "bias_relu", True, 3600, 1300),
    ("dec_res3", 8192, 512, 64, "bfloat8_b", "bias_relu", True, 4650, 1300),
    ("aspp_1x1", 2048, 2048, 256, "bfloat8_b", "bias_relu", True, 13780, 5000),
    ("aspp_proj", 2048, 1280, 256, "bfloat8_b", "bias_relu", True, 16830, 3200),
    ("res3_short", 8192, 256, 512, "bfloat16", "bias", True, 13050, 5000),
    ("res4_short", 2048, 512, 1024, "bfloat16", "bias", True, 16210, 5000),
]


def _force_split(monkeypatch):
    # GENIFY_MATMUL_FORCE="grid_m,grid_n[,placement]": measure one candidate split
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    force = os.environ.get("GENIFY_MATMUL_FORCE")
    if not force:
        return
    fm, fn, *fp = force.split(",")
    orig = pd.choose_split

    def _forced(geom, gx, gy, budget):
        for grid in pd.grid_candidates(geom, gx, gy):
            if (grid.grid_m, grid.grid_n) == (int(fm), int(fn)) and (not fp or grid.placement == fp[0]):
                return grid, pd.choose_blocking(geom, grid, budget)
        raise RuntimeError(f"no candidate {force}")

    monkeypatch.setattr(pd, "choose_split", _forced)
    # keep the PLAN trace (it wraps whatever choose_split is now)
    if os.environ.get("GENIFY_MATMUL_PLAN") == "1":
        inner = pd.choose_split

        def _traced(geom, gx, gy, budget):
            grid, blk = inner(geom, gx, gy, budget)
            print(f"PLAN forced mode={pd.operand_mode(geom, grid, blk)} {grid} {blk}")
            return grid, blk

        monkeypatch.setattr(pd, "choose_split", _traced)


@pytest.mark.parametrize("case", R19_CASES, ids=[c[0] for c in R19_CASES])
def test_perf_r19(case, device, monkeypatch, request):
    _apply_knobs(monkeypatch, request)
    _force_split(monkeypatch)
    _, m, k, n, a_dt, epilogue, a_l1, _, _ = case
    torch.manual_seed(0)
    A = torch.randn(1, 1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    a_mc = ttnn.L1_MEMORY_CONFIG if a_l1 else ttnn.DRAM_MEMORY_CONFIG
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    bias = create_ttnn_input_tensor(torch.randn(1, n), device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    kwargs = {"bias": bias}
    if epilogue == "bias_relu":
        kwargs["activation"] = "relu"
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(
        ta, tb, dtype=ttnn.bfloat8_b, compute_kernel_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG, **kwargs
    )
    ttnn.synchronize_device(device)
    if os.environ.get("GENIFY_MATMUL_CHECK") == "1":
        ref = ttnn.to_torch(ta).float() @ ttnn.to_torch(tb).float() + ttnn.to_torch(bias).float()
        if epilogue == "bias_relu":
            ref = torch.relu(ref)
        got = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        print(f"CHECK {request.node.callspec.id} pcc={pcc:.6f}")
        assert pcc > 0.99, pcc
