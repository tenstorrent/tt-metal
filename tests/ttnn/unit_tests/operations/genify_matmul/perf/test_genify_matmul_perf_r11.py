# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 11: the interleaved model cases still slower than stock (narrow N, mid M), at each model's own
config (FLUX.2 / text encoder: bf16 A, HiFi2; Qwen3.8 prefill: bf16 x bfp4, LoFi), fp32 DEST, DRAM interleaved.

Skipped unless GENIFY_MATMUL_PERF=1. GENIFY_MATMUL_KNOBS="NAME=value,..." overrides program-descriptor knobs
(python literals); GENIFY_MATMUL_ONLY="sub1,sub2" selects cases by id substring. GENIFY_MATMUL_CHECK=1 also
checks the output against torch (PCC).
"""
import ast
import math
import os
import pytest
import torch
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf guard: set GENIFY_MATMUL_PERF=1"),
]

# (name, m, k, n, B dtype, fidelity, epilogue, stock achievable_ns @ 1350 MHz)
R11_CASES = [
    ("proj_out", 2048, 3072, 128, "bfloat16", "HiFi2", "none", 60950),
    ("te_qkv", 512, 5120, 3072, "bfloat8_b", "HiFi2", "none", 139200),
    ("te_o", 512, 4096, 2560, "bfloat8_b", "HiFi2", "none", 79430),
    ("te_down", 512, 16384, 5120, "bfloat8_b", "HiFi2", "none", 508620),
    ("te_gate", 512, 5120, 16384, "bfloat8_b", "HiFi2", "none", 501000),
    ("s_qkv", 2304, 6144, 9216, "bfloat16", "HiFi2", "none", 1244630),
    ("s_out", 2304, 12288, 6144, "bfloat16", "HiFi2", "none", 1688530),
    ("s_ff1", 2304, 6144, 18432, "bfloat16", "HiFi2", "swiglu", 2398220),
    ("q4096_qkvg", 4096, 5120, 3584, "bfloat4_b", "LoFi", "none", 371880),
    ("q1024_in", 1024, 5120, 4160, "bfloat4_b", "LoFi", "none", 139020),
    ("q512_in", 512, 5120, 4160, "bfloat4_b", "LoFi", "none", 108750),
    ("q512_out", 512, 1536, 5120, "bfloat4_b", "LoFi", "none", 52140),
    ("q512_gu", 512, 5120, 8704, "bfloat4_b", "LoFi", "swiglu", 214380),
    ("q512_down", 512, 4352, 5120, "bfloat4_b", "LoFi", "none", 111660),
    ("q512_qkvg", 512, 5120, 3584, "bfloat4_b", "LoFi", "none", 101270),
]


def _apply_knobs(monkeypatch, request):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    only = [t for t in os.environ.get("GENIFY_MATMUL_ONLY", "").split(",") if t]
    if only and not any(t == request.node.callspec.id for t in only):
        pytest.skip("not selected by GENIFY_MATMUL_ONLY")
    spec = os.environ.get("GENIFY_MATMUL_KNOBS", "")
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, value = item.split("=", 1)
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, ast.literal_eval(value))
    force = os.environ.get("GENIFY_MATMUL_FORCE")  # "grid_m,grid_n[,placement]": measure one candidate split
    if force:
        fm, fn, *fp = force.split(",")

        def _forced(geom, gx, gy, budget):
            for grid in pd.grid_candidates(geom, gx, gy):
                if (grid.grid_m, grid.grid_n) == (int(fm), int(fn)) and (not fp or grid.placement == fp[0]):
                    return grid, pd.choose_blocking(geom, grid, budget)
            raise RuntimeError(f"no candidate {force}")

        monkeypatch.setattr(pd, "choose_split", _forced)
    if os.environ.get("GENIFY_MATMUL_PLAN") == "1":
        orig = pd.choose_split

        def _traced(geom, gx, gy, budget):
            grid, blk = orig(geom, gx, gy, budget)
            print(f"PLAN {request.node.callspec.id} mode={pd.operand_mode(geom, grid, blk)} {grid} {blk}")
            return grid, blk

        monkeypatch.setattr(pd, "choose_split", _traced)


@pytest.mark.parametrize("case", R11_CASES, ids=[c[0] for c in R11_CASES])
def test_perf_r11(case, device, monkeypatch, request):
    _apply_knobs(monkeypatch, request)
    _, m, k, n, w_dt, fid, epilogue, _ = case
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    kwargs = {"activation": "swiglu"} if epilogue == "swiglu" else {}
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, **kwargs)
    ttnn.synchronize_device(device)
    if os.environ.get("GENIFY_MATMUL_CHECK") == "1":
        ref = A.float() @ ttnn.to_torch(tb).float()  # the quantized (bfp8 / bfp4) weight the op saw
        if epilogue == "swiglu":
            t = ref.reshape(1, m, n // 64, 2, 32)
            ref = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(1, m, n // 2)
        got = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"CHECK {request.node.callspec.id} pcc={pcc:.6f} rel_rms={rms:.5f}")
        assert pcc > 0.999, pcc
