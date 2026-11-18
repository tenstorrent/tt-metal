# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 15: small-M weight streaming at the DRAM roofline. Every case of the refinement at its model's
own golden config (Qwen3.8 prefill M = 128: bf16 x bfp4, LoFi, A in L1 for the GDN layers; FLUX.2: bf16 x bf16,
HiFi2; fp32 DEST, DRAM interleaved, output in DRAM).

Skipped unless GENIFY_MATMUL_PERF=1. GENIFY_MATMUL_KNOBS="NAME=value,..." overrides program-descriptor knobs
(python literals); GENIFY_MATMUL_ONLY="id1,id2" selects cases by exact id; GENIFY_MATMUL_PLAN=1 prints the
chosen split; GENIFY_MATMUL_CHECK=1 also checks the output against torch (PCC).
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

# (id, m, k, n, A dtype, B dtype, out dtype, fidelity, epilogue, A in L1, roofline_ns)
R15_CASES = [
    ("q128_gu", 128, 5120, 8704, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "swiglu", True, 56800),
    ("q128_in", 128, 5120, 4160, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none", True, 28300),
    ("q128_down", 128, 4352, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none", True, 30000),
    ("q128_out", 128, 1536, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none", True, 12400),
    ("q128_qkvg", 128, 5120, 3584, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none", False, 27200),
    ("f256_qkv", 256, 6144, 9216, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none", False, 262800),
    ("f256_attn", 256, 6144, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "addcmul", False, 96000),
    ("f256_ff1", 256, 6144, 18432, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "swiglu", False, 508600),
    ("f256_ff2", 256, 9216, 6144, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none", False, 262800),
    ("f256_ctx", 256, 15360, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none", False, 225300),
    ("f1_lin2_fp32", 1, 6144, 6144, "float32", "bfloat16", "float32", "HiFi2", "none", False, 167300),
    ("f1_lin2", 1, 6144, 6144, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none", False, 165500),
    ("f1_tmod", 1, 6144, 52224, "float32", "bfloat16", "bfloat16", "HiFi2", "bias", False, 1409000),
    ("f1_lin1", 1, 256, 6144, "float32", "bfloat16", "float32", "HiFi2", "none", False, 7700),
    ("x_embedder", 2048, 128, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none", False, 30200),
    ("proj_out", 2048, 3072, 128, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none", False, 30200),
    # Guard: a mid-M shape the bank split competes for (test_genify_matmul_mcast's text-encoder qkv in bf16).
    ("te_qkv_bf16_hifi4", 512, 5120, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi4", "none", False, 0),
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
    if os.environ.get("GENIFY_MATMUL_PLAN") == "1":
        orig = pd.choose_split

        def _traced(geom, gx, gy, budget):
            grid, blk = orig(geom, gx, gy, budget)
            terms = [round(t / 1350, 1) for t in pd.split_cost_terms(geom, grid, blk)]
            print(f"PLAN {request.node.callspec.id} mode={pd.operand_mode(geom, grid, blk)} us={terms} {grid} {blk}")
            return grid, blk

        monkeypatch.setattr(pd, "choose_split", _traced)


@pytest.mark.parametrize("case", R15_CASES, ids=[c[0] for c in R15_CASES])
def test_perf_r15(case, device, monkeypatch, request):
    _apply_knobs(monkeypatch, request)
    _, m, k, n, a_dt, w_dt, o_dt, fid, epilogue, a_l1, _ = case
    torch.manual_seed(0)
    A = torch.randn((1, k) if m == 1 else (1, m, k))
    B = torch.randn(k, n) / math.sqrt(k)
    a_mc = ttnn.L1_MEMORY_CONFIG if a_l1 else ttnn.DRAM_MEMORY_CONFIG
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    o_dtype = getattr(ttnn, o_dt)
    n_out = n // 2 if epilogue == "swiglu" else n
    kwargs = {}
    if epilogue == "swiglu":
        kwargs["activation"] = "swiglu"
    elif epilogue == "bias":
        kwargs["bias"] = create_ttnn_input_tensor(
            torch.randn(1, n), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
    elif epilogue == "addcmul":
        R = torch.randn(*A.shape[:-1], n_out)
        G = torch.randn(1, n_out)
        kwargs["residual"] = create_ttnn_input_tensor(R, device, dtype=o_dtype, layout=ttnn.TILE_LAYOUT)
        kwargs["gate"] = create_ttnn_input_tensor(G, device, dtype=o_dtype, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(
        ta, tb, dtype=o_dtype, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs
    )
    ttnn.synchronize_device(device)
    if os.environ.get("GENIFY_MATMUL_CHECK") == "1":
        ref = ttnn.to_torch(ta).float() @ ttnn.to_torch(tb).float()
        if epilogue == "swiglu":
            t = ref.reshape(*ref.shape[:-1], n // 64, 2, 32)
            ref = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(*ref.shape[:-1], n // 2)
        elif epilogue == "bias":
            ref = ref + ttnn.to_torch(kwargs["bias"]).float()
        elif epilogue == "addcmul":
            ref = ttnn.to_torch(kwargs["residual"]).float() + ttnn.to_torch(kwargs["gate"]).float() * ref
        got = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"CHECK {request.node.callspec.id} pcc={pcc:.6f} rel_rms={rms:.5f}")
        assert pcc > 0.999, pcc
