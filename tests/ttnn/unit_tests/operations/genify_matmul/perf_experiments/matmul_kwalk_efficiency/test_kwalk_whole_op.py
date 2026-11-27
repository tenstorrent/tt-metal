# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op A/B of the kw K-walk on PERF FOCUS 1 (1x2048x5120 bf16 @ 5120x8704 bf4b, LoFi, fp32 DEST,
swiglu, DRAM interleaved). `base` runs genify_matmul as-is; `kw` monkeypatches the op's descriptor
builder to swap ONLY the compute kernel source for perf_experiments/matmul_kwalk_efficiency/kernels/
op_compute_kw.cpp (the op's compute with the K-walk replaced). Both are checked vs torch.

    scripts/run_safe_pytest.sh <this file>
Env: KWO_ORDER (default "base,kw"), KWO_FIDELITY (default LoFi).
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import math
from pathlib import Path

import torch
import ttnn
from loguru import logger

import importlib

from ttnn.operations.genify_matmul import genify_matmul

gm_mod = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

ROOT = Path(__file__).resolve().parents[7]
KW_COMPUTE = (
    ROOT / "ttnn/ttnn/operations/genify_matmul/perf_experiments/matmul_kwalk_efficiency/kernels/op_compute_kw.cpp"
)
_ORIG = gm_mod.create_program_descriptor


def _patched(*args, **kwargs):
    d = _ORIG(*args, **kwargs)
    n = 0
    for k in d.kernels:
        if str(k.kernel_source).endswith("genify_matmul_compute.cpp"):
            k.kernel_source = str(KW_COMPUTE)
            n += 1
    assert n == 1, n
    return d


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    vals = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            e = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                vals.append(float(e.duration))
    return vals


def test_kwalk_whole_op(device):
    m, k, n = 2048, 5120, 8704
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("KWO_FIDELITY", "LoFi")),
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    Aq, Bq = ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float()
    y = Aq @ Bq
    g = y.reshape(1, m, n // 64, 2, 32)
    ref = (torch.nn.functional.silu(g[..., 0, :]) * g[..., 1, :]).reshape(1, m, n // 2)
    _read_kernel_ns(device)
    res, outs = {}, {}
    for v in os.environ.get("KWO_ORDER", "base,kw").split(","):
        gm_mod.create_program_descriptor = _patched if v == "kw" else _ORIG
        try:
            out = genify_matmul(ta, tb, compute_kernel_config=cfg, activation="swiglu")
            ttnn.synchronize_device(device)
        finally:
            gm_mod.create_program_descriptor = _ORIG
        ns = _read_kernel_ns(device)
        got = ttnn.to_torch(out).float()
        outs[v] = got
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        res[v] = (ns, pcc, rms)
        assert pcc > 0.9995, (v, pcc)
    lines = [f"KWO {v:<5} ns {r[0]}  pcc {r[1]:.6f}  rel_rms {r[2]:.5f}" for v, r in res.items()]
    if "base" in outs and "kw" in outs:
        lines.append(f"KWO max|kw - base| = {(outs['kw'] - outs['base']).abs().max().item():.3e}")
    logger.info("\n" + "\n".join(lines))
