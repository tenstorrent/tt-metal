"""
Exhaustive config sweep for the ttnn.matmul precision bug on Wormhole.

Tests every combination of math_fidelity x fp32_dest_acc_en, plus
the "no config" default path, to determine if any matmul configuration
produces exact results for: flat_index = row_idx * 1280 + col_idx.

The matmul docstring (matmul_nanobind.cpp) says:
  "Due to a hardware bug on Wormhole (fixed on Blackhole), when
   fp32_acc_to_dest is enabled, output values can rarely be off by
   a negative power of two (e.g. -128). This bug happens most
   frequently at HiFi4, and in decreasing frequency as math fidelity
   is reduced (e.g. HiFi3, HiFi2, LoFi). If affected, consider either
   disabling fp32_acc_to_dest, or decreasing fidelity."

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_matmul_bug_sweep.py
"""

import pytest
import torch
import ttnn

from gather_deepseek_ocr_codegen import utils
from models.common.utility_functions import comp_pcc

S = 913
D = 1280
N = 903
DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

FIDELITIES = [
    ttnn.MathFidelity.HiFi4,
    ttnn.MathFidelity.HiFi3,
    ttnn.MathFidelity.HiFi2,
    ttnn.MathFidelity.LoFi,
]

FIDELITY_NAMES = {
    ttnn.MathFidelity.HiFi4: "HiFi4",
    ttnn.MathFidelity.HiFi3: "HiFi3",
    ttnn.MathFidelity.HiFi2: "HiFi2",
    ttnn.MathFidelity.LoFi: "LoFi",
}


def _build_matmul_inputs(seed=42):
    torch.manual_seed(seed)
    row_indices = torch.randint(0, N, (S,), dtype=torch.int32)
    col_indices = torch.arange(D, dtype=torch.int32)
    row_3d = row_indices.reshape(S, 1, 1).expand(S, D, 1)
    col_3d = col_indices.reshape(1, D, 1).expand(S, D, 1)
    lhs = torch.cat([row_3d, col_3d], dim=2).float()
    rhs = torch.tensor([[float(D)], [1.0]], dtype=torch.float32)
    cpu_result = torch.matmul(lhs, rhs)
    return lhs, rhs, cpu_result


def _to_tt(tensor, dtype, device, layout=ttnn.Layout.ROW_MAJOR):
    t = ttnn.from_torch(tensor, dtype=dtype, layout=layout)
    return ttnn.to_device(t, device, memory_config=DRAM)


def _to_torch(tt_tensor):
    return ttnn.to_torch(ttnn.from_device(tt_tensor))


def _run_matmul(lhs, rhs, device, compute_kernel_config=None):
    tt_lhs = _to_tt(lhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_rhs = _to_tt(rhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)

    kwargs = dict(
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
    )
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    return _to_torch(ttnn.matmul(tt_lhs, tt_rhs, **kwargs))


def _report(tt_torch, cpu_torch, label):
    a = tt_torch.float().flatten()
    b = cpu_torch.float().flatten()
    diff = (a - b).abs()
    mismatches = torch.sum(diff > 0).item()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    _, pcc_msg = comp_pcc(cpu_torch, tt_torch, 0.0)
    passed_099, _ = comp_pcc(cpu_torch, tt_torch, 0.99)
    print(f"\n  {label}:")
    print(f"    PCC:        {pcc_msg}")
    print(f"    mismatches: {mismatches}/{a.numel()} ({100*mismatches/a.numel():.1f}%)")
    print(f"    max_diff:   {max_d}")
    print(f"    mean_diff:  {mean_d:.4f}")
    print(f"    pcc>=0.99:  {'PASS' if passed_099 else 'FAIL'}")
    return passed_099, pcc_msg


@pytest.fixture(scope="module")
def device():
    return utils.DeviceGetter.get_device((1, 1))


# ---- Parametrized sweep: fidelity x fp32_dest_acc_en ----

@pytest.mark.parametrize("fidelity", FIDELITIES, ids=lambda f: FIDELITY_NAMES[f])
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32acc_ON", "fp32acc_OFF"])
def test_matmul_config_sweep(device, fidelity, fp32_acc):
    lhs, rhs, cpu_out = _build_matmul_inputs()
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc
    )
    label = f"{FIDELITY_NAMES[fidelity]} fp32_acc={fp32_acc}"
    result = _run_matmul(lhs, rhs, device, compute_kernel_config=config)
    passed, msg = _report(result, cpu_out, label)
    assert passed, f"{label}: {msg}"


# ---- No config (use matmul defaults) ----

def test_matmul_no_config(device):
    """No compute_kernel_config: defaults to HiFi3+fp32_acc on Wormhole for F32 inputs."""
    lhs, rhs, cpu_out = _build_matmul_inputs()
    result = _run_matmul(lhs, rhs, device, compute_kernel_config=None)
    passed, msg = _report(result, cpu_out, "no_config (default)")
    assert passed, f"no_config: {msg}"
