"""
Same tests as test_matmul_bug.py but with the Wormhole fix applied:
  HiFi4 + fp32_dest_acc_en=True  ->  HiFi3 + fp32_dest_acc_en=True

The warning in compute_kernel_config.cpp says:
  "On Wormhole with fp32 accumulation, output accuracy can be worse
   with HiFi4 than HiFi3 due to a hardware bug.
   Prefer using HiFi3 with fp32 accumulation on Wormhole."

If HiFi3 fixes the matmul precision, all three tests should pass
with 0 diff and PCC >= 0.99.

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_matmul_bug_fix.py
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from gather_deepseek_ocr_codegen import utils
from models.common.utility_functions import comp_pcc

S = 913
D = 1280
N = 903
DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

MATMUL_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True
)


# ---------------------------------------------------------------------------
# Input generation (identical to test_matmul_bug.py)
# ---------------------------------------------------------------------------

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


def _build_source(seed=42):
    torch.manual_seed(seed)
    return torch.randn(N * D, 1, dtype=torch.bfloat16)


def _to_tt(tensor, dtype, device, layout=ttnn.Layout.ROW_MAJOR):
    if dtype == ttnn.DataType.UINT32:
        t = ttnn.from_torch(
            tensor.to(torch.int32), dtype=ttnn.DataType.INT32, layout=layout
        )
        t = ttnn.to_device(t, device, memory_config=DRAM)
        return ttnn.typecast(t, ttnn.DataType.UINT32, memory_config=DRAM)
    t = ttnn.from_torch(tensor, dtype=dtype, layout=layout)
    return ttnn.to_device(t, device, memory_config=DRAM)


def _to_torch(tt_tensor):
    return ttnn.to_torch(ttnn.from_device(tt_tensor))


def _report(tt_torch, cpu_torch, label):
    a = tt_torch.float().flatten()
    b = cpu_torch.float().flatten()
    diff = (a - b).abs()
    mismatches = torch.sum(diff > 0).item()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    _, pcc_msg = comp_pcc(cpu_torch, tt_torch, 0.0)
    passed_099, _ = comp_pcc(cpu_torch, tt_torch, 0.99)
    print(f"  {label}:")
    print(f"    PCC:        {pcc_msg}")
    print(f"    mismatches: {mismatches}/{a.numel()} ({100*mismatches/a.numel():.1f}%)")
    print(f"    max_diff:   {max_d}")
    print(f"    mean_diff:  {mean_d:.4f}")
    print(f"    pcc>=0.99:  {'PASS' if passed_099 else 'FAIL'}")
    return passed_099, pcc_msg


@pytest.fixture(scope="module")
def device():
    return utils.DeviceGetter.get_device((1, 1))


# ---------------------------------------------------------------------------
# Test 1: Matmul only with HiFi3 fix
# ---------------------------------------------------------------------------

def test_matmul_only_hifi3(device):
    """ttnn.matmul [913,1280,2] @ [2,1] with HiFi3+fp32_acc (fix)."""
    lhs, rhs, cpu_out = _build_matmul_inputs()

    tt_lhs = _to_tt(lhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_rhs = _to_tt(rhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_out = ttnn.matmul(
        tt_lhs, tt_rhs,
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
        compute_kernel_config=MATMUL_KERNEL_CONFIG,
    )
    result = _to_torch(tt_out)

    passed, msg = _report(result, cpu_out, "Op6 matmul only (HiFi3 fix)")
    assert passed, msg


# ---------------------------------------------------------------------------
# Test 2: Matmul + flatten with HiFi3 fix
# ---------------------------------------------------------------------------

def test_matmul_then_flatten_hifi3(device):
    """Op6 -> Op7 with HiFi3+fp32_acc (fix)."""
    lhs, rhs, cpu_matmul = _build_matmul_inputs()
    cpu_flat = cpu_matmul.reshape(1, S * D).to(torch.int32)

    tt_lhs = _to_tt(lhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_rhs = _to_tt(rhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)

    tt = ttnn.matmul(
        tt_lhs, tt_rhs,
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
        compute_kernel_config=MATMUL_KERNEL_CONFIG,
    )
    tt = ttnn.reshape(tt, [1, S * D], memory_config=DRAM)
    tt = ttnn.typecast(tt, ttnn.DataType.UINT32, memory_config=DRAM)
    tt = ttnn.to_layout(tt, ttnn.Layout.ROW_MAJOR, None, memory_config=None)

    result = _to_torch(tt)
    passed, msg = _report(result, cpu_flat, "Op6->Op7 matmul+flatten (HiFi3 fix)")
    assert passed, msg


# ---------------------------------------------------------------------------
# Test 3: Matmul + flatten + embedding with HiFi3 fix
# ---------------------------------------------------------------------------

def test_matmul_then_flatten_then_embedding_hifi3(device):
    """Op6 -> Op7 -> Op8 with HiFi3+fp32_acc (fix)."""
    lhs, rhs, cpu_matmul = _build_matmul_inputs()
    cpu_flat = cpu_matmul.reshape(1, S * D).to(torch.int32)
    src_flat = _build_source()

    cpu_embed = F.embedding(
        cpu_flat.long().reshape(-1), src_flat.reshape(-1, 1)
    )
    cpu_out = cpu_embed.reshape(S, D)

    tt_lhs = _to_tt(lhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_rhs = _to_tt(rhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)

    tt = ttnn.matmul(
        tt_lhs, tt_rhs,
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
        compute_kernel_config=MATMUL_KERNEL_CONFIG,
    )
    tt = ttnn.reshape(tt, [1, S * D], memory_config=DRAM)
    tt = ttnn.typecast(tt, ttnn.DataType.UINT32, memory_config=DRAM)
    tt = ttnn.to_layout(tt, ttnn.Layout.ROW_MAJOR, None, memory_config=None)

    tt_src = _to_tt(src_flat, ttnn.DataType.BFLOAT16, device, layout=ttnn.Layout.ROW_MAJOR)

    tt = ttnn.embedding(
        tt, tt_src,
        padding_idx=None, layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM,
    )
    tt = ttnn.reshape(tt, [S, D], memory_config=DRAM)

    result = _to_torch(tt)
    passed, msg = _report(result, cpu_out, "Op6->Op7->Op8 full pipeline (HiFi3 fix)")
    assert passed, msg
