"""
Standalone reproduction of ttnn.matmul precision bug on Wormhole.

No model or gather dependencies — just the matmul that computes
flat_index = row_idx * 1280 + col_idx, and its downstream effects.

Three tests that prove Op6 (matmul) is the sole root cause:

  1. test_matmul_only (Op6)
     Matmul alone: diff is massive (94.6% wrong, max_diff=258),
     but PCC stays high (~1.0) because values are linearly correlated.

  2. test_matmul_then_flatten (Op6 -> Op7)
     Matmul + reshape/typecast/to_layout: diff carries through unchanged,
     PCC still high. Op7 just passes through Op6's errors.

  3. test_matmul_then_flatten_then_embedding (Op6 -> Op7 -> Op8)
     Matmul + flatten + embedding: PCC crashes to ~0.05.
     The corrupted flat indices cause embedding to grab wrong values.

Combined with the per-op isolation tests (test_per_op_pcc.py) that show
Op7 and Op8 pass individually with correct inputs, this proves the matmul
is the sole source of the end-to-end PCC drop.

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_matmul_bug.py
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


# ---------------------------------------------------------------------------
# Input generation — no model dependencies, just integer-valued tensors
# ---------------------------------------------------------------------------

def _build_matmul_inputs(seed=42):
    """Build the exact inputs to the matmul that linearizes 2D indices.

    Returns:
        lhs:        [913, 1280, 2] float32  — each element is [row_idx, col_idx]
        rhs:        [2, 1]         float32  — weights [1280.0, 1.0]
        cpu_result: [913, 1280, 1] float32  — row_idx * 1280 + col_idx
    """
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
    """Build the flattened source for embedding lookup."""
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
    """Print PCC and diff stats. Returns (pcc_passed_099, pcc_value_str)."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return utils.DeviceGetter.get_device((1, 1))


# ---------------------------------------------------------------------------
# Test 1: Matmul only (Op6)
# ---------------------------------------------------------------------------

def test_matmul_only(device):
    """ttnn.matmul [913,1280,2] @ [2,1] with HiFi4+fp32_acc on Wormhole.

    Computes flat_index = row_idx * 1280 + col_idx.
    All values are exact integers within float32 range (max ~1.15M < 2^24).
    Any diff is a hardware precision bug.
    """
    lhs, rhs, cpu_out = _build_matmul_inputs()

    tt_lhs = _to_tt(lhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_rhs = _to_tt(rhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_out = ttnn.matmul(
        tt_lhs, tt_rhs,
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    result = _to_torch(tt_out)

    passed, msg = _report(result, cpu_out, "Op6 matmul only")
    assert passed, msg


# ---------------------------------------------------------------------------
# Test 2: Matmul + flatten (Op6 -> Op7)
# ---------------------------------------------------------------------------

def test_matmul_then_flatten(device):
    """Op6 -> Op7: matmul then reshape+typecast(UINT32)+to_layout(ROW_MAJOR).

    The corrupted float indices from matmul get cast to uint32.
    Diff carries through unchanged. PCC still high (errors are correlated).
    """
    lhs, rhs, cpu_matmul = _build_matmul_inputs()
    cpu_flat = cpu_matmul.reshape(1, S * D).to(torch.int32)

    tt_lhs = _to_tt(lhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_rhs = _to_tt(rhs, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)

    tt = ttnn.matmul(
        tt_lhs, tt_rhs,
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    tt = ttnn.reshape(tt, [1, S * D], memory_config=DRAM)
    tt = ttnn.typecast(tt, ttnn.DataType.UINT32, memory_config=DRAM)
    tt = ttnn.to_layout(tt, ttnn.Layout.ROW_MAJOR, None, memory_config=None)

    result = _to_torch(tt)
    passed, msg = _report(result, cpu_flat, "Op6->Op7 matmul+flatten")
    assert passed, msg


# ---------------------------------------------------------------------------
# Test 3: Matmul + flatten + embedding (Op6 -> Op7 -> Op8)
# ---------------------------------------------------------------------------

def test_matmul_then_flatten_then_embedding(device):
    """Op6 -> Op7 -> Op8: full index-linearize-then-lookup pipeline.

    The corrupted flat indices from matmul cause embedding to grab wrong
    values from the source. PCC crashes to ~0.05.
    """
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
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
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
    passed, msg = _report(result, cpu_out, "Op6->Op7->Op8 matmul+flatten+embedding")
    assert passed, msg
