"""
Element-wise fix for the ttnn.matmul precision bug on Wormhole.

Replaces the matmul-based index linearization:
    matmul([S,D,2], [[1280],[1]])  ->  row_idx * 1280 + col_idx

with element-wise ops that use the SFPU pipeline (not the matrix engine):
    ttnn.add(ttnn.multiply(row_idx, 1280.0), col_idx)

This avoids the Wormhole hardware bug entirely.

Three tests mirror test_matmul_bug.py to prove the fix:
  1. test_eltwise_only           — multiply+add (replaces Op6)
  2. test_eltwise_then_flatten   — + reshape/typecast/to_layout (Op6->Op7)
  3. test_eltwise_then_flatten_then_embedding — full pipeline (Op6->Op7->Op8)

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_matmul_bug_eltwise_fix.py
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


def _build_inputs(seed=42):
    """Build row_idx [S,D] and col_idx [S,D] as float32, plus CPU golden."""
    torch.manual_seed(seed)
    row_indices = torch.randint(0, N, (S,), dtype=torch.int32)
    col_indices = torch.arange(D, dtype=torch.int32)

    row_2d = row_indices.reshape(S, 1).expand(S, D).float()
    col_2d = col_indices.reshape(1, D).expand(S, D).float()

    cpu_result = (row_2d * D + col_2d).unsqueeze(-1)  # [S, D, 1] to match matmul shape
    return row_2d, col_2d, cpu_result


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


def _eltwise_linearize(tt_row, tt_col, device):
    """Compute flat_index = row_idx * D + col_idx using element-wise ops."""
    tt_product = ttnn.multiply(tt_row, float(D))
    tt_flat = ttnn.add(tt_product, tt_col)
    ttnn.deallocate(tt_product, False)
    return tt_flat


@pytest.fixture(scope="module")
def device():
    return utils.DeviceGetter.get_device((1, 1))


# ---------------------------------------------------------------------------
# Test 1: Element-wise only (replaces Op6 matmul)
# ---------------------------------------------------------------------------

def test_eltwise_only(device):
    """ttnn.multiply + ttnn.add: row_idx * 1280 + col_idx via SFPU."""
    row_2d, col_2d, cpu_out = _build_inputs()

    tt_row = _to_tt(row_2d, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_col = _to_tt(col_2d, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)

    tt_flat = _eltwise_linearize(tt_row, tt_col, device)
    result = _to_torch(tt_flat).unsqueeze(-1)  # [S,D] -> [S,D,1]

    passed, msg = _report(result, cpu_out, "Eltwise: multiply+add only")
    assert passed, msg


# ---------------------------------------------------------------------------
# Test 2: Element-wise + flatten (Op6 -> Op7)
# ---------------------------------------------------------------------------

def test_eltwise_then_flatten(device):
    """Eltwise -> reshape + typecast(UINT32) + to_layout(ROW_MAJOR)."""
    row_2d, col_2d, cpu_matmul = _build_inputs()
    cpu_flat = cpu_matmul.reshape(1, S * D).to(torch.int32)

    tt_row = _to_tt(row_2d, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_col = _to_tt(col_2d, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)

    tt = _eltwise_linearize(tt_row, tt_col, device)
    tt = ttnn.reshape(tt, [1, S * D], memory_config=DRAM)
    tt = ttnn.typecast(tt, ttnn.DataType.UINT32, memory_config=DRAM)
    tt = ttnn.to_layout(tt, ttnn.Layout.ROW_MAJOR, None, memory_config=None)

    result = _to_torch(tt)
    passed, msg = _report(result, cpu_flat, "Eltwise->flatten (Op6->Op7)")
    assert passed, msg


# ---------------------------------------------------------------------------
# Test 3: Element-wise + flatten + embedding (Op6 -> Op7 -> Op8)
# ---------------------------------------------------------------------------

def test_eltwise_then_flatten_then_embedding(device):
    """Eltwise -> flatten -> embedding: full pipeline with SFPU fix."""
    row_2d, col_2d, cpu_matmul = _build_inputs()
    cpu_flat = cpu_matmul.reshape(1, S * D).to(torch.int32)
    src_flat = _build_source()

    cpu_embed = F.embedding(
        cpu_flat.long().reshape(-1), src_flat.reshape(-1, 1)
    )
    cpu_out = cpu_embed.reshape(S, D)

    tt_row = _to_tt(row_2d, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_col = _to_tt(col_2d, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)

    tt = _eltwise_linearize(tt_row, tt_col, device)
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
    passed, msg = _report(result, cpu_out, "Eltwise->flatten->embedding (full pipeline)")
    assert passed, msg
