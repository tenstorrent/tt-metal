"""
Cumulative per-op PCC test for the gather decomposition (Approach A).

Runs TT ops cumulatively: Op1, Op1->Op2, Op1->Op2->Op3, etc.
At each stage, pulls back the TT result and compares against
the CPU golden intermediate.  Shows exactly where accumulated
error causes PCC/diff to degrade.

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_cumulative_pcc.py
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from gather_deepseek_ocr_codegen import utils
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

S = 913
D = 1280
N = 903
DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_inputs(seed=42):
    torch.manual_seed(seed)
    source = torch.randn(N, D, dtype=torch.bfloat16)
    mask_1d = torch.zeros(S, dtype=torch.bool)
    mask_1d[:N] = True
    mask_1d = mask_1d[torch.randperm(S)]
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, N - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand(S, D).contiguous()
    return source, source_idx_2d


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


def _compare(tt_torch, cpu_torch, label):
    """Print PCC and diff stats. Returns (pcc_passed, pcc_value)."""
    a = tt_torch.float().flatten()
    b = cpu_torch.float().flatten()

    diff = (a - b).abs()
    mismatches = torch.sum(diff > 0).item()
    max_d = diff.max().item()
    mean_d = diff.mean().item()

    pcc_passed, pcc_msg = comp_pcc(cpu_torch, tt_torch, 0.99)
    print(f"  {label}: {pcc_msg}, "
          f"mismatches={mismatches}/{a.numel()}, "
          f"max_diff={max_d}, mean_diff={mean_d:.4f}")
    return pcc_passed, pcc_msg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return utils.DeviceGetter.get_device((1, 1))


@pytest.fixture(scope="module")
def cpu_intermediates():
    """All CPU golden intermediates, computed once."""
    source, source_idx_2d = _build_inputs(seed=42)
    idx_i32 = source_idx_2d.to(torch.int32)

    idx_u32 = idx_i32.clone()
    idx_3d = idx_u32.reshape(S, D, 1)

    col_offsets = torch.arange(D, dtype=torch.int32) \
                       .reshape(1, D, 1).expand(S, D, 1).contiguous()
    concat = torch.cat([idx_3d, col_offsets], dim=2)

    src_flat = source.reshape(N * D, 1)

    concat_f32 = concat.float()
    weight = torch.tensor([[float(D)], [1.0]], dtype=torch.float32)
    matmul = torch.matmul(concat_f32, weight)

    flat_f32 = matmul.reshape(1, S * D)
    flat_idx = flat_f32.to(torch.int32)

    embed_out = F.embedding(
        flat_idx.long().reshape(-1), src_flat.reshape(-1, 1)
    )
    embed_result = embed_out.reshape(S, D)

    return {
        "source": source,
        "idx_i32": idx_i32,
        "col_offsets": col_offsets,
        "weight": weight,
        "op1": idx_u32,
        "op2": idx_3d,
        "op3": concat,
        "op4": src_flat,
        "op5": concat_f32,
        "op6": matmul,
        "op7": flat_idx,
        "op8": embed_result,
    }


@pytest.fixture(scope="module")
def tt_intermediates(device, cpu_intermediates):
    """Run the full TT pipeline once, capturing torch copies at every stage."""
    idx_i32 = cpu_intermediates["idx_i32"]
    source = cpu_intermediates["source"]
    col_offsets = cpu_intermediates["col_offsets"]
    weight = cpu_intermediates["weight"]

    results = {}

    # --- Index path ---

    # Op 1: to_layout(TILE) + typecast(UINT32)
    tt = _to_tt(idx_i32, ttnn.DataType.INT32, device)
    tt = ttnn.to_layout(tt, ttnn.Layout.TILE, None, memory_config=None)
    tt = ttnn.typecast(tt, ttnn.DataType.UINT32, memory_config=DRAM)
    results["op1"] = _to_torch(tt)

    # Op 2: reshape [913,1280,1]
    tt = ttnn.reshape(tt, [S, D, 1], memory_config=DRAM)
    results["op2"] = _to_torch(tt)

    # Op 3: concat with col_offsets
    tt_col = _to_tt(col_offsets, ttnn.DataType.UINT32, device, layout=ttnn.Layout.TILE)
    tt = ttnn.concat([tt, tt_col], 2, memory_config=DRAM)
    results["op3"] = _to_torch(tt)

    # --- Source path (independent) ---

    # Op 4: to_layout(TILE) + reshape [1155840,1]
    tt_src = _to_tt(source, ttnn.DataType.BFLOAT16, device)
    tt_src = ttnn.to_layout(tt_src, ttnn.Layout.TILE, None, memory_config=None)
    tt_src = ttnn.reshape(tt_src, [N * D, 1], memory_config=DRAM)
    results["op4"] = _to_torch(tt_src)

    # --- Index path continues ---

    # Op 5: typecast UINT32 -> FLOAT32
    tt = ttnn.typecast(tt, ttnn.DataType.FLOAT32, memory_config=DRAM)
    results["op5"] = _to_torch(tt)

    # Op 6: matmul [913,1280,2] @ [2,1]
    tt_w = _to_tt(weight, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt = ttnn.matmul(
        tt, tt_w,
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    results["op6"] = _to_torch(tt)

    # Op 7: reshape + typecast(UINT32) + to_layout(ROW_MAJOR)
    tt = ttnn.reshape(tt, [1, S * D], memory_config=DRAM)
    tt = ttnn.typecast(tt, ttnn.DataType.UINT32, memory_config=DRAM)
    tt = ttnn.to_layout(tt, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    results["op7"] = _to_torch(tt)

    # Op 8: embedding + reshape
    tt_src = ttnn.to_layout(tt_src, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    tt_out = ttnn.embedding(
        tt, tt_src,
        padding_idx=None, layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM,
    )
    tt_out = ttnn.reshape(tt_out, [S, D], memory_config=DRAM)
    results["op8"] = _to_torch(tt_out)

    return results


# ---------------------------------------------------------------------------
# Tests — cumulative, each checks TT output after ops 1..N vs CPU golden
# ---------------------------------------------------------------------------

def test_cumul_through_op1(cpu_intermediates, tt_intermediates):
    """Op1 only: to_layout + typecast on index."""
    passed, msg = _compare(
        tt_intermediates["op1"], cpu_intermediates["op1"], "Through Op1"
    )
    assert passed, msg


def test_cumul_through_op2(cpu_intermediates, tt_intermediates):
    """Op1 -> Op2: + reshape index."""
    passed, msg = _compare(
        tt_intermediates["op2"], cpu_intermediates["op2"], "Through Op2"
    )
    assert passed, msg


def test_cumul_through_op3(cpu_intermediates, tt_intermediates):
    """Op1 -> Op2 -> Op3: + concat with col_offsets."""
    passed, msg = _compare(
        tt_intermediates["op3"], cpu_intermediates["op3"], "Through Op3"
    )
    assert passed, msg


def test_cumul_through_op4(cpu_intermediates, tt_intermediates):
    """Op4 (source path): to_layout + reshape source."""
    passed, msg = _compare(
        tt_intermediates["op4"], cpu_intermediates["op4"], "Through Op4"
    )
    assert passed, msg


def test_cumul_through_op5(cpu_intermediates, tt_intermediates):
    """Op1 -> .. -> Op5: + typecast to FLOAT32."""
    passed, msg = _compare(
        tt_intermediates["op5"], cpu_intermediates["op5"], "Through Op5"
    )
    assert passed, msg


def test_cumul_through_op6(cpu_intermediates, tt_intermediates):
    """Op1 -> .. -> Op6: + matmul. EXPECT DROP HERE."""
    passed, msg = _compare(
        tt_intermediates["op6"], cpu_intermediates["op6"], "Through Op6"
    )
    assert passed, msg


def test_cumul_through_op7(cpu_intermediates, tt_intermediates):
    """Op1 -> .. -> Op7: + flatten indices."""
    passed, msg = _compare(
        tt_intermediates["op7"], cpu_intermediates["op7"], "Through Op7"
    )
    assert passed, msg


def test_cumul_through_op8(cpu_intermediates, tt_intermediates):
    """Op1 -> .. -> Op7 + Op4 -> Op8: full pipeline."""
    passed, msg = _compare(
        tt_intermediates["op8"], cpu_intermediates["op8"], "Through Op8"
    )
    assert passed, msg
