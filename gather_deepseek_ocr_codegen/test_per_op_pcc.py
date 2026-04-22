"""
Per-op PCC isolation for the gather decomposition (Approach B).

For each ttnn op in the codegen _main graph, we:
  1. Compute the CPU golden input/output for that op using torch
  2. Send the CPU input to TT device
  3. Run only that single ttnn op
  4. Pull the result back and compare against the CPU output

This isolates each op with zero error propagation between ops.

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_per_op_pcc.py
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from gather_deepseek_ocr_codegen import utils
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
    """Send a torch tensor to TT device.  Handles UINT32 via INT32+typecast."""
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


def _int_equal(tt_result, cpu_expected, label):
    """Compare integer-valued tensors exactly."""
    a = tt_result.to(torch.int32).flatten()
    b = cpu_expected.to(torch.int32).flatten()
    mismatches = torch.sum(a != b).item()
    if mismatches:
        diff = (a.long() - b.long()).abs()
        print(f"  {label}: {mismatches}/{a.numel()} mismatches, "
              f"max_diff={diff.max().item()}, mean_diff={diff.float().mean().item():.2f}")
    assert mismatches == 0, (
        f"{label}: {mismatches}/{a.numel()} exact mismatches"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return utils.DeviceGetter.get_device((1, 1))


@pytest.fixture(scope="module")
def cpu_intermediates():
    """All CPU intermediates for the gather decomposition, computed once."""
    source, source_idx_2d = _build_inputs(seed=42)
    idx_i32 = source_idx_2d.to(torch.int32)

    # Op 1 output: typecast index (int32 -> effectively uint32, same values)
    idx_u32 = idx_i32.clone()

    # Op 2 output: reshape index
    idx_3d = idx_u32.reshape(S, D, 1)

    # Constant: column offsets
    col_offsets = torch.arange(D, dtype=torch.int32).reshape(1, D, 1) \
                       .expand(S, D, 1).contiguous()

    # Op 3 output: concat [row_idx, col_idx]
    concat = torch.cat([idx_3d, col_offsets], dim=2)

    # Op 4 output: flatten source
    src_flat = source.reshape(N * D, 1)

    # Op 5 output: typecast concat to float32
    concat_f32 = concat.float()

    # Op 6 output: matmul to linearize indices
    weight = torch.tensor([[float(D)], [1.0]], dtype=torch.float32)
    matmul = torch.matmul(concat_f32, weight)

    # Op 7 output: reshape + typecast to flat int indices
    flat_f32 = matmul.reshape(1, S * D)
    flat_idx = flat_f32.to(torch.int32)

    # Op 8 output: embedding lookup
    embed_out = F.embedding(
        flat_idx.long().reshape(-1), src_flat.reshape(-1, 1)
    )
    embed_result = embed_out.reshape(S, D)

    return {
        "source": source,
        "idx_i32": idx_i32,
        "idx_u32": idx_u32,
        "idx_3d": idx_3d,
        "col_offsets": col_offsets,
        "concat": concat,
        "src_flat": src_flat,
        "concat_f32": concat_f32,
        "weight": weight,
        "matmul": matmul,
        "flat_f32": flat_f32,
        "flat_idx": flat_idx,
        "embed_result": embed_result,
    }


# ---------------------------------------------------------------------------
# Tests — one per op group
# ---------------------------------------------------------------------------

def test_op1_typecast_index(device, cpu_intermediates):
    """to_layout(TILE) + typecast(INT32 -> UINT32) on index [913,1280]."""
    cpu_in = cpu_intermediates["idx_i32"]
    cpu_out = cpu_intermediates["idx_u32"]

    tt_in = _to_tt(cpu_in, ttnn.DataType.INT32, device)
    tt_tiled = ttnn.to_layout(tt_in, ttnn.Layout.TILE, None, memory_config=None)
    tt_out = ttnn.typecast(tt_tiled, ttnn.DataType.UINT32, memory_config=DRAM)

    _int_equal(_to_torch(tt_out), cpu_out, "Op1 typecast index")


def test_op2_reshape_index(device, cpu_intermediates):
    """reshape index [913,1280] -> [913,1280,1] UINT32."""
    cpu_in = cpu_intermediates["idx_u32"]
    cpu_out = cpu_intermediates["idx_3d"]

    tt_in = _to_tt(cpu_in, ttnn.DataType.UINT32, device, layout=ttnn.Layout.TILE)
    tt_out = ttnn.reshape(tt_in, [S, D, 1], memory_config=DRAM)

    _int_equal(_to_torch(tt_out), cpu_out, "Op2 reshape index")


def test_op3_concat(device, cpu_intermediates):
    """concat([index_3d, col_offsets], dim=2) -> [913,1280,2] UINT32."""
    cpu_idx = cpu_intermediates["idx_3d"]
    cpu_col = cpu_intermediates["col_offsets"]
    cpu_out = cpu_intermediates["concat"]

    tt_idx = _to_tt(cpu_idx, ttnn.DataType.UINT32, device, layout=ttnn.Layout.TILE)
    tt_col = _to_tt(cpu_col, ttnn.DataType.UINT32, device, layout=ttnn.Layout.TILE)
    tt_out = ttnn.concat([tt_idx, tt_col], 2, memory_config=DRAM)

    _int_equal(_to_torch(tt_out), cpu_out, "Op3 concat")


def test_op4_reshape_source(device, cpu_intermediates):
    """to_layout(TILE) + reshape source [903,1280] -> [1155840,1] BF16."""
    cpu_in = cpu_intermediates["source"]
    cpu_out = cpu_intermediates["src_flat"]

    tt_in = _to_tt(cpu_in, ttnn.DataType.BFLOAT16, device)
    tt_tiled = ttnn.to_layout(tt_in, ttnn.Layout.TILE, None, memory_config=None)
    tt_out = ttnn.reshape(tt_tiled, [N * D, 1], memory_config=DRAM)

    result = _to_torch(tt_out)
    passed, msg = assert_with_pcc(cpu_out, result, pcc=0.99)
    assert passed, f"Op4 reshape source: {msg}"


def test_op5_typecast_to_f32(device, cpu_intermediates):
    """typecast concat [913,1280,2] UINT32 -> FLOAT32."""
    cpu_in = cpu_intermediates["concat"]
    cpu_out = cpu_intermediates["concat_f32"]

    tt_in = _to_tt(cpu_in, ttnn.DataType.UINT32, device, layout=ttnn.Layout.TILE)
    tt_out = ttnn.typecast(tt_in, ttnn.DataType.FLOAT32, memory_config=DRAM)

    result = _to_torch(tt_out)
    assert torch.equal(result, cpu_out), (
        f"Op5 typecast f32: max_diff="
        f"{torch.max(torch.abs(result - cpu_out)).item()}"
    )


def test_op6_matmul(device, cpu_intermediates):
    """matmul [913,1280,2] @ [2,1] -> [913,1280,1]. PRIMARY SUSPECT.

    Computes flat_index = row_idx * 1280 + col_idx.
    Values up to 1,155,839 — within float32 exact-integer range (2^24),
    so any mismatch is a TT matmul precision issue.
    """
    cpu_in = cpu_intermediates["concat_f32"]
    cpu_w = cpu_intermediates["weight"]
    cpu_out = cpu_intermediates["matmul"]

    tt_in = _to_tt(cpu_in, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_w = _to_tt(cpu_w, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_out = ttnn.matmul(
        tt_in, tt_w,
        transpose_a=False, transpose_b=False,
        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
        program_config=None, activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )

    result = _to_torch(tt_out)
    diff = (result - cpu_out).abs()
    mismatches = torch.sum(diff > 0).item()
    print(f"  Op6 matmul: {mismatches}/{result.numel()} inexact, "
          f"max_diff={diff.max().item()}, mean_diff={diff.float().mean().item():.4f}")

    passed, msg = assert_with_pcc(cpu_out, result, pcc=0.99)
    assert passed, f"Op6 matmul: {msg}"


def test_op7_flatten_indices(device, cpu_intermediates):
    """reshape [913,1280,1] -> [1,1168640] + typecast F32->UINT32 + to_layout ROW_MAJOR."""
    cpu_in = cpu_intermediates["matmul"]
    cpu_out = cpu_intermediates["flat_idx"]

    tt_in = _to_tt(cpu_in, ttnn.DataType.FLOAT32, device, layout=ttnn.Layout.TILE)
    tt_reshaped = ttnn.reshape(tt_in, [1, S * D], memory_config=DRAM)
    tt_cast = ttnn.typecast(tt_reshaped, ttnn.DataType.UINT32, memory_config=DRAM)
    tt_out = ttnn.to_layout(tt_cast, ttnn.Layout.ROW_MAJOR, None, memory_config=None)

    _int_equal(_to_torch(tt_out), cpu_out, "Op7 flatten indices")


def test_op8_embedding(device, cpu_intermediates):
    """embedding(flat_indices, flat_source) -> [913,1280]. SECONDARY SUSPECT."""
    cpu_idx = cpu_intermediates["flat_idx"]
    cpu_src = cpu_intermediates["src_flat"]
    cpu_out = cpu_intermediates["embed_result"]

    tt_idx = _to_tt(cpu_idx, ttnn.DataType.UINT32, device, layout=ttnn.Layout.ROW_MAJOR)
    tt_src = _to_tt(cpu_src, ttnn.DataType.BFLOAT16, device, layout=ttnn.Layout.ROW_MAJOR)

    tt_out = ttnn.embedding(
        tt_idx, tt_src,
        padding_idx=None, layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM,
    )
    tt_result = ttnn.reshape(tt_out, [S, D], memory_config=DRAM)

    result = _to_torch(tt_result)
    passed, msg = assert_with_pcc(cpu_out, result, pcc=0.99)
    assert passed, f"Op8 embedding: {msg}"
