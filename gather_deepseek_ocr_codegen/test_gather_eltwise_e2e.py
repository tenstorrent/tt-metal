"""
End-to-end gather PCC test with the element-wise fix applied to the
full codegen pipeline.

Patches the original codegen _main() by replacing the ttnn.matmul call
(Op6: [S,D,2] @ [2,1]) with element-wise ttnn.multiply + ttnn.add,
then runs the complete gather decomposition and compares against CPU.

This validates that the eltwise fix resolves the PCC drop in the full
codegen context, not just in isolation.

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_gather_eltwise_e2e.py
"""

import pytest
import torch
import ttnn

from gather_deepseek_ocr_codegen import utils
from gather_deepseek_ocr_codegen.main import consteval__main, load_inputs_for__main
from models.common.utility_functions import comp_pcc

S = 913
D = 1280
DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _main_eltwise(input):
    """Same as codegen _main() but replaces ttnn.matmul with eltwise ops."""
    ce_cache = consteval__main({}, input)
    var_0 = input[0]
    var_1 = input[1]

    # Op1: typecast index INT32 -> UINT32
    ttnn_to_layout_0 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_to_layout_0, ttnn.DataType.UINT32, memory_config=DRAM,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)

    # Op2: reshape index [913,1280] -> [913,1280,1]
    ttnn_reshape_1 = ttnn.reshape(ttnn_typecast_0, [S, D, 1], memory_config=DRAM)
    ttnn.deallocate(ttnn_typecast_0, False)

    # Op3: concat [row_idx, col_idx] -> [913,1280,2]
    ttnn_concat_0 = ttnn.concat(
        [ttnn_reshape_1, ce_cache["main_const_eval_0"][0]], 2, memory_config=DRAM,
    )
    ttnn.deallocate(ttnn_reshape_1, False)

    # Op4: reshape source [903,1280] -> [1155840,1]
    ttnn_to_layout_1 = ttnn.to_layout(var_1, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_1, False)
    ttnn_reshape_2 = ttnn.reshape(ttnn_to_layout_1, [S * D - S * D + 903 * D, 1], memory_config=DRAM)
    ttnn.deallocate(ttnn_to_layout_1, False)

    # Op5: typecast concat -> FLOAT32
    ttnn_typecast_1 = ttnn.typecast(ttnn_concat_0, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(ttnn_concat_0, False)

    # --- FIX: replace matmul with element-wise multiply + add ---
    # Extract row_idx (dim 2, index 0) and col_idx (dim 2, index 1)
    # ttnn_typecast_1 is [913, 1280, 2] FLOAT32
    # Slice along last dim to get row and col components
    row_idx = ttnn_typecast_1[:, :, 0:1]  # [913, 1280, 1]
    col_idx = ttnn_typecast_1[:, :, 1:2]  # [913, 1280, 1]
    ttnn.deallocate(ttnn_typecast_1, False)

    tt_product = ttnn.multiply(row_idx, float(D))
    ttnn.deallocate(row_idx, False)
    ttnn_eltwise_result = ttnn.add(tt_product, col_idx)
    ttnn.deallocate(tt_product, False)
    ttnn.deallocate(col_idx, False)
    # ttnn_eltwise_result: [913, 1280, 1] FLOAT32 — same shape as matmul output

    # Op7: reshape + typecast + to_layout (flatten indices)
    ttnn_reshape_3 = ttnn.reshape(ttnn_eltwise_result, [1, S * D], memory_config=DRAM)
    ttnn.deallocate(ttnn_eltwise_result, False)
    ttnn_typecast_2 = ttnn.typecast(ttnn_reshape_3, ttnn.DataType.UINT32, memory_config=DRAM)
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_to_layout_2 = ttnn.to_layout(ttnn_typecast_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_typecast_2, False)

    # Op4 continued: source to ROW_MAJOR
    ttnn_to_layout_3 = ttnn.to_layout(ttnn_reshape_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_2, False)

    # Op8: embedding
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_2, ttnn_to_layout_3,
        padding_idx=None, layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM,
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn.deallocate(ttnn_to_layout_2, False)

    ttnn_reshape_4 = ttnn.reshape(ttnn_embedding_0, [S, D], memory_config=DRAM)
    ttnn.deallocate(ttnn_embedding_0, False)

    return [ttnn_reshape_4]


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


def test_gather_e2e_eltwise_fix(device):
    """Full codegen pipeline with eltwise fix, compared against CPU torch.gather."""
    tt_inputs = load_inputs_for__main()

    # CPU golden: extract torch tensors from the TT inputs
    idx_torch = ttnn.to_torch(ttnn.from_device(tt_inputs[0]))  # [913, 1280] INT32
    src_torch = ttnn.to_torch(ttnn.from_device(tt_inputs[1]))  # [903, 1280] BFLOAT16

    cpu_expected = torch.gather(
        src_torch.float(),
        0,
        idx_torch.long().expand_as(src_torch[:1].expand(913, -1)),
    )
    # Simpler: idx_torch values are row indices, replicated across cols
    cpu_expected = torch.gather(src_torch.float(), 0, idx_torch.long())

    # Run patched TT pipeline
    tt_outputs = _main_eltwise(tt_inputs)
    tt_result = ttnn.to_torch(ttnn.from_device(tt_outputs[0]))

    passed, msg = _report(tt_result, cpu_expected.bfloat16(), "E2E gather (eltwise fix)")
    assert passed, msg
