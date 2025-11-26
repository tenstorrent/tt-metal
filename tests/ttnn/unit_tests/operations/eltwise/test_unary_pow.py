import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose


@pytest.mark.parametrize("exponent", [2.0, -2.0, 3.56, -3.56, 0.5, -0.5])
def test_pow(exponent, device):
    torch.manual_seed(42)
    torch_base = torch.rand([4, 4], dtype=torch.bfloat16)
    torch_output = torch.pow(torch_base, exponent)
    ttnn_base = ttnn.from_torch(torch_base, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_output = ttnn.pow(ttnn_base, exponent)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # # Print results in tabular format
    # print(f"\n{'='*70}")
    # print(f"Test Results for Exponent: {exponent}")
    # print(f"{'='*70}")
    # print(f"{'Base':<15} {'Torch Output':<15} {'TTNN Output':<15} {'Diff':<15}")
    # print(f"{'-'*70}")

    # for i in range(torch_base.shape[0]):
    #     for j in range(torch_base.shape[1]):
    #         base_val = torch_base[i, j].item()
    #         torch_val = torch_output[i, j].item()
    #         ttnn_val = ttnn_output[i, j].item()
    #         diff = abs(torch_val - ttnn_val)
    #         print(f"{base_val:<15.6f} {torch_val:<15.6f} {ttnn_val:<15.6f} {diff:<15.6f}")

    # print(f"{'='*70}\n")

    assert assert_with_ulp(torch_output, ttnn_output, 2 if exponent == 3.56 else 1)


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0, 0.65, -1.0, -2.0, -0.65, 3.56, -3.56])
def test_pow_arange_masking(exponent, device):
    # Generate all possible bit pattern for bf16
    fp32 = torch.arange(0, 2**16, dtype=torch.int32).to(torch.float32)
    nan_mask = torch.isnan(fp32)
    neg_zero_mask = (fp32 == 0.0) & torch.signbit(fp32)
    pos_inf_mask = fp32 == float("inf")
    neg_inf_mask = fp32 == float("-inf")

    mask = nan_mask | neg_zero_mask | pos_inf_mask | neg_inf_mask
    fp32[mask] = 1.0
    clean_bf16 = fp32.to(torch.bfloat16)
    clean_bits = clean_bf16.view(torch.uint16)
    tt_input = clean_bits.view(torch.bfloat16)

    tt_in = ttnn.from_torch(
        tt_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(tt_input, exponent, device=device)

    tt_result = ttnn.pow(tt_in, exponent)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)
