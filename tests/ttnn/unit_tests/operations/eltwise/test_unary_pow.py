import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose


@pytest.mark.parametrize("exponent", [2.0, -2.0])
def test_pow(exponent):
    """Test ttnn.pow operation with various exponents"""
    torch.manual_seed(42)

    # Use values > 0.1 to avoid issues with negative exponents
    torch_base = torch.rand([4, 4], dtype=torch.bfloat16)
    torch_output = torch.pow(torch_base, exponent)

    with ttnn.manage_device(device_id=0) as device:
        ttnn_base = ttnn.from_torch(torch_base, layout=ttnn.TILE_LAYOUT, device=device)

        print(f"Test Results for Exponent: {exponent}")
        ttnn_output = ttnn.pow(ttnn_base, exponent)
        ttnn_output = ttnn.to_torch(ttnn_output)

        # Print results in tabular format
        print(f"\n{'='*70}")
        print(f"{'='*70}")
        print(f"{'Base':<15} {'Torch Output':<15} {'TTNN Output':<15} {'Diff':<15}")
        print(f"{'-'*70}")

        for i in range(torch_base.shape[0]):
            for j in range(torch_base.shape[1]):
                base_val = torch_base[i, j].item()
                torch_val = torch_output[i, j].item()
                ttnn_val = ttnn_output[i, j].item()
                diff = abs(torch_val - ttnn_val)
                print(f"{base_val:<15.6f} {torch_val:<15.6f} {ttnn_val:<15.6f} {diff:<15.6f}")

        print(f"{'='*70}\n")

        # Compare outputs with appropriate tolerance
        assert torch.allclose(
            torch_output, ttnn_output, rtol=1e-2, atol=1e-2
        ), f"Output mismatch for exponent {exponent}"


@pytest.mark.parametrize("exponent", [3.56, -3.56])
def test_pow_arange_masking(exponent, device):
    # Generate all possible bit pattersn for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)

    # Mask NaN
    mask = torch.isnan(input_tensor) | ((input_tensor == 0) & torch.signbit(input_tensor))
    input_tensor[mask] = 1.0
    # for -0.0, we get inf but expected is nan

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(input_tensor, exponent, device=device)

    tt_result = ttnn.pow(tt_in, exponent)
    result = ttnn.to_torch(tt_result)

    # Debug: Find problematic values
    print(f"\n{'='*80}")
    print(f"Exponent: {exponent}")
    print(f"{'='*80}")

    # Check for non-finite mismatches
    golden_nan = torch.isnan(golden)
    golden_inf = torch.isinf(golden)
    result_nan = torch.isnan(result)
    result_inf = torch.isinf(result)

    # Find positions where finite/non-finite status differs
    finite_mismatch = (golden_nan != result_nan) | (golden_inf != result_inf)

    if finite_mismatch.any():
        print(f"\nFound {finite_mismatch.sum()} finite/non-finite mismatches!")
        print(f"\nFirst 20 problematic values:")
        print(f"{'Input':<20} {'Golden':<20} {'Result':<20} {'Issue':<30}")
        print(f"{'-'*90}")

        count = 0
        for idx in torch.where(finite_mismatch)[0][:20]:
            inp = input_tensor[idx].item()
            gld = golden[idx].item()
            res = result[idx].item()

            issue = ""
            if torch.isnan(golden[idx]) and not torch.isnan(result[idx]):
                issue = "Golden=NaN, Result=finite"
            elif not torch.isnan(golden[idx]) and torch.isnan(result[idx]):
                issue = "Golden=finite, Result=NaN"
            elif torch.isinf(golden[idx]) and not torch.isinf(result[idx]):
                issue = "Golden=Inf, Result=finite"
            elif not torch.isinf(golden[idx]) and torch.isinf(result[idx]):
                issue = "Golden=finite, Result=Inf"

            print(f"{inp:<20.6f} {gld:<20} {res:<20} {issue:<30}")
            count += 1

    # Also check for unexpected NaN/Inf in result
    unexpected_nonfinite = (~golden_nan & ~golden_inf) & (result_nan | result_inf)
    if unexpected_nonfinite.any():
        print(f"\n\nFound {unexpected_nonfinite.sum()} unexpected non-finite values in result!")
        print(f"\nFirst 20 unexpected non-finite values:")
        print(f"{'Input':<20} {'Input(hex)':<15} {'Golden':<20} {'Result':<20}")
        print(f"{'-'*80}")

        for idx in torch.where(unexpected_nonfinite)[0][:20]:
            inp = input_tensor[idx].item()
            inp_hex = hex(all_bitpatterns[idx].item())
            gld = golden[idx].item()
            res = result[idx].item()
            print(f"{inp:<20.6f} {inp_hex:<15} {gld:<20.6f} {res:<20}")

    print(f"{'='*80}\n")

    assert_with_ulp(golden, result, 1)
