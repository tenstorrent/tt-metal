"""Debug: check what rsqrt factor the device is applying."""
import torch
import ttnn
import pytest
from ttnn.operations.layernorm import layernorm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="32x32"),
    ],
)
def test_debug_ratio(device, shape):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Run on device
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = layernorm(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # Expected
    expected = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]], eps=1e-5)

    # Manual computation in bfloat16
    x = torch_input.float()
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    rsqrt_val = torch.rsqrt(var + 1e-5)

    # Check ratio actual/expected for each row
    for row in range(min(4, shape[2])):
        row_actual = actual[0, 0, row, :]
        row_expected = expected[0, 0, row, :]
        mask = row_expected.abs() > 0.1
        if mask.sum() > 0:
            ratios = row_actual[mask] / row_expected[mask]
            print(f"Row {row}: ratio mean={ratios.mean():.6f}, std={ratios.std():.6f}")
            print(f"  true_rsqrt={rsqrt_val[0,0,row,0]:.6f}")
            implied_rsqrt = row_actual[mask] / centered[0, 0, row, :][mask]
            print(f"  implied_rsqrt mean={implied_rsqrt.mean():.6f}")
            print(f"  expected_rsqrt / implied = {rsqrt_val[0,0,row,0] / implied_rsqrt.mean():.6f}")

    max_diff = (actual - expected).abs().max()
    print(f"\nMax diff: {max_diff}")
