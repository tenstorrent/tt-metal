# Debug: test if variance/rsqrt computation is correct by outputting
# the rsqrt-scaled identity: output = 1.0 * rsqrt (by passing all-ones centered)
# Actually, let's test by computing centered * rsqrt for just 1 tile
# by forcing the output to skip the multiply and just show centered values

# Actually, let's use a simpler test: compute layernorm for shapes (32,32) and (32,128)
# and compare element-by-element ratios

import torch
import ttnn
import pytest
from ttnn.operations.layernorm import layernorm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 128), id="32x128"),
    ],
)
def test_debug_var(device, shape):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Expected
    expected = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]], eps=1e-5)

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

    # Manual computation
    x = torch_input.float()
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    rsqrt_val = torch.rsqrt(var + 1e-5)

    # Check: what rsqrt does the device seem to be applying?
    # actual = centered * device_rsqrt
    # device_rsqrt = actual / centered (for each element)
    centered_bf16 = (torch_input.float() - mean).to(torch.bfloat16).float()

    for row in range(min(4, shape[2])):
        # Compute implied rsqrt for each element in the row
        row_actual = actual[0, 0, row, :]
        row_centered = centered_bf16[0, 0, row, :]
        # Avoid division by near-zero
        mask = row_centered.abs() > 0.01
        if mask.sum() > 0:
            implied_rsqrt = row_actual[mask] / row_centered[mask]
            true_rsqrt = rsqrt_val[0, 0, row, 0].item()
            print(
                f"Row {row}: true_rsqrt={true_rsqrt:.4f}, "
                f"implied_rsqrt mean={implied_rsqrt.mean():.4f}, "
                f"std={implied_rsqrt.std():.4f}, "
                f"min={implied_rsqrt.min():.4f}, max={implied_rsqrt.max():.4f}"
            )

    # Also check per-tile grouping
    Wt = shape[-1] // 32
    for tile in range(Wt):
        start = tile * 32
        end = start + 32
        row0_actual = actual[0, 0, 0, start:end]
        row0_centered = centered_bf16[0, 0, 0, start:end]
        mask = row0_centered.abs() > 0.01
        if mask.sum() > 0:
            ratios = row0_actual[mask] / row0_centered[mask]
            print(
                f"Tile {tile} (row 0, cols {start}-{end-1}): " f"ratio mean={ratios.mean():.4f}, std={ratios.std():.4f}"
            )

    max_diff = (actual - expected).abs().max()
    print(f"\nMax diff: {max_diff}")
