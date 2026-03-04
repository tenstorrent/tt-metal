"""Debug: detailed analysis for Wt=4 (32x128) case."""
import torch
import ttnn
import pytest
from ttnn.operations.layernorm import layernorm


def test_debug_wt4(device):
    shape = (1, 1, 32, 128)
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

    # Manual computation in float32
    x = torch_input.float()
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    rsqrt_val = torch.rsqrt(var + 1e-5)

    # Check per-tile ratios for row 0
    Wt = shape[-1] // 32
    for tile in range(Wt):
        start = tile * 32
        end = start + 32
        row0_actual = actual[0, 0, 0, start:end]
        row0_centered = centered[0, 0, 0, start:end]
        mask = row0_centered.abs() > 0.1
        if mask.sum() > 0:
            implied_rsqrt = row0_actual[mask] / row0_centered[mask]
            true_rsqrt = rsqrt_val[0, 0, 0, 0].item()
            print(
                f"Tile {tile} (cols {start}-{end-1}): "
                f"implied_rsqrt mean={implied_rsqrt.mean():.6f}, "
                f"std={implied_rsqrt.std():.6f}, "
                f"true_rsqrt={true_rsqrt:.6f}, "
                f"ratio={implied_rsqrt.mean()/true_rsqrt:.6f}"
            )

    # Check if there's a pattern per-row across all tiles
    print("\nPer-row analysis (all tiles combined):")
    for row in range(min(4, shape[2])):
        row_actual = actual[0, 0, row, :]
        row_centered = centered[0, 0, row, :]
        mask = row_centered.abs() > 0.1
        if mask.sum() > 0:
            implied = row_actual[mask] / row_centered[mask]
            true_r = rsqrt_val[0, 0, row, 0].item()
            print(f"  Row {row}: implied={implied.mean():.6f}, true={true_r:.6f}, ratio={implied.mean()/true_r:.6f}")

    expected = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]], eps=1e-5)
    max_diff = (actual - expected).abs().max()
    print(f"\nMax diff: {max_diff}")
