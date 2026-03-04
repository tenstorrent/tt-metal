"""Debug: detailed analysis for (4,2,64,64) case."""
import torch
import ttnn
import pytest
from ttnn.operations.layernorm import layernorm


def test_debug_4264(device):
    shape = (4, 2, 64, 64)
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

    # Reference
    expected = torch.nn.functional.layer_norm(torch_input.float(), [shape[-1]], eps=1e-5)

    # Reshape both to 2D for analysis
    actual_2d = actual.reshape(-1, shape[-1])
    expected_2d = expected.reshape(-1, shape[-1])

    # Check per-row diffs (512 rows total, 16 blocks of 32 rows each)
    per_row_max_diff = (actual_2d - expected_2d).abs().max(dim=1).values

    # Show per-block summary
    for block in range(16):
        start_row = block * 32
        end_row = start_row + 32
        block_max = per_row_max_diff[start_row:end_row].max().item()
        block_mean = per_row_max_diff[start_row:end_row].mean().item()
        status = "FAIL" if block_max > 0.2 else "ok"
        print(
            f"Block {block:2d} (rows {start_row:3d}-{end_row-1:3d}): max_diff={block_max:.4f}, mean_diff={block_mean:.4f} [{status}]"
        )

    # Show worst rows
    worst_rows = per_row_max_diff.topk(5)
    print("\nWorst 5 rows:")
    for idx, val in zip(worst_rows.indices, worst_rows.values):
        row = idx.item()
        block = row // 32
        print(f"  Row {row} (block {block}): max_diff={val:.4f}")
        # Show per-tile ratio for this row
        x = torch_input.float().reshape(-1, shape[-1])
        mean_val = x[row].mean()
        centered = x[row] - mean_val
        var_val = centered.pow(2).mean()
        true_rsqrt = (var_val + 1e-5).rsqrt().item()
        mask = centered.abs() > 0.1
        if mask.sum() > 0:
            implied = actual_2d[row][mask] / centered[mask]
            print(
                f"    true_rsqrt={true_rsqrt:.6f}, implied_rsqrt={implied.mean():.6f}, ratio={implied.mean()/true_rsqrt:.6f}"
            )

    max_diff = (actual - expected).abs().max()
    print(f"\nOverall max diff: {max_diff}")
