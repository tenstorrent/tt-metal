# Bisect test: dump full reduce output to understand fp32 DEST layout issue
import pytest
import torch
import ttnn
from .layer_norm_rm import layer_norm_rm


def make_input(shape):
    """Each element = row + col/W, so values vary within each row."""
    t = torch.zeros(shape, dtype=torch.bfloat16)
    W = shape[-1]
    flat = t.view(-1, W)
    for r in range(flat.shape[0]):
        for c in range(W):
            flat[r, c] = float(r + 1) + float(c) / W
    return t


def test_reduce_output_fp32(device):
    """Dump full reduce output to see the pattern in fp32 dest mode."""
    shape = (1, 1, 32, 32)
    W = shape[-1]

    torch_input = make_input(shape)
    torch_gamma = torch.ones(1, 1, 1, W, dtype=torch.bfloat16)
    torch_beta = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Phase 13 = reduce mean only
    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta, bisect_phase=13)
    torch_output = ttnn.to_torch(ttnn_output)

    # Expected means: row r -> mean = (r+1) + 15.5/32
    print("\n=== FULL REDUCE OUTPUT (Phase 13) ===")
    print("Expected means per row: (r+1) + 15.5/32 ≈ (r+1) + 0.484")
    print("\nRow : col0      col1      col2      col3      | expected_mean")
    print("-" * 70)
    for r in range(32):
        vals = torch_output[0, 0, r, :4].tolist()
        expected_mean = (r + 1) + 15.5 / 32
        all_zero = all(abs(v) < 0.001 for v in vals)
        marker = "" if not all_zero else " *** ZERO ***"
        match = ""
        if abs(vals[0]) > 0.001:
            # Find which row this mean corresponds to
            for exp_r in range(32):
                exp_mean = (exp_r + 1) + 15.5 / 32
                if abs(vals[0] - exp_mean) < 0.1:
                    if exp_r != r:
                        match = f" [MAPS TO ROW {exp_r}!]"
                    else:
                        match = " [correct]"
                    break
        print(
            f"  {r:2d} : {vals[0]:9.4f} {vals[1]:9.4f} {vals[2]:9.4f} {vals[3]:9.4f} | {expected_mean:6.3f}{marker}{match}"
        )
