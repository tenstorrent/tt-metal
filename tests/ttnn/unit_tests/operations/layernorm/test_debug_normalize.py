"""Debug test to diagnose normalize stage accuracy."""
import torch
import ttnn
from ttnn.operations.layernorm import layernorm


def test_debug_normalize(device):
    torch.manual_seed(42)

    shapes = [(1, 1, 32, 32), (1, 1, 32, 128), (1, 1, 32, 256)]

    for shape in shapes:
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        expected = torch.nn.functional.layer_norm(torch_input, [shape[-1]], eps=1e-5)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_output = layernorm(ttnn_input)
        torch_output = ttnn.to_torch(ttnn_output)

        diff = (torch_output.float() - expected.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Find worst element
        worst_idx = diff.argmax()
        worst_coords = []
        idx = worst_idx.item()
        for d in reversed(shape):
            worst_coords.insert(0, idx % d)
            idx //= d

        print(f"\nShape {shape}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        print(
            f"  Worst at {worst_coords}: got={torch_output.flatten()[worst_idx].item():.6f}, "
            f"expected={expected.flatten()[worst_idx].item():.6f}"
        )

        # Check per-row stats for first tile-row
        for row in range(min(4, shape[-2])):
            row_diff = diff[0, 0, row, :].max().item()
            row_mean_diff = diff[0, 0, row, :].mean().item()
            print(f"  Row {row}: max_diff={row_diff:.6f}, mean_diff={row_mean_diff:.6f}")
