# Minimal test for DPRINT debugging of group_norm mean computation
import torch
import ttnn
from .group_norm import group_norm


def test_debug_mean(device):
    """Run with: TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=TR2 scripts/tt-test.sh --dev <this_file>"""
    shape = (1, 1, 32, 32)
    G = 4
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    N, _, HW, C = shape
    # Print expected means per group
    x_r = torch_input.reshape(N, G, C // G, HW)
    for g in range(G):
        group_mean = x_r[:, g, :, :].mean().item()
        print(f"PyTorch expected mean[{g}] = {group_mean:.6f}")

    gamma = torch.ones(1, 1, 1, C, dtype=torch.bfloat16)
    beta = torch.zeros(1, 1, 1, C, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = group_norm(ttnn_input, num_groups=G, gamma=gamma, beta=beta, eps=1e-5)
    torch_output = ttnn.to_torch(ttnn_output).float()

    # Reference
    m = x_r.float().mean(dim=[2, 3], keepdim=True)
    v = x_r.float().var(dim=[2, 3], unbiased=False, keepdim=True)
    expected = (torch_input.float() - m.expand_as(x_r).reshape(N, 1, HW, C).float()) / torch.sqrt(
        v.expand_as(x_r).reshape(N, 1, HW, C).float() + 1e-5
    )

    diff = (torch_output - expected).abs()
    print(f"\nmax_diff = {diff.max().item():.6f}")
    print(f"mean_diff = {diff.mean().item():.6f}")
