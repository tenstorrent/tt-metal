"""Quick diagnostic for gamma multiply debugging."""
import torch
import ttnn
from ttnn.operations.rms_norm import rms_norm


def test_diag_gamma(device):
    torch.manual_seed(42)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    # Reference: normalize only (no gamma)
    normalize_ref = (
        torch_input.float() * torch.rsqrt(torch.mean(torch_input.float() ** 2, dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)

    # Reference: with gamma
    gamma_ref = (
        torch_input.float()
        * torch.rsqrt(torch.mean(torch_input.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        * gamma.float()
    ).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_gamma = ttnn.from_torch(
        gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma)
    torch_output = ttnn.to_torch(ttnn_output)

    diff_gamma = (torch_output.float() - gamma_ref.float()).abs().max().item()
    diff_norm = (torch_output.float() - normalize_ref.float()).abs().max().item()

    print(f"\nMax diff vs gamma_ref: {diff_gamma}")
    print(f"Max diff vs normalize_ref: {diff_norm}")
    print(f"\nOutput[0,0,0,:8]: {torch_output[0,0,0,:8]}")
    print(f"Gamma_ref[0,0,0,:8]: {gamma_ref[0,0,0,:8]}")
    print(f"Normalize_ref[0,0,0,:8]: {normalize_ref[0,0,0,:8]}")
    print(f"Gamma[0,0,0,:8]: {gamma[0,0,0,:8]}")

    # Check if output matches normalize (gamma not applied)
    if diff_norm < 0.01:
        print("\n*** OUTPUT MATCHES NORMALIZE (no gamma) -- gamma multiply NOT applied ***")
    elif diff_gamma < 0.2:
        print("\n*** OUTPUT MATCHES GAMMA REF -- gamma multiply working ***")
    else:
        print(f"\n*** OUTPUT MATCHES NEITHER -- something else wrong ***")

    assert diff_gamma < 0.2, f"Max diff vs gamma_ref: {diff_gamma}"
