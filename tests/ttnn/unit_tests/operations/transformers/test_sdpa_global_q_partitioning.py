import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "B,NH,S,DH,q_chunk,k_chunk",
    [
        # Even distribution cases
        (1, 10, 2368, 128, 64, 128),  # Sprint shape
        (1, 8, 256, 64, 32, 32),  # Simple multi-head
        (1, 40, 1024, 128, 64, 128),  # Many heads
        # Uneven distribution cases (regression tests)
        (1, 10, 2368, 128, 64, 256),  # Prime-ish chunk count
        (2, 8, 512, 64, 64, 64),  # Multi-batch
        (1, 3, 256, 64, 32, 32),  # Few heads
    ],
)
def test_global_q_partitioning_correctness(device, B, NH, S, DH, q_chunk, k_chunk):
    """Verify correctness with various shapes exercising different work distributions."""
    torch.manual_seed(42)

    Q = torch.randn(B, NH, S, DH)
    K = torch.randn(B, NH, S, DH)
    V = torch.randn(B, NH, S, DH)

    # PyTorch reference
    scale = 1.0 / (DH**0.5)
    attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
    expected = torch.matmul(attn, V)

    # TTNN with full grid
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )

    Q_tt = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    K_tt = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    V_tt = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tt = ttnn.transformer.scaled_dot_product_attention(
        Q_tt,
        K_tt,
        V_tt,
        is_causal=False,
        program_config=program_config,
    )

    output = ttnn.to_torch(output_tt)
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    assert passing, f"PCC check failed: {pcc}"
