import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("seq_len", [3410, 4095, 6820, 8190, 9450, 13640, 16380, 18900, 27280, 32760, 37800, 75600])
@pytest.mark.parametrize("hidden_dim", [640, 1280, 2560, 5120])
def test_fused_muladd(seq_len, hidden_dim, device):
    torch_dtype = torch.bfloat16

    # Create random inputs in torch
    tensor_shape = (1, 1, seq_len, hidden_dim)
    vector_shape = (1, 1, 1, hidden_dim)
    A = torch.randn(tensor_shape, dtype=torch_dtype)
    B = torch.randn(tensor_shape, dtype=torch_dtype)
    C = torch.randn(vector_shape, dtype=torch_dtype)

    # Torch reference: out = A + B * C
    torch_out = A + B * C

    # Move inputs to ttnn
    tt_A = ttnn.from_torch(A, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_B = ttnn.from_torch(B, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Compute in ttnn: A = B + C * D
    tt_out = tt_A + tt_B * tt_C

    # Bring result back to torch
    tt_out_torch = ttnn.to_torch(tt_out)

    # Compare
    assert_with_pcc(torch_out, tt_out_torch, pcc=0.999)
