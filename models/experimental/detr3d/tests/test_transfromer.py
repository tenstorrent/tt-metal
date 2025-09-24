import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.detr3d.ttnn.transformer import TTNNMultiheadAttention


from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("d_model", [256])
@pytest.mark.parametrize("nhead", [4])
def test_ttnn_multihead_attention_vs_torch(device, batch_size, seq_len, d_model, nhead, reset_seeds):
    """Test TTNN MultiheadAttention against PyTorch reference implementation"""

    # Create PyTorch reference model
    torch_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, bias=True).eval()

    # Create TTNN model
    ttnn_mha = TTNNMultiheadAttention(d_model, nhead, device)

    # Extract and convert PyTorch weights to TTNN format
    with torch.no_grad():
        # Get PyTorch weights
        torch_qkv_weight = torch.cat(
            [
                torch_mha.in_proj_weight[:d_model],  # query
                torch_mha.in_proj_weight[d_model : 2 * d_model],  # key
                torch_mha.in_proj_weight[2 * d_model :],  # value
            ],
            dim=0,
        )
        torch_qkv_bias = torch.cat(
            [
                torch_mha.in_proj_bias[:d_model],  # query
                torch_mha.in_proj_bias[d_model : 2 * d_model],  # key
                torch_mha.in_proj_bias[2 * d_model :],  # value
            ],
            dim=0,
        )

        # Convert to TTNN tensors
        ttnn_mha.qkv_weight = ttnn.from_torch(
            torch_qkv_weight.T,  # Transpose for linear layer
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.qkv_bias = ttnn.from_torch(
            torch_qkv_bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_mha.out_weight = ttnn.from_torch(
            torch_mha.out_proj.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_mha.out_bias = ttnn.from_torch(
            torch_mha.out_proj.bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    # Create test inputs
    torch_input = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output, _ = torch_mha(torch_input, torch_input, torch_input)

    # TTNN forward pass
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_output = ttnn_mha(ttnn_input, ttnn_input, ttnn_input)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Calculate and log PCC before assertion
    passing, pcc_message = comp_pcc(torch_output, ttnn_output_torch, 0.99)
    logger.info(f"PCC: {pcc_message}")
    logger.info(f"PCC: {torch_output.shape}")
    logger.info(f"PCC: {ttnn_output_torch.shape}")

    # Compare outputs with PCC (Pearson Correlation Coefficient)
    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)
