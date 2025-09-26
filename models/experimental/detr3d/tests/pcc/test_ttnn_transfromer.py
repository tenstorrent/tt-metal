import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.detr3d.ttnn.transformer import TTNNMultiheadAttention
from models.experimental.detr3d.reference.detr3d_model import TransformerDecoderLayer
from models.experimental.detr3d.ttnn.transformer import TTTransformerDecoderLayer

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
    # torch_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead).eval()
    torch_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=False).eval()
    # torch_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, bias=True).eval()

    # Create TTNN model
    ttnn_mha = TTNNMultiheadAttention(d_model, nhead, device)

    # Extract and convert PyTorch weights to TTNN format
    with torch.no_grad():
        # Get PyTorch weights

        # torch_qkv_weight = torch.cat(
        #     [
        #         torch_mha.in_proj_weight[:d_model],  # query
        #         torch_mha.in_proj_weight[d_model : 2 * d_model],  # key
        #         torch_mha.in_proj_weight[2 * d_model :],  # value
        #     ],
        #     dim=0,
        # )
        # torch_qkv_bias = torch.cat(
        #     [
        #         torch_mha.in_proj_bias[:d_model],  # query
        #         torch_mha.in_proj_bias[d_model : 2 * d_model],  # key
        #         torch_mha.in_proj_bias[2 * d_model :],  # value
        #     ],
        #     dim=0,
        # )

        # Convert to TTNN tensors
        ttnn_mha.q_weight = ttnn.from_torch(
            torch_mha.in_proj_weight[:d_model].T,  # Transpose for linear layer
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.k_weight = ttnn.from_torch(
            torch_mha.in_proj_weight[d_model : 2 * d_model].T,  # Transpose for linear layer
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.v_weight = ttnn.from_torch(
            torch_mha.in_proj_weight[2 * d_model :].T,  # Transpose for linear layer
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.q_bias = ttnn.from_torch(
            torch_mha.in_proj_bias[:d_model].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        ttnn_mha.k_bias = ttnn.from_torch(
            torch_mha.in_proj_bias[d_model : 2 * d_model].reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ttnn_mha.v_bias = ttnn.from_torch(
            torch_mha.in_proj_bias[2 * d_model :].reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
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
    ttnn_input = ttnn.from_torch(
        torch_input.permute(1, 0, 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    print(f"{ttnn_input.shape=}")
    print(f"{torch_output.shape=}")

    ttnn_output = ttnn_mha(ttnn_input, ttnn_input, ttnn_input)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    ttnn_output_torch = torch.permute(ttnn_output_torch, (1, 0, 2))

    print(f"{ttnn_output_torch.shape=}")

    # Calculate and log PCC before assertion
    passing, pcc_message = comp_pcc(torch_output, ttnn_output_torch, 0.99)
    logger.info(f"PCC: {pcc_message}")
    logger.info(f"PCC: {torch_output.shape}")
    logger.info(f"PCC: {ttnn_output_torch.shape}")

    # Compare outputs with PCC (Pearson Correlation Coefficient)
    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)


from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


def create_transformer_decoder_layer_preprocessor(device, weight_dtype=ttnn.bfloat16):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}

        if hasattr(torch_model, "self_attn"):
            # Preprocess self-attention parameters
            parameters["self_attn"] = {}

            # Handle QKV weights for self-attention
            if hasattr(torch_model.self_attn, "in_proj_weight"):
                # Split combined QKV weight into separate Q, K, V
                qkv_weight = torch_model.self_attn.in_proj_weight
                d_model = qkv_weight.shape[1]
                q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
                qkv_bias = torch_model.self_attn.in_proj_bias
                q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

                parameters["self_attn"]["q_weight"] = ttnn.from_torch(
                    q_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["self_attn"]["k_weight"] = ttnn.from_torch(
                    k_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["self_attn"]["v_weight"] = ttnn.from_torch(
                    v_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["self_attn"]["q_bias"] = ttnn.from_torch(
                    q_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["self_attn"]["k_bias"] = ttnn.from_torch(
                    k_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["self_attn"]["v_bias"] = ttnn.from_torch(
                    v_bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )

            if hasattr(torch_model.self_attn, "out_proj"):
                parameters["self_attn"]["out_weight"] = ttnn.from_torch(
                    torch_model.self_attn.out_proj.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["self_attn"]["out_bias"] = None
                if torch_model.self_attn.out_proj.bias is not None:
                    parameters["self_attn"]["out_bias"] = ttnn.from_torch(
                        torch_model.self_attn.out_proj.bias.reshape(1, -1),
                        dtype=weight_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

        if hasattr(torch_model, "multihead_attn"):
            # Preprocess cross-attention parameters
            parameters["multihead_attn"] = {}

            if hasattr(torch_model.multihead_attn, "in_proj_weight"):
                qkv_weight = torch_model.multihead_attn.in_proj_weight
                q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)

                parameters["multihead_attn"]["q_weight"] = ttnn.from_torch(
                    q_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["multihead_attn"]["k_weight"] = ttnn.from_torch(
                    k_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                parameters["multihead_attn"]["v_weight"] = ttnn.from_torch(
                    v_weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )

            if hasattr(torch_model.multihead_attn, "out_proj"):
                parameters["multihead_attn"]["out_weight"] = ttnn.from_torch(
                    torch_model.multihead_attn.out_proj.weight.T,
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

        # Preprocess layer normalization parameters
        for norm_name in ["norm1", "norm2", "norm3"]:
            if hasattr(torch_model, norm_name):
                norm_layer = getattr(torch_model, norm_name)
                parameters[norm_name] = {}
                parameters[norm_name]["weight"] = ttnn.from_torch(
                    norm_layer.weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
                if hasattr(norm_layer, "bias") and norm_layer.bias is not None:
                    parameters[norm_name]["bias"] = ttnn.from_torch(
                        norm_layer.bias, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                    )

        # Preprocess feedforward parameters
        if hasattr(torch_model, "linear1"):
            parameters["linear1"] = {}
            parameters["linear1"]["weight"] = ttnn.from_torch(
                torch_model.linear1.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            if torch_model.linear1.bias is not None:
                parameters["linear1"]["bias"] = ttnn.from_torch(
                    torch_model.linear1.bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )

        if hasattr(torch_model, "linear2"):
            parameters["linear2"] = {}
            parameters["linear2"]["weight"] = ttnn.from_torch(
                torch_model.linear2.weight.T, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            if torch_model.linear2.bias is not None:
                parameters["linear2"]["bias"] = ttnn.from_torch(
                    torch_model.linear2.bias.reshape(1, -1), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )

        return parameters

    return custom_preprocessor


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead, normalize_before",
    [
        # (1, 32, 256, 4, True),
        (1, 128, 256, 4, False),
        # (2, 32, 512, 8, True),
        # Add only the combinations you care about
    ],
)
def test_transformer_decoder_layer_inference(
    batch_size,
    seq_len,
    d_model,
    nhead,
    normalize_before,
):
    """Test TtTransformerDecoderLayer against PyTorch reference implementation"""

    dtype = ttnn.bfloat16
    dim_feedforward = d_model * 4

    # Initialize reference model
    reference_model = TransformerDecoderLayer(
        d_model, nhead, dim_feedforward, dropout=0.0, normalize_before=normalize_before
    ).eval()

    # Create test inputs with consistent dimensions
    tgt_input = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)
    memory_input = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)  # Same seq_len as tgt

    # Create proper positional embeddings
    query_pos = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)
    pos = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)  # Match memory_input shape

    # Get reference output with explicit None masks
    with torch.no_grad():
        ref_output = reference_model(
            tgt_input,
            memory_input,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=pos,
            query_pos=query_pos,
        )
    mesh_device = ttnn.open_device(device_id=0, l1_small_size=32768)
    preprocessor = create_transformer_decoder_layer_preprocessor(mesh_device)
    parameters = preprocessor(reference_model, "decoder_layer", {})

    # Initialize TTNN model with preprocessed parameters
    tt_model = TTTransformerDecoderLayer(
        mesh_device,
        d_model,
        nhead,
        dim_feedforward,
        normalize_before=normalize_before,
        parameters=parameters,  # Pass preprocessed weights
    )

    # print(f"{parameters=}")

    # Load weights from reference model to TTNN model
    # (In practice, this would be done through state_dict loading)
    state_dict = reference_model.state_dict()

    # Convert inputs to TTNN tensors
    tt_tgt = ttnn.from_torch(
        tgt_input.permute(1, 0, 2),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_memory = ttnn.from_torch(
        memory_input.permute(1, 0, 2),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_query_pos = ttnn.from_torch(
        query_pos.permute(1, 0, 2),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_pos = ttnn.from_torch(
        pos.permute(1, 0, 2),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN model
    tt_output, _ = tt_model(tt_tgt, tt_memory, query_pos=tt_query_pos, pos=tt_pos, return_attn_weights=False)

    if isinstance(ref_output, tuple):
        ref_output = ref_output[0]  # Get the tensor, ignore attention weights
    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = torch.permute(tt_output_torch, (1, 0, 2))

    # Compare outputs
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc=0.99)

    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    logger.info(f"Batch: {batch_size}, Seq: {seq_len}, D_model: {d_model}, Heads: {nhead}")
    logger.info(f"Normalize before: {normalize_before}")

    if passing:
        logger.info("TransformerDecoderLayer Test Passed!")
    else:
        logger.warning("TransformerDecoderLayer Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Check implementation! {pcc_message}"
