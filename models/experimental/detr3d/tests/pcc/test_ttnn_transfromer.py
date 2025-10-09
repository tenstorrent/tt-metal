import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.experimental.detr3d.ttnn.transformer import TTNNMultiheadAttention
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.detr3d.reference.detr3d_model import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    build_decoder,
)
from models.experimental.detr3d.ttnn.transformer import (
    TTTransformerDecoderLayer,
    TTTransformerEncoderLayer,
    build_ttnn_decoder,
)


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
    torch_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=False).eval()

    # Create TTNN model
    ttnn_mha = TTNNMultiheadAttention(d_model, nhead, device)

    # Extract and convert PyTorch weights to TTNN format
    with torch.no_grad():
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


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead, normalize_before",
    [
        (1, 128, 256, 4, True),
        (1, 128, 256, 4, False),
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
    dim_feedforward = 256

    # Initialize reference model
    reference_model = TransformerDecoderLayer(
        d_model, nhead, dim_feedforward, dropout=0.0, normalize_before=normalize_before
    ).eval()

    # Create test inputs with consistent dimensions
    tgt_input = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)
    memory_input = torch.randn(seq_len * 8, batch_size, d_model, dtype=torch.float32)

    # Create proper positional embeddings
    query_pos = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)
    pos = torch.randn(seq_len * 8, batch_size, d_model, dtype=torch.float32)

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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=mesh_device,
    )

    # Initialize TTNN model with preprocessed parameters
    tt_model = TTTransformerDecoderLayer(
        mesh_device,
        d_model,
        nhead,
        dim_feedforward,
        normalize_before=normalize_before,
        parameters=parameters,  # Pass preprocessed weights
    )

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


def compute_mask(device, xyz, radius, dist=None):
    with torch.no_grad():
        if dist is None or dist.shape[1] != xyz.shape[1]:
            dist = torch.cdist(xyz, xyz, p=2)
        # entries that are True in the mask do not contribute to self-attention
        # so points outside the radius are not considered
        mask = dist >= radius
    mask_ttnn = torch.zeros_like(mask, dtype=torch.float).masked_fill_(mask, float("-inf"))
    mask_ttnn = ttnn.from_torch(mask_ttnn, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return mask, mask_ttnn


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead, normalize_before",
    [
        (1, 2048, 256, 4, True),
        (1, 1024, 256, 4, True),
        (1, 128, 256, 4, False),
    ],
)
def test_transformer_encoder_layer_inference(
    batch_size,
    seq_len,
    d_model,
    nhead,
    normalize_before,
    device,
):
    """Test TTTransformerEncoderLayer against PyTorch reference implementation"""

    torch.manual_seed(0)
    mesh_device = device
    dtype = ttnn.bfloat16
    dim_feedforward = 128

    # Initialize reference model
    reference_model = TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.0,
        normalize_before=normalize_before,
        # activation = 'relu'
    ).eval()

    # Create test inputs
    src_input = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)
    xyz = torch.randn(batch_size, seq_len, 3, dtype=torch.float32)
    attn_mask, attn_mask_ttnn = compute_mask(mesh_device, xyz, 0.16000000000000003, None)
    # mask must be tiled to num_heads of the transformer
    bsz, n, n = attn_mask.shape
    attn_mask = attn_mask.unsqueeze(1)
    attn_mask = attn_mask.repeat(1, nhead, 1, 1)
    attn_mask = attn_mask.view(bsz * nhead, n, n)
    attn_mask_ttnn = ttnn.unsqueeze(attn_mask_ttnn, 1)

    # Create positional embeddings
    pos = torch.randn(seq_len, batch_size, d_model, dtype=torch.float32)

    # Get reference output with explicit None masks
    with torch.no_grad():
        ref_output = reference_model(
            src_input,
            src_mask=attn_mask,
            src_key_padding_mask=None,
            pos=pos,
        )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=mesh_device,
    )

    # Initialize TTNN model with preprocessed parameters
    tt_model = TTTransformerEncoderLayer(
        mesh_device,
        d_model,
        nhead,
        dim_feedforward,
        normalize_before=normalize_before,
        parameters=parameters,  # Pass preprocessed weights
    )

    # Convert inputs to TTNN tensors
    tt_src = ttnn.from_torch(
        src_input.permute(1, 0, 2),
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Run TTNN model
    tt_output = tt_model(tt_src, src_mask=attn_mask_ttnn, pos=None, return_attn_weights=False)

    if isinstance(ref_output, tuple):
        ref_output = ref_output[0]  # Get the tensor, ignore attention weights

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = torch.permute(tt_output_torch, (1, 0, 2))

    # Compare outputs
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc=0.99)

    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"Batch: {batch_size}, Seq: {seq_len}, D_model: {d_model}, Heads: {nhead}")
    logger.info(f"Normalize before: {normalize_before}")

    if passing:
        logger.info("TransformerEncoderLayer Test Passed!")
    else:
        logger.warning("TransformerEncoderLayer Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Check implementation! {pcc_message}"

    ttnn.close_device(mesh_device)


class Args:
    """Mock args class to match the build_decoder function"""

    def __init__(self):
        self.dec_dim = 256
        self.dec_nhead = 4
        self.dec_ffn_dim = 256
        self.dec_dropout = 0.0
        self.dec_nlayers = 8


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_transformer_decoder_inference(device):
    """Test TTTransformerDecoder against PyTorch reference implementation"""

    dtype = ttnn.bfloat16
    args = Args()

    # Build reference decoder
    reference_model = build_decoder(args)
    reference_model.eval()

    # Create test inputs with the specified shapes
    tgt = torch.randn(128, 1, 256, dtype=torch.float32)
    enc_features = torch.randn(1024, 1, 256, dtype=torch.float32)
    query_embed = torch.randn(128, 1, 256, dtype=torch.float32)
    enc_pos = torch.randn(1024, 1, 256, dtype=torch.float32)

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(tgt, enc_features, query_pos=query_embed, pos=enc_pos)[0]
    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    # Build TTNN decoder
    tt_decoder = build_ttnn_decoder(args, device, parameters)

    # Convert inputs to TTNN tensors (convert to batch-first format)
    tt_tgt = ttnn.from_torch(
        tgt.permute(1, 0, 2),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_enc_features = ttnn.from_torch(
        enc_features.permute(1, 0, 2),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_query_embed = ttnn.from_torch(
        query_embed.permute(1, 0, 2),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_enc_pos = ttnn.from_torch(
        enc_pos.permute(1, 0, 2),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN decoder
    tt_output, _ = tt_decoder(
        tt_tgt,
        tt_enc_features,
        query_pos=tt_query_embed,
        pos=tt_enc_pos,
        return_attn_weights=False,
    )

    # Handle intermediate results - take the last one to match reference [0] indexing
    if isinstance(tt_output, list):
        tt_output = tt_output[-1]  # Take last intermediate result

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch.permute(1, 0, 2)  # Convert back to [seq_len, batch, d_model]

    # Handle reference output format
    if isinstance(ref_output, torch.Tensor):
        # If ref_output is the stacked intermediate results, take the last one
        if ref_output.dim() == 4:  # [num_layers, seq_len, batch, d_model]
            ref_output = ref_output[-1]  # Take last layer output

    # Compare outputs
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc=0.99)

    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"Input shapes - tgt: {tgt.shape}, enc_features: {enc_features.shape}")
    logger.info(f"Query embed: {query_embed.shape}, enc_pos: {enc_pos.shape}")
    logger.info(f"Num layers: {args.dec_nlayers}, d_model: {args.dec_dim}, nhead: {args.dec_nhead}")

    if passing:
        logger.info("TransformerDecoder Test Passed!")
    else:
        logger.warning("TransformerDecoder Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Check implementation! {pcc_message}"
