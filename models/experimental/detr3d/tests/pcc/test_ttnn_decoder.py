# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.detr3d.reference.model_3detr import (
    TransformerDecoderLayer,
    build_decoder,
)
from models.experimental.detr3d.ttnn.transformer_decoder import TtnnTransformerDecoderLayer
from models.experimental.detr3d.common import load_torch_model_state, DotAccessibleDict
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.ttnn.model_3detr import build_ttnn_decoder


class Tt3DetrArgs(Detr3dArgs):
    def __init__(self):
        self.parameters = None
        self.device = None


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead, normalize_before, weight_key_prefix",
    [
        (1, 128, 256, 4, True, "decoder.layers.0"),  # 7 more repeated blocks can be tested
        (1, 128, 256, 4, False, "decoder.layers.0"),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_transformer_decoder_layer_inference(
    batch_size,
    seq_len,
    d_model,
    nhead,
    normalize_before,
    weight_key_prefix,
    device,
):
    mesh_device = device
    dtype = ttnn.bfloat16
    dim_feedforward = 256

    # Initialize reference model
    reference_model = TransformerDecoderLayer(
        d_model, nhead, dim_feedforward, dropout=0.0, normalize_before=normalize_before
    )
    load_torch_model_state(reference_model, weight_key_prefix)

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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=mesh_device,
    )

    # Initialize TTNN model with preprocessed parameters
    tt_model = TtnnTransformerDecoderLayer(
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


@pytest.mark.parametrize(
    "tgt_shape, enc_features_shape",
    [
        ((128, 1, 256), (1024, 1, 256)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_transformer_decoder_inference(tgt_shape, enc_features_shape, device):
    dtype = ttnn.bfloat16

    # Build reference decoder
    torch_args = Detr3dArgs()
    reference_model = build_decoder(torch_args)
    load_torch_model_state(reference_model, "decoder")

    # Create test inputs with the specified shapes
    tgt = torch.randn(tgt_shape, dtype=torch.float32)
    enc_features = torch.randn(enc_features_shape, dtype=torch.float32)
    query_embed = torch.randn(tgt_shape, dtype=torch.float32)
    enc_pos = torch.randn(enc_features_shape, dtype=torch.float32)

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(tgt, enc_features, query_pos=query_embed, pos=enc_pos)[0]
    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    tt_args = Tt3DetrArgs()
    tt_args.device = device
    tt_args.parameters = DotAccessibleDict({"decoder": parameters})
    tt_decoder = build_ttnn_decoder(tt_args)

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

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    tt_output_torch = tt_output_torch.permute(0, 2, 1, 3)  # Convert back to [seq_len, batch, d_model]

    # Compare outputs
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc=0.999)

    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"Input shapes - tgt: {tgt.shape}, enc_features: {enc_features.shape}")
    logger.info(f"Query embed: {query_embed.shape}, enc_pos: {enc_pos.shape}")
    logger.info(f"Num layers: {torch_args.dec_nlayers}, d_model: {torch_args.dec_dim}, nhead: {torch_args.dec_nhead}")

    if passing:
        logger.info("TransformerDecoder Test Passed!")
    else:
        logger.warning("TransformerDecoder Test Failed!")

    assert passing, f"PCC value is lower than 0.99. Check implementation! {pcc_message}"
