# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.embedding_model import EmbeddingTransformer
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.pooling import MeanPooling


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 512, 1024),
)
@pytest.mark.parametrize(
    "batch_size",
    (1, 2),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mean_pooling(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc):
    """Test MeanPooling layer functionality"""
    dtype = ttnn.bfloat16

    # Create model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, cache_hf=True)

    # Initialize pooling layer
    pooling = MeanPooling(mesh_device=mesh_device, args=model_args)

    # Create test input: [batch, 1, seq_len, hidden_dim]
    hidden_dim = model_args.dim
    input_tensor = torch.randn(batch_size, 1, seq_len, hidden_dim, dtype=torch.bfloat16)

    # Create attention mask: [batch, 1, seq_len, 1] - all 1s (no masking)
    attention_mask = torch.ones(batch_size, 1, seq_len, 1, dtype=torch.bfloat16)

    # Convert to ttnn tensors
    tt_input = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_attention_mask = ttnn.from_torch(
        attention_mask,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run pooling
    tt_output = pooling(tt_input, attention_mask=tt_attention_mask)

    # Convert back to torch
    output_tensor = ttnn.to_torch(tt_output)

    # Expected output shape: [batch, 1, 1, hidden_dim]
    assert output_tensor.shape == (batch_size, 1, 1, hidden_dim)

    # Compute reference mean pooling
    ref_output = torch.mean(input_tensor, dim=2, keepdim=True)
    assert output_tensor.shape == ref_output.shape

    # Check numerical accuracy
    passing, pcc = comp_pcc(output_tensor, ref_output, pcc=0.99)
    logger.info(f"MeanPooling PCC: {pcc}")
    assert passing, f"MeanPooling PCC {pcc} is too low"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 512),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_embedding_transformer(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc):
    """Test EmbeddingTransformer model functionality"""
    dtype = ttnn.bfloat8_b

    # Create model args for Qwen3-Embedding-8B
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, cache_hf=True)
    model_args.model_name = "Qwen3-Embedding-8B"
    model_args.n_layers = 1  # Use only 1 layer for testing

    # Load state dict
    state_dict = model_args.load_state_dict()

    # Create embedding transformer
    embedding_model = EmbeddingTransformer(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    # Create test input tokens
    tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))

    # Prepare prefill inputs
    (
        tt_tokens,
        tt_rot_mats_global,
        tt_rot_mats_local,
        tt_page_table,
        tt_chunk_page_table,
    ) = embedding_model.prepare_inputs_prefill(tokens)

    # Create attention mask
    attention_mask = torch.ones(batch_size, 1, seq_len, 1, dtype=torch.float32)
    tt_attention_mask = ttnn.from_torch(
        attention_mask,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run forward pass
    tt_embeddings = embedding_model.forward(
        x=tt_tokens,
        current_pos=None,
        rot_mats_global=tt_rot_mats_global,
        rot_mats_local=tt_rot_mats_local,
        mode="prefill",
        attention_mask=tt_attention_mask,
    )

    # Convert to torch
    embeddings = ttnn.to_torch(tt_embeddings)

    # Check output shape: [batch, 1, 1, hidden_dim]
    expected_shape = (batch_size, 1, 1, model_args.dim)
    assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"

    # Check that embeddings are not all zeros
    assert not torch.allclose(embeddings, torch.zeros_like(embeddings)), "Embeddings should not be all zeros"

    logger.info(f"EmbeddingTransformer output shape: {embeddings.shape}")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (64, 128),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_pooling_with_attention_mask(seq_len, mesh_device, reset_seeds, ensure_gc):
    """Test MeanPooling with attention mask (partial masking)"""
    dtype = ttnn.bfloat16
    batch_size = 1

    # Create model args
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, cache_hf=True)

    # Initialize pooling layer
    pooling = MeanPooling(mesh_device=mesh_device, args=model_args)

    # Create test input
    hidden_dim = model_args.dim
    input_tensor = torch.randn(batch_size, 1, seq_len, hidden_dim, dtype=torch.bfloat16)

    # Create attention mask with some positions masked (simulating padding)
    attention_mask = torch.ones(batch_size, 1, seq_len, 1, dtype=torch.bfloat16)
    # Mask last half of sequence
    attention_mask[:, :, seq_len // 2 :, :] = 0

    # Convert to ttnn tensors
    tt_input = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_attention_mask = ttnn.from_torch(
        attention_mask,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run pooling
    tt_output = pooling(tt_input, attention_mask=tt_attention_mask)

    # Convert back to torch
    output_tensor = ttnn.to_torch(tt_output)

    # Expected output shape: [batch, 1, 1, hidden_dim]
    assert output_tensor.shape == (batch_size, 1, 1, hidden_dim)

    # Compute reference mean pooling with mask
    masked_input = input_tensor * attention_mask.unsqueeze(-1)  # Broadcast to hidden_dim
    mask_sum = attention_mask.sum(dim=2, keepdim=True)
    mask_sum = torch.clamp(mask_sum, min=1e-9)  # Avoid division by zero
    ref_output = masked_input.sum(dim=2, keepdim=True) / mask_sum.unsqueeze(-1)

    assert output_tensor.shape == ref_output.shape

    # Check numerical accuracy
    passing, pcc = comp_pcc(output_tensor, ref_output, pcc=0.99)
    logger.info(f"MeanPooling with mask PCC: {pcc}")
    assert passing, f"MeanPooling with mask PCC {pcc} is too low"
