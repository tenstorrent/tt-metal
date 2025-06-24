# models/demos/deepseek_v3/tests/test_embedding_1d.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.reference.embeddings_and_head import DeepseekV3Embeddings
from models.demos.deepseek_v3.tt.embedding_1d import Embedding_1D
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing"""
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
    return config


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek Embedding model using local implementation."""
    return DeepseekV3Embeddings(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        padding_idx=getattr(hf_config, "pad_token_id", None),
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 1),  # Single token decode
        ("decode", 32),  # Batch decode
        ("prefill", 128),  # Short prefill
        ("prefill", 512),  # Medium prefill
        ("prefill", 2048),  # Long prefill
    ],
)
def test_embedding_forward_pass(
    mode,
    seq_len,
    reference_model,
    hf_config,
    temp_dir,
    mesh_device,
):
    """Test embedding forward pass against reference model."""
    # Setup: Convert weights and get weight_config
    hf_state_dict = reference_model.state_dict()
    weight_config = Embedding_1D.convert_weights(hf_config, hf_state_dict, temp_dir, mesh_device)

    # Generate appropriate config
    if mode == "prefill":
        model_config = Embedding_1D.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = Embedding_1D.decode_model_config(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, mesh_device)

    # Instantiate the model
    tt_embedding = Embedding_1D(hf_config, mesh_device)

    # Prepare input - in decode mode batch is placed into seq_len dimension anyway
    torch_input_ids = torch.randint(0, min(1000, hf_config.vocab_size), (1, seq_len))

    tt_input_ids = ttnn.from_torch(
        torch_input_ids,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # TTNN forward pass
    tt_output = tt_embedding.forward(tt_input_ids, run_config, mesh_device)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Reference forward pass
    reference_output = reference_model(torch_input_ids)

    # Compare outputs
    pcc_required = 0.99  # Embedding should be exact match (just lookup)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"Reference shape: {reference_output.shape}")
    logger.info(f"TTNN output shape: {tt_output_torch.shape}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"Embedding output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input_ids)


if __name__ == "__main__":
    pytest.main([__file__])
