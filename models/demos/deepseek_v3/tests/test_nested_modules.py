# models/demos/deepseek_v3/tests/test_nested_modules.py
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger

# Import from local reference files instead of HuggingFace
from torch.nn import Embedding
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.embedding_1d import Embedding1D
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.utility_functions import comp_pcc


class TestModule(AbstractModule):
    """A test module that contains an embedding layer.
    This is used to test nested module functionality.
    Assumes Embedding1D passes its tests already.
    """

    @classmethod
    def convert_weights(cls, hf_config, state_dict, output_path, mesh_device):
        return {"embedding": Embedding1D.convert_weights(hf_config, state_dict, output_path, mesh_device)}

    @classmethod
    def prefill_model_config(cls, hf_config, mesh_device):
        return {"embedding": Embedding1D.prefill_model_config(hf_config, mesh_device)}

    @classmethod
    def decode_model_config(cls, hf_config, mesh_device):
        return {"embedding": Embedding1D.decode_model_config(hf_config, mesh_device)}

    @classmethod
    def forward_prefill(cls, x, cfg):
        return Embedding1D.forward_prefill(x, cfg["embedding"])

    @classmethod
    def forward_decode(cls, x, cfg):
        return Embedding1D.forward_decode(x, cfg["embedding"])


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing."""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.fixture
def reference_model(hf_config):
    """DeepSeek just uses the standard Embedding module."""
    return Embedding(
        hf_config.vocab_size,
        hf_config.hidden_size,
        hf_config.pad_token_id,
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
        ("prefill", 128),  # Short prefill
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
    weight_config = TestModule.convert_weights(hf_config, hf_state_dict, temp_dir, mesh_device)

    # Generate appropriate config
    model_prefill_config = TestModule.prefill_model_config(hf_config, mesh_device)
    model_decode_config = TestModule.decode_model_config(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_prefill_config, run_decode_config = TestModule.run_config(
        model_prefill_config, model_decode_config, weight_config, mesh_device
    )

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
    if mode == "prefill":
        tt_output = TestModule.forward_prefill(tt_input_ids, run_prefill_config)
    else:
        tt_output = TestModule.forward_decode(tt_input_ids, run_decode_config)

    logger.info(tt_output)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            dims=(-2, -1),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )

    # Reference forward pass
    reference_output = reference_model(torch_input_ids)
    print(reference_output.shape)

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
