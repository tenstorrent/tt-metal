# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3RMSNorm
from models.demos.deepseek_v3.tt.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.config_helpers import NORM_CATEGORIES
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


def get_hidden_size_for_norm_category(hf_config, norm_category):
    """Helper function to determine hidden_size based on norm_category."""
    if norm_category == "attention_norm" or norm_category == "mlp_norm":
        return hf_config.hidden_size
    elif norm_category == "q_norm":
        return hf_config.q_lora_rank
    elif norm_category == "k_norm":
        return hf_config.kv_lora_rank
    else:
        raise ValueError(f"Invalid norm category: {norm_category}")


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
def reference_model(hf_config, request):
    """Get the actual DeepSeek RMSNorm model using local implementation."""
    # Get norm_category from the test function parameters
    norm_category = None
    if hasattr(request, "node"):
        # Extract norm_category from the test function's parametrization
        for key, value in request.node.callspec.params.items():
            if key == "norm_category":
                norm_category = value
                break

    if norm_category is None:
        raise ValueError("norm_category parameter not found in test")

    hidden_size = get_hidden_size_for_norm_category(hf_config, norm_category)
    return DeepseekV3RMSNorm(
        hidden_size=hidden_size,
        eps=hf_config.rms_norm_eps,
    )


@pytest.mark.parametrize(
    "mode, batch, seq_len, norm_category",
    [
        ("decode", 32, 1, "attention_norm"),  # Batch decode with distributed and sharded inputs
        ("prefill", 1, 128, "attention_norm"),  # Prefill with distributed and interleaved inputs
        ("decode", 32, 1, "q_norm"),  # Q norm test (uses q_lora_rank)
        ("prefill", 1, 128, "q_norm"),  # Q norm prefill test
        ("decode", 32, 1, "k_norm"),  # K norm test (uses kv_lora_rank)
        ("prefill", 1, 128, "k_norm"),  # K norm test prefill test
    ],
)
def test_rmsnorm_forward_pass(
    mode,
    batch,
    seq_len,
    norm_category,
    reference_model,
    hf_config,
    temp_dir,
    galaxy_or_t3k_mesh,
):
    mesh_device = galaxy_or_t3k_mesh

    """Test rmsnorm forward pass against reference model."""
    assert norm_category in NORM_CATEGORIES, f"Invalid norm category: {norm_category}"
    is_decoder_norm = norm_category == "attention_norm" or norm_category == "mlp_norm"
    # Setup: Convert weights and get weight_config
    hf_state_dict = reference_model.state_dict()
    weight_config = RMSNorm.convert_weights(
        hf_config, hf_state_dict, temp_dir, mesh_device, norm_category=norm_category
    )

    # Generate appropriate config
    if mode == "prefill":
        model_config = RMSNorm.prefill_model_config(hf_config, mesh_device, norm_category=norm_category)
    else:
        model_config = RMSNorm.decode_model_config(hf_config, mesh_device, norm_category=norm_category)

    # Create a new model state
    model_state = RMSNorm.create_state(hf_config, mesh_device=mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Determine hidden_size based on norm_category
    hidden_size = get_hidden_size_for_norm_category(hf_config, norm_category)

    # Prepare input - in decode mode batch is placed into seq_len dimension anyway
    if mode == "decode":
        torch_input = torch.randn(1, 1, batch, hidden_size)
    else:
        torch_input = torch.randn(1, 1, batch * seq_len, hidden_size)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=list(mesh_device.shape))
        if is_decoder_norm
        else ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    if is_decoder_norm and mode == "decode":
        shard_core_grid = ttnn.CoreGrid(x=4, y=7)
        sharded_memory_config = ttnn.create_sharded_memory_config(
            shape=(
                ttnn.core.roundup(tt_input.shape[0] * tt_input.shape[1] * tt_input.shape[2], ttnn.TILE_SIZE),
                tt_input.shape[3] // shard_core_grid.num_cores,
            ),
            core_grid=shard_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_input = ttnn.to_memory_config(tt_input, memory_config=sharded_memory_config)

    # TTNN forward pass
    if mode == "decode":
        tt_output = RMSNorm.forward_decode(tt_input, run_config)
    else:
        tt_output = RMSNorm.forward_prefill(tt_input, run_config)

    if is_decoder_norm:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=list(mesh_device.shape)),
        )
    else:
        tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    tt_output_torch = tt_output_torch[..., : (batch * seq_len), :]
    # Reference forward pass
    reference_output = reference_model(torch_input)

    # Compare outputs
    pcc_required = 0.99  # Embedding should be exact match (just lookup)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"Reference shape: {reference_output.shape}")
    logger.info(f"TTNN output shape: {tt_output_torch.shape}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"RMS output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
