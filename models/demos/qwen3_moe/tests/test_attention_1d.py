# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from random import randint

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.qwen3_moe.reference.modeling_qwen3_moe import Qwen3MoeAttention as ReferenceAttention
from models.demos.qwen3_moe.reference.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding as ReferenceRope
from models.demos.qwen3_moe.tt.attention_1d import Attention1D
from models.demos.qwen3_moe.tt.ccl_1d import CCL1D
from models.demos.qwen3_moe.tt.rope import RotarySetup
from models.demos.qwen3_moe.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


@pytest.fixture
def hf_config_short(hf_config):
    """Load Qwen3MoE config for testing."""
    hf_config.num_hidden_layers = 1
    hf_config.max_seq_len = 5 * 1024  # Set max sequence length for testing
    return hf_config


@pytest.fixture
def reference_model(hf_config_short, reset_seeds):
    """Get the actual Qwen3MoE Attention model using local implementation."""
    hf_config = hf_config_short
    model = ReferenceAttention(hf_config, layer_idx=0)
    return model


def get_cache_on_host(tt_cache, mesh_device):
    """
    Get the KV cache on the host from the TTNN cache.
    This specifically retrieves the first row of the mesh_device.
    """
    host_cache = []
    for i, t in enumerate(
        ttnn.get_device_tensors(tt_cache)[: list(mesh_device.shape)[0]]
    ):  # Only get from first row of mesh_device
        host_t = t.cpu().to_torch()
        host_cache.append(host_t)
    host_cache = torch.concat(host_cache, dim=0)

    return host_cache


@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1, 32),
        ("prefill", 128, 1),
    ],
)
@pytest.mark.parametrize(
    "check_cache",
    [False],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
# @pytest.mark.skip(reason="FIXME: Reduce-scatter was unable to identify either a sender or receiver device ID")
def test_forward_pass(
    mode,
    seq_len,
    batch_size,
    check_cache,
    reference_model,
    hf_config_short,
    temp_dir,
    mesh_device,
):
    hf_config = hf_config_short
    mesh_shape = list(mesh_device.shape)

    if mode == "prefill":
        assert batch_size == 1, "Prefill mode only supports batch size of 1"

    ############################
    ### Set up configs
    ############################
    # Setup: Convert weights and get weight_config
    logger.info(f"Converting weights for Attention1D to {temp_dir}")
    weight_config = Attention1D.convert_weights(hf_config, reference_model.state_dict(), temp_dir, mesh_device)

    # Generate appropriate configs
    ccl = CCL1D(hf_config, mesh_device)
    if mode == "prefill":
        model_config = Attention1D.prefill_model_config(hf_config, mesh_device, ccl)
    else:
        model_config = Attention1D.decode_model_config(hf_config, mesh_device, ccl)

    # Create a new model state
    model_state = Attention1D.create_state(hf_config, mesh_device=mesh_device)

    # Create RunConfig using both weight_config and model_config
    logger.info(f"model_config = {model_config}")
    logger.info(f"weight_config = {weight_config}")
    # logger.info(f"model_state = {model_state}")
    run_config = create_run_config(model_config, weight_config, model_state)

    ############################
    ### Torch inputs
    ############################
    start_pos = randint(0, hf_config.max_seq_len // 2) if mode == "decode" else 0

    position_ids = torch.arange(hf_config.max_seq_len).unsqueeze(0)
    rotary_emb = ReferenceRope(hf_config)

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size).to(dtype=torch.bfloat16)
    position_embeddings = rotary_emb(torch_input, position_ids)

    cos, sin = position_embeddings

    ############################
    ### Torch reference
    ############################
    # TODO: Save reference output?
    cur_position_embeddings = (
        cos[:, start_pos : start_pos + 1, :],  # shape: [1, 1, dim]
        sin[:, start_pos : start_pos + 1, :],
    )
    reference_model = reference_model.to(dtype=torch.bfloat16)
    reference_output, _ = reference_model(
        torch_input,
        position_embeddings=cur_position_embeddings,
        attention_mask=None,
        past_key_value=DynamicCache(),
        cache_position=start_pos,
    )

    ############################
    ### TTNN inputs
    ############################
    if mode == "decode":
        # TT Shape: [1, seq_len, batch_size, hidden_size]
        torch_input = torch_input.permute(1, 0, 2)
    torch_input = torch_input.unsqueeze(0)

    # TODO: Need to handle padding for batch
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=mesh_shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    position_idxs = (
        [start_pos] * batch_size if mode == "decode" else None
    )  # TODO: Only support same position for all users

    if mode == "decode":
        position_idxs_tensor = ttnn.from_torch(
            torch.tensor(position_idxs),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(None, None), mesh_shape=mesh_shape
            ),  # TODO: Shard on batch when DP
            dtype=ttnn.int32,
        )

    # RoPE stuff
    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size=batch_size,
        hf_config=hf_config,
    )
    rope_setup_dp = RotarySetup(
        device=mesh_device,
        batch_size=batch_size,
        hf_config=hf_config,
        use_dp=True,
    )

    if mode == "prefill":
        rot_mats = rope_setup.get_rot_mats_table(seq_len)
        rot_mats_dp = [None, None, None]
    else:
        rot_idxs = torch.tensor(position_idxs, dtype=torch.int32)
        rot_mats = rope_setup.get_rot_mats(rot_idxs)
        rot_mats_dp = rope_setup_dp.get_rot_mats(rot_idxs)
    rope_tensors = {
        "cos_matrix": rot_mats[0],
        "sin_matrix": rot_mats[1],
        "trans_matrix": rot_mats[2],
        "cos_matrix_dp": rot_mats_dp[0],
        "sin_matrix_dp": rot_mats_dp[1],
        "trans_matrix_dp": rot_mats_dp[2],
    }

    user_id = None if mode == "decode" else 0  # TODO: randint(0, Attention1D.MAX_BATCH_SIZE)

    ############################
    ### TTNN forward pass
    ############################
    if mode == "prefill":
        tt_output = Attention1D.forward_prefill(tt_input, user_id, rope_tensors, run_config)
    else:
        tt_output = Attention1D.forward_decode(tt_input, position_idxs_tensor, rope_tensors, run_config)
    logger.info(f"tt_output shape: {tt_output.shape}")
    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-1, 0), mesh_shape=mesh_shape)
    )[:1, ...]

    if mode == "decode":
        # Torch Shape: [batch_size, seq_len, hidden_size]
        tt_output_torch = tt_output_torch.squeeze(0).permute(1, 0, 2)
    logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
    logger.info(f"tt_output_torch : {tt_output_torch}")

    ############################
    ### Validation
    ############################
    logger.info("Validating Attention 1D output")
    all_passing = True
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    all_passing = all_passing and passing

    logger.info(f"Mode: {mode}, Seq len: {seq_len}, Batch size: {batch_size}")
    logger.info(f"PCC: {pcc_message}")

    if check_cache:
        logger.info("Checking KV cache PCC")

        pcc_required_kv = 0.999
        range_to_check = range(start_pos, start_pos + seq_len)

        tt_k_cache = get_cache_on_host(run_config["k_cache"], mesh_device)[: Attention1D.MAX_BATCH_SIZE, ...].squeeze(
            1
        )  # [bsz, max_seq_len, head_dim + rope_head_dim]
        tt_v_cache = get_cache_on_host(run_config["v_cache"], mesh_device)[: Attention1D.MAX_BATCH_SIZE, ...].squeeze(1)

        ref_cache_k = reference_model.k_cache[:, :, range_to_check, :]  # [bsz, _, head_dim]
        ref_cache_v = reference_model.v_cache[:, :, range_to_check, :]  # [bsz, _, head_dim]

        k_passing, k_pcc_message = comp_pcc(ref_cache_k, tt_k_cache, pcc_required_kv)
        v_passing, v_pcc_message = comp_pcc(ref_cache_v, tt_v_cache, pcc_required_kv)

        logger.info(f"Cache K PCC: {k_pcc_message}")
        logger.info(f"Cache V PCC: {v_pcc_message}")

        all_passing = all_passing and k_passing and v_passing

    assert (
        all_passing
    ), f"Attention 1D output does not meet PCC requirement {pcc_required} or KV Cache PCC requirement {pcc_required_kv if check_cache else 'N/A'}: {pcc_message}"


if __name__ == "__main__":
    pytest.main([__file__])
