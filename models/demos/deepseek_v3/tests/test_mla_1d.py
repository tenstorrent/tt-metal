# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path
from random import randint

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.rope import RotarySetup

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3_impl.model import MLA, ModelArgs, precompute_freqs_cis
from models.utility_functions import comp_pcc, get_mesh_device


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing"""
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    config.max_seq_len = 5 * 1024  # Set max sequence length for testing

    return config


@pytest.fixture
def reference(hf_config, reset_seeds):
    """Get the actual DeepSeek MLA model using local implementation."""

    config_path = "models/demos/deepseek_v3_impl/configs/config_671B.json"
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))

    model_args.n_heads = hf_config.num_attention_heads
    model_args.max_seq_len = hf_config.max_seq_len

    model = MLA(model_args)
    model.init_weights_with_random()

    return model_args, model


def get_cache_on_host(tt_cache, mesh_device):
    """
    Get the KVPE cache on the host from the TTNN cache.
    This specifically retrieves the first row of the mesh_device,
    which currently only supports DP on the first row.
    """
    host_cache = []
    for i, t in enumerate(
        ttnn.get_device_tensors(tt_cache)[: list(mesh_device.shape)[0]]
    ):  # Only get from first row of mesh_device
        host_t = t.cpu().to_torch()
        host_cache.append(host_t)
    host_cache = torch.concat(host_cache, dim=0)

    return host_cache


# Unit Tests
@pytest.mark.parametrize(
    "mesh_device",
    [
        get_mesh_device(),
    ],
    indirect=True,
)
def test_convert_weights(reference, hf_config, temp_dir, mesh_device):
    """Test that weights are correctly converted to TTNN format."""

    logger.info(f"Converting weights for MLA1D to {temp_dir}")

    _, reference_model = reference

    # Convert weights - now returns weight_config
    weight_config = MLA1D.convert_weights(hf_config, reference_model.state_dict(), temp_dir, mesh_device)

    assert "wq_a" in weight_config
    assert "wq_b" in weight_config
    assert "wkv_a" in weight_config
    assert "wkv_b1" in weight_config
    assert "wkv_b2" in weight_config
    assert "wo" in weight_config


# Integration Tests
@pytest.mark.parametrize(
    "mesh_device",
    [get_mesh_device()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1, 32),
        # ("prefill", 128, 1),
    ],
)
@pytest.mark.parametrize(
    "check_cache",
    [True],
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
def test_forward_pass(
    mode,
    seq_len,
    batch_size,
    check_cache,
    reference,
    hf_config,
    temp_dir,
    mesh_device,
):
    mesh_shape = list(mesh_device.shape)

    reference_args, reference_model = reference

    if mode == "prefill":
        assert batch_size == 1, "Prefill mode only supports batch size of 1"

    ############################
    ### Set up configs
    ############################
    # Setup: Convert weights and get weight_config
    logger.info(f"Converting weights for MLA1D to {temp_dir}")
    weight_config = MLA1D.convert_weights(hf_config, reference_model.state_dict(), temp_dir, mesh_device)

    # Generate appropriate configs
    ccl = CCL1D(mesh_device, hf_config)
    model_prefill_config = MLA1D.prefill_model_config(hf_config, mesh_device, ccl)
    model_decode_config = MLA1D.decode_model_config(hf_config, mesh_device, ccl)

    # Create RunConfig using both weight_config and model_config
    run_prefill_config, run_decode_config = MLA1D.run_config(
        model_prefill_config, model_decode_config, weight_config, mesh_device
    )
    run_config = run_prefill_config if mode == "prefill" else run_decode_config

    ############################
    ### Torch inputs
    ############################
    start_pos = randint(0, hf_config.max_seq_len) if mode == "decode" else 0

    freqs_cis = precompute_freqs_cis(reference_args)
    if mode == "prefill":
        freqs_cis = freqs_cis[:seq_len, :]
    else:
        freqs_cis = freqs_cis[start_pos, :]

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)
    torch_input = torch_input.to(dtype=torch.bfloat16)

    # Generate the mask
    mask = None
    if mode == "prefill":
        # For prefill, we need to create a causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    ############################
    ### Torch reference
    ############################
    reference_output = reference_model(
        torch_input,
        start_pos=start_pos,
        freqs_cis=freqs_cis,
        mask=mask,
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

    user_id = None if mode == "decode" else 0  # TODO: randint(0, MLA1D.MAX_BATCH_SIZE)

    ############################
    ### TTNN forward pass
    ############################
    if mode == "prefill":
        tt_output = MLA1D.forward_prefill(tt_input, user_id, rope_tensors, run_config)
        tt_output_torch = ttnn.to_torch(tt_output)
    else:
        tt_output = MLA1D.forward_decode(tt_input, position_idxs_tensor, rope_tensors, run_config)
        tt_output_torch = ttnn.to_torch(
            tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-1, 0), mesh_shape=mesh_shape)
        )[:1, ...]
        tt_output_torch = tt_output_torch.squeeze(0).permute(1, 0, 2)

    ############################
    ### Validation
    ############################
    logger.info("Validating MLA 1D output")
    all_passing = True
    pcc_required = 0.98  # Slightly lower due to bfloat conversions
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    all_passing = all_passing and passing

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    if check_cache:
        logger.info("Checking KVPE cache PCC")

        pcc_required_kvpe = 0.98
        if mode == "prefill":
            range_to_check = range(start_pos, seq_len)
        else:
            range_to_check = range(start_pos, start_pos + 1)

        tt_cache = get_cache_on_host(run_config["kvpe_cache"], mesh_device)[: MLA1D.MAX_BATCH_SIZE, ...].squeeze(
            1
        )  # [bsz, max_seq_len, head_dim + rope_head_dim]
        tt_cache_kv = tt_cache[:, range_to_check, : hf_config.kv_lora_rank]
        tt_cache_pe = tt_cache[:, range_to_check, hf_config.kv_lora_rank :]

        ref_cache_kv = reference_model.kv_cache[:, range_to_check, :]  # [bsz, _, head_dim]
        ref_cache_pe = reference_model.pe_cache[:, range_to_check, :]  # [bsz, _, rope_head_dim]

        kv_passing, kv_pcc_message = comp_pcc(ref_cache_kv, tt_cache_kv, pcc_required_kvpe)
        pe_passing, pe_pcc_message = comp_pcc(ref_cache_pe, tt_cache_pe, pcc_required_kvpe)

        logger.info(f"Cache KV PCC: {kv_pcc_message}")
        logger.info(f"Cache PE PCC: {pe_pcc_message}")

        all_passing = all_passing and kv_passing and pe_passing

    assert (
        all_passing
    ), f"MLA output does not meet PCC requirement {pcc_required} or KVPE Cache PCC requirement {pcc_required_kvpe}"


if __name__ == "__main__":
    pytest.main([__file__])
