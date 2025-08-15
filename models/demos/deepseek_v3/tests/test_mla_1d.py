# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from random import randint

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import dequantize_state_dict
from models.demos.deepseek_v3.utils.reference_forwards import reference_forward_mla as reference_forward
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import MAX_START_POS, load_state_dict
from models.utility_functions import comp_pcc


@pytest.fixture
def hf_config_short(hf_config):
    """Load DeepSeek config for testing."""
    hf_config.num_hidden_layers = 1
    hf_config.max_seq_len = 5 * 1024  # Set max sequence length for testing
    return hf_config


@pytest.fixture
def reference(hf_config_short, reset_seeds):
    """Get the actual DeepSeek MLA model using local implementation."""

    model = DeepseekV3Attention(hf_config_short, layer_idx=0).eval()
    return model


def get_cache_on_host(tt_cache: ttnn.Tensor, row_idx: int, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """
    Get the KVPE cache on the host from the TTNN cache.

    Args:
        tt_cache (ttnn.Tensor): The TTNN cache tensor.
        row_idx (int): The row index to get the cache from.
        mesh_device (ttnn.MeshDevice): The mesh device to get the cache from.
    Returns:
        torch.Tensor: The cache tensor on the host.
    """
    mesh_shape = list(mesh_device.shape)
    select_devices = slice(row_idx * mesh_shape[1], (row_idx + 1) * mesh_shape[1])

    host_cache = []
    for i, t in enumerate(ttnn.get_device_tensors(tt_cache)[select_devices]):
        host_t = t.cpu().to_torch()
        host_cache.append(host_t)
    host_cache = torch.concat(host_cache, dim=0)

    return host_cache


@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1, 32),
        ("prefill", 128, 1),
        ("prefill", 2048, 1),
    ],
)
@pytest.mark.parametrize(
    "check_cache",
    [True],
)
@pytest.mark.parametrize(
    "weights_type",
    ["random", "real"],
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
@pytest.mark.parametrize(
    "module_path",
    ["model.layers.0.self_attn"],
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size,
    check_cache,
    reference,
    hf_config_short,
    tmp_path,
    mesh_device,
    ccl,
    weights_type,
    model_path,
    module_path,
):
    # Hang workaround for large shapes
    if seq_len > 1024:
        os.environ["TT_MM_THROTTLE_PERF"] = "5"

    hf_config = hf_config_short
    mesh_shape = list(mesh_device.shape)

    dp_factor = mesh_shape[1]
    paged_config = MLA1D.get_valid_paged_config(hf_config.max_seq_len, MLA1D.MAX_BATCH_SIZE, dp_factor)

    reference_model = reference
    if weights_type == "real":
        state_dict = load_state_dict(model_path, module_path)
        dequantized_state_dict = dequantize_state_dict(state_dict, hf_config)
        reference_model.load_state_dict(dequantized_state_dict)

    if mode == "prefill":
        assert batch_size == 1, "Prefill mode only supports batch size of 1"

    ############################
    ### Set up configs
    ############################
    # Setup: Convert weights and get weight_config
    logger.info(f"Converting weights for MLA1D to {tmp_path}")
    state_dicts = [reference_model.to(torch.bfloat16).state_dict()] * mesh_shape[
        0
    ]  # Duplicate state dicts for each row in the mesh
    weight_config = MLA1D.convert_weights(hf_config, state_dicts, tmp_path, mesh_device)

    # Generate appropriate configs
    if mode == "prefill":
        model_config = MLA1D.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = MLA1D.decode_model_config(hf_config, mesh_device)

    # Create a new model state
    model_state = MLA1D.create_state(hf_config, mesh_device, paged_config, ccl)

    # Create a new model shared state
    model_shared_state = MLA1D.create_shared_state(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    ############################
    ### Torch inputs
    ############################
    logger.info("Preparing Torch inputs")
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size).to(dtype=torch.bfloat16)
    if mode == "prefill":
        position_idxs = torch.tensor([seq_len for _ in range(batch_size)])
    else:
        position_idxs = torch.tensor([randint(0, MAX_START_POS) for _ in range(batch_size)])

    ############################
    ### Torch reference
    ############################
    logger.info("Running Torch reference forward pass")
    # TODO: Save reference output?
    reference_output, reference_cache = reference_forward(
        reference_model,
        torch_input,
        position_ids=position_idxs,
        mode=mode,
    )

    ############################
    ### TTNN inputs
    ############################
    logger.info("Preparing TTNN inputs")

    user_id = None if mode == "decode" else randint(0, MLA1D.MAX_BATCH_SIZE - 1)

    if mode == "decode":
        # TT Shape: [1, seq_len, batch_size, hidden_size]
        torch_input = torch_input.permute(1, 0, 2)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_page_table, page_table = MLA1D.create_page_table(
        MLA1D.MAX_BATCH_SIZE,
        dp_factor=dp_factor,
        config=paged_config,
        mesh_device=mesh_device,
    )

    if mode == "decode":
        position_idxs_tensor = ttnn.from_torch(
            torch.tensor(position_idxs),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_shape),
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

    ############################
    ### TTNN forward pass
    ############################
    logger.info("Running TTNN forward pass")

    cur_row_idx = randint(0, mesh_shape[0] - 1)

    if mode == "prefill":
        tt_output = MLA1D.forward_prefill(tt_input, run_config, user_id, rope_tensors, tt_page_table, cur_row_idx)
    else:
        tt_output = MLA1D.forward_decode(
            tt_input, run_config, position_idxs_tensor, rope_tensors, tt_page_table, cur_row_idx
        )

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_shape)
    )[cur_row_idx, ...]

    if mode == "decode":
        # Torch Shape: [batch_size, seq_len, hidden_size]
        tt_output_torch = tt_output_torch.permute(1, 0, 2)

    ############################
    ### Validation
    ############################
    logger.info("Validating MLA 1D output")
    all_passing = True
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    all_passing = all_passing and passing

    logger.info(f"Mode: {mode}, Seq len: {seq_len}, Batch size: {batch_size}")
    logger.info(f"PCC: {pcc_message}")

    if check_cache:
        logger.info("Checking KVPE cache PCC")

        pcc_required_kvpe = 0.999

        tt_cache = get_cache_on_host(
            run_config["kvpe_cache"], cur_row_idx, mesh_device
        )  # [DP Factor * max_num_blocks, nh, block_size, head_dim + rope_head_dim]
        tt_cache = MLA1D.from_paged_cache(tt_cache, page_table, dp_factor).squeeze(
            1
        )  # [bsz, max_seq_len, head_dim + rope_head_dim]

        # Slice the correct region of the cache
        if mode == "decode":
            # Advanced indexing to get the correct position for each user
            batch_indices = torch.arange(batch_size)
            tt_cache = tt_cache[batch_indices, position_idxs, :].unsqueeze(
                1
            )  # [bsz, 1(seq_len), head_dim + rope_head_dim]
        else:
            tt_cache = tt_cache[user_id, :seq_len, :].unsqueeze(1)  # [1(bsz), seq_len, head_dim + rope_head_dim]

        tt_cache_kv = tt_cache[..., : hf_config.kv_lora_rank]
        tt_cache_pe = tt_cache[..., hf_config.kv_lora_rank :]

        ref_cache_kv = reference_cache[..., : hf_config.kv_lora_rank]  # [bsz, _, head_dim]
        ref_cache_pe = reference_cache[..., hf_config.kv_lora_rank :]  # [bsz, _, rope_head_dim]

        kv_passing, kv_pcc_message = comp_pcc(ref_cache_kv, tt_cache_kv, pcc_required_kvpe)
        pe_passing, pe_pcc_message = comp_pcc(ref_cache_pe, tt_cache_pe, pcc_required_kvpe)

        logger.info(f"Cache KV PCC: {kv_pcc_message}")
        logger.info(f"Cache PE PCC: {pe_pcc_message}")

        all_passing = all_passing and kv_passing and pe_passing

        # Check if the rest of the cache is empty
        row_idxs = list(range(mesh_shape[0]))
        row_idxs.remove(cur_row_idx)

        for r in row_idxs:
            other_cache = get_cache_on_host(run_config["kvpe_cache"], r, mesh_device)
            passing = torch.all(other_cache == 0)

            if not passing:
                logger.error(f"Cache for row {r} is not empty")
                all_passing = all_passing and passing

        # Check if cache is empty for other users in prefill mode
        if mode == "prefill":
            other_cache = get_cache_on_host(run_config["kvpe_cache"], cur_row_idx, mesh_device)

            # Convert to paged cache format so we can index by user_id
            other_cache = MLA1D.from_paged_cache(other_cache, page_table, dp_factor)

            # Mask out the user_id and it's sequence length
            mask = torch.ones_like(other_cache, dtype=torch.bool)
            mask[user_id, :, :seq_len, :] = False  # Mask out
            passing = torch.all(other_cache[mask] == 0)

            if not passing:
                logger.error(f"Cache for users other than {user_id} are not empty in prefill mode")
                all_passing = all_passing and passing

    assert (
        all_passing
    ), f"MLA output does not meet PCC requirement {pcc_required} or KVPE Cache PCC requirement {pcc_required_kvpe}"


if __name__ == "__main__":
    pytest.main([__file__])
