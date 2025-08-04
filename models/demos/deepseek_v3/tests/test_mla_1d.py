# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from random import randint

import pytest
import torch
from loguru import logger

# Import from local reference files instead of HuggingFace
from transformers import DynamicCache

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc

MAX_START_POS = 512


@pytest.fixture
def hf_config_short(hf_config):
    """Load DeepSeek config for testing."""
    hf_config.num_hidden_layers = 1
    hf_config.max_seq_len = 5 * 1024  # Set max sequence length for testing
    return hf_config


@pytest.fixture
def reference(hf_config_short, reset_seeds):
    """Get the actual DeepSeek MLA model using local implementation."""

    model = DeepseekV3Attention(hf_config_short, layer_idx=0)
    model.init_weights_with_random()  # Initialize weights with random values
    return model


def reference_forward(
    reference_model: DeepseekV3Attention,
    torch_input: torch.Tensor,
    position_ids: torch.LongTensor,
    mode: str,
) -> torch.Tensor:
    """Run the reference model forward pass.

    This function specifically uses MLA with the "absorption" implementation,
    called with `forward_mla`, which is simply an extension on top of HF's DeepSeek implementation.

    Notes for Decode Mode:
        NOTE: This function currently does not support multi-iteration decoding.

        It is critical to note that since the HF implementation uses DynamicCache, we cannot just
        perform decode on users with arbitrry position_ids. Instead, if we want to simulate this situation,
        we need to pad the input to the largest position id in the batch, and then use it for the forward pass.

        This has the same effect as doing prefill. As such, on the output side, we must also
        slice the output to only return the last token for each user.

        The reference cache is also sliced in a similar manner to only return the last token's cache.

    Notes on position_ids_expanded:
        The HF implementation uses position_ids_expanded, which is a tensor of shape [bsz, q_len].


    Args:
        reference_model (DeepseekV3Attention): The reference model to run.
        torch_input (torch.Tensor): The input tensor to the model.
        position_ids (torch.LongTensor): The position ids for the input.
        mode (str): The mode of operation, either "decode" or "prefill".
    Returns:
        torch.Tensor: The output tensor from the model.
    """

    config = reference_model.config
    bsz, q_len, dim = torch_input.shape
    torch_input = torch_input.to(dtype=torch.float32)

    assert bsz == position_ids.shape[0], "Batch size of input and position_ids must match"

    # Generate the cache
    cache = DynamicCache()

    if mode == "prefill":
        # Create the mask
        mask = torch.triu(torch.full((bsz, 1, q_len, q_len), float("-inf")), diagonal=1)
    else:
        assert q_len == 1, "Decode mode should have sequence length of 1"

        # Perform special padding for decode mode
        max_position_idx = position_ids.max().item()
        new_torch_input = torch.zeros(bsz, max_position_idx + 1, dim)
        for b in range(bsz):
            pos = position_ids[b].item()
            new_torch_input[b, pos] = torch_input[b, 0]
        torch_input = new_torch_input
        q_len = torch_input.shape[1]  # Update q_len to the new sequence length

        # Create the mask
        mask = torch.full((bsz, 1, q_len, q_len), float("-inf"))
        for i in range(bsz):
            usable_len = position_ids[i].item() + 1
            mask[i, 0, :usable_len, :usable_len] = torch.triu(
                torch.full((usable_len, usable_len), float("-inf")), diagonal=1
            )

    position_ids_expanded = torch.arange(0, q_len, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)

    out, _, past_key_value = reference_model.forward_mla(
        hidden_states=torch_input,
        attention_mask=mask,
        position_ids=position_ids_expanded,
        past_key_value=cache,
    )
    cache = past_key_value.key_cache[reference_model.layer_idx].squeeze(1)

    if mode == "decode":
        # Get last token
        batch_indices = torch.arange(bsz)
        out = out[batch_indices, position_ids, :].unsqueeze(1)  # [bsz, 1, hidden_size]
        cache = cache[batch_indices, position_ids, :].unsqueeze(1)  # [bsz, 1, head_dim + rope_head_dim]

    return out, cache


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
    for i, t in enumerate(ttnn.get_device_tensors(tt_cache)[select_devices]):  # Only get from first row of mesh_device
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
    hf_config_short,
    tmp_path,
    mesh_device,
):
    hf_config = hf_config_short
    mesh_shape = list(mesh_device.shape)

    dp_factor = mesh_shape[1] if mode == "decode" else 1
    paged_config = (
        MLA1D.get_valid_paged_config(hf_config.max_seq_len, batch_size, dp_factor) if mode == "decode" else None
    )

    reference_model = reference

    if mode == "prefill":
        assert batch_size == 1, "Prefill mode only supports batch size of 1"

    ############################
    ### Set up configs
    ############################
    # Setup: Convert weights and get weight_config
    logger.info(f"Converting weights for MLA1D to {tmp_path}")
    state_dicts = [reference_model.state_dict()] * mesh_shape[0]  # Duplicate state dicts for each row in the mesh
    weight_config = MLA1D.convert_weights(hf_config, state_dicts, tmp_path, mesh_device)

    # Generate appropriate configs
    ccl = CCL1D(mesh_row)
    if mode == "prefill":
        model_config = MLA1D.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = MLA1D.decode_model_config(hf_config, mesh_device)

    # Create a new model state
    model_state = MLA1D.create_state(hf_config, mesh_device, dp_factor, paged_config, ccl)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    ############################
    ### Torch inputs
    ############################

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size).to(dtype=torch.bfloat16)
    if mode == "prefill":
        position_idxs = torch.tensor([seq_len for _ in range(batch_size)])
    else:
        position_idxs = torch.tensor([randint(0, MAX_START_POS) for _ in range(batch_size)])

    ############################
    ### Torch reference
    ############################
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
    if mode == "decode":
        # TT Shape: [1, seq_len, batch_size, hidden_size]
        torch_input = torch_input.permute(1, 0, 2)
    torch_input = torch_input.unsqueeze(0)

    # TODO: Need to handle padding for batch
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_page_table, page_table = None, None

    if mode == "decode":
        position_idxs_tensor = ttnn.from_torch(
            torch.tensor(position_idxs),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(None, 0), mesh_shape=mesh_shape
            ),  # TODO: Shard on batch when DP
            dtype=ttnn.int32,
        )

        tt_page_table, page_table = MLA1D.create_page_table(
            batch_size,
            dp_factor=dp_factor,
            config=paged_config,
            mesh_device=mesh_device,
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
    logger.info("Running TTNN forward pass")

    cur_row_idx = randint(0, mesh_shape[0] - 1)

    if mode == "prefill":
        tt_output = MLA1D.forward_prefill(tt_input, run_config, user_id, rope_tensors)
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

        if mode == "decode":
            tt_cache = get_cache_on_host(
                run_config["kvpe_cache"], cur_row_idx, mesh_device
            )  # [DP Factor * max_num_blocks, nh, block_size, head_dim + rope_head_dim]
            tt_cache = MLA1D.from_paged_cache(tt_cache, page_table, dp_factor).squeeze(
                1
            )  # [bsz, max_seq_len, head_dim + rope_head_dim]
        else:
            tt_cache = get_cache_on_host(run_config["kvpe_cache"], cur_row_idx, mesh_device)[
                : MLA1D.MAX_BATCH_SIZE, ...
            ].squeeze(
                1
            )  # [bsz, max_seq_len, head_dim + rope_head_dim]

        # Slice the correct region of the cache
        if mode == "decode":
            batch_indices = torch.arange(batch_size)
            tt_cache = tt_cache[batch_indices, position_idxs, :].unsqueeze(1)  # [bsz, 1, head_dim + rope_head_dim]
        else:
            tt_cache = tt_cache[:batch_size, :seq_len, :]

        tt_cache_kv = tt_cache[..., : hf_config.kv_lora_rank]
        tt_cache_pe = tt_cache[..., hf_config.kv_lora_rank :]

        ref_cache_kv = reference_cache[..., : hf_config.kv_lora_rank]  # [bsz, _, head_dim]
        ref_cache_pe = reference_cache[..., hf_config.kv_lora_rank :]  # [bsz, _, rope_head_dim]

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
