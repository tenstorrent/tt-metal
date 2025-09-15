# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    dequantize_state_dict,
    get_model_config,
    load_state_dict,
    paged_cache_from_torch,
    run_reference_with_attention,
    torch_cache_from_paged,
    torch_cache_from_transformers_single_layer,
)


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

    host_cache = []
    for i, t in enumerate(ttnn.get_device_tensors(tt_cache)[row_idx * mesh_shape[1] : (row_idx + 1) * mesh_shape[1]]):
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
    [None, "model.layers.0.self_attn"],
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size,
    hf_config_short,
    tmp_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    set_deterministic_env,
):
    # Hardcoded arguments; can later change them to test arguments if needed
    check_cache = True
    layer_idx = 0

    # Hang workaround for large shapes
    if seq_len > 1024:
        os.environ["TT_MM_THROTTLE_PERF"] = "5"

    # Check params
    if mode == "prefill":
        assert batch_size == 1, "Prefill only supports a batch size of 1"
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"

    # Get reference IO
    logger.info("Setting up reference IO")
    if module_path is None:
        reference_model = DeepseekV3Attention(hf_config_short, layer_idx=layer_idx).eval().to(torch.bfloat16)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config_short.quantization_config["weight_block_size"],
        )

        torch_input = torch.randn(batch_size, seq_len, hf_config_short.hidden_size, dtype=torch.bfloat16)
        if mode == "prefill":
            position_ids = torch.tensor([seq_len])
        else:
            position_ids = torch.randint(0, hf_config_short.max_seq_len - 1, (batch_size,))
        reference_output, input_cache, output_cache = run_reference_with_attention(
            reference_model, torch_input, position_ids, layer_idx, hf_config_short, mode, check_cache
        )
        input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
        output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)
    else:
        # TODO: once reference IO is integrated, remove generating it
        reference_model = DeepseekV3Attention(hf_config_short, layer_idx=layer_idx).eval().to(torch.bfloat16)
        state_dict = load_state_dict(model_path, module_path)
        dequantized_state_dict = dequantize_state_dict(state_dict, hf_config_short)
        reference_model.load_state_dict(dequantized_state_dict)

        torch_input = torch.randn(batch_size, seq_len, hf_config_short.hidden_size, dtype=torch.bfloat16)
        if mode == "prefill":
            position_ids = torch.tensor([seq_len])
        else:
            position_ids = torch.randint(0, hf_config_short.max_seq_len - 1, (batch_size,))
        reference_output, input_cache, output_cache = run_reference_with_attention(
            reference_model, torch_input, position_ids, layer_idx, hf_config_short, mode, check_cache
        )
        input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
        output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)

    # Set up page config
    logger.info("Setting up model configs")
    _, dp_factor = mesh_device.shape
    user_id = None if mode == "decode" else torch.randint(0, MAX_BATCH_SIZE, ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, MAX_BATCH_SIZE, dp_factor)
    paged_input_cache, torch_page_table = paged_cache_from_torch(input_cache, dp_factor, paged_config, user_id)

    # Set up model config
    weight_config = MLA1D.convert_weights(hf_config_short, [state_dict] * mesh_device.shape[0], tmp_path, mesh_device)
    model_config = get_model_config(MLA1D, mode, hf_config_short, mesh_device)
    model_state = MLA1D.create_state(
        hf_config_short, paged_config, mesh_device, ccl, (paged_input_cache,) * mesh_device.shape[0]
    )
    run_config = create_run_config(model_config, weight_config, model_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [1, seq_len, batch_size, hidden_size]

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    position_ids_tensor = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_device.shape),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    tt_page_table = MLA1D.create_page_table(torch_page_table, paged_config, mesh_device)

    # RoPE setup
    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size=batch_size,
        hf_config=hf_config_short,
    )

    if mode == "prefill":
        rot_mats = rope_setup.get_rot_mats_table(seq_len)
    else:
        rot_idxs = torch.tensor(position_ids, dtype=torch.int32)
        rot_mats = rope_setup.get_rot_mats(rot_idxs)
    rope_tensors = {
        "cos_matrix": rot_mats[0],
        "sin_matrix": rot_mats[1],
        "trans_matrix": rot_mats[2],
    }

    # Forward pass
    logger.info("Running TTNN forward pass")

    cur_row_idx = torch.randint(0, mesh_device.shape[0], ()).item()
    if mode == "prefill":
        tt_output = MLA1D.forward_prefill(tt_input, user_id, cur_row_idx, run_config, rope_tensors, tt_page_table)
    else:
        tt_output = MLA1D.forward_decode(
            tt_input, position_ids_tensor, cur_row_idx, run_config, rope_tensors, tt_page_table
        )

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
    )[cur_row_idx]
    if mode == "decode":
        tt_output_torch = tt_output_torch.permute(1, 0, 2)  # Torch Shape: [batch_size, seq_len, hidden_size]

    # Check output PCC
    logger.info("Validating output")
    all_passing = True
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    all_passing &= passing

    logger.info(f"Mode: {mode}, Seq len: {seq_len}, Batch size: {batch_size}")
    logger.info(f"PCC: {pcc_message}")

    # Check cache PCC
    if check_cache:
        logger.info("Checking KVPE cache PCC")

        pcc_required_kvpe = 0.999

        tt_cache = get_cache_on_host(
            run_config["kvpe_cache"], cur_row_idx, mesh_device
        )  # [DP Factor * max_num_blocks, nh, block_size, head_dim + rope_head_dim]
        tt_cache = torch_cache_from_paged(tt_cache, torch_page_table, dp_factor).squeeze(
            1
        )  # [bsz, max_seq_len, head_dim + rope_head_dim]
        output_cache = output_cache.squeeze(1)  # [bsz, max_seq_len, head_dim + rope_head_dim]

        # Slice the correct region of the cache
        if mode == "decode":
            # Advanced indexing to get the correct position for each user
            batch_indices = torch.arange(batch_size)
            tt_cache = tt_cache[batch_indices, position_ids, :].unsqueeze(
                1
            )  # [bsz, 1(seq_len), head_dim + rope_head_dim]
            output_cache = output_cache[:, -1, :].unsqueeze(1)  # [bsz, 1(seq_len), head_dim + rope_head_dim]
        else:
            tt_cache = tt_cache[user_id, :seq_len, :].unsqueeze(1)  # [1(bsz), seq_len, head_dim + rope_head_dim]
            output_cache = output_cache[0, :seq_len, :].unsqueeze(1)  # [1(bsz), seq_len, head_dim + rope_head_dim]

        tt_cache_kv = tt_cache[..., : hf_config_short.kv_lora_rank]
        tt_cache_pe = tt_cache[..., hf_config_short.kv_lora_rank :]

        ref_cache_kv = output_cache[..., : hf_config_short.kv_lora_rank]  # [bsz, _, head_dim]
        ref_cache_pe = output_cache[..., hf_config_short.kv_lora_rank :]  # [bsz, _, rope_head_dim]

        kv_passing, kv_pcc_message = comp_pcc(ref_cache_kv, tt_cache_kv, pcc_required_kvpe)
        pe_passing, pe_pcc_message = comp_pcc(ref_cache_pe, tt_cache_pe, pcc_required_kvpe)

        logger.info(f"Cache KV PCC: {kv_pcc_message}")
        logger.info(f"Cache PE PCC: {pe_pcc_message}")

        all_passing = all_passing and kv_passing and pe_passing

        # Check if the rest of the cache is empty
        for r in range(mesh_device.shape[0]):
            if r == cur_row_idx:
                continue
            other_cache = get_cache_on_host(run_config["kvpe_cache"], r, mesh_device)
            passing = torch.all(other_cache == 0)

            if not passing:
                logger.error(f"Cache for row {r} is not empty")
                all_passing = all_passing and passing

        # Check if cache is empty for other users in prefill mode
        if mode == "prefill":
            other_cache = get_cache_on_host(run_config["kvpe_cache"], cur_row_idx, mesh_device)

            # Convert to paged cache format so we can index by user_id
            other_cache = torch_cache_from_paged(other_cache, torch_page_table, dp_factor)

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
