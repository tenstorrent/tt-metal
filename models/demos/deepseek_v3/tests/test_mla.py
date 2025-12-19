# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_cache_from_torch,
    run_reference_with_attention,
    torch_cache_from_paged,
    torch_cache_from_transformers_single_layer,
)

PCC_REQUIRED = 0.99
PCC_REQUIRED_KVPE = 0.999


def get_cache_on_host(tt_cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """
    Get the KVPE cache on the host from the TTNN cache.

    Args:
        tt_cache (ttnn.Tensor): The TTNN cache tensor.
        mesh_device (ttnn.MeshDevice): The mesh device to get the cache from.
        row_idx (int | None): The row index to get the cache from. If None, gets the entire cache.
    Returns:
        torch.Tensor: The cache tensor on the host.
    """
    return ttnn.to_torch(
        tt_cache,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


def generate_reference_io(
    model_path: Path,
    module_path: str | None,
    hf_config: PretrainedConfig,
    layer_idx: int,
    seq_len: int,
    batch_size: int,
    mode: str,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if module_path is None:
        reference_model = DeepseekV3Attention(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
    else:
        reference_model = DeepseekV3Attention(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)
        state_dict = sub_state_dict(state_dict, module_path + ".")
        dequantized_state_dict = dequantize_state_dict(state_dict, hf_config)
        reference_model.load_state_dict(dequantized_state_dict)

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.tensor([seq_len])
    else:
        position_ids = position_ids_or_seq_lens = torch.randint(0, hf_config.max_seq_len - 1, (batch_size,))
    reference_output, input_cache, output_cache = run_reference_with_attention(
        reference_model, torch_input, position_ids_or_seq_lens, layer_idx, hf_config, mode, zeroed_cache=True
    )
    input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
    output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        reference_output = reference_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED):
    passing, pcc = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(f"PCC: {pcc}")
    return passing


def check_cache_matches(tt_cache, reference_cache, lora_rank, pcc_required=PCC_REQUIRED_KVPE):
    # Check correct shapes
    dims_correct = True
    if len(tt_cache.shape) != 4:
        logger.error(
            f"Expected tt_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {tt_cache.shape=}"
        )
        dims_correct = False
    if len(reference_cache.shape) != 4:
        logger.error(
            f"Expected reference_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {reference_cache.shape=}"
        )
        dims_correct = False
    if not dims_correct:
        return False

    if tt_cache.shape != reference_cache.shape:
        logger.error(f"Cache shape mismatch: {tt_cache.shape=} vs {reference_cache.shape=}")
        return False

    if torch.all(tt_cache == 0) != torch.all(reference_cache == 0):
        if torch.all(tt_cache == 0):
            logger.error("TTNN cache is all zeros, but reference cache is not.")
        else:
            logger.error("Reference cache is all zeros, but TTNN cache is not.")
        return False

    # Check PCC
    tt_cache_kv = tt_cache[..., :lora_rank]
    tt_cache_pe = tt_cache[..., lora_rank:]

    ref_cache_kv = reference_cache[..., :lora_rank]  # [bsz, _, _, head_dim]
    ref_cache_pe = reference_cache[..., lora_rank:]  # [bsz, _, _, rope_head_dim]

    kv_passing, kv_pcc_message = comp_pcc(ref_cache_kv, tt_cache_kv, pcc_required)
    pe_passing, pe_pcc_message = comp_pcc(ref_cache_pe, tt_cache_pe, pcc_required)

    logger.info(f"Cache KV PCC: {kv_pcc_message}")
    logger.info(f"Cache PE PCC: {pe_pcc_message}")
    return kv_passing and pe_passing


def check_cache_unchanged(tt_cache, exclusion_area: tuple[slice, slice, slice, slice]):
    if len(tt_cache.shape) != 4:
        logger.error(
            f"Expected tt_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {tt_cache.shape=}"
        )
        return False

    tt_cache = tt_cache.clone()
    tt_cache[exclusion_area] = 0  # Zero out the area we don't want to check
    all_zeros = True
    for user_id in range(tt_cache.shape[0]):
        if not torch.all(tt_cache[user_id] == 0):
            logger.error(
                f"Cache for user {user_id % USERS_PER_ROW} row {user_id // USERS_PER_ROW} not empty outside exclusion area"
            )
            all_zeros = False

    return all_zeros


def run_test_forward_pass_mla1d(
    layer_idx,
    mode,
    seq_len,
    batch_size,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config,
    state_dict,
):
    # Check params
    if mode == "prefill":
        assert batch_size == 1, "Prefill only supports a batch size of 1"
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"

    # Get reference IO
    logger.info("Setting up reference IO")
    state_dict, position_ids, torch_input, reference_output, input_cache, output_cache = generate_reference_io(
        model_path, module_path, hf_config_short, layer_idx, seq_len, batch_size, mode, state_dict
    )

    # Set up page config
    logger.info("Setting up model configs")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, (1, mesh_device.shape[1]), paged_config, user_id
    )

    # Set up model config
    weight_config = get_test_weight_config(
        MLA1D,
        hf_config_short,
        (state_dict,) * mesh_device.shape[0],
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(MLA1D, mode, hf_config_short, mesh_device)
    model_state = MLA1D.create_state(
        hf_config_short, paged_config, mesh_device, ccl, (paged_input_cache,) * mesh_device.shape[0]
    )
    run_config = create_run_config(model_config, weight_config, model_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")

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

    tt_page_table = MLA1D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )
    tt_rope_tensors = get_rope_tensors(hf_config_short, batch_size, seq_len, position_ids, mesh_device)

    # Forward pass
    logger.info("Running TTNN forward pass")

    cur_row_idx = torch.randint(0, mesh_device.shape[0], ()).item()
    if mode == "prefill":
        tt_output = MLA1D.forward_prefill(tt_input, user_id, cur_row_idx, run_config, tt_rope_tensors, tt_page_table)
    else:
        tt_output = MLA1D.forward_decode(
            tt_input, position_ids_tensor, cur_row_idx, run_config, tt_rope_tensors, tt_page_table
        )

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
    )[cur_row_idx]

    # Check PCC
    tt_cache = torch_cache_from_paged(
        get_cache_on_host(run_config["kvpe_cache"], mesh_device), torch_page_table, mesh_device.get_num_devices()
    )
    if mode == "prefill":
        batch_id = user_id + cur_row_idx * USERS_PER_ROW
        assert (
            check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)
            and check_cache_matches(
                tt_cache[batch_id : batch_id + 1, :, :seq_len],
                output_cache,
                hf_config_short.kv_lora_rank,
                pcc_required=PCC_REQUIRED_KVPE,
            )
            and check_cache_unchanged(
                tt_cache, (slice(batch_id, batch_id + 1), slice(None), slice(None, seq_len), slice(None))
            )
        ), f"MLA output for prefill {seq_len=} {user_id=} {cur_row_idx=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"
    else:
        assert (
            check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)
            and check_cache_matches(
                tt_cache[torch.arange(batch_size) + cur_row_idx * USERS_PER_ROW, :, position_ids, :].unsqueeze(2),
                output_cache[:, :, -1:, :],
                hf_config_short.kv_lora_rank,
                pcc_required=PCC_REQUIRED_KVPE,
            )
            and check_cache_unchanged(
                tt_cache,
                (
                    slice(cur_row_idx * USERS_PER_ROW, (cur_row_idx + 1) * USERS_PER_ROW),
                    slice(None),
                    slice(None),
                    slice(None),
                ),
            )
        ), f"MLA output for decode {batch_size=} {position_ids=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"


def run_test_forward_pass_mla2d(
    layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config,
    state_dict,
):
    # Check params
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    # Get reference IO
    logger.info("Setting up reference IO")
    state_dict, position_ids, torch_input, reference_output, input_cache, output_cache = generate_reference_io(
        model_path, module_path, hf_config_short, layer_idx, seq_len, batch_size, mode, state_dict
    )

    # Set up page config
    logger.info("Setting up model configs")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW * mesh_device.shape[0], ()).item()
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )

    # Set up model config
    weight_config = get_test_weight_config(
        MLA2D,
        hf_config_short,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(MLA2D, mode, hf_config_short, mesh_device)
    model_state = MLA2D.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_cache)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    position_ids_tensor = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    tt_page_table = MLA2D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )
    tt_rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)

    # Forward pass
    logger.info("Running TTNN forward pass")

    if mode == "prefill":
        tt_output = MLA2D.forward_prefill(tt_input, user_id, run_config, tt_rope_tensors, tt_page_table)
    else:
        tt_output = MLA2D.forward_decode(tt_input, position_ids_tensor, run_config, tt_rope_tensors, tt_page_table)

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
    ).reshape(
        -1, seq_len, hf_config_short.hidden_size
    )  # Concatenate all batches together

    # Check PCC
    tt_cache = torch_cache_from_paged(
        get_cache_on_host(run_config["mla1d"]["kvpe_cache"], mesh_device),
        torch_page_table,
        mesh_device.get_num_devices(),
    )
    if mode == "prefill":
        assert (
            check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)
            and check_cache_matches(
                tt_cache[user_id : user_id + 1, :, :seq_len],
                output_cache,
                hf_config_short.kv_lora_rank,
                pcc_required=PCC_REQUIRED_KVPE,
            )
            and check_cache_unchanged(
                tt_cache, (slice(user_id, user_id + 1), slice(None), slice(None, seq_len), slice(None))
            )
        ), f"MLA output for prefill {seq_len=} {user_id=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"
    else:
        assert check_output_matches(
            tt_output_torch, reference_output, pcc_required=PCC_REQUIRED
        ) and check_cache_matches(
            tt_cache[torch.arange(batch_size), :, position_ids, :].unsqueeze(2),
            output_cache[:, :, -1:, :],
            hf_config_short.kv_lora_rank,
            pcc_required=PCC_REQUIRED_KVPE,
        ), f"MLA output for decode {batch_size=} {position_ids=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"


@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row",
    [
        ("decode", 1, USERS_PER_ROW),
    ]
    + [("prefill", seq_len, 1) for seq_len in PREFILL_SEQ_LENS],
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
@pytest.mark.parametrize(
    "test_closure",
    [run_test_forward_pass_mla1d, run_test_forward_pass_mla2d],
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config,
    test_closure,
    set_deterministic_env,
    state_dict,
):
    # Skip all prefill seq lengths except 128 to avoid exceeding CI workload time
    if mode == "prefill" and seq_len != 128:
        pytest.skip(
            f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
        )

    # Hardcoded arguments; can later change them to test arguments if needed
    layer_idx = 0

    test_closure(
        layer_idx,
        mode,
        seq_len,
        batch_size_per_row,
        hf_config_short,
        cache_path,
        mesh_device,
        ccl,
        model_path,
        module_path,
        force_recalculate_weight_config,
        state_dict,
    )


if __name__ == "__main__":
    pytest.main([__file__])
