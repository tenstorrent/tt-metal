# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.tt.model.row_pipelined_model import RowPipelinedModel
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_caches_from_torch,
    run_reference_with_attention,
    torch_cache_from_transformers,
)


def generate_reference_io(
    use_real_weights: bool,
    mode: str,
    seq_len: int,
    batch_size: int,
    hf_config: PretrainedConfig,
    model_path: str,
    state_dict: dict[str, torch.Tensor],
):
    """Generate reference input and output for the given mode using either real or random weights."""
    # This needs to be disabled as deterministic way to quantize weights is not supported
    torch.use_deterministic_algorithms(False)

    if use_real_weights:
        torch.use_deterministic_algorithms(False)

        state_dict = sub_state_dict(state_dict, "", hf_config.num_hidden_layers)

        logger.info(f"Creating reference model")
        # Create model on meta device (no weight initialization or memory allocation)
        with torch.device("meta"):
            reference_model = DeepseekV3ForCausalLM(hf_config).eval()

        # Move to target device without allocating memory for parameters
        reference_model = reference_model.to_empty(device=torch.device("cpu"))

        logger.info(f"Loading state dict into reference model")
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
        reference_model = reference_model.to(torch.bfloat16)
    else:
        logger.info("Creating reference model with random weights")
        reference_model = DeepseekV3ForCausalLM(hf_config).eval().to(torch.bfloat16)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.to(torch.bfloat16).state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )

    torch_input = torch.randint(0, hf_config.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.tensor([seq_len])
    else:
        # position_ids = torch.randint(0, hf_config.max_seq_len - 1, (batch_size,))
        position_ids = position_ids_or_seq_lens = torch.zeros(
            (batch_size,), dtype=torch.long
        )  # TODO: investigate the PCC issue with real weights

    logger.info(
        f"Running reference model with torch_input shape: {torch_input.shape} and position_ids shape: {position_ids_or_seq_lens.shape}"
    )
    reference_output, input_cache, output_cache = run_reference_with_attention(
        reference_model, torch_input, position_ids_or_seq_lens, None, hf_config, mode, False
    )
    logger.info(f"Reference model output shape: {reference_output.shape}")
    input_cache = torch_cache_from_transformers(input_cache)
    output_cache = torch_cache_from_transformers(output_cache)

    if mode == "decode":
        torch_input = torch_input.transpose(1, 0)  # [seq_len, batch_size]
        reference_output = reference_output.transpose(1, 0)  # [seq_len, batch_size]
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def run_test_forward_pass_ppmodel(
    use_real_weights,
    mode,
    seq_len,
    batch_size,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
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
        use_real_weights, mode, seq_len, batch_size, hf_config_short, model_path, state_dict
    )

    # Set up page config
    logger.info("Setting up model configs")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_caches, torch_page_tables = paged_caches_from_torch(
        input_cache, (1, mesh_device.shape[1]), paged_config, user_id
    )

    # Set up model config
    weight_config = get_test_weight_config(
        RowPipelinedModel, hf_config_short, (state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    model_config = get_model_config(RowPipelinedModel, mode, hf_config_short, mesh_device)
    model_state = RowPipelinedModel.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowPipelinedModel.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
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

    tt_page_tables = tuple(
        MLA1D.create_page_table(page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device)
        for torch_page_table in torch_page_tables
    )
    rope_tensors = get_rope_tensors(hf_config_short, batch_size, seq_len, position_ids, mesh_device)
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    # Forward pass
    logger.info("Running TTNN forward pass")
    if mode == "prefill":
        tt_output = RowPipelinedModel.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = RowPipelinedModel.forward_decode(
            tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_tables
        )

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    assert (
        tt_output_torch.shape[-1] == hf_config_short.vocab_size
    ), f"Output shape mismatch: {tt_output_torch.shape} vs {hf_config_short.vocab_size}"

    # Check output PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.97)


def run_test_forward_pass_dpmodel(
    use_real_weights,
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
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
        use_real_weights, mode, seq_len, batch_size, hf_config_short, model_path, state_dict
    )

    # Set up page config
    logger.info("Setting up model configs")
    _, dp_factor = mesh_device.shape
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW, ()).item()
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, dp_factor)
    paged_input_caches, torch_page_tables = paged_caches_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )

    # Set up model config
    weight_config = get_test_weight_config(
        RowBatchedModel, hf_config_short, (state_dict,), cache_path, mesh_device, force_recalculate_weight_config
    )
    model_config = get_model_config(RowBatchedModel, mode, hf_config_short, mesh_device)
    model_state = RowBatchedModel.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
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

    tt_page_tables = tuple(
        MLA2D.create_page_table(page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device)
        for torch_page_table in torch_page_tables
    )
    rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    # Forward pass
    logger.info("Running TTNN forward pass")
    if mode == "prefill":
        tt_output = RowBatchedModel.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = RowBatchedModel.forward_decode(
            tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_tables
        )

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape)
    )
    assert (
        tt_output_torch.shape[-1] == hf_config_short.vocab_size
    ), f"Output shape mismatch: {tt_output_torch.shape} vs {hf_config_short.vocab_size}"

    # Check output PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.97)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_real_weights",
    [True],  # Test only with real weights for now
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row",
    [
        ("decode", 1, 32),
    ]
    + [("prefill", seq_len, 1) for seq_len in PREFILL_SEQ_LENS],
)
@pytest.mark.parametrize("test_closure", [run_test_forward_pass_ppmodel, run_test_forward_pass_dpmodel])
def test_forward_pass(
    use_real_weights,
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
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
    # Set less layers and shorter max length for the sake of testing
    hf_config_short.num_hidden_layers = 8

    test_closure(
        use_real_weights,
        mode,
        seq_len,
        batch_size_per_row,
        hf_config_short,
        cache_path,
        mesh_device,
        model_path,
        ccl,
        force_recalculate_weight_config,
        state_dict,
    )


if __name__ == "__main__":
    pytest.main([__file__])
