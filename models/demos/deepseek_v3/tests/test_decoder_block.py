# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from random import randint

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3DecoderLayer
from models.demos.deepseek_v3.tt.decoder_block.decoder_block import DecoderBlock
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.reference_forwards import reference_forward_decode as reference_forward
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    MAX_START_POS,
    add_inv_scale_to_state_dict,
    load_reference_io_tensors_for_module,
    load_state_dict,
)
from models.utility_functions import comp_pcc


@pytest.fixture
def hf_config(hf_config):
    """Load DeepSeek config for testing."""
    hf_config.max_seq_len = 5 * 1024  # Set max sequence length for testing
    return hf_config


def load_reference_model(hf_config, layer_idx):
    """Load the reference model for testing."""

    model = DeepseekV3DecoderLayer(hf_config, layer_idx=layer_idx).eval()

    return model


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "DecoderBlockClass, module_path, reference_layer_idx",
    [
        (DecoderBlock, None, 0),
        # (MoEDecoderBlock, None, 3), # TODO: Uncomment once PCC is fixed for MoE
        # (DecoderBlock, "model.layers.0", None),
        # (MoEDecoderBlock, "model.layers.3", None), # TODO: Uncomment once PCC is fixed for MoE
    ],
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1, 32),
        # ("prefill", 512), # TODO: Uncomment once MLA prefill works
        # ("prefill", 2048),  # Test chunking # TODO: Uncomment once MLA prefill works
    ],
)
def test_forward_pass(
    DecoderBlockClass,
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size,
    hf_config,
    tmp_path,
    mesh_device,
    model_path,
    ccl,
):
    mesh_shape = list(mesh_device.shape)
    num_rows, sdpa_dp_factor = mesh_shape

    paged_config = MLA1D.get_valid_paged_config(hf_config.max_seq_len, MLA1D.MAX_BATCH_SIZE, sdpa_dp_factor)

    ############################
    ### Set up reference
    ############################
    logger.info("Setting up reference model")
    if module_path is None:
        reference_model = load_reference_model(hf_config, reference_layer_idx)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.to(torch.bfloat16).state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
    else:
        state_dict = load_state_dict(model_path, module_path)

    ############################
    ### Torch inputs
    ############################
    logger.info("Preparing Torch inputs")
    if module_path is None:
        reference_output = None
        torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)
    else:
        sequence_length = seq_len + 1 if mode == "decode" else seq_len + 1
        torch_input, reference_output = load_reference_io_tensors_for_module(
            mode, module_path, sequence_length, num_rows
        )

    if mode == "prefill":
        position_idxs = torch.tensor([seq_len for _ in range(batch_size)])
    else:
        position_idxs = torch.tensor([randint(0, MAX_START_POS) for _ in range(batch_size)])

    ############################
    ### Torch reference
    ############################
    logger.info("Running Torch reference forward pass")
    if module_path is None:
        reference_output, reference_cache = reference_forward(
            reference_model,
            torch_input,
            position_ids=position_idxs,
            mode=mode,
        )

    ############################
    ### Set up TTNN configs
    ############################
    logger.info("Setting up TTNN configs")

    # For now, since we're only loading one layer, we can replicate
    state_dicts = [state_dict] * num_rows
    weight_config = DecoderBlockClass.convert_weights(hf_config, state_dicts, tmp_path, mesh_device)

    is_padding_layer = [False] * num_rows
    if mode == "prefill":
        model_config = DecoderBlockClass.prefill_model_config(hf_config, mesh_device, is_padding_layer)
    else:
        model_config = DecoderBlockClass.decode_model_config(hf_config, mesh_device, is_padding_layer)

    # Create a new model state
    model_state = DecoderBlockClass.create_state(
        hf_config,
        mesh_device,
        paged_config=paged_config,
        is_padding_layer=is_padding_layer,
        ccl=ccl,
    )

    # Create run config
    run_config = create_run_config(model_config, weight_config, model_state)

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
        dp_factor=sdpa_dp_factor,
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

    if mode == "prefill":
        rot_mats = rope_setup.get_rot_mats_table(seq_len)
    else:
        rot_idxs = torch.tensor(position_idxs, dtype=torch.int32)
        rot_mats = rope_setup.get_rot_mats(rot_idxs)
    rope_tensors = {
        "cos_matrix": rot_mats[0],
        "sin_matrix": rot_mats[1],
        "trans_matrix": rot_mats[2],
    }

    ############################
    ### TTNN forward pass
    ############################
    logger.info("Running TTNN forward pass")
    cur_row_idx = 0  # randint(0, mesh_shape[0] - 1)

    if mode == "prefill":
        tt_output = DecoderBlockClass.forward_prefill(
            tt_input, run_config, user_id, rope_tensors, tt_page_table, cur_row_idx
        )
    else:
        tt_output = DecoderBlockClass.forward_decode(
            tt_input, cur_row_idx, position_idxs_tensor, rope_tensors, tt_page_table, run_config
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
    logger.info("Validating output")
    all_passing = True
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(f"PCC for {DecoderBlockClass.__name__} in {mode} mode: {pcc_message}")

    all_passing = all_passing and passing

    assert all_passing, f"Test failed for {DecoderBlockClass.__name__} because PCC < {pcc_required} in {mode} mode."


if __name__ == "__main__":
    pytest.main([__file__])
