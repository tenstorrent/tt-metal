# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3DecoderLayer
from models.demos.deepseek_v3.tt.decoder_block.decoder_block import DecoderBlock
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block import MoEDecoderBlock
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    get_model_config,
    paged_cache_from_torch,
    run_reference_with_attention,
    torch_cache_from_transformers_single_layer,
)


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
        (MoEDecoderBlock, None, 3),
        # (DecoderBlock, "model.layers.0", None),
        # (MoEDecoderBlock, "model.layers.3", None),
    ],
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1, 32),
        ("prefill", 128, 1),
        # ("prefill", 2048, 1),  # Test chunking # TODO: Uncomment once MLA prefill works
    ],
)
def test_forward_pass(
    DecoderBlockClass: type[DecoderBlockBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size,
    hf_config_short,
    tmp_path,
    mesh_device,
    model_path,
    ccl,
    set_deterministic_env,
):
    # CCL workaround (remove once persistent buffers are added)
    mesh_device.disable_and_clear_program_cache()

    # Check params
    if mode == "prefill":
        assert batch_size == 1, "Prefill only supports a batch size of 1"
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"

    # Get reference IO
    logger.info("Setting up reference IO")
    if module_path is None:
        reference_model = (
            DeepseekV3DecoderLayer(hf_config_short, layer_idx=reference_layer_idx).eval().to(torch.bfloat16)
        )
        # This needs to be disabled as deterministic way to quantize weights is not supported
        torch.use_deterministic_algorithms(False)
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
            reference_model, torch_input, position_ids, reference_layer_idx, hf_config_short, mode, False
        )
        input_cache = torch_cache_from_transformers_single_layer(input_cache, reference_layer_idx)
        output_cache = torch_cache_from_transformers_single_layer(output_cache, reference_layer_idx)
    else:
        raise NotImplementedError("Loading reference IO is not implemented yet")

    # Set up page config
    logger.info("Setting up model configs")
    _, dp_factor = mesh_device.shape
    user_id = None if mode == "decode" else torch.randint(0, MAX_BATCH_SIZE, ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, MAX_BATCH_SIZE, dp_factor)
    paged_input_cache, torch_page_table = paged_cache_from_torch(input_cache, dp_factor, paged_config, user_id)

    # Set up model config
    is_padding_layer = [False] * mesh_device.shape[0]
    weight_config = DecoderBlockClass.convert_weights(
        hf_config_short, [state_dict] * mesh_device.shape[0], tmp_path, mesh_device
    )
    model_config = get_model_config(DecoderBlockClass, mode, hf_config_short, mesh_device, is_padding_layer)
    model_state = DecoderBlockClass.create_state(
        hf_config_short, paged_config, mesh_device, ccl, is_padding_layer, (paged_input_cache,) * mesh_device.shape[0]
    )
    model_shared_state = DecoderBlockClass.create_shared_state(hf_config_short, mesh_device, is_padding_layer)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [1, seq_len, batch_size, hidden_size]

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, MAX_BATCH_SIZE, mesh_device.shape[1])

    # Forward pass
    logger.info("Running TTNN forward pass")

    cur_row_idx = torch.randint(0, mesh_device.shape[0], ()).item()
    if mode == "prefill":
        tt_output = DecoderBlockClass.forward_prefill(
            tt_input, user_id, cur_row_idx, run_config, rope_tensors, tt_page_table
        )
    else:
        tt_output = DecoderBlockClass.forward_decode(
            tt_input, position_ids_tensor, cur_row_idx, run_config, rope_tensors, tt_page_table
        )

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
    )[cur_row_idx]
    if mode == "decode":
        tt_output_torch = tt_output_torch.permute(1, 0, 2)  # Torch Shape: [batch_size, seq_len, hidden_size]

    # Check output PCC
    logger.info("Validating output")
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}, Batch size: {batch_size}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"Test failed for {DecoderBlockClass.__name__} because PCC < {pcc_required} in {mode} mode."


if __name__ == "__main__":
    pytest.main([__file__])
