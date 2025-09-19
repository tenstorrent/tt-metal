# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import itertools

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.model_1d import Model1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    dequantize_state_dict,
    get_model_config,
    load_state_dict,
    paged_caches_from_torch,
    run_reference_with_attention,
    torch_cache_from_transformers,
)


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
    "mode, seq_len, batch_size",
    [
        ("decode", 1, 32),
        ("prefill", 128, 1),
        # ("prefill", 2048),  # Test chunking # TODO: Uncomment once MLA prefill works
    ],
)
def test_forward_pass(
    use_real_weights,
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
    # Set less layers and shorter max length for the sake of testing
    hf_config_short.num_hidden_layers = 8

    # CCL workaround (remove once persistent buffers are added)
    mesh_device.disable_and_clear_program_cache()

    # Check params
    if mode == "prefill":
        assert batch_size == 1, "Prefill only supports a batch size of 1"
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"

    # Get reference IO
    logger.info("Setting up reference IO")
    if use_real_weights:
        torch.use_deterministic_algorithms(False)

        logger.info("Loading real weights from disk")
        state_dict = load_state_dict(model_path, "")
        state_dict = {
            k: v
            for k, v in state_dict.items()
            for layer_idx_str in ["".join(itertools.takewhile(str.isdigit, k.removeprefix("model.layers.")))]
            if not layer_idx_str or int(layer_idx_str) < hf_config_short.num_hidden_layers
        }  # Trim the loaded state dict to not run out of memory

        logger.info("Creating reference model")
        reference_model = DeepseekV3ForCausalLM(hf_config_short).eval().to(torch.bfloat16)
        logger.info("Loading real weights into reference model")
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config_short))

        torch_input = torch.randint(0, hf_config_short.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
        if mode == "prefill":
            position_ids = torch.tensor([seq_len])
        else:
            position_ids = torch.randint(0, hf_config_short.max_seq_len - 1, (batch_size,))
            position_ids = torch.zeros(
                (batch_size,), dtype=torch.long
            )  # TODO: investigate the PCC issue with real weights

        logger.info("Running the reference model")
        reference_output, input_cache, output_cache = run_reference_with_attention(
            reference_model, torch_input, position_ids, None, hf_config_short, mode, False
        )
        input_cache = torch_cache_from_transformers(input_cache)
        output_cache = torch_cache_from_transformers(output_cache)
    else:
        logger.info("Creating reference model with random weights")
        reference_model = DeepseekV3ForCausalLM(hf_config_short).eval().to(torch.bfloat16)
        # This needs to be disabled as deterministic way to quantize weights is not supported
        torch.use_deterministic_algorithms(False)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.to(torch.bfloat16).state_dict(),
            block_shape=hf_config_short.quantization_config["weight_block_size"],
        )

        torch_input = torch.randint(0, hf_config_short.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
        if mode == "prefill":
            position_ids = torch.tensor([seq_len])
        else:
            # position_ids = torch.randint(0, hf_config_short.max_seq_len - 1, (batch_size,))
            position_ids = torch.zeros(
                (batch_size,), dtype=torch.long
            )  # TODO: investigate the PCC issue with real weights
        reference_output, input_cache, output_cache = run_reference_with_attention(
            reference_model, torch_input, position_ids, None, hf_config_short, mode, False
        )
        input_cache = torch_cache_from_transformers(input_cache)
        output_cache = torch_cache_from_transformers(output_cache)

    # Set up page config
    logger.info("Setting up model configs")
    _, dp_factor = mesh_device.shape
    user_id = None if mode == "decode" else torch.randint(0, MAX_BATCH_SIZE, ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, MAX_BATCH_SIZE, dp_factor)
    paged_input_caches, torch_page_tables = paged_caches_from_torch(input_cache, dp_factor, paged_config, user_id)

    # Set up model config
    weight_config = Model1D.convert_weights(hf_config_short, [state_dict], tmp_path, mesh_device)
    logger.info("Weight conversion done")
    model_config = get_model_config(Model1D, mode, hf_config_short, mesh_device)
    logger.info(f"Model config created for {mode} mode")
    model_state = Model1D.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_caches)
    logger.info("Model state created")
    model_shared_state = Model1D.create_shared_state(hf_config_short, mesh_device)
    logger.info("Model shared state created")
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)
    logger.info("Run config created")

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    if mode == "decode":
        # TT Shape: [1, seq_len, batch_size]
        torch_input = torch_input.transpose(-1, -2)

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
        MLA1D.create_page_table(torch_page_table, paged_config, mesh_device) for torch_page_table in torch_page_tables
    )

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
    if mode == "prefill":
        tt_output = Model1D.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = Model1D.forward_decode(tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_tables)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    if mode == "decode":
        tt_output_torch = tt_output_torch.squeeze(0).permute(1, 0, 2)  # Torch Shape: [batch_size, seq_len, hidden_size]
    assert (
        tt_output_torch.shape[-1] == hf_config_short.vocab_size
    ), f"Output shape mismatch: {tt_output_torch.shape} vs {hf_config_short.vocab_size}"

    # Check output PCC
    logger.info("Validating output")
    pcc_required = 0.91
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}, Batch size: {batch_size}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"Test failed for Model1D because PCC < {pcc_required} in {mode} mode."


if __name__ == "__main__":
    pytest.main([__file__])
