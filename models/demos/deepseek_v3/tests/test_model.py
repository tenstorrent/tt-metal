# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from random import randint

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.model_1d import Model1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import dequantize_state_dict
from models.demos.deepseek_v3.utils.reference_forwards import reference_forward_model as reference_forward
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    load_reference_io_tensors_for_module,
    load_state_dict,
)
from models.utility_functions import comp_pcc


def merge_dicts(parent, child, prefix):
    for k, v in child.items():
        if isinstance(v, dict):
            merge_dicts(parent.setdefault(k, {}), v, prefix + f"{k}.")
        else:
            parent[prefix + f"{k}"] = v


def create_whole_model_state_dict(model_path, hf_config, prefix="model."):
    state_dict = {}

    state_dict_temp = load_state_dict(model_path, "model.embed_tokens")
    merge_dicts(state_dict, state_dict_temp, prefix + "embed_tokens.")
    for li in range(hf_config.num_hidden_layers):
        state_dict_temp = load_state_dict(model_path, f"model.layers.{li}")
        merge_dicts(state_dict, state_dict_temp, prefix + f"layers.{li}.")
    state_dict_temp = load_state_dict(model_path, "model.norm")
    merge_dicts(state_dict, state_dict_temp, prefix + "norm.")
    state_dict_temp = load_state_dict(model_path, "lm_head")
    merge_dicts(state_dict, state_dict_temp, "lm_head.")

    return state_dict


@pytest.fixture
def hf_config(hf_config):
    """Load DeepSeek config for testing."""
    hf_config.num_hidden_layers = 3
    hf_config.max_seq_len = 5 * 1024  # Set max sequence length for testing
    return hf_config


def load_reference_model(hf_config):
    """Load the reference model for testing."""

    return DeepseekV3ForCausalLM(hf_config).eval()


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "module_path",
    [
        None,
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
@pytest.mark.parametrize(
    "weights_type",
    ["real", "random"],
)
def test_forward_pass(
    module_path, mode, seq_len, batch_size, hf_config, tmp_path, mesh_device, model_path, ccl, reset_seeds, weights_type
):
    mesh_device.disable_and_clear_program_cache()
    mesh_shape = list(mesh_device.shape)
    num_rows, sdpa_dp_factor = mesh_shape

    paged_config = MLA1D.get_valid_paged_config(hf_config.max_seq_len, MLA1D.MAX_BATCH_SIZE, sdpa_dp_factor)

    ############################
    ### Set up reference
    ############################
    logger.info("Setting up reference model")
    reference_model = load_reference_model(hf_config)
    if weights_type == "random":
        state_dict = add_inv_scale_to_state_dict(
            reference_model.to(torch.bfloat16).state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
    else:
        state_dict = create_whole_model_state_dict(model_path, hf_config)
        dequantized_state_dict = dequantize_state_dict(state_dict, hf_config)
        reference_model.load_state_dict(dequantized_state_dict)

    ############################
    ### Torch inputs
    ############################
    logger.info("Preparing Torch inputs")
    if module_path is None:
        reference_output = None
        torch_input = torch.randint(0, hf_config.vocab_size - 1, (batch_size, seq_len)).long()
    else:
        raise NotImplementedError(
            "Loading reference IO tensors for module path is not implemented. "
            "Please implement this if you want to test with a specific module path."
        )
        sequence_length = seq_len + 1 if mode == "decode" else seq_len + 1
        torch_input, reference_output = load_reference_io_tensors_for_module(
            mode, module_path, sequence_length, num_rows
        )

    if mode == "prefill":
        position_idxs = torch.tensor([seq_len for _ in range(batch_size)])
    else:
        # TODO: Why do we need to set zero position ids?
        position_idxs = torch.tensor([0 for _ in range(batch_size)])

    ############################
    ### Torch reference
    ############################
    logger.info("Running Torch reference forward pass")
    if module_path is None:
        reference_output = reference_forward(
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
    weight_config = Model1D.convert_weights(hf_config, state_dict, tmp_path, mesh_device, state_dict_prefix="model.")

    if mode == "prefill":
        model_config = Model1D.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = Model1D.decode_model_config(hf_config, mesh_device)

    # Create a new model state
    model_state = Model1D.create_state(
        hf_config,
        mesh_device,
        paged_config=paged_config,
        ccl=ccl,
    )

    model_shared_state = Model1D.create_shared_state(
        hf_config,
        mesh_device,
    )

    # Create run config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    ############################
    ### TTNN inputs
    ############################
    logger.info("Preparing TTNN inputs")
    user_id = None if mode == "decode" else randint(0, MLA1D.MAX_BATCH_SIZE - 1)

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

    output_row_idx = 0  # TODO: Make this configurable
    if mode == "prefill":
        tt_output = Model1D.forward_prefill(tt_input, run_config, user_id, rope_tensors, tt_page_table)
    else:
        tt_output = Model1D.forward_decode(tt_input, position_idxs_tensor, rope_tensors, tt_page_table, run_config)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    assert (
        tt_output_torch.shape[-1] == hf_config.vocab_size
    ), f"Output shape mismatch: {tt_output_torch.shape} vs {hf_config.vocab_size}"
    reference_output = reference_output.permute(1, 0, 2).unsqueeze(0)  # [1,1,B,V]

    ############################
    ### Validation
    ############################
    logger.info("Validating output")
    all_passing = True
    pcc_required = 0.94
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(f"PCC for {Model1D.__name__} in {mode} mode: {pcc_message}")

    all_passing = all_passing and passing

    assert all_passing, f"Test failed for {Model1D.__name__} because PCC < {pcc_required} in {mode} mode."


if __name__ == "__main__":
    pytest.main([__file__])
