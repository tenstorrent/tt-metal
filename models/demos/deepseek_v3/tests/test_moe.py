# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import math
import os
from copy import deepcopy

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN
from models.demos.deepseek_v3.tt.model.row_batched_model import get_fabric_config
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    load_reference_io,
    load_reference_io_tensors_for_module,
    run_module_forward,
)


@pytest.fixture
def reference_model(hf_config):
    """Build the routed-experts-only MoE reference used by the TT MoE test."""
    torch.use_deterministic_algorithms(True)
    moe_config = deepcopy(hf_config)
    moe_config.n_shared_experts = None
    return DeepseekV3MoE(moe_config).eval()


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def load_real_moe_input(mode: str, module_path: str, num_tokens: int) -> torch.Tensor:
    if mode == "prefill":
        torch_input, _ = load_reference_io_tensors_for_module(mode, module_path, num_tokens, 1)
        return torch_input.squeeze(0).to(torch.bfloat16)

    reference_io = load_reference_io(mode, module_path)
    assert all(len(logs) <= 1 for logs in reference_io), f"Expected a non-range module, got {module_path}"
    assert all(len(logs) > 0 for logs in reference_io), f"Some logs for module {module_path} {mode} were not generated."

    io_module_paths, torch_args, _, _ = zip(*[logs[0] for logs in reference_io])
    (torch_inputs,) = zip(*torch_args)
    assert set(io_module_paths) == {module_path}

    torch_input = torch.concat(torch_inputs, dim=1).unsqueeze(0)

    if torch_input.shape[2] < num_tokens:
        repeats = math.ceil(num_tokens / torch_input.shape[2])
        torch_input = torch_input.repeat(1, 1, repeats, 1)

    return torch_input[:, :, :num_tokens, :].squeeze(0).to(torch.bfloat16)


def generate_reference_io(
    mode: str,
    num_tokens: int,
    reference_model: DeepseekV3MoE,
    hf_config,
    weight_type: str,
    checkpoint_state_dict: dict[str, torch.Tensor] | None = None,
    module_path: str | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if weight_type == "random":
        # Preserve random-init dtypes, especially the fp32 gate score-correction bias.
        state_dict_out = _clone_state_dict(reference_model.state_dict())
        torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
    else:
        assert weight_type == "real"
        assert checkpoint_state_dict is not None
        assert module_path is not None
        moe_state_dict = {
            name: tensor
            for name, tensor in sub_state_dict(checkpoint_state_dict, module_path + ".").items()
            if not name.startswith("shared_experts.")
        }
        if not moe_state_dict:
            pytest.skip(f"Checkpoint does not contain routed MoE weights under '{module_path}'")

        state_dict_out = moe_state_dict
        reference_model.load_state_dict(state_dict_out)
        torch_input = load_real_moe_input(mode, module_path, num_tokens)

    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = reference_model(torch_input)

    return state_dict_out, torch_input, reference_output


_max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_prefill_seq_len = int(_max_seq_len_env) if _max_seq_len_env is not None else DEFAULT_PREFILL_SEQ_LEN


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": get_fabric_config()},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, batch_size_per_row, seq_len",
    [
        ("decode", USERS_PER_ROW, 1),
        ("prefill", 1, _prefill_seq_len),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
@pytest.mark.parametrize("weight_type", ["real"])
def test_forward_pass(
    device_params,
    mode,
    batch_size_per_row,
    seq_len,
    set_deterministic_env,
    reference_model,
    hf_config,
    request,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
    weight_type,
    force_recalculate_weight_config,
):
    """Test forward pass against reference model."""

    module_path = "model.layers.3.mlp" if weight_type == "real" else None
    checkpoint_state_dict = request.getfixturevalue("state_dict") if weight_type == "real" else None
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len
    state_dict, torch_input, reference_output = generate_reference_io(
        mode=mode,
        num_tokens=num_tokens,
        reference_model=reference_model,
        hf_config=hf_config,
        weight_type=weight_type,
        checkpoint_state_dict=checkpoint_state_dict,
        module_path=module_path,
    )

    weight_config = get_test_weight_config(
        MoE,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="test_moe",
        real_weights=weight_type == "real",
        layer_id=module_path,
    )

    model_config = get_model_config(
        MoE, mode, hf_config, mesh_device, device_params["fabric_config"], topk_fallback=topk_fallback
    )
    model_state = MoE.create_state(hf_config, mesh_device, ccl)
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(MoE, mode, tt_input, run_config, handle_tensor_parallel=True)

    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    logger.info(f"Mode: {mode}, Num tokens: {num_tokens}, Weight type: {weight_type}")
    assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.97)


if __name__ == "__main__":
    pytest.main([__file__])
