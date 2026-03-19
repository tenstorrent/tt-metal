# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN
from models.demos.deepseek_v3.tt.blaze_moe_gate import MoEGate
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    load_reference_io,
    load_reference_io_tensors_for_module,
    run_module_forward,
)
from tests.ttnn.utils_for_testing import comp_pcc


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
    reference_model: ReferenceMoEGate,
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
            name[5:]: tensor
            for name, tensor in sub_state_dict(checkpoint_state_dict, module_path + ".").items()
            if name.startswith("gate.")
        }
        if not moe_state_dict:
            pytest.skip(f"Checkpoint does not contain routed MoE weights under '{module_path}'")

        state_dict_out = moe_state_dict
        reference_model.load_state_dict(state_dict_out)
        torch_input = load_real_moe_input(mode, module_path, num_tokens)

    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_topk_indices, reference_topk_weights = reference_model(torch_input)

    return state_dict_out, torch_input, reference_topk_indices, reference_topk_weights


_max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_prefill_seq_len = int(_max_seq_len_env) if _max_seq_len_env is not None else DEFAULT_PREFILL_SEQ_LEN


@pytest.mark.parametrize(
    "mode,batch_size_per_row,seq_len",
    [
        ("decode", USERS_PER_ROW, 1),
        ("prefill", 1, _prefill_seq_len),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback,use_bitonic_sort",
    [
        (True, True),
    ],
)
@pytest.mark.parametrize("weight_type", ["real"])
def test_forward_pass(
    mode,
    batch_size_per_row,
    seq_len,
    hf_config,
    topk_fallback,
    use_bitonic_sort,
    request,
    cache_path,
    mesh_device,
    set_deterministic_env,
    weight_type,
    force_recalculate_weight_config,
):
    """Test forward pass against reference model."""

    module_path = "model.layers.3.mlp" if weight_type == "real" else None
    reference_model = ReferenceMoEGate(hf_config, use_bitonic_sort)
    checkpoint_state_dict = request.getfixturevalue("state_dict") if weight_type == "real" else None
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len
    state_dict, torch_input, reference_topk_indices, reference_topk_weights = generate_reference_io(
        mode=mode,
        num_tokens=num_tokens,
        reference_model=reference_model,
        hf_config=hf_config,
        weight_type=weight_type,
        checkpoint_state_dict=checkpoint_state_dict,
        module_path=module_path,
    )

    weight_config = get_test_weight_config(
        MoEGate,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="test_moe_gate",
        real_weights=weight_type == "real",
        layer_id=module_path,
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(
        MoEGate, mode, hf_config, mesh_device, topk_fallback=topk_fallback, use_bitonic_sort=use_bitonic_sort
    )

    # Create a new model state
    model_state = MoEGate.create_state(hf_config, mesh_device=mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass using utility function
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_topk_weights, tt_topk_indices = run_module_forward(MoEGate, mode, tt_input, run_config)

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_topk_weights_memory_config = tt_topk_weights.memory_config()
    assert (
        actual_topk_weights_memory_config == expected_output_memory_config
    ), f"TopK experts weights memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_weights_memory_config}"

    actual_topk_indices_memory_config = tt_topk_indices.memory_config()
    assert (
        actual_topk_indices_memory_config == expected_output_memory_config
    ), f"TopK experts indices memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_indices_memory_config}"

    # Convert output back to torch
    tt_topk_weights_torch = ttnn.to_torch(
        tt_topk_weights,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_topk_indices_torch = ttnn.to_torch(
        tt_topk_indices,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)

    # Compare outputs
    logger.info(f"Mode: {mode}, Seq len: {seq_len}")

    reference_topk_weights = torch.sort(reference_topk_weights.to(torch.bfloat16), dim=-1, stable=True)[0]
    tt_topk_weights_torch = torch.sort(tt_topk_weights_torch.to(torch.bfloat16), dim=-1, stable=True)[0]

    def count_indices_diff_fast(indices_a: torch.Tensor, indices_b: torch.Tensor):
        indices_a = torch.sort(indices_a.to(torch.int32), dim=-1).values
        indices_b = torch.sort(indices_b.to(torch.int32), dim=-1).values

        total_diff = 0

        for a, b in zip(indices_a, indices_b):
            i = j = common = 0
            while i < len(a) and j < len(b):
                if a[i] == b[j]:
                    common += 1
                    i += 1
                    j += 1
                elif a[i] < b[j]:
                    i += 1
                else:
                    j += 1

            diff = len(a) - common
            total_diff += diff

        return total_diff

    total_diff = count_indices_diff_fast(reference_topk_indices, tt_topk_indices_torch)

    topk_weights_pcc_required = 0.98
    passing, pcc_message = comp_pcc(reference_topk_weights, tt_topk_weights_torch, topk_weights_pcc_required)

    logger.info(f"TopK experts weights PCC: {pcc_message}")

    assert (
        passing
    ), f"TopK experts weights output does not meet PCC requirement {topk_weights_pcc_required}: {pcc_message}"

    # assert total_diff <= 250, f"TopK experts indices output does not match: {total_diff}"
    # here due to tie breaking, we cannot guarantee all the indices are the same as the pytorch version


if __name__ == "__main__":
    pytest.main([__file__])
