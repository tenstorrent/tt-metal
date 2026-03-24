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
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
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
    checkpoint_state_dict: dict[str, torch.Tensor],
    module_path: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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
def test_forward_pass(
    mode,
    batch_size_per_row,
    seq_len,
    hf_config,
    request,
    cache_path,
    mesh_device,
    set_deterministic_env,
    force_recalculate_weight_config,
):
    """Test forward pass against reference model."""

    module_path = "model.layers.3.mlp"
    reference_model = ReferenceMoEGate(hf_config)
    checkpoint_state_dict = request.getfixturevalue("state_dict")
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len
    state_dict, torch_input, reference_topk_indices, reference_topk_weights = generate_reference_io(
        mode=mode,
        num_tokens=num_tokens,
        reference_model=reference_model,
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
        real_weights=True,
        layer_id=module_path,
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(MoEGate, mode, hf_config, mesh_device)

    # Create a new model state
    model_state = MoEGate.create_shared_state(mesh_device)

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

    # sort reference
    ref_weights = reference_topk_weights.to(torch.bfloat16)
    ref_indices = reference_topk_indices.to(torch.int32)

    ref_sorted_weights, ref_sort_idx = torch.sort(ref_weights, dim=-1, descending=True, stable=True)
    ref_sorted_indices = torch.gather(ref_indices, -1, ref_sort_idx)

    # sort tt
    tt_weights = tt_topk_weights_torch.to(torch.bfloat16)
    tt_indices = tt_topk_indices_torch.to(torch.int32)

    tt_sorted_weights, tt_sort_idx = torch.sort(tt_weights, dim=-1, descending=True, stable=True)
    tt_sorted_indices = torch.gather(tt_indices, -1, tt_sort_idx)

    # compare
    topk_weights_pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_sorted_weights, tt_sorted_weights, topk_weights_pcc_required)

    def count_different_indices_vectorized(ref_weights, ref_indices, tt_weights, tt_indices, rtol=1e-5, atol=1e-8):
        """
        Compute the number of different TopK indices while considering tie-breaking.

        Args:
            ref_weights (torch.Tensor): Reference weights [B, K], assumed sorted in descending order.
            ref_indices (torch.Tensor): Reference indices [B, K], aligned with weights.
            tt_weights (torch.Tensor): Target weights [B, K], assumed sorted in descending order.
            tt_indices (torch.Tensor): Target indices [B, K], aligned with weights.
            rtol (float): Relative tolerance for considering weights as tied.
            atol (float): Absolute tolerance for considering weights as tied.

        Returns:
            total_diff (int): Total number of positions where indices differ, considering tie-breaking.
            total_positions (int): Total number of positions (B*K).
            accuracy (float): Fraction of positions that match (tie-aware TopK accuracy).
        """
        B, K = ref_weights.shape
        total_positions = B * K
        total_diff = 0

        # Compute tie mask: True where adjacent weights are considered equal
        close_mask = torch.isclose(ref_weights[:, 1:], ref_weights[:, :-1], rtol=rtol, atol=atol)

        # Pad with False to mark the end of last group
        padded_mask = torch.cat([close_mask, torch.zeros((B, 1), dtype=torch.bool, device=ref_weights.device)], dim=1)

        # Loop over each row (batch) and count differences per tie group
        for i in range(B):
            start = 0
            for j in range(K):
                if not padded_mask[i, j]:
                    # Current tie group: indices from start to j (inclusive)
                    ref_set = set(ref_indices[i, start : j + 1].tolist())
                    tt_set = set(tt_indices[i, start : j + 1].tolist())
                    total_diff += len(ref_set - tt_set)  # count positions in ref not in tt
                    start = j + 1

        accuracy = 1 - total_diff / total_positions
        return total_diff, total_positions, accuracy

    total_diff, total_positions, accuracy = count_different_indices_vectorized(
        ref_sorted_weights, ref_sorted_indices, tt_sorted_weights, tt_sorted_indices
    )

    logger.info(f"TopK experts weights PCC: {pcc_message}")
    logger.info(f"TopK experts indices accuracy: {accuracy}")
    assert (
        passing
    ), f"TopK experts weights output does not meet PCC requirement {topk_weights_pcc_required}: {pcc_message}"

    assert accuracy >= 0.59, f"TopK experts indices output does not match: {accuracy}"
    # due to tie breaking, we cannot guarantee all the indices are the same as the pytorch version


if __name__ == "__main__":
    pytest.main([__file__])
