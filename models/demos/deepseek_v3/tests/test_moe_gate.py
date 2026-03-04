# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN
from models.demos.deepseek_v3.tt.grouped_moe_gate import MoEGate as GroupedMoEGate
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.tt.new_moe_gate import MoEGate as NewMoEGate
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, run_module_forward
from tests.ttnn.utils_for_testing import comp_pcc

_max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_prefill_seq_len = int(_max_seq_len_env) if _max_seq_len_env is not None else DEFAULT_PREFILL_SEQ_LEN


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
        ("prefill", _prefill_seq_len),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback,use_bitonic_sort",
    [
        (True, True),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    hf_config,
    topk_fallback,
    use_bitonic_sort,
    cache_path,
    mesh_device,
    set_deterministic_env,
):
    """Test forward pass against reference model."""

    batch_size = 1

    # Get state dict from actual model - pass directly to convert_weights
    torch.use_deterministic_algorithms(True)
    reference_model = ReferenceMoEGate(hf_config, use_bitonic_sort).eval()
    # If testing old MoE gate, remove below two lines and uncomment the two lines in the weight_config
    if hasattr(reference_model, "e_score_correction_bias"):
        reference_model.e_score_correction_bias.data = torch.zeros_like(reference_model.e_score_correction_bias.data)
    hf_state_dict = reference_model.state_dict()

    weight_config = get_test_weight_config(
        MoEGate,
        hf_config,
        (hf_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=True,
        test_name="test_new_moe_gate",
        real_weights=True,
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(
        MoEGate, mode, hf_config, mesh_device, topk_fallback=topk_fallback, use_bitonic_sort=use_bitonic_sort
    )

    # Create a new model state
    model_state = MoEGate.create_state(hf_config, mesh_device=mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Create input tensor
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    reference_topk_indices, reference_topk_weights = reference_model(torch_input)

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
    tt_topk_weights_new, tt_topk_indices_new = run_module_forward(NewMoEGate, mode, tt_input, run_config)
    tt_topk_weights_grouped, tt_topk_indices_grouped = run_module_forward(GroupedMoEGate, mode, tt_input, run_config)

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_topk_weights_memory_config = tt_topk_weights.memory_config()
    """
    assert (
        actual_topk_weights_memory_config == expected_output_memory_config
    ), f"TopK experts weights memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_weights_memory_config}"

    actual_topk_indices_memory_config = tt_topk_indices.memory_config()
    assert (
        actual_topk_indices_memory_config == expected_output_memory_config
    ), f"TopK experts indices memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_indices_memory_config}"
    """
    # Convert output back to torch
    tt_topk_weights_torch = ttnn.to_torch(
        tt_topk_weights,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_topk_indices_torch = ttnn.to_torch(
        tt_topk_indices,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_topk_weights_torch_new = ttnn.to_torch(
        tt_topk_weights_new,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_topk_indices_torch_new = ttnn.to_torch(
        tt_topk_indices_new,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_topk_weights_torch_grouped = ttnn.to_torch(
        tt_topk_weights_grouped,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_topk_indices_torch_grouped = ttnn.to_torch(
        tt_topk_indices_grouped,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)
    ttnn.deallocate(tt_topk_weights_new)
    ttnn.deallocate(tt_topk_indices_new)
    ttnn.deallocate(tt_topk_weights_grouped)
    ttnn.deallocate(tt_topk_indices_grouped)

    # Compare outputs
    logger.info(f"Mode: {mode}, Seq len: {seq_len}")

    topk_weights_pcc_required = 0.99

    reference_topk_weights = torch.sort(reference_topk_weights.to(torch.bfloat16), dim=-1, stable=True)[0]
    tt_topk_weights_torch = torch.sort(tt_topk_weights_torch, dim=-1, stable=True)[0]
    tt_topk_weights_torch_new = torch.sort(tt_topk_weights_torch_new, dim=-1, stable=True)[0]
    tt_topk_weights_torch_grouped = torch.sort(tt_topk_weights_torch_grouped, dim=-1, stable=True)[0]
    reference_topk_indices = torch.sort(reference_topk_indices.to(torch.int32), dim=-1, stable=True)[0]
    tt_topk_indices_torch = torch.sort(tt_topk_indices_torch.to(torch.int32), dim=-1, stable=True)[0]
    tt_topk_indices_torch_new = torch.sort(tt_topk_indices_torch_new.to(torch.int32), dim=-1, stable=True)[0]
    tt_topk_indices_torch_grouped = torch.sort(tt_topk_indices_torch_grouped.to(torch.int32), dim=-1, stable=True)[0]
    print("Original MoE Gate")
    print(f"TopK weights PCC: {comp_pcc(reference_topk_weights, tt_topk_weights_torch, topk_weights_pcc_required)}")
    temp = reference_topk_indices != tt_topk_indices_torch
    print(f"Number of indices that do not match: {temp.sum()}")
    print("New MoE Gate")
    print(f"TopK weights PCC: {comp_pcc(reference_topk_weights, tt_topk_weights_torch_new, topk_weights_pcc_required)}")
    temp = reference_topk_indices != tt_topk_indices_torch_new
    print(f"Number of indices that do not match: {temp.sum()}")
    print("Grouped MoE Gate")
    print(
        f"TopK weights PCC: {comp_pcc(reference_topk_weights, tt_topk_weights_torch_grouped, topk_weights_pcc_required)}"
    )
    temp = reference_topk_indices != tt_topk_indices_torch_grouped
    print(f"Number of indices that do not match: {temp.sum()}")
    import pdb

    pdb.set_trace()

    """
    logger.info(f"TopK experts weights PCC: {pcc_message}")
    assert (
        passing
    ), f"TopK experts weights output does not meet PCC requirement {topk_weights_pcc_required}: {pcc_message}"

    # stable sort both reference and ttnn indices to avoid random tie breaking for better comparison
    assert torch.equal(reference_topk_indices, tt_topk_indices_torch), "TopK experts indices output does not match"
    """


if __name__ == "__main__":
    pytest.main([__file__])
