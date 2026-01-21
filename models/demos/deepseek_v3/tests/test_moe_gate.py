# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, run_module_forward
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
    ]
    + [("prefill", seq_len) for seq_len in PREFILL_SEQ_LENS],
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

    # Skip all prefill seq lengths except 128 to avoid exceeding CI workload time
    if mode == "prefill" and seq_len != 128:
        pytest.skip(
            f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
        )
    batch_size = 1

    # Get state dict from actual model - pass directly to convert_weights
    torch.use_deterministic_algorithms(True)
    reference_model = ReferenceMoEGate(hf_config, use_bitonic_sort).eval()
    hf_state_dict = reference_model.state_dict()

    weight_config = get_test_weight_config(
        MoEGate, hf_config, (hf_state_dict,), cache_path, mesh_device, force_recalculate=False
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

    topk_weights_pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_topk_weights, tt_topk_weights_torch, topk_weights_pcc_required)

    logger.info(f"TopK experts weights PCC: {pcc_message}")
    assert (
        passing
    ), f"TopK experts weights output does not meet PCC requirement {topk_weights_pcc_required}: {pcc_message}"

    topk_indices_pcc_required = 1.0
    # stable sort both reference and ttnn indices to avoid random tie breaking for better comparison
    reference_topk_indices = torch.sort(reference_topk_indices.to(torch.short), dim=-1, stable=True)[0]
    tt_topk_indices_torch = torch.sort(tt_topk_indices_torch, dim=-1, stable=True)[0]
    assert torch.allclose(reference_topk_indices, tt_topk_indices_torch), f"TopK experts indices output does not match"


if __name__ == "__main__":
    pytest.main([__file__])
