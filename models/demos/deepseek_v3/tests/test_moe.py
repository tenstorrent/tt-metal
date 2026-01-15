# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    torch.use_deterministic_algorithms(True)
    # Note : Running Reference MoE without shared experts
    hf_config.n_shared_experts = None
    return DeepseekV3MoE(hf_config).eval()


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 128),
    ]
    + [("prefill", seq_len) for seq_len in PREFILL_SEQ_LENS],
)
def test_forward_pass(
    mode,
    seq_len,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
):
    """Test forward pass against reference model."""

    # Skip all prefill seq lengths except 128 to avoid exceeding CI workload time
    if mode == "prefill" and seq_len != 128:
        pytest.skip(
            f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
        )

    batch_size = 1

    # Get state dict from actual model - pass directly to convert_weights
    state_dict = add_inv_scale_to_state_dict(
        reference_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Create input tensor
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = reference_model(torch_input)

    weight_config = get_test_weight_config(
        MoE, hf_config, (state_dict,), cache_path, mesh_device, force_recalculate=False
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(MoE, mode, hf_config, mesh_device, topk_fallback=topk_fallback)

    # Create a new model state with CCL
    model_state = MoE.create_state(hf_config, mesh_device, ccl)

    # Create a new model shared state
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Convert input to TTNN, DP=4 and Replicated
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass using utility function
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(MoE, mode, tt_input, run_config)

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Compare outputs using utility function
    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
