# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn

# Import from local reference files instead of HuggingFace
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
        ("prefill", 2048),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    set_deterministic_env,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
    test_timer,  # Added timing fixture
    test_cache_manager,  # Unified cache fixture
):
    """Test forward pass against reference model."""
    batch_size = 1

    with test_timer.time("1. state_dict_processing"):
        # Note: Running Reference MoE without shared experts
        hf_config.n_shared_experts = None
        # Get reference model from cache
        cache_key = "DeepseekV3MoE:random"
        reference_model = test_cache_manager.get_reference_model(DeepseekV3MoE, hf_config, None, cache_key)

        # Cache quantized state dict for random weights mode
        if test_cache_manager.use_random_weights:
            state_dict_key = f"MoE:quantized:{hf_config.quantization_config['weight_block_size']}"
            if state_dict_key not in test_cache_manager.dequantized_state_dicts:
                state_dict = add_inv_scale_to_state_dict(
                    reference_model.state_dict(),
                    block_shape=hf_config.quantization_config["weight_block_size"],
                )
                test_cache_manager.dequantized_state_dicts[state_dict_key] = state_dict
            else:
                state_dict = test_cache_manager.dequantized_state_dicts[state_dict_key]
        else:
            # Real weights path (if we add support later)
            state_dict = add_inv_scale_to_state_dict(
                reference_model.state_dict(),
                block_shape=hf_config.quantization_config["weight_block_size"],
            )

    with test_timer.time("2. reference_forward"):
        # Create input tensor
        torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

        # Reference forward pass
        reference_model.eval()
        reference_model.to(torch.bfloat16)
        with torch.no_grad():
            reference_output = reference_model(torch_input)

    with test_timer.time("3. get_test_weight_config"):
        weight_config = get_test_weight_config(
            MoE, hf_config, (state_dict,), cache_path, mesh_device, force_recalculate=False
        )

    with test_timer.time("4. get_model_config"):
        # Generate appropriate config using utility function
        model_config = get_model_config(MoE, mode, hf_config, mesh_device, topk_fallback=topk_fallback)

    with test_timer.time("5. create_state"):
        # Create a new model state with CCL
        model_state = MoE.create_state(hf_config, mesh_device, ccl)

    with test_timer.time("6. create_shared_state"):
        # Create a new model shared state
        model_shared_state = MoE.create_shared_state(hf_config, mesh_device)

    with test_timer.time("7. create_run_config"):
        # Create RunConfig using both weight_config and model_config
        cached_ttnn_weights = test_cache_manager.ttnn_weight_tensors if test_cache_manager else None
        run_config = create_run_config(
            model_config, weight_config, model_state, model_shared_state, cached_ttnn_weights=cached_ttnn_weights
        )

    with test_timer.time("8. ttnn_from_torch"):
        # Convert input to TTNN, DP=4 and Replicated
        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(1),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    with test_timer.time("9. ttnn_forward"):
        # TTNN forward pass using utility function
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = run_module_forward(MoE, mode, tt_input, run_config)

    with test_timer.time("10. verify_memory_config"):
        # Verify output memory config matches expected
        expected_output_memory_config = run_config["output_memory_config"]
        actual_output_memory_config = tt_output.memory_config()
        assert (
            actual_output_memory_config == expected_output_memory_config
        ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    with test_timer.time("11. ttnn_to_torch"):
        # Convert output back to torch
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )

    with test_timer.time("12. cleanup"):
        # Cleanup
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

    with test_timer.time("13. pcc_check"):
        # Compare outputs using utility function
        logger.info(f"Mode: {mode}, Seq len: {seq_len}")
        assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
