# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Quick smoke test for MoE on single galaxy (TG).

This is a simplified version of the full TG test for rapid validation during development.
Run this for quick feedback, then run the full test suite before merging.

Usage:
    export MESH_DEVICE=TG
    pytest models/demos/deepseek_v3/tests/test_moe_tg_smoke.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tests.test_moe_single_galaxy import (
    TG_NUM_EXPERTS,
    _clone_state_dict,
    create_scaled_config,
    validate_mesh_for_tg_test,
)
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_fabric_config
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)


@pytest.mark.requires_device("TG")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": get_fabric_config()}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode",
    ["decode", "prefill"],
)
def test_moe_tg_smoke(
    device_params,
    mode,
    set_deterministic_env,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
):
    """
    Quick smoke test for MoE on TG.

    Tests basic functionality with random weights for rapid validation.
    """
    validate_mesh_for_tg_test(mesh_device)

    # Create scaled config
    scaled_config = create_scaled_config(hf_config, num_experts=TG_NUM_EXPERTS)

    # Create reference model
    torch.use_deterministic_algorithms(True)
    moe_config = create_scaled_config(hf_config, num_experts=TG_NUM_EXPERTS)
    moe_config.n_shared_experts = None
    reference_model = DeepseekV3MoE(moe_config).eval()

    # Simple test config
    seq_len = 512 if mode == "prefill" else 1
    batch_size_per_row = 1 if mode == "prefill" else USERS_PER_ROW
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len

    # Generate random input
    state_dict = _clone_state_dict(reference_model.state_dict())
    torch_input = torch.randn(1, num_tokens, scaled_config.hidden_size, dtype=torch.bfloat16)

    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = reference_model(torch_input)

    # Setup model
    weight_config = get_test_weight_config(
        MoE,
        scaled_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="smoke_test",
        real_weights=False,
    )

    model_config = get_model_config(
        MoE, mode, scaled_config, mesh_device, device_params["fabric_config"], topk_fallback=True
    )
    model_state = MoE.create_state(scaled_config, mesh_device, ccl)
    model_shared_state = MoE.create_shared_state(scaled_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    logger.info(f"Smoke test: mode={mode}, num_tokens={num_tokens}, experts={TG_NUM_EXPERTS}")

    # Run forward pass
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

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Validate
    assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.97)
    logger.info("✓ Smoke test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
