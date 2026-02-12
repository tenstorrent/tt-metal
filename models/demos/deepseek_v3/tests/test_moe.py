# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os

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

TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
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
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10485760},
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
    "mode,num_tokens",
    [
        ("decode", 128),
    ]
    + [
        ("prefill", seq_len)
        if seq_len == 128
        else pytest.param(
            "prefill",
            seq_len,
            marks=pytest.mark.skipif(
                CI_ACTIVE,
                reason=(
                    f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
                ),
            ),
        )
        for seq_len in PREFILL_SEQ_LENS
    ],
)
@pytest.mark.parametrize(
    "trace_mode",
    [
        pytest.param(False, id="eager"),
        pytest.param(True, id="tracing"),
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [
        pytest.param(True, id="real_weights"),
        pytest.param(False, id="random_weights"),
    ],
)
def test_forward_pass(
    mode,
    num_tokens,
    trace_mode,
    use_real_weights,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
):
    """Test forward pass against reference model."""
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")
    if trace_mode and topk_fallback:
        pytest.skip("Tracing not supported with topk_fallback.")

    if use_real_weights:
        model_for_reference = reference_model
    else:
        model_for_reference = DeepseekV3MoE(hf_config).eval()

    # Get state dict from model - pass directly to convert_weights
    state_dict = add_inv_scale_to_state_dict(
        model_for_reference.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Create input tensor
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward pass
    model_for_reference.eval()
    model_for_reference.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = model_for_reference(torch_input)

    weight_config = get_test_weight_config(
        MoE,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=not use_real_weights,
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

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    def to_torch_output(tt_output: ttnn.Tensor) -> torch.Tensor:
        expected_output_memory_config = run_config["output_memory_config"]
        actual_output_memory_config = tt_output.memory_config()
        assert (
            actual_output_memory_config == expected_output_memory_config
        ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"
        return ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )

    logger.info(f"Mode: {mode}, Num tokens: {num_tokens}")
    if trace_mode:
        # Iteration 0: eager compile run (not traced)
        tt_output = run_module_forward(MoE, mode, tt_input, run_config)
        ttnn.synchronize_device(mesh_device)
        tt_output_torch = to_torch_output(tt_output)
        assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.98)
        ttnn.deallocate(tt_output)

        # Reset CCL semaphore counters before trace capture
        ccl.reset_sem_counters()

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_module_forward(MoE, mode, tt_input, run_config)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        tt_output_torch = to_torch_output(trace_output)
        assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.98)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_output)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_output = run_module_forward(MoE, mode, tt_input, run_config)
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                tt_output_torch = to_torch_output(tt_output)
                assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.98)
            ttnn.deallocate(tt_output)

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
