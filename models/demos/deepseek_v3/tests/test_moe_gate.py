# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os

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

TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
)


@pytest.mark.parametrize(
    "mode,seq_len",
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
        pytest.param(False, marks=_CI_SKIP_MARK, id="random_weights"),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10485760},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "topk_fallback,use_bitonic_sort",
    [
        pytest.param(True, True, id="topk_fallback_true"),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    trace_mode,
    use_real_weights,
    hf_config,
    topk_fallback,
    use_bitonic_sort,
    cache_path,
    mesh_device,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """Test forward pass against reference model."""
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")
    if trace_mode and topk_fallback:
        pytest.skip("Tracing not supported with topk_fallback.")

    batch_size = 1
    effective_topk_fallback = topk_fallback and not trace_mode

    # Get state dict from actual model - pass directly to convert_weights
    torch.use_deterministic_algorithms(True)
    reference_model = ReferenceMoEGate(hf_config, use_bitonic_sort).eval()
    hf_state_dict = reference_model.state_dict()

    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        MoEGate,
        hf_config,
        (hf_state_dict,),
        weight_cache_root,
        mesh_device,
        force_recalculate=force_recalculate_weight_config or not use_real_weights,
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(
        MoEGate,
        mode,
        hf_config,
        mesh_device,
        topk_fallback=effective_topk_fallback,
        use_bitonic_sort=use_bitonic_sort,
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
    reference_topk_indices_short = reference_topk_indices.to(torch.short)
    reference_sort_idx = torch.argsort(reference_topk_indices_short, dim=-1, stable=True)
    reference_topk_indices_sorted = torch.gather(reference_topk_indices_short, -1, reference_sort_idx)
    reference_topk_weights_sorted = torch.gather(reference_topk_weights, -1, reference_sort_idx)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    def check_outputs(tt_topk_weights, tt_topk_indices) -> None:
        expected_output_memory_config = run_config["output_memory_config"]
        actual_topk_weights_memory_config = tt_topk_weights.memory_config()
        assert (
            actual_topk_weights_memory_config == expected_output_memory_config
        ), f"TopK experts weights memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_weights_memory_config}"

        actual_topk_indices_memory_config = tt_topk_indices.memory_config()
        assert (
            actual_topk_indices_memory_config == expected_output_memory_config
        ), f"TopK experts indices memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_indices_memory_config}"

        tt_topk_weights_torch = ttnn.to_torch(
            tt_topk_weights,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].squeeze(0)
        tt_topk_indices_torch = ttnn.to_torch(
            tt_topk_indices,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].squeeze(0)

        tt_topk_indices_short = tt_topk_indices_torch.to(torch.short)
        tt_sort_idx = torch.argsort(tt_topk_indices_short, dim=-1, stable=True)
        tt_topk_indices_sorted = torch.gather(tt_topk_indices_short, -1, tt_sort_idx)
        tt_topk_weights_sorted = torch.gather(tt_topk_weights_torch, -1, tt_sort_idx)

        topk_weights_pcc_required = 0.99
        passing, pcc_message = comp_pcc(
            reference_topk_weights_sorted, tt_topk_weights_sorted, topk_weights_pcc_required
        )
        logger.info(f"TopK experts weights PCC: {pcc_message}")
        assert (
            passing
        ), f"TopK experts weights output does not meet PCC requirement {topk_weights_pcc_required}: {pcc_message}"

        assert torch.allclose(
            reference_topk_indices_sorted, tt_topk_indices_sorted
        ), "TopK experts indices output does not match"

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    if trace_mode:
        tt_topk_weights, tt_topk_indices = run_module_forward(MoEGate, mode, tt_input, run_config)
        ttnn.synchronize_device(mesh_device)
        check_outputs(tt_topk_weights, tt_topk_indices)
        ttnn.deallocate(tt_topk_weights)
        ttnn.deallocate(tt_topk_indices)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_topk_weights, trace_topk_indices = run_module_forward(MoEGate, mode, tt_input, run_config)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        check_outputs(trace_topk_weights, trace_topk_indices)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_topk_weights)
        ttnn.deallocate(trace_topk_indices)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_topk_weights, tt_topk_indices = run_module_forward(MoEGate, mode, tt_input, run_config)
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                check_outputs(tt_topk_weights, tt_topk_indices)
            ttnn.deallocate(tt_topk_weights)
            ttnn.deallocate(tt_topk_indices)

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
