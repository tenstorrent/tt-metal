# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Fused op unit test for gpt_oss_moe (full MLP: router + experts).

This fused op tests the entire MLP module (MoE Mixture of Experts) which includes:
1. Router: TopKRouter that computes expert indices and routing weights
   - Linear projection from hidden_size to num_experts
   - TopK selection
   - Softmax normalization of routing weights
2. Experts: ThroughputExperts with all_to_all operations
   - all_to_all_dispatch - Route tokens to expert devices (CCL)
   - moe_expert_token_remap - Create sparsity pattern
   - Expert computation - Gate/Up/Down projections with sparse matmul + SwiGLU
   - all_to_all_combine - Route expert outputs back (CCL)
   - Apply routing weights and reduce across experts
   - all_reduce - Aggregate across columns (CCL)

Contains CCL ops so single device test is skipped.
"""


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from models.demos.gpt_oss.tests.test_factory import TestFactory
from models.demos.gpt_oss.tt.mlp import MLP
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

DEVICE_PERF_ENV_VAR = "GPT_OSS_MOE_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
PERF_MEASURE_ITERS = 100
DEVICE_PERF_ITERS = 10
DEVICE_PERF_MARGIN = 0.1


def gpt_oss_moe_reference(
    hidden_states: torch.Tensor,
    reference_mlp,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for gpt_oss_moe.

    Uses the HuggingFace GptOss MLP module (router + experts) as the reference.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        reference_mlp: HuggingFace GptOss MLP module (reference_layer.mlp)

    Returns:
        Tuple of (output tensor [batch, seq_len, hidden_size], routing_scores)
    """
    with torch.no_grad():
        output, routing_scores = reference_mlp(hidden_states)
    return output, routing_scores


def gpt_oss_moe_ttnn(
    hidden_states: ttnn.Tensor,
    tt_mlp: MLP,
    is_decode: bool,
) -> ttnn.Tensor:
    """TTNN implementation for gpt_oss_moe.

    This is the full MLP __call__ method which includes router + experts.

    Args:
        hidden_states: Input tensor [batch, 1, seq_len, hidden_size] in TTNN format
        tt_mlp: TTNN MLP module (router + experts)
        is_decode: Whether this is decode mode (seq_len=1) or prefill

    Returns:
        Output tensor [1, batch, seq_len, hidden_size] or similar
    """
    return tt_mlp(hidden_states, is_decode=is_decode)


def _compare_with_reference(
    tt_output: torch.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    atol: float,
    rtol: float,
    name: str = "",
) -> tuple[bool, float]:
    """Compare TT output with reference, returning pass status and PCC."""
    passing, pcc = comp_pcc(ref_output.float(), tt_output.float(), expected_pcc)
    logger.info(f"PCC {name}: {pcc}")
    return passing, pcc


def _measure_perf_us(
    mesh_device: ttnn.MeshDevice,
    op_fn,
    warmup_iters: int,
    measure_iters: int,
    trace_mode: bool = False,
) -> float:
    """Measure performance in microseconds."""
    ttnn.synchronize_device(mesh_device)

    if trace_mode:
        # Initial run to allocate tensors
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(warmup_iters):
                output = op_fn()
                ttnn.deallocate(output)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(measure_iters):
            output = op_fn()
            ttnn.deallocate(output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        profiler.clear()
        profiler.start("gpt_oss_moe_perf")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiler.end("gpt_oss_moe_perf", PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get("gpt_oss_moe_perf") * 1e6

    # Non-trace mode
    for _ in range(warmup_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    profiler.clear()
    profiler.start("gpt_oss_moe_perf")
    for _ in range(measure_iters):
        output = op_fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    profiler.end("gpt_oss_moe_perf", PERF_CNT=measure_iters)
    return profiler.get("gpt_oss_moe_perf") * 1e6


def _create_reference_mlp_and_state_dict(config, layer_idx=0):
    """Create HuggingFace reference MLP and extract state dict.

    Returns the reference MLP module and its state dict.
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    reference_layer = GptOssDecoderLayer(config, layer_idx=layer_idx)
    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)

    reference_mlp = reference_layer.mlp
    reference_mlp.eval()

    # Extract the MLP state dict
    mlp_state_dict = {
        name: param.clone() for name, param in reference_layer.state_dict().items() if name.startswith("mlp.")
    }
    # Remove 'mlp.' prefix
    mlp_state_dict = {name.replace("mlp.", ""): param for name, param in mlp_state_dict.items()}

    return reference_mlp, mlp_state_dict


def _run_moe_test(
    mesh_device: ttnn.MeshDevice,
    config,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    expected_pcc: float,
    expected_atol: float,
    expected_rtol: float,
    expected_perf_us: float,
    trace_mode: bool,
    program_cache_enabled: bool,
    use_real_weights: bool,
    step_prefix: str,
):
    """Run the full MoE fused op test."""

    # Determine batch per device and row sharding
    mesh_shape = mesh_device.shape
    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_shape[0] == 0, "Batch size must be divisible by mesh rows"
        batch_size_per_device = batch_size // mesh_shape[0]
    else:
        is_row_sharded = False
        batch_size_per_device = batch_size

    is_decode = seq_len == 1

    # Create reference MLP and state dict
    reference_mlp, mlp_state_dict = _create_reference_mlp_and_state_dict(config, layer_idx=0)

    # Create input tensor
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Get reference output (uses float32)
    ref_output, routing_scores = gpt_oss_moe_reference(hidden_states, reference_mlp)

    # Convert to bfloat16 for TTNN
    hidden_states_bf16 = hidden_states.to(torch.bfloat16)

    # Create TTNN MLP
    # For throughput experts, need high throughput mode when using 4x8 mesh
    use_throughput_experts = mesh_shape[0] > 1 and batch_size * seq_len > 1

    tt_mlp = MLP(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=mlp_state_dict,
        ccl_manager=None,  # Not used for throughput experts
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        use_throughput_experts=use_throughput_experts,
    )

    # Convert input to TTNN tensor
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states_bf16.unsqueeze(1),  # [B, 1, S, H]
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # Run TTNN implementation
    tt_output = gpt_oss_moe_ttnn(tt_hidden_states, tt_mlp, is_decode=is_decode)

    # Convert output to torch
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., : batch_size * seq_len, :hidden_size]

    # Compare with reference
    passing, pcc = _compare_with_reference(
        tt_output_torch,
        ref_output,
        expected_pcc,
        expected_atol,
        expected_rtol,
        "moe",
    )
    assert passing, f"MoE test failed. PCC: {pcc} < {expected_pcc}"

    # Performance measurement
    if not trace_mode or program_cache_enabled:
        perf_profiler = BenchmarkProfiler()
        benchmark_data = BenchmarkData()
        trace_suffix = "trace" if trace_mode else "no_trace"
        cache_suffix = "pcache" if program_cache_enabled else "no_pcache"
        step_name = f"{step_prefix}_{trace_suffix}_{cache_suffix}"

        def op_fn():
            return gpt_oss_moe_ttnn(tt_hidden_states, tt_mlp, is_decode=is_decode)

        perf_profiler.start("run")
        perf_profiler.start(step_name)
        perf_us = _measure_perf_us(
            mesh_device,
            op_fn,
            PERF_WARMUP_ITERS,
            PERF_MEASURE_ITERS,
            trace_mode=trace_mode,
        )
        logger.info(f"Perf avg: {perf_us:.3f} us over {PERF_MEASURE_ITERS} iters (warmup {PERF_WARMUP_ITERS})")
        perf_profiler.end(step_name)
        perf_profiler.end("run")

        benchmark_data.add_measurement(
            perf_profiler,
            0,
            step_name,
            f"{step_name}-avg_us",
            perf_us,
            step_warm_up_num_iterations=PERF_WARMUP_ITERS,
            target=expected_perf_us if expected_perf_us > 0 and not trace_mode and program_cache_enabled else None,
        )
        benchmark_data.save_partial_run_json(
            perf_profiler,
            run_type="gpt_oss_fused_ops",
            ml_model_name="gpt-oss",
            batch_size=batch_size,
            input_sequence_length=seq_len,
        )

        if expected_perf_us > 0 and not trace_mode and program_cache_enabled:
            perf_margin = 0.2
            assert perf_us <= expected_perf_us * (
                1 + perf_margin
            ), f"Perf regression: {perf_us:.3f}us exceeds expected {expected_perf_us:.3f}us"
        elif expected_perf_us == 0 and not trace_mode and program_cache_enabled:
            logger.warning("TODO: Set expected_perf_us using a measured baseline.")

    return pcc


def _skip_single_device_ccl():
    """Skip single device test because this fused op contains CCL ops."""
    pytest.skip(
        "Single-device test is not applicable because gpt_oss_moe includes CCL ops "
        "(all_to_all_dispatch, all_to_all_combine, all_reduce in throughput experts)."
    )


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        # Decode mode - batch 128, seq_len 1
        # TODO: Replace expected_perf_us baselines with theoretical targets.
        ("decode", 1, 0.92, 0.5, 0.5, 168570.044),  # Measured PCC from module test: 0.927
        # Prefill mode - batch 1, seq_len 128
        # ("prefill", 128, 0.92, 0.5, 0.5, 0.0),  # TODO: Enable once prefill is verified
    ],
)
@pytest.mark.parametrize("use_real_weights", [False], ids=["random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 8)],
    ids=["mesh_4x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_gpt_oss_moe(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Test the gpt_oss_moe fused op (full MLP: router + experts).

    This tests the complete MLP forward pass including:
    - TopK Router (linear projection + topk + softmax)
    - ThroughputExperts with all_to_all operations (CCL)
    - Sparse matmul (gate/up/down) + SwiGLU
    - Routing weight application and reduction
    """
    if not program_cache_enabled:
        mesh_device.disable_and_clear_program_cache()

    # Get HF config from TestFactory
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Use config values for dimensions
    hidden_size = config.hidden_size

    # Batch size depends on mode
    if mode == "decode":
        assert seq_len == 1, "Decode mode always has seq_len=1"
        batch_size = 128  # matches decode_128 test configuration
    else:  # prefill
        batch_size = 1

    # Run test
    pcc = _run_moe_test(
        mesh_device,
        config,
        batch_size,
        seq_len,
        hidden_size,
        expected_pcc,
        expected_atol,
        expected_rtol,
        expected_perf_us,
        trace_mode,
        program_cache_enabled,
        use_real_weights,
        f"gpt_oss_moe_{mode}_seq{seq_len}",
    )

    logger.info(f"Test passed with PCC: {pcc}")


@pytest.mark.parametrize(
    "mode, seq_len, expected_pcc, expected_atol, expected_rtol, expected_perf_us",
    [
        ("decode", 1, 0.92, 0.5, 0.5, 168570.044),
    ],
)
@pytest.mark.parametrize("use_real_weights", [False], ids=["random_weights"])
@pytest.mark.parametrize("program_cache_enabled", [True, False], ids=["program_cache", "no_program_cache"])
@pytest.mark.parametrize("trace_mode", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 8)],
    ids=["mesh_4x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_gpt_oss_moe_single_device(
    mode,
    seq_len,
    expected_pcc,
    expected_atol,
    expected_rtol,
    expected_perf_us,
    use_real_weights,
    program_cache_enabled,
    trace_mode,
    mesh_device,
):
    """Single device test for gpt_oss_moe.

    This test is skipped because gpt_oss_moe contains CCL ops:
    - all_to_all_dispatch
    - all_to_all_combine
    - all_reduce
    """
    _skip_single_device_ccl()


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
        # ("prefill", 128),  # TODO: Enable once prefill is verified
    ],
)
def test_gpt_oss_moe_device_perf(mode, seq_len):
    """Device performance test for gpt_oss_moe.

    This test runs the device profiler to measure kernel duration and op-to-op latency.
    """
    # This test will be enabled after verifying the main test
    pytest.skip("Device perf test not yet implemented - run main test first")


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_gpt_oss_moe_single_device_device_perf(mode, seq_len):
    """Single device device performance test for gpt_oss_moe."""
    _skip_single_device_ccl()
