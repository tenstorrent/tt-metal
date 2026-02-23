# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-device unit tests and device performance test for the fused MoE
compute kernel (moe_gpt).

Uses the weight preparation functions from
``models.demos.gpt_oss.tt.experts_throughput.weights`` to transform random
(or real) weights into the DRAM-sharded format required by the kernel, then
checks PCC against a torch reference with SwiGLU activation.

When the ``GPT_OSS_FUSED_MOE_KERNEL_DEVICE_PERF`` environment variable is
set, the functional test switches to device-perf mode: it runs warmup
iterations followed by signpost-bracketed measurement iterations (no
accuracy checks).  The companion ``test_gpt_oss_fused_moe_kernel_device_perf``
test invokes Tracy on that code path.

Dimensions (GPT-OSS 20b):
    M  = 32   (tokens)
    K  = 2880 (hidden_size)
    N  = 2880 (intermediate_size)
    E  = 4    (experts per device)
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tt.experts_throughput.weights import (
    extract_fused_moe_kernel_output,
    prepare_fused_moe_kernel_input,
    prepare_fused_moe_kernel_weights,
)

PCC_THRESHOLD = 0.984

# GPT-OSS dimensions
E = 4  # experts per device
K = 2880  # hidden_size
N = 2880  # intermediate_size
M = 32  # tokens

# Device performance constants
DEVICE_PERF_ENV_VAR = "GPT_OSS_FUSED_MOE_KERNEL_DEVICE_PERF"
PERF_WARMUP_ITERS = 10
DEVICE_PERF_ITERS = 10


def swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """GPT-OSS SwiGLU activation reference.

    result = (clamp(up, -7, 7) + 1) * clamp(gate, max=7) * sigmoid(1.702 * clamp(gate, max=7))
    """
    gate_c = torch.clamp(gate, max=clamp_limit)
    up_c = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def torch_reference(activation, w0, w1, w2, w0_bias=None, w1_bias=None, w2_bias=None):
    """Compute torch reference output: SwiGLU(act @ w0 + b0, act @ w1 + b1) @ w2 + b2."""
    w0_out = activation @ w0  # [E, M, N]
    w1_out = activation @ w1  # [E, M, N]
    if w0_bias is not None:
        w0_out = w0_out + w0_bias
    if w1_bias is not None:
        w1_out = w1_out + w1_bias
    intermediate = swiglu_reference(w0_out, w1_out)
    result = intermediate @ w2  # [E, M, K]
    if w2_bias is not None:
        result = result + w2_bias
    return result


def _load_expert_weights_for_layer(model_path, layer_idx=0, device_idx=0, num_devices=32):
    """Load and unfuse real expert weights for one layer and one device's experts.

    Loads the HF checkpoint, extracts expert weights for *layer_idx*, unfuses
    the interleaved ``gate_up_proj`` into separate gate (w0) and up (w1)
    tensors, and slices to the experts owned by *device_idx*.

    Returns:
        ``(w0, w1, w2, w0_bias, w1_bias, w2_bias, num_experts_per_device)``
        where w0/w1 are ``[E_local, K, N]`` and w2 is ``[E_local, N, K]``.
        Biases are ``[E_local, 1, dim]`` or ``None``.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from models.demos.gpt_oss.utils.substate import substate

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_experts = config.num_local_experts
    epd = num_experts // num_devices  # experts per device

    logger.info(f"Loading HF model from {model_path} for layer {layer_idx} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    state_dict = model.state_dict()
    del model  # free memory

    # Convert to bfloat16 if needed
    state_dict = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in state_dict.items()}

    layer_sd = substate(state_dict, f"model.layers.{layer_idx}.mlp.experts")

    # Unfuse gate_up_proj: [num_experts, hidden_size, 2*intermediate_size] (interleaved)
    gate_up = layer_sd["gate_up_proj"]  # [E_total, K, 2*N]
    w0_all = gate_up[..., ::2]  # gate: even columns [E_total, K, N]
    w1_all = gate_up[..., 1::2]  # up: odd columns   [E_total, K, N]
    w2_all = layer_sd["down_proj"]  # [E_total, N, K]

    # Biases
    w0_bias_all = None
    w1_bias_all = None
    w2_bias_all = None

    if "gate_up_proj_bias" in layer_sd:
        gate_up_bias = layer_sd["gate_up_proj_bias"]  # [E_total, 2*N]
        w0_bias_all = gate_up_bias[..., ::2].unsqueeze(1)  # [E_total, 1, N]
        w1_bias_all = gate_up_bias[..., 1::2].unsqueeze(1)  # [E_total, 1, N]

    if "down_proj_bias" in layer_sd:
        w2_bias_all = layer_sd["down_proj_bias"].unsqueeze(1)  # [E_total, 1, K]

    # Slice to this device's experts
    start = device_idx * epd
    end = start + epd
    w0 = w0_all[start:end].contiguous()
    w1 = w1_all[start:end].contiguous()
    w2 = w2_all[start:end].contiguous()

    w0_bias = w0_bias_all[start:end].contiguous() if w0_bias_all is not None else None
    w1_bias = w1_bias_all[start:end].contiguous() if w1_bias_all is not None else None
    w2_bias = w2_bias_all[start:end].contiguous() if w2_bias_all is not None else None

    return w0, w1, w2, w0_bias, w1_bias, w2_bias, epd


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW},
            id="dispatch_row",
        )
    ],
    indirect=True,
)
class TestFusedMoeKernel:
    """Tests for the fused MoE kernel with real weight preparation."""

    def test_fused_moe_kernel_accuracy(self, device):
        """L1 sharded output path: check PCC per expert >= threshold."""
        # Random weights
        w0 = torch.rand(E, K, N, dtype=torch.bfloat16) - 0.5
        w1 = torch.rand(E, K, N, dtype=torch.bfloat16) - 0.5
        w2 = torch.rand(E, N, K, dtype=torch.bfloat16) - 0.5

        # Prepare weights for fused kernel
        tt_w0_w1, tt_w2, ring2cores = prepare_fused_moe_kernel_weights(device, w0, w1, w2)

        # Random input
        activation = torch.rand(E, M, K, dtype=torch.bfloat16) - 0.5

        # Prepare input
        tt_input = prepare_fused_moe_kernel_input(device, activation, ring2cores)

        # Run fused kernel (output in-place on input tensor)
        ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=0,
        )

        # Extract output
        output = extract_fused_moe_kernel_output(tt_input, E, M, K, ring2cores)

        # Torch reference (zero biases)
        with torch.no_grad():
            output_ref = torch_reference(activation, w0, w1, w2)

        # Check PCC per expert
        passing = True
        for expert_id in range(E):
            _passed, pcc_val = comp_pcc(output_ref[expert_id], output[expert_id])
            if pcc_val < PCC_THRESHOLD:
                passing = False
                logger.warning(f"Expert {expert_id}: PCC={pcc_val:.6f} < {PCC_THRESHOLD}")
            else:
                logger.info(f"Expert {expert_id}: PCC={pcc_val:.6f} (Passed)")

        assert passing, "Some experts did not pass the PCC check"

    def test_fused_moe_kernel_dram_output(self, device):
        """DRAM output path (enable_dram_output=True): check shape and PCC."""
        w0 = torch.rand(E, K, N, dtype=torch.bfloat16) - 0.5
        w1 = torch.rand(E, K, N, dtype=torch.bfloat16) - 0.5
        w2 = torch.rand(E, N, K, dtype=torch.bfloat16) - 0.5

        tt_w0_w1, tt_w2, ring2cores = prepare_fused_moe_kernel_weights(device, w0, w1, w2)

        activation = torch.rand(E, M, K, dtype=torch.bfloat16) - 0.5
        tt_input = prepare_fused_moe_kernel_input(device, activation, ring2cores)

        # Run with DRAM output
        tt_dram_output = ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=0,
            enable_dram_output=True,
        )

        # Verify DRAM output properties
        assert tt_dram_output.shape == [E, 1, M, K], f"Expected shape [{E}, 1, {M}, {K}], got {tt_dram_output.shape}"
        assert tt_dram_output.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {tt_dram_output.layout}"

        # Convert to ROW_MAJOR for comparison
        tt_dram_output_rm = ttnn.to_layout(tt_dram_output, ttnn.ROW_MAJOR_LAYOUT)
        tt_to_torch_output = ttnn.to_torch(tt_dram_output_rm)  # (E, 1, M, K)
        tt_to_torch_output = tt_to_torch_output.squeeze(1)  # (E, M, K)

        # Torch reference
        with torch.no_grad():
            output_ref = torch_reference(activation, w0, w1, w2)

        # Check PCC per expert
        passing = True
        for expert_id in range(E):
            _passed, pcc_val = comp_pcc(output_ref[expert_id], tt_to_torch_output[expert_id])
            if pcc_val < PCC_THRESHOLD:
                passing = False
                logger.warning(f"DRAM output - Expert {expert_id}: PCC={pcc_val:.6f} < {PCC_THRESHOLD}")
            else:
                logger.info(f"DRAM output - Expert {expert_id}: PCC={pcc_val:.6f} (Passed)")

        assert passing, "Some experts did not pass the PCC check (DRAM output path)"

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("HF_MODEL", "")),
        reason="HF_MODEL not set or path does not exist — real weights unavailable",
    )
    @pytest.mark.parametrize("layer_idx", [0])
    def test_fused_moe_kernel_real_weights(self, device, layer_idx):
        """Test fused MoE kernel with real HF checkpoint weights.

        Requires ``HF_MODEL`` environment variable pointing to a GPT-OSS
        checkpoint (e.g. ``/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b``).

        The torch reference includes biases; the fused kernel does not support
        biases so we expect the PCC to be lower than the random-weight tests
        when biases are non-zero.
        """
        model_path = os.environ["HF_MODEL"]

        w0, w1, w2, w0_bias, w1_bias, w2_bias, epd = _load_expert_weights_for_layer(model_path, layer_idx=layer_idx)

        tt_w0_w1, tt_w2, ring2cores = prepare_fused_moe_kernel_weights(device, w0, w1, w2)

        activation = torch.rand(epd, M, w0.shape[1], dtype=torch.bfloat16) - 0.5
        tt_input = prepare_fused_moe_kernel_input(device, activation, ring2cores)

        ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=epd,
            layer_id=0,
        )

        output = extract_fused_moe_kernel_output(tt_input, epd, M, w0.shape[1], ring2cores)

        # Reference WITHOUT biases (matches what the kernel computes)
        with torch.no_grad():
            output_ref_no_bias = torch_reference(activation, w0, w1, w2)

        # Reference WITH biases (for informational comparison)
        with torch.no_grad():
            output_ref_with_bias = torch_reference(activation, w0, w1, w2, w0_bias, w1_bias, w2_bias)

        passing = True
        for expert_id in range(epd):
            _passed, pcc_no_bias = comp_pcc(output_ref_no_bias[expert_id], output[expert_id])
            _passed2, pcc_with_bias = comp_pcc(output_ref_with_bias[expert_id], output[expert_id])
            status = "Passed" if pcc_no_bias >= PCC_THRESHOLD else "FAILED"
            logger.info(
                f"Real weights L{layer_idx} Expert {expert_id}: "
                f"PCC(no bias)={pcc_no_bias:.6f}  PCC(with bias)={pcc_with_bias:.6f}  [{status}]"
            )
            if pcc_no_bias < PCC_THRESHOLD:
                passing = False

        assert passing, "Some experts did not pass the PCC check with real weights"


# ===========================================================================
# Standalone functional test (used by device perf profiler)
# ===========================================================================


@pytest.mark.parametrize(
    "mode, seq_len",
    [("decode", 1)],
)
@pytest.mark.parametrize("program_cache_enabled", [True], ids=["program_cache"])
@pytest.mark.parametrize("trace_mode", [False], ids=["eager"])
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW},
            id="dispatch_row",
        )
    ],
    indirect=True,
)
def test_gpt_oss_fused_moe_kernel(device, mode, seq_len, program_cache_enabled, trace_mode):
    """Functional test for fused MoE kernel, with device-perf mode support.

    When ``GPT_OSS_FUSED_MOE_KERNEL_DEVICE_PERF`` is set, runs warmup +
    signpost-bracketed iterations for Tracy profiling.  Otherwise, runs a
    single iteration with accuracy checks (same as
    ``TestFusedMoeKernel.test_fused_moe_kernel_accuracy``).

    The ``mode``, ``seq_len``, ``program_cache_enabled``, and ``trace_mode``
    parameters exist so the test ID contains the keywords expected by the
    unified device-perf runner's ``-k`` filter.  They are not used by the
    test logic itself (the fused kernel always operates on M=32 tokens in
    eager mode).
    """
    w0 = torch.rand(E, K, N, dtype=torch.bfloat16) - 0.5
    w1 = torch.rand(E, K, N, dtype=torch.bfloat16) - 0.5
    w2 = torch.rand(E, N, K, dtype=torch.bfloat16) - 0.5

    tt_w0_w1, tt_w2, ring2cores = prepare_fused_moe_kernel_weights(device, w0, w1, w2)

    activation = torch.rand(E, M, K, dtype=torch.bfloat16) - 0.5

    def run_once():
        tt_input = prepare_fused_moe_kernel_input(device, activation, ring2cores)
        ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=0,
        )
        return tt_input

    # Device performance measurement mode
    if os.getenv(DEVICE_PERF_ENV_VAR) is not None:
        logger.info("Device-perf mode: warmup + signposted iterations")
        from tracy import signpost

        for _ in range(PERF_WARMUP_ITERS):
            tt_out = run_once()
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)

        ttnn.synchronize_device(device)
        signpost("start")
        for _ in range(DEVICE_PERF_ITERS):
            tt_out = run_once()
            ttnn.synchronize_device(device)
            ttnn.deallocate(tt_out)
        signpost("stop")
        return

    # Standard accuracy mode
    tt_input = run_once()
    output = extract_fused_moe_kernel_output(tt_input, E, M, K, ring2cores)

    with torch.no_grad():
        output_ref = torch_reference(activation, w0, w1, w2)

    passing = True
    for expert_id in range(E):
        _passed, pcc_val = comp_pcc(output_ref[expert_id], output[expert_id])
        if pcc_val < PCC_THRESHOLD:
            passing = False
            logger.warning(f"Expert {expert_id}: PCC={pcc_val:.6f} < {PCC_THRESHOLD}")
        else:
            logger.info(f"Expert {expert_id}: PCC={pcc_val:.6f} (Passed)")

    assert passing, "Some experts did not pass the PCC check"


# ===========================================================================
# Device performance test (invokes Tracy on the functional test above)
# ===========================================================================


def test_gpt_oss_fused_moe_kernel_device_perf():
    """Device performance test for the fused MoE kernel.

    Runs ``test_gpt_oss_fused_moe_kernel`` under Tracy profiler and reports
    kernel duration and op-to-op latency.
    """
    from models.demos.gpt_oss.tests.fused_op_unit_tests.fused_op_device_perf_utils import (
        DEVICE_PERF_ITERS,
        collect_device_perf,
    )
    from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

    test_path = "models/demos/gpt_oss/tests/fused_op_unit_tests/test_gpt_oss_fused_moe_kernel.py"
    expr = "program_cache and not no_program_cache and eager and decode and 1"
    command = f'pytest {test_path}::test_gpt_oss_fused_moe_kernel -k "{expr}"'

    step_name = "gpt_oss_fused_moe_kernel_device_perf"

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    perf_profiler.start("run")
    perf_profiler.start(step_name)

    os.environ[DEVICE_PERF_ENV_VAR] = "1"
    try:
        op_stats, total_kernel_ns, total_op_to_op_ns = collect_device_perf(
            command,
            subdir="gpt_oss_fused_moe_kernel_device_perf",
            warmup_iters=0,
            use_signposts=True,
        )
    finally:
        os.environ.pop(DEVICE_PERF_ENV_VAR, None)

    perf_profiler.end(step_name)
    perf_profiler.end("run")

    assert op_stats, "No device perf stats captured."
    total_kernel_us = total_kernel_ns / 1000.0
    total_op_to_op_us = total_op_to_op_ns / 1000.0
    avg_kernel_us = total_kernel_us / DEVICE_PERF_ITERS
    avg_op_to_op_us = total_op_to_op_us / DEVICE_PERF_ITERS

    logger.info(f"Device perf per-op averages (ns): {json.dumps(op_stats, indent=2)}")
    logger.info(
        f"Device perf totals ({DEVICE_PERF_ITERS} iterations): "
        f"kernel={total_kernel_us:.3f} us, op_to_op={total_op_to_op_us:.3f} us"
    )
    logger.info(
        f"Device perf per-iteration averages: " f"kernel={avg_kernel_us:.3f} us, op_to_op={avg_op_to_op_us:.3f} us"
    )

    assert total_kernel_ns > 0, "Total kernel duration must be positive."

    benchmark_data.add_measurement(perf_profiler, 0, step_name, "total_kernel_duration_us", total_kernel_us)
    benchmark_data.add_measurement(perf_profiler, 0, step_name, "total_op_to_op_latency_us", total_op_to_op_us)
    benchmark_data.add_measurement(perf_profiler, 0, step_name, "avg_kernel_duration_us", avg_kernel_us)
    benchmark_data.add_measurement(perf_profiler, 0, step_name, "avg_op_to_op_latency_us", avg_op_to_op_us)

    benchmark_data.save_partial_run_json(
        perf_profiler,
        run_type="gpt_oss_fused_moe_kernel_device_perf",
        ml_model_name="gpt-oss",
        batch_size=M,
        input_sequence_length=M,
    )
