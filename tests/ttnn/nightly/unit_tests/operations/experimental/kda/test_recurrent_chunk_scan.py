# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA recurrent chunk scan."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    BF16_ALLOWED,
    CHUNK_SIZE,
    PROTOCOL_NAMES,
    assert_outputs_accurate,
    assert_runtime_contract,
    device_protocol,
    host_protocol,
    initial_state,
    one_core_height_sharded,
    recurrent_oracle,
    run_recurrent,
    to_device,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
    collect_accuracy_and_determinism_results,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 2_000_000}),
]


@dataclass(frozen=True)
class _PerformanceCase:
    case_id: str
    batch_heads: int
    num_chunks: int
    key_dim: int
    value_dim: int
    expected_duration_ns: int


_PERF_REGRESSION_MARGIN = 0.05
_REGRESSION_CASE = _PerformanceCase(
    "bh2-n4-k32-v64",
    batch_heads=2,
    num_chunks=4,
    key_dim=32,
    value_dim=64,
    expected_duration_ns=16_269,
)
_PRODUCTION_CASE = _PerformanceCase(
    "pr7-leaf-bh96-n20-k128-v128",
    batch_heads=96,
    num_chunks=20,
    key_dim=128,
    value_dim=128,
    expected_duration_ns=363_886,
)
_PRODUCTION_BF16 = frozenset({"kd", "q_decay", "final_decay"})


def _recurrent_chunk_scan_ops(
    inputs: Sequence[torch.Tensor | ttnn.Tensor],
    state: torch.Tensor | ttnn.Tensor,
    outputs: Sequence[torch.Tensor | ttnn.Tensor],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(inputs) != 7 or len(outputs) != 2:
        raise ValueError("recurrent chunk scan requires seven protocol inputs and two outputs")
    tensors = (*inputs, state, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("recurrent chunk-scan tensor shapes must be positive")
    if any(len(tensor.shape) != 4 for tensor in inputs) or len(state.shape) != 3:
        raise ValueError("recurrent chunk-scan tensor shapes are inconsistent")

    batch_heads, num_chunks, chunk_size, value_dim = inputs[0].shape
    key_dim = inputs[1].shape[-1]
    expected_input_shapes = (
        (batch_heads, num_chunks, CHUNK_SIZE, value_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, 1),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if (
        chunk_size != CHUNK_SIZE
        or any(tensor.shape != expected for tensor, expected in zip(inputs, expected_input_shapes, strict=True))
        or state.shape != (batch_heads, key_dim, value_dim)
        or outputs[0].shape != (batch_heads, num_chunks, CHUNK_SIZE, value_dim)
        or outputs[1].shape != state.shape
    ):
        raise ValueError("recurrent chunk-scan tensor shapes are inconsistent")

    instances = batch_heads * num_chunks
    return (
        perf_model.FpuOps(
            matrix_flops=instances * (6 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim),
            multiply_ops=instances * key_dim * value_dim,
            add_ops=instances * (2 * CHUNK_SIZE * value_dim + key_dim * value_dim),
        ),
        perf_model.SfpuOps(),
    )


def _recurrent_chunk_scan_performance(
    inputs: Sequence[ttnn.Tensor],
    state: ttnn.Tensor,
    outputs: Sequence[ttnn.Tensor],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _recurrent_chunk_scan_ops(inputs, state, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=(*inputs, state),
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_recurrent_chunk_scan_work_golden() -> None:
    inputs = (
        torch.empty((1, 1, 32, 1)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 32)),
        torch.empty((1, 1, 2, 32)),
        torch.empty((1, 1, 2, 1)),
        torch.empty((1, 1, 32, 32)),
    )
    fpu, sfpu = _recurrent_chunk_scan_ops(
        inputs,
        torch.empty((1, 2, 1)),
        (torch.empty((1, 1, 32, 1)), torch.empty((1, 2, 1))),
    )

    assert fpu == perf_model.FpuOps(matrix_flops=4480, multiply_ops=2, add_ops=66)
    assert sfpu == perf_model.SfpuOps()


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "key_dim", "value_dim", "bf16_names", "input_memory", "output_memory"),
    [
        pytest.param(
            2, 1, 32, 32, frozenset(), ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, id="single-chunk-fp32"
        ),
        pytest.param(
            2, 3, 32, 64, BF16_ALLOWED, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, id="three-chunk-all-allowed-bf16"
        ),
        pytest.param(
            2,
            4,
            32,
            64,
            frozenset({"kd", "q_decay", "final_decay"}),
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            id="production-four-chunk",
        ),
        pytest.param(6, 2, 64, 32, frozenset(), ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, id="grouped-batch-heads"),
        pytest.param(
            9,
            2,
            32,
            64,
            frozenset(),
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            id="value-sharding-above-eight-heads",
        ),
    ],
)
def test_recurrent_chunk_scan_contract_and_trace(
    device: ttnn.Device,
    batch_heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    bf16_names: frozenset[str],
    output_memory: ttnn.MemoryConfig,
    input_memory: ttnn.MemoryConfig,
) -> None:
    host_inputs = host_protocol(batch_heads, num_chunks, key_dim, value_dim, bf16_names=bf16_names)
    host_state = initial_state(batch_heads, key_dim, value_dim)
    expected = recurrent_oracle(host_inputs, host_state)
    inputs = tuple(to_device(tensor, device, memory_config=input_memory) for tensor in host_inputs)
    state = to_device(host_state, device, memory_config=input_memory)

    assert_runtime_contract(
        device,
        (*inputs, state),
        lambda: run_recurrent(inputs, state, memory_config=output_memory),
        expected,
        names=("token_output", "final_state"),
        dtypes=(ttnn.bfloat16, ttnn.float32),
        shapes=((batch_heads, num_chunks, CHUNK_SIZE, value_dim), (batch_heads, key_dim, value_dim)),
        expected_memory_config=output_memory,
    )


def _regression_inputs(
    device: ttnn.Device,
    *,
    protocol_seed: int,
    state_seed: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, tuple[ttnn.Tensor, ...], ttnn.Tensor]:
    case = _REGRESSION_CASE
    host_inputs = host_protocol(
        case.batch_heads,
        case.num_chunks,
        case.key_dim,
        case.value_dim,
        seed=protocol_seed,
    )
    host_state = initial_state(case.batch_heads, case.key_dim, case.value_dim, seed=state_seed)
    return host_inputs, host_state, device_protocol(host_inputs, device), to_device(host_state, device)


def _production_inputs(
    device: ttnn.Device,
    *,
    protocol_seed: int,
    state_seed: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, tuple[ttnn.Tensor, ...], ttnn.Tensor]:
    case = _PRODUCTION_CASE
    host_inputs = host_protocol(
        case.batch_heads,
        case.num_chunks,
        case.key_dim,
        case.value_dim,
        bf16_names=_PRODUCTION_BF16,
        seed=protocol_seed,
    )
    host_state = initial_state(case.batch_heads, case.key_dim, case.value_dim, seed=state_seed)
    return host_inputs, host_state, device_protocol(host_inputs, device), to_device(host_state, device)


def _production_compute_config(device: ttnn.Device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def test_recurrent_chunk_scan_is_device_deterministic(device: ttnn.Device) -> None:
    case = _REGRESSION_CASE
    host_inputs, host_state, inputs, state = _regression_inputs(device, protocol_seed=1441, state_seed=1442)
    expected = recurrent_oracle(host_inputs, host_state)
    reference, outputs, mismatch_marker = collect_accuracy_and_determinism_results(
        device, lambda: run_recurrent(inputs, state)
    )
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="recurrent outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("token_output", "final_state"), expected, outputs, strict=True):
        assert_accurate(golden, output, name=f"deterministic recurrent reference {name}", pcc_threshold=0.999)
    assert tuple(reference[0].shape) == (
        case.batch_heads,
        case.num_chunks,
        CHUNK_SIZE,
        case.value_dim,
    )
    for output in reference:
        ttnn.deallocate(output)


def test_recurrent_chunk_scan_cache_hit_rebinds_fresh_tensors(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _REGRESSION_CASE
    host_a, state_a_host, inputs_a, state_a = _regression_inputs(device, protocol_seed=1911, state_seed=1913)
    host_b, state_b_host, inputs_b, state_b = _regression_inputs(device, protocol_seed=1912, state_seed=1914)
    outputs_a = run_recurrent(inputs_a, state_a)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = run_recurrent(inputs_b, state_b)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(inputs_a, inputs_b, strict=True))
    assert state_a.buffer_address() != state_b.buffer_address()
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(outputs_a, outputs_b, strict=True))
    assert_outputs_accurate(
        recurrent_oracle(host_a, state_a_host),
        outputs_a,
        names=("token_output", "final_state"),
        context="cache miss tensors",
    )
    assert_outputs_accurate(
        recurrent_oracle(host_b, state_b_host),
        outputs_b,
        names=("token_output", "final_state"),
        context="cache hit fresh tensors",
    )


def test_recurrent_chunk_scan_default_compute_config_matches_explicit_defaults(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _REGRESSION_CASE
    _, _, inputs, state = _regression_inputs(device, protocol_seed=817, state_seed=817)
    implicit = run_recurrent(inputs, state)
    entries = device.num_program_cache_entries()
    explicit_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )
    explicit = run_recurrent(inputs, state, compute_kernel_config=explicit_config)
    assert device.num_program_cache_entries() == entries
    for name, implicit_tt, explicit_tt in zip(("token_output", "final_state"), implicit, explicit, strict=True):
        assert_bit_identical(ttnn.to_torch(implicit_tt), ttnn.to_torch(explicit_tt), name=f"{name} explicit defaults")


def test_recurrent_chunk_scan_approximate_math_uses_distinct_accurate_program(
    device: ttnn.Device, isolated_program_cache: None
) -> None:
    case = _REGRESSION_CASE
    host_inputs, host_state, inputs, state = _regression_inputs(device, protocol_seed=818, state_seed=818)
    exact = run_recurrent(inputs, state)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = run_recurrent(inputs, state, compute_kernel_config=approximate_config)
    assert device.num_program_cache_entries() == entries + 1
    expected = recurrent_oracle(host_inputs, host_state)
    assert_outputs_accurate(expected, exact, names=("token_output", "final_state"), context="exact recurrent math")
    assert_outputs_accurate(
        expected,
        approximate,
        names=("token_output", "final_state"),
        context="approximate recurrent math",
    )


def test_recurrent_chunk_scan_rejects_unsupported_compute_config(device: ttnn.Device, expect_error: Callable) -> None:
    case = _REGRESSION_CASE
    _, _, inputs, state = _regression_inputs(device, protocol_seed=819, state_seed=819)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        run_recurrent(inputs, state, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_recurrent_chunk_scan_regression_performance(device: ttnn.Device) -> None:
    case = _REGRESSION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrent chunk-scan performance checks")
    _, _, inputs, state = _regression_inputs(device, protocol_seed=117, state_seed=117)

    def run() -> list[ttnn.Tensor]:
        return run_recurrent(inputs, state)

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(outputs[0].shape) == (
        case.batch_heads,
        case.num_chunks,
        CHUNK_SIZE,
        case.value_dim,
    )
    performance = _recurrent_chunk_scan_performance(
        inputs,
        state,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"recurrent chunk scan regression {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    upper = case.expected_duration_ns * (1 + _PERF_REGRESSION_MARGIN)
    assert duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns exceeds {upper:.0f} ns "
        f"(reference {case.expected_duration_ns} ns, upper margin {_PERF_REGRESSION_MARGIN * 100:.0f}%)"
    )


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_recurrent_chunk_scan_production_performance(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrent chunk-scan performance checks")
    _, _, inputs, state = _production_inputs(device, protocol_seed=117, state_seed=117)
    compute_config = _production_compute_config(device)

    def run() -> list[ttnn.Tensor]:
        return run_recurrent(
            inputs,
            state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(outputs[0].shape) == (
        case.batch_heads,
        case.num_chunks,
        CHUNK_SIZE,
        case.value_dim,
    )
    assert outputs[0].memory_config() == ttnn.DRAM_MEMORY_CONFIG
    performance = _recurrent_chunk_scan_performance(
        inputs,
        state,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )
    logger.info(
        f"recurrent chunk scan production {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    upper = case.expected_duration_ns * (1 + _PERF_REGRESSION_MARGIN)
    assert duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns exceeds {upper:.0f} ns "
        f"(reference {case.expected_duration_ns} ns, upper margin {_PERF_REGRESSION_MARGIN * 100:.0f}%)"
    )


@pytest.mark.parametrize("host_index", range(7))
def test_recurrent_chunk_scan_rejects_host_protocol_inputs(
    device: ttnn.Device, expect_error: Callable, host_index: int
) -> None:
    host_inputs = host_protocol(2, 2, 32, 32)
    inputs = list(device_protocol(host_inputs, device))
    host = host_inputs[host_index]
    dtype = ttnn.bfloat16 if host.dtype == torch.bfloat16 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    state = to_device(initial_state(2, 32, 32), device)
    with expect_error(RuntimeError, f"{PROTOCOL_NAMES[host_index]} must be an allocated device tensor"):
        run_recurrent(tuple(inputs), state)


def test_recurrent_chunk_scan_rejects_host_initial_state(device: ttnn.Device, expect_error: Callable) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    state = ttnn.from_torch(initial_state(2, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "initial_state must be an allocated device tensor"):
        run_recurrent(inputs, state)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("v_beta_dtype", "v_beta must be FLOAT32 or BFLOAT16"),
        ("intra_dtype", "intra must be FLOAT32"),
        ("t_inv_dtype", "t_inv must be FLOAT32"),
        ("layout", "kd must use TILE layout"),
        ("rank", "v_beta must be rank 4"),
        ("shape", "kd shape mismatch"),
        ("chunk", "v_beta shape mismatch"),
        ("key_alignment", "K and V must be positive and tile aligned"),
        ("value_alignment", "K and V must be positive and tile aligned"),
        ("sharded", "v_beta must use interleaved memory"),
        ("state_dtype", "initial_state must be FLOAT32"),
        ("state_rank", "initial_state must be rank 3"),
        ("state_shape", "initial_state shape mismatch"),
        ("output_sharded", "output memory layout must be INTERLEAVED, got HEIGHT_SHARDED"),
    ],
)
def test_recurrent_chunk_scan_rejects_invalid_inputs(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    host_inputs = list(host_protocol(2, 2, 32, 32))
    inputs = list(device_protocol(host_inputs, device))
    host_state = initial_state(2, 32, 32)
    state = to_device(host_state, device)
    memory_config = None
    if case == "v_beta_dtype":
        inputs[0] = to_device(host_inputs[0], device, dtype=ttnn.bfloat8_b)
    elif case == "intra_dtype":
        inputs[3] = to_device(host_inputs[3], device, dtype=ttnn.bfloat16)
    elif case == "t_inv_dtype":
        inputs[6] = to_device(host_inputs[6], device, dtype=ttnn.bfloat16)
    elif case == "layout":
        inputs[1] = to_device(host_inputs[1], device, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "rank":
        inputs[0] = to_device(host_inputs[0].reshape(2, 64, 32), device)
    elif case == "shape":
        inputs[1] = to_device(host_inputs[1][:, :1], device)
    elif case == "chunk":
        inputs[0] = to_device(torch.randn(2, 2, 64, 32), device)
    elif case == "key_alignment":
        inputs[1] = to_device(torch.randn(2, 2, 32, 48), device)
    elif case == "value_alignment":
        inputs[0] = to_device(torch.randn(2, 2, 32, 48), device)
    elif case == "sharded":
        inputs[0] = to_device(host_inputs[0], device, memory_config=one_core_height_sharded((128, 32)))
    elif case == "state_dtype":
        state = to_device(host_state, device, dtype=ttnn.bfloat16)
    elif case == "state_rank":
        state = to_device(host_state.reshape(2, 1, 32, 32), device)
    elif case == "state_shape":
        state = to_device(host_state[:, :, :16], device)
    elif case == "output_sharded":
        memory_config = one_core_height_sharded((128, 32))
    with expect_error(RuntimeError, message):
        run_recurrent(tuple(inputs), state, memory_config=memory_config)


def test_recurrent_chunk_scan_requires_initial_state(device: ttnn.Device, expect_error: Callable) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.recurrent_chunk_scan(*inputs)


@pytest.mark.parametrize(
    "removed_keyword", ["chunk_size", "state_only", "identity_tile", "summary_pair", "output_bf16"]
)
def test_recurrent_chunk_scan_does_not_expose_prototype_modes(
    device: ttnn.Device, expect_error: Callable, removed_keyword: str
) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    state = to_device(initial_state(2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.recurrent_chunk_scan(*inputs, state, **{removed_keyword: True})
