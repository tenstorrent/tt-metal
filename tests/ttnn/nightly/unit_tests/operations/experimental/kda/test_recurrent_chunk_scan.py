# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA recurrent chunk scan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    BF16_ALLOWED,
    CHUNK_SIZE,
    PROTOCOL_NAMES,
    assert_device_deterministic,
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
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 2_000_000}], indirect=True),
]


@dataclass(frozen=True)
class _ProductionCase:
    case_id: str
    batch_heads: int
    num_chunks: int
    key_dim: int
    value_dim: int
    expected_duration_ns: int | None


_PRODUCTION_PERF_MARGIN = 0.05
_PRODUCTION_CASE = _ProductionCase(
    "bh2-n4-k32-v64",
    batch_heads=2,
    num_chunks=4,
    key_dim=32,
    value_dim=64,
    expected_duration_ns=16_917,
)


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "key_dim", "value_dim", "bf16_names", "output_memory"),
    [
        pytest.param(2, 1, 32, 32, frozenset(), ttnn.DRAM_MEMORY_CONFIG, id="single-chunk-fp32"),
        pytest.param(2, 3, 32, 64, BF16_ALLOWED, ttnn.L1_MEMORY_CONFIG, id="three-chunk-all-allowed-bf16"),
        pytest.param(
            2,
            4,
            32,
            64,
            frozenset({"kd", "q_decay", "final_decay"}),
            ttnn.DRAM_MEMORY_CONFIG,
            id="production-four-chunk",
        ),
        pytest.param(6, 2, 64, 32, frozenset(), ttnn.L1_MEMORY_CONFIG, id="grouped-batch-heads"),
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
) -> None:
    host_inputs = host_protocol(batch_heads, num_chunks, key_dim, value_dim, bf16_names=bf16_names)
    host_state = initial_state(batch_heads, key_dim, value_dim)
    expected = recurrent_oracle(host_inputs, host_state)
    inputs = device_protocol(host_inputs, device)
    state = to_device(host_state, device)

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
        seed=protocol_seed,
    )
    host_state = initial_state(case.batch_heads, case.key_dim, case.value_dim, seed=state_seed)
    return host_inputs, host_state, device_protocol(host_inputs, device), to_device(host_state, device)


def test_recurrent_chunk_scan_is_device_deterministic(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    host_inputs, host_state, inputs, state = _production_inputs(device, protocol_seed=1441, state_seed=1442)
    expected = recurrent_oracle(host_inputs, host_state)
    reference = assert_device_deterministic(
        device,
        lambda: run_recurrent(inputs, state),
        names=("token_output", "final_state"),
    )
    assert_outputs_accurate(
        expected,
        reference,
        names=("token_output", "final_state"),
        context="deterministic recurrent reference",
    )
    assert tuple(reference[0].shape) == (
        case.batch_heads,
        case.num_chunks,
        CHUNK_SIZE,
        case.value_dim,
    )


def test_recurrent_chunk_scan_cache_hit_rebinds_fresh_tensors(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    host_a, state_a_host, inputs_a, state_a = _production_inputs(device, protocol_seed=1911, state_seed=1913)
    host_b, state_b_host, inputs_b, state_b = _production_inputs(device, protocol_seed=1912, state_seed=1914)
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


def test_recurrent_chunk_scan_default_compute_config_matches_explicit_defaults(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    _, _, inputs, state = _production_inputs(device, protocol_seed=817, state_seed=817)
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


def test_recurrent_chunk_scan_approximate_math_uses_distinct_accurate_program(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    host_inputs, host_state, inputs, state = _production_inputs(device, protocol_seed=818, state_seed=818)
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
    case = _PRODUCTION_CASE
    _, _, inputs, state = _production_inputs(device, protocol_seed=819, state_seed=819)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        run_recurrent(inputs, state, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_recurrent_chunk_scan_production_performance(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrent chunk-scan performance checks")
    _, _, inputs, state = _production_inputs(device, protocol_seed=117, state_seed=117)

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
    logger.info(
        f"recurrent chunk scan {case.case_id}: duration={duration_ns:.0f} ns, "
        f"profiler_runtime_id={perf_record['runtime_id']}"
    )
    if case.expected_duration_ns is not None:
        lower = case.expected_duration_ns * (1 - _PRODUCTION_PERF_MARGIN)
        upper = case.expected_duration_ns * (1 + _PRODUCTION_PERF_MARGIN)
        assert lower <= duration_ns <= upper, (
            f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
            f"(reference {case.expected_duration_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
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
        ("output_sharded", "output memory must be interleaved"),
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
