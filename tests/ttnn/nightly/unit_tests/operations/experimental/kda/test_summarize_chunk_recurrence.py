# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA recurrence summaries."""

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
    PROTOCOL_NAMES,
    assert_outputs_accurate,
    assert_runtime_contract,
    assert_summary_reconstructs_state,
    device_protocol,
    group_summary_height_sharded,
    host_protocol,
    run_summary,
    summary_oracle,
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
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 2_000_000}], indirect=True),
]


@dataclass(frozen=True)
class _ProductionCase:
    case_id: str
    batch_heads: int
    num_chunks: int
    dim: int
    expected_duration_ns: int | None


_PRODUCTION_PERF_MARGIN = 0.05
_PRODUCTION_CASE = _ProductionCase(
    "bh8-n4-d32",
    batch_heads=8,
    num_chunks=4,
    dim=32,
    expected_duration_ns=26_214,
)


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "dim", "bf16_names"),
    [
        pytest.param(2, 1, 32, frozenset(), id="single-chunk-fp32"),
        pytest.param(4, 3, 64, BF16_ALLOWED, id="three-chunk-all-allowed-bf16"),
        pytest.param(8, 4, 32, frozenset({"v_beta", "kd", "final_decay"}), id="grouped-four-chunk"),
    ],
)
def test_summarize_chunk_recurrence_contract_trace_and_semantics(
    device: ttnn.Device,
    batch_heads: int,
    num_chunks: int,
    dim: int,
    bf16_names: frozenset[str],
) -> None:
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, bf16_names=bf16_names, seed=811)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.float32, ttnn.float32),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


def _production_protocol(
    device: ttnn.Device,
    *,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[ttnn.Tensor, ...]]:
    case = _PRODUCTION_CASE
    host_inputs = host_protocol(case.batch_heads, case.num_chunks, case.dim, case.dim, seed=seed)
    return host_inputs, device_protocol(host_inputs, device)


def test_summarize_chunk_recurrence_is_device_deterministic(device: ttnn.Device) -> None:
    host_inputs, inputs = _production_protocol(device, seed=1441)
    reference, outputs, mismatch_marker = collect_accuracy_and_determinism_results(device, lambda: run_summary(inputs))
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="summary outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("affine_a", "affine_b"), summary_oracle(host_inputs), outputs, strict=True):
        assert_accurate(golden, output, name=f"deterministic summary reference {name}", pcc_threshold=0.999)
    assert_summary_reconstructs_state(host_inputs, outputs[0], outputs[1])
    for output in reference:
        ttnn.deallocate(output)


def test_summarize_chunk_recurrence_cache_hit_rebinds_fresh_tensors(device: ttnn.Device) -> None:
    host_a, inputs_a = _production_protocol(device, seed=1911)
    host_b, inputs_b = _production_protocol(device, seed=1912)
    outputs_a = run_summary(inputs_a)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = run_summary(inputs_b)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(inputs_a, inputs_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(outputs_a, outputs_b, strict=True))
    assert_outputs_accurate(
        summary_oracle(host_a),
        outputs_a,
        names=("affine_a", "affine_b"),
        context="summary cache miss tensors",
    )
    assert_outputs_accurate(
        summary_oracle(host_b),
        outputs_b,
        names=("affine_a", "affine_b"),
        context="summary cache hit fresh tensors",
    )


def test_summarize_chunk_recurrence_default_compute_config_matches_explicit_defaults(device: ttnn.Device) -> None:
    _, inputs = _production_protocol(device, seed=817)
    implicit = run_summary(inputs)
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
    explicit = run_summary(inputs, compute_kernel_config=explicit_config)
    assert device.num_program_cache_entries() == entries
    for name, implicit_tt, explicit_tt in zip(("affine_a", "affine_b"), implicit, explicit, strict=True):
        assert_bit_identical(ttnn.to_torch(implicit_tt), ttnn.to_torch(explicit_tt), name=f"{name} explicit defaults")


def test_summarize_chunk_recurrence_approximate_math_uses_distinct_accurate_program(device: ttnn.Device) -> None:
    host_inputs, inputs = _production_protocol(device, seed=818)
    exact = run_summary(inputs)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = run_summary(inputs, compute_kernel_config=approximate_config)
    assert device.num_program_cache_entries() == entries + 1
    expected = summary_oracle(host_inputs)
    assert_outputs_accurate(expected, exact, names=("affine_a", "affine_b"), context="exact summary math")
    assert_outputs_accurate(
        expected,
        approximate,
        names=("affine_a", "affine_b"),
        context="approximate summary math",
    )


def test_summarize_chunk_recurrence_rejects_unsupported_compute_config(
    device: ttnn.Device, expect_error: Callable
) -> None:
    _, inputs = _production_protocol(device, seed=819)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        run_summary(inputs, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_summarize_chunk_recurrence_production_performance(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrence-summary performance checks")
    _, inputs = _production_protocol(device, seed=117)

    def run() -> list[ttnn.Tensor]:
        return run_summary(inputs)

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(outputs[0].shape) == (case.batch_heads, case.dim, case.dim)
    logger.info(
        f"recurrence summary {case.case_id}: duration={duration_ns:.0f} ns, "
        f"profiler_runtime_id={perf_record['runtime_id']}"
    )
    if case.expected_duration_ns is not None:
        lower = case.expected_duration_ns * (1 - _PRODUCTION_PERF_MARGIN)
        upper = case.expected_duration_ns * (1 + _PRODUCTION_PERF_MARGIN)
        assert lower <= duration_ns <= upper, (
            f"{case.case_id} duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
            f"(reference {case.expected_duration_ns} ns, margin +/- {_PRODUCTION_PERF_MARGIN * 100:.0f}%)"
        )


def test_summarize_chunk_recurrence_height_sharded_l1_output(device: ttnn.Device) -> None:
    batch_heads, num_chunks, dim = 4, 2, 32
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, seed=812)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)
    output_memory = group_summary_height_sharded(device, batch_heads, dim)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs, memory_config=output_memory),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.float32, ttnn.float32),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
        expected_memory_config=output_memory,
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


@pytest.mark.parametrize("host_index", range(7))
def test_summarize_chunk_recurrence_rejects_host_protocol_inputs(
    device: ttnn.Device, expect_error: Callable, host_index: int
) -> None:
    host_inputs = host_protocol(2, 2, 32, 32)
    inputs = list(device_protocol(host_inputs, device))
    host = host_inputs[host_index]
    dtype = ttnn.bfloat16 if host.dtype == torch.bfloat16 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, f"{PROTOCOL_NAMES[host_index]} must be an allocated device tensor"):
        run_summary(tuple(inputs))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("key_value_mismatch", "K must equal V"),
        ("q_decay_dtype", "q_decay must be FLOAT32 or BFLOAT16"),
        ("intra_dtype", "intra must be FLOAT32"),
    ],
)
def test_summarize_chunk_recurrence_rejects_invalid_inputs(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    host_inputs = list(host_protocol(2, 2, 32, 32))
    inputs = list(device_protocol(host_inputs, device))
    memory_config = None
    if case == "key_value_mismatch":
        host_inputs = list(host_protocol(2, 2, 32, 64))
        inputs = list(device_protocol(host_inputs, device))
    elif case == "q_decay_dtype":
        inputs[2] = to_device(host_inputs[2], device, dtype=ttnn.bfloat8_b)
    elif case == "intra_dtype":
        inputs[3] = to_device(host_inputs[3], device, dtype=ttnn.bfloat16)
    with expect_error(RuntimeError, message):
        run_summary(tuple(inputs), memory_config=memory_config)


@pytest.mark.parametrize(
    "removed_keyword",
    ["chunk_size", "initial_state", "state_only", "identity_tile", "summary_pair", "output_bf16", "raw_seed"],
)
def test_summarize_chunk_recurrence_does_not_expose_prototype_modes(
    device: ttnn.Device, expect_error: Callable, removed_keyword: str
) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, **{removed_keyword: True})
