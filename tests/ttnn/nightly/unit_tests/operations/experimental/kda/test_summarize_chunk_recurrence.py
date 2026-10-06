# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA recurrence summaries."""

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
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]


@dataclass(frozen=True)
class _PerformanceCase:
    case_id: str
    batch_heads: int
    num_chunks: int
    dim: int
    expected_duration_ns: int


_PERF_REGRESSION_MARGIN = 0.05
_REGRESSION_CASE = _PerformanceCase(
    "bh8-n4-d32",
    batch_heads=8,
    num_chunks=4,
    dim=32,
    # Median of 5 Blackhole runs with the homogeneous-chain summary (g1b.5.17) and the shared per-chunk decay
    # diagonal (tt_metal_tracker-g1b.4.15).
    expected_duration_ns=19_212,
)
_PRODUCTION_CASE = _PerformanceCase(
    "pr7-leaf-bh96-n20-d128",
    batch_heads=96,
    num_chunks=20,
    dim=128,
    # Median of 5 Blackhole runs with the homogeneous-chain summary (g1b.5.17) and the shared per-chunk decay
    # diagonal (tt_metal_tracker-g1b.4.15).
    expected_duration_ns=301_008,
)
_PRODUCTION_BF16 = frozenset({"kd", "q_decay", "final_decay"})


def _summarize_chunk_recurrence_ops(
    inputs: Sequence[torch.Tensor | ttnn.Tensor],
    outputs: Sequence[torch.Tensor | ttnn.Tensor],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(inputs) != 7 or len(outputs) != 2:
        raise ValueError("chunk-recurrence summary requires seven inputs and two outputs")
    tensors = (*inputs, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("chunk-recurrence summary tensor shapes must be positive")
    if any(len(tensor.shape) != 4 for tensor in inputs):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")

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
        or key_dim != value_dim
        or any(tensor.shape != expected for tensor, expected in zip(inputs, expected_input_shapes, strict=True))
        or outputs[0].shape != (batch_heads, key_dim, key_dim)
        or outputs[1].shape != (batch_heads, key_dim, value_dim)
    ):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")

    instances = batch_heads * num_chunks
    return (
        perf_model.FpuOps(
            matrix_flops=instances * (8 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim),
            multiply_ops=instances * 2 * key_dim * value_dim,
            add_ops=instances * (2 * CHUNK_SIZE * value_dim + 2 * key_dim * value_dim)
            + batch_heads * key_dim * value_dim,
        ),
        perf_model.SfpuOps(),
    )


def _summarize_chunk_recurrence_performance(
    inputs: Sequence[ttnn.Tensor],
    outputs: Sequence[ttnn.Tensor],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _summarize_chunk_recurrence_ops(inputs, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=inputs,
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_summarize_chunk_recurrence_work_golden() -> None:
    inputs = (
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 32)),
        torch.empty((1, 1, 2, 32)),
        torch.empty((1, 1, 2, 1)),
        torch.empty((1, 1, 32, 32)),
    )
    fpu, sfpu = _summarize_chunk_recurrence_ops(
        inputs,
        (torch.empty((1, 2, 2)), torch.empty((1, 2, 2))),
    )

    assert fpu == perf_model.FpuOps(matrix_flops=9216, multiply_ops=8, add_ops=140)
    assert sfpu == perf_model.SfpuOps()


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "dim", "bf16_names"),
    [
        pytest.param(2, 1, 32, frozenset(), id="single-chunk-fp32"),
        pytest.param(4, 3, 64, BF16_ALLOWED, id="three-chunk-all-allowed-bf16"),
        pytest.param(8, 4, 32, frozenset({"v_beta", "kd", "final_decay"}), id="grouped-four-chunk"),
    ],
)
def test_summarize_chunk_recurrence_contract_trace_and_semantics(
    zero_actual_start,
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
        lambda: run_summary(inputs, actual_start=zero_actual_start),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.bfloat16, ttnn.bfloat16),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


def _regression_protocol(
    device: ttnn.Device,
    *,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[ttnn.Tensor, ...]]:
    case = _REGRESSION_CASE
    host_inputs = host_protocol(case.batch_heads, case.num_chunks, case.dim, case.dim, seed=seed)
    return host_inputs, device_protocol(host_inputs, device)


def _production_protocol(
    device: ttnn.Device,
    *,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[ttnn.Tensor, ...]]:
    case = _PRODUCTION_CASE
    host_inputs = host_protocol(
        case.batch_heads,
        case.num_chunks,
        case.dim,
        case.dim,
        bf16_names=_PRODUCTION_BF16,
        seed=seed,
    )
    return host_inputs, device_protocol(host_inputs, device)


def _production_compute_config(device: ttnn.Device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _log_summary_subtraction_conditioning(
    case_id: str,
    expected: tuple[torch.Tensor, torch.Tensor],
    actual: list[ttnn.Tensor],
) -> None:
    expected_a, expected_b = expected
    actual_a, actual_b = (ttnn.to_torch(output).float() for output in actual)
    raw_identity_state = expected_a + expected_b
    a_scale = expected_a.abs().max().item()
    subtraction_scale = max(raw_identity_state.abs().max().item(), expected_b.abs().max().item())
    a_max_abs_error = (actual_a - expected_a).abs().max().item()
    b_max_abs_error = (actual_b - expected_b).abs().max().item()
    logger.info(
        f"summary conditioning {case_id}: "
        f"A=[{expected_a.min().item():.6e},{expected_a.max().item():.6e}] max_abs={a_scale:.6e}; "
        f"B=[{expected_b.min().item():.6e},{expected_b.max().item():.6e}] "
        f"max_abs={expected_b.abs().max().item():.6e}; "
        f"raw_identity_state_max_abs={raw_identity_state.abs().max().item():.6e}; "
        f"subtraction_amplification={subtraction_scale / a_scale:.6e}; "
        f"A_max_abs_error={a_max_abs_error:.6e} ({a_max_abs_error / a_scale:.6e} of A scale); "
        f"B_max_abs_error={b_max_abs_error:.6e}"
    )
    assert_outputs_accurate(
        expected,
        actual,
        names=("affine_a", "affine_b"),
        context=f"summary subtraction conditioning {case_id}",
    )


@pytest.mark.parametrize("case_id", ["regression", "production"])
def test_summarize_chunk_recurrence_subtraction_conditioning(
    zero_actual_start, device: ttnn.Device, case_id: str
) -> None:
    if case_id == "production":
        host_inputs, inputs = _production_protocol(device, seed=117)
        outputs = run_summary(
            inputs, compute_kernel_config=_production_compute_config(device), actual_start=zero_actual_start
        )
    else:
        host_inputs, inputs = _regression_protocol(device, seed=117)
        outputs = run_summary(inputs, actual_start=zero_actual_start)
    _log_summary_subtraction_conditioning(case_id, summary_oracle(host_inputs), outputs)


# Outlier value channel in a long-memory, high-beta head (tt_metal_tracker-g1b.5.17, Qwen3.8-2.4T layer 0 V head 17 on
# real text: one value channel of the state reaches |S| ~ 166 against a head RMS ~ 2). The transition A must not
# depend on v_beta: formed as F(I) - B - I, the large B column cancels out of A's same column and leaves the chains'
# rounding there (device: 0.25 against true entries ~1e-2), which E @ carry multiplies by the state. A homogeneous
# identity-seeded chain keeps E's error independent of B (device on the real head: 4e-3).
_OUTLIER_CHANNEL = 62
_OUTLIER_VALUE = 90.0
# Peak |E error| gate. E's error comes from reading the near-identity transition chain through the source registers
# (about 2^-9 of the per-chunk erase term, summed over the group); it does not scale with B. On this protocol
# (|B| up to 197, |E| up to 0.97) the homogeneous chain measured 2.6e-3 and the F(I) - B - I form 0.21 in column 62.
_OUTLIER_E_MAX_ABS_ERROR = 0.02


def _outlier_channel_protocol(heads: int, chunks: int, dim: int, *, seed: int) -> tuple[torch.Tensor, ...]:
    """Prepared chunk terms of a scalar-decay delta rule (GDN form) in FP64, cast to the production dtypes.

    Unit-norm keys, beta in [0.85, 0.95], per-token log decay ~ -3e-4 (|G_last| ~ 1e-2 per chunk: long memory) and
    unit-scale values except one channel near _OUTLIER_VALUE.
    """
    generator = torch.Generator().manual_seed(seed)
    tokens = (heads, chunks, CHUNK_SIZE)
    k = torch.randn(*tokens, dim, generator=generator, dtype=torch.float64)
    k = k / k.norm(dim=-1, keepdim=True)
    q = torch.randn(*tokens, dim, generator=generator, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True) * dim**-0.5
    v = torch.randn(*tokens, dim, generator=generator, dtype=torch.float64)
    v[..., _OUTLIER_CHANNEL] = _OUTLIER_VALUE + 5.0 * v[..., _OUTLIER_CHANNEL]
    beta = 0.85 + 0.1 * torch.rand(*tokens, generator=generator, dtype=torch.float64)
    g = -3e-4 * (0.5 + torch.rand(*tokens, generator=generator, dtype=torch.float64))
    cumulative = torch.cumsum(g, dim=-1)
    final = cumulative[..., -1:]
    causal = torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool).tril()
    pairwise = torch.exp((cumulative[..., :, None] - cumulative[..., None, :]).masked_fill(~causal, float("-inf")))
    terms = {
        "v_beta": beta[..., None] * v,
        "kd": beta[..., None] * k * torch.exp(cumulative)[..., None],
        "q_decay": q * torch.exp(cumulative)[..., None],
        "intra": q @ k.transpose(-1, -2) * pairwise,
        "k_dec_t": (k * torch.exp(final - cumulative)[..., None]).transpose(-1, -2),
        "final_decay": torch.expm1(final)[..., None].expand(heads, chunks, dim, 1),
        "t_inv": torch.linalg.inv(
            torch.eye(CHUNK_SIZE, dtype=torch.float64)
            + torch.tril(beta[..., None] * (k @ k.transpose(-1, -2)) * pairwise, diagonal=-1)
        ),
    }
    return tuple(
        terms[name].to(torch.bfloat16) if name in _PRODUCTION_BF16 else terms[name].float() for name in PROTOCOL_NAMES
    )


def _homogeneous_summary_oracle(protocol: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 (E = A - I, B) of the protocol as given; A from the v_beta-free recurrence, independent of B."""
    v_beta, kd, _, _, k_dec_t, final_decay, t_inv = (tensor.double() for tensor in protocol)
    heads, chunks, _, dim = kd.shape
    identity = torch.eye(dim, dtype=torch.float64).expand(heads, dim, dim)
    transition, offset = identity.clone(), torch.zeros(heads, dim, v_beta.shape[-1], dtype=torch.float64)
    for chunk in range(chunks):
        write = k_dec_t[:, chunk] @ t_inv[:, chunk]
        step = identity + final_decay[:, chunk] * identity - write @ kd[:, chunk]
        transition = step @ transition
        offset = step @ offset + write @ v_beta[:, chunk]
    return transition - identity, offset


def test_summarize_chunk_recurrence_outlier_value_channel(zero_actual_start, device: ttnn.Device) -> None:
    heads, chunks, dim = 2, 10, 128  # one production summary group (10 chunks) at K = V = 128
    host_inputs = _outlier_channel_protocol(heads, chunks, dim, seed=517)
    expected_e, expected_b = _homogeneous_summary_oracle(host_inputs)
    affine_e, affine_b = (
        ttnn.to_torch(output).double()
        for output in run_summary(
            device_protocol(host_inputs, device),
            compute_kernel_config=_production_compute_config(device),
            actual_start=zero_actual_start,
        )
    )
    column_error = (affine_e - expected_e).abs().amax(dim=(0, 1))  # peak |E error| per column
    worst = int(column_error.argmax())
    logger.info(
        f"outlier-channel summary: |B| max {expected_b.abs().max().item():.1f} (column {_OUTLIER_CHANNEL}), "
        f"|E| max {expected_e.abs().max().item():.4f}; peak |E error| {column_error.max().item():.3e} at column "
        f"{worst}, column {_OUTLIER_CHANNEL} {column_error[_OUTLIER_CHANNEL].item():.3e}, other columns median "
        f"{column_error.median().item():.3e}; B peak rel error "
        f"{((affine_b - expected_b).abs().max() / expected_b.abs().max()).item():.3e}"
    )
    assert column_error.max().item() <= _OUTLIER_E_MAX_ABS_ERROR, (
        f"summary transition E column {worst} peak error {column_error.max().item():.3e} > "
        f"{_OUTLIER_E_MAX_ABS_ERROR} (|B| column {_OUTLIER_CHANNEL} ~ {expected_b.abs().max().item():.0f}: E must not "
        "inherit B's rounding)"
    )
    assert_accurate(expected_b.float(), affine_b.float(), name="outlier-channel summary B", pcc_threshold=0.9999)


def test_summarize_chunk_recurrence_is_device_deterministic(zero_actual_start, device: ttnn.Device) -> None:
    host_inputs, inputs = _regression_protocol(device, seed=1441)
    reference, outputs, mismatch_marker = collect_accuracy_and_determinism_results(
        device, lambda: run_summary(inputs, actual_start=zero_actual_start)
    )
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="summary outputs device-side exact-value determinism marker",
    )
    for name, golden, output in zip(("affine_a", "affine_b"), summary_oracle(host_inputs), outputs, strict=True):
        assert_accurate(
            golden.float(), output.float(), name=f"deterministic summary reference {name}", pcc_threshold=0.999
        )
    assert_summary_reconstructs_state(host_inputs, outputs[0], outputs[1])
    for output in reference:
        ttnn.deallocate(output)


def test_summarize_chunk_recurrence_cache_hit_rebinds_fresh_tensors(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    host_a, inputs_a = _regression_protocol(device, seed=1911)
    host_b, inputs_b = _regression_protocol(device, seed=1912)
    outputs_a = run_summary(inputs_a, actual_start=zero_actual_start)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = run_summary(inputs_b, actual_start=zero_actual_start)
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


def test_summarize_chunk_recurrence_default_compute_config_matches_explicit_defaults(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    _, inputs = _regression_protocol(device, seed=817)
    implicit = run_summary(inputs, actual_start=zero_actual_start)
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
    explicit = run_summary(inputs, compute_kernel_config=explicit_config, actual_start=zero_actual_start)
    assert device.num_program_cache_entries() == entries
    for name, implicit_tt, explicit_tt in zip(("affine_a", "affine_b"), implicit, explicit, strict=True):
        assert_bit_identical(ttnn.to_torch(implicit_tt), ttnn.to_torch(explicit_tt), name=f"{name} explicit defaults")


def test_summarize_chunk_recurrence_approximate_math_uses_distinct_accurate_program(
    zero_actual_start, device: ttnn.Device, isolated_program_cache: None
) -> None:
    host_inputs, inputs = _regression_protocol(device, seed=818)
    exact = run_summary(inputs, actual_start=zero_actual_start)
    entries = device.num_program_cache_entries()
    approximate_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    approximate = run_summary(inputs, compute_kernel_config=approximate_config, actual_start=zero_actual_start)
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
    zero_actual_start, device: ttnn.Device, expect_error: Callable
) -> None:
    _, inputs = _regression_protocol(device, seed=819)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        run_summary(inputs, compute_kernel_config=unsupported_config, actual_start=zero_actual_start)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_summarize_chunk_recurrence_regression_performance(zero_actual_start, device: ttnn.Device) -> None:
    case = _REGRESSION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrence-summary performance checks")
    _, inputs = _regression_protocol(device, seed=117)

    def run() -> list[ttnn.Tensor]:
        return run_summary(inputs, actual_start=zero_actual_start)

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(outputs[0].shape) == (case.batch_heads, case.dim, case.dim)
    performance = _summarize_chunk_recurrence_performance(
        inputs,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"recurrence summary regression {case.case_id}: measured_ns={duration_ns:.0f}, "
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
def test_summarize_chunk_recurrence_production_performance(zero_actual_start, device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for recurrence-summary performance checks")
    _, inputs = _production_protocol(device, seed=117)
    output_memory = group_summary_height_sharded(device, case.batch_heads, case.dim)
    compute_config = _production_compute_config(device)

    def run() -> list[ttnn.Tensor]:
        return run_summary(
            inputs, memory_config=output_memory, compute_kernel_config=compute_config, actual_start=zero_actual_start
        )

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert tuple(outputs[0].shape) == (case.batch_heads, case.dim, case.dim)
    assert outputs[0].memory_config() == output_memory
    performance = _summarize_chunk_recurrence_performance(
        inputs,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"recurrence summary production {case.case_id}: measured_ns={duration_ns:.0f}, "
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


def test_summarize_chunk_recurrence_height_sharded_l1_output(zero_actual_start, device: ttnn.Device) -> None:
    batch_heads, num_chunks, dim = 4, 2, 32
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, seed=812)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)
    output_memory = group_summary_height_sharded(device, batch_heads, dim)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs, memory_config=output_memory, actual_start=zero_actual_start),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.bfloat16, ttnn.bfloat16),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
        expected_memory_config=output_memory,
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


@pytest.mark.parametrize("host_index", range(7))
def test_summarize_chunk_recurrence_rejects_host_protocol_inputs(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, host_index: int
) -> None:
    host_inputs = host_protocol(2, 2, 32, 32)
    inputs = list(device_protocol(host_inputs, device))
    host = host_inputs[host_index]
    dtype = ttnn.bfloat16 if host.dtype == torch.bfloat16 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, f"{PROTOCOL_NAMES[host_index]} must be an allocated device tensor"):
        run_summary(tuple(inputs), actual_start=zero_actual_start)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("key_value_mismatch", "K must equal V"),
        ("q_decay_dtype", "q_decay must be FLOAT32 or BFLOAT16"),
        ("intra_dtype", "intra must be FLOAT32"),
    ],
)
def test_summarize_chunk_recurrence_rejects_invalid_inputs(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, case: str, message: str
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
        run_summary(tuple(inputs), memory_config=memory_config, actual_start=zero_actual_start)


@pytest.mark.parametrize(
    "removed_keyword",
    ["chunk_size", "initial_state", "state_only", "identity_tile", "summary_pair", "output_bf16", "raw_seed"],
)
def test_summarize_chunk_recurrence_does_not_expose_prototype_modes(
    zero_actual_start, device: ttnn.Device, expect_error: Callable, removed_keyword: str
) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs, actual_start=zero_actual_start, **{removed_keyword: True}
        )
