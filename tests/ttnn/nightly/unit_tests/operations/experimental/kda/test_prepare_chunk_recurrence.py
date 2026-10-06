# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA chunk-recurrence preparation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
    collect_accuracy_and_determinism_results,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device,
]

CHUNK_SIZE = 32
OUTPUT_NAMES = ("v_beta", "kd", "q_decay", "intra", "k_dec_t", "final_decay", "t_inv")


@dataclass(frozen=True)
class _TestCase:
    case_id: str
    num_heads: int
    num_chunks: int
    key_dim: int
    value_dim: int


_PERFORMANCE_MARGIN = 0.05
_PRODUCTION_OUTPUT_BF16_MASK = 0x26
_PRODUCTION_EXPECTED_DURATION_NS = 816_534
_T_INV_MAX_ABS = 0.01
_UNIT_TEST_CASE = _TestCase("unit-h2-n4-k32-v64", 2, 4, 32, 64)
_PRODUCTION_CASE = _TestCase("sp2-tp4-h24-n80-k128-v128", 24, 80, 128, 128)


def _prepare_chunk_recurrence_ops(
    inputs: Sequence[torch.Tensor | ttnn.Tensor],
    outputs: Sequence[torch.Tensor | ttnn.Tensor],
) -> tuple[perf_model.FpuOps, perf_model.SfpuOps]:
    if len(inputs) != 5 or len(outputs) != 7:
        raise ValueError("chunk-recurrence preparation requires five inputs and seven outputs")
    tensors = (*inputs, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("chunk-recurrence preparation tensor shapes must be positive")

    q, k, v, g, beta = inputs
    if len(q.shape) != 3 or len(v.shape) != 3 or len(beta.shape) != 4:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    num_heads, num_chunks, chunk_size, trailing = beta.shape
    if chunk_size != CHUNK_SIZE or trailing != 1 or q.shape[-1] % num_heads or v.shape[-1] % num_heads:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    key_dim = q.shape[-1] // num_heads
    value_dim = v.shape[-1] // num_heads
    if (
        q.shape != (1, num_chunks * CHUNK_SIZE, num_heads * key_dim)
        or k.shape != q.shape
        or g.shape != q.shape
        or v.shape != (1, num_chunks * CHUNK_SIZE, num_heads * value_dim)
    ):
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    expected_output_shapes = (
        (num_heads, num_chunks, CHUNK_SIZE, value_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, 1),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if any(output.shape != expected for output, expected in zip(outputs, expected_output_shapes, strict=True)):
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")

    # Canonical work, independent of the C++ mirror. cumsum(g) is (C-1)*K additions and G_last is its last
    # row; the kernel's prefix-mask and sum-broadcast matmuls are implementation cost, not matrix work here.
    instances = num_heads * num_chunks
    inverse_flops = CHUNK_SIZE * (CHUNK_SIZE - 1) * (CHUNK_SIZE + 1) // 3
    return (
        perf_model.FpuOps(
            matrix_flops=instances * (4 * CHUNK_SIZE**2 * key_dim + inverse_flops),
            multiply_ops=instances * (10 * CHUNK_SIZE * key_dim + CHUNK_SIZE * value_dim),
            add_ops=instances * (2 * CHUNK_SIZE + (CHUNK_SIZE - 1) * key_dim + CHUNK_SIZE * key_dim + CHUNK_SIZE**2),
            reduction_ops=instances * 2 * CHUNK_SIZE * (key_dim - 1),
        ),
        perf_model.SfpuOps(
            exp_ops=instances * (3 * CHUNK_SIZE * key_dim + key_dim),
            rsqrt_ops=instances * 2 * CHUNK_SIZE,
        ),
    )


def _prepare_chunk_recurrence_performance(
    inputs: Sequence[ttnn.Tensor],
    outputs: Sequence[ttnn.Tensor],
    *,
    measured_ns: float,
    math_fidelity: ttnn.MathFidelity,
) -> perf_model.KdaPerformance:
    fpu, sfpu = _prepare_chunk_recurrence_ops(inputs, outputs)
    return perf_model.performance(
        fpu=fpu,
        sfpu=sfpu,
        inputs=inputs,
        outputs=outputs,
        measured_ns=measured_ns,
        math_fidelity=math_fidelity,
    )


def test_prepare_chunk_recurrence_work_golden() -> None:
    inputs = (
        torch.empty((1, 32, 2)),
        torch.empty((1, 32, 2)),
        torch.empty((1, 32, 1)),
        torch.empty((1, 32, 2)),
        torch.empty((1, 1, 32, 1)),
    )
    outputs = (
        torch.empty((1, 1, 32, 1)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 32)),
        torch.empty((1, 1, 2, 32)),
        torch.empty((1, 1, 2, 1)),
        torch.empty((1, 1, 32, 32)),
    )

    fpu, sfpu = _prepare_chunk_recurrence_ops(inputs, outputs)
    assert fpu == perf_model.FpuOps(
        matrix_flops=19104,
        multiply_ops=672,
        add_ops=1214,
        reduction_ops=64,
    )
    assert sfpu == perf_model.SfpuOps(exp_ops=194, rsqrt_ops=64)


def _host_inputs(
    num_heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    *,
    seed: int = 1731,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    sequence = num_chunks * CHUNK_SIZE
    q = (0.3 * torch.randn(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    k = (0.3 * torch.randn(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    v = (0.2 * torch.randn(1, sequence, num_heads * value_dim, generator=generator)).to(torch.bfloat16).float()
    g = (-0.001 - 0.05 * torch.rand(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    beta = torch.sigmoid(torch.randn(num_heads, num_chunks, CHUNK_SIZE, 1, generator=generator)).float()
    return q, k, v, g, beta


def _reshape_flat(tensor: torch.Tensor, num_heads: int, num_chunks: int, dim: int) -> torch.Tensor:
    return (
        tensor.float()
        .reshape(num_chunks * CHUNK_SIZE, num_heads, dim)
        .permute(1, 0, 2)
        .reshape(num_heads, num_chunks, CHUNK_SIZE, dim)
    )


def _causal_decayed_products(left: torch.Tensor, right: torch.Tensor, cumulative_g: torch.Tensor) -> torch.Tensor:
    """tril(sum_k left[i,k] * right[j,k] * exp(G[i,k] - G[j,k])) in FP64, masking G_i - G_j before exp.

    The difference form is finite at any decay; a separable exp(G_i) * exp(-G_j) form overflows
    even FP64 once the per-chunk cumulative decay exceeds ~700. Inputs are [chunks, C, K].
    """
    difference = cumulative_g[:, :, None, :] - cumulative_g[:, None, :, :]
    causal = torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool).tril()[None, :, :, None]
    decay = torch.exp(difference.masked_fill(~causal, float("-inf")))
    return torch.einsum("nik,njk,nijk->nij", left, right.double(), decay)


def _oracle(
    inputs: tuple[torch.Tensor, ...],
    num_heads: int,
    output_bf16_mask: int,
) -> tuple[torch.Tensor, ...]:
    q, k, v, g, beta, *_ = inputs
    num_chunks = beta.shape[1]
    key_dim = q.shape[-1] // num_heads
    value_dim = v.shape[-1] // num_heads
    q = _reshape_flat(q, num_heads, num_chunks, key_dim)
    k = _reshape_flat(k, num_heads, num_chunks, key_dim)
    v = _reshape_flat(v, num_heads, num_chunks, value_dim)
    g = _reshape_flat(g, num_heads, num_chunks, key_dim)
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6) * (key_dim**-0.5)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    cumulative_g = torch.cumsum(g, dim=2)
    decay = torch.exp(cumulative_g)
    final_g = cumulative_g[:, :, -1]

    v_beta = beta * v
    kd = beta * k * decay
    q_decay = q * decay
    k_dec_t = (k * torch.exp(final_g.unsqueeze(2) - cumulative_g)).transpose(-1, -2)
    final_decay = torch.expm1(final_g).unsqueeze(-1)  # complement form: exp(G_last) - 1
    cumulative_g_fp64 = cumulative_g.double()
    akk = torch.stack(
        [
            _causal_decayed_products(beta[h].double() * k[h].double(), k[h], cumulative_g_fp64[h])
            for h in range(num_heads)
        ]
    )
    intra = torch.stack(
        [_causal_decayed_products(q[h].double(), k[h], cumulative_g_fp64[h]) for h in range(num_heads)]
    ).float()
    identity = torch.eye(CHUNK_SIZE, dtype=torch.float64).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE)
    t_inv = torch.linalg.inv(identity + torch.tril(akk, diagonal=-1)).float()
    outputs = (v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv)
    return tuple(
        output.to(torch.bfloat16) if output_bf16_mask & (1 << index) else output.float()
        for index, output in enumerate(outputs)
    )


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType,
    *,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(inputs: tuple[torch.Tensor, ...], device: ttnn.Device) -> tuple[ttnn.Tensor, ...]:
    return tuple(
        _to_device(tensor, device, ttnn.bfloat16 if index < 4 else ttnn.float32) for index, tensor in enumerate(inputs)
    )


def _run(
    inputs: tuple[ttnn.Tensor, ...],
    num_heads: int,
    *,
    num_key_heads: int | None = None,
    output_bf16_mask: int = 0,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    actual_start: ttnn.Tensor | None = None,
    actual_end: ttnn.Tensor | None = None,
) -> list[ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.prepare_chunk_recurrence(
            *inputs,
            num_heads,
            num_key_heads=num_key_heads,
            output_bf16_mask=output_bf16_mask,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            actual_start=actual_start,
            actual_end=actual_end,
        )


def _case_host_inputs(case: _TestCase, *, seed: int) -> tuple[torch.Tensor, ...]:
    return _host_inputs(case.num_heads, case.num_chunks, case.key_dim, case.value_dim, seed=seed)


def _t_inv_numerical_stress_inputs() -> tuple[torch.Tensor, ...]:
    """Reproduce the captured inverse instability without loading captured values."""
    inputs = list(_host_inputs(1, 1, 128, 128, seed=0))
    generator = torch.Generator().manual_seed(0)
    # Layer 13's failing chunk has cosine similarity around 0.997, key norm
    # around 0.36, and beta around 0.918. A shared Gaussian direction plus
    # small independent noise approximates that geometry; exact gate values
    # are unnecessary to expose the inverse's cancellation.
    direction = torch.randn(1, 1, 128, generator=generator)
    noise = torch.randn(1, CHUNK_SIZE, 128, generator=generator)
    inputs[1] = ((0.36 / 128**0.5) * (direction + 0.07 * noise)).to(torch.bfloat16).float()
    inputs[3] = torch.full_like(inputs[3], -0.05).to(torch.bfloat16).float()
    inputs[4] = 0.90 + 0.03 * torch.rand(1, 1, CHUNK_SIZE, 1, generator=generator)
    return tuple(inputs)


def _production_compute_config(device: ttnn.Device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _assert_output_accurate(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    context: str,
    t_inv_max_abs_threshold: float = _T_INV_MAX_ABS,
) -> None:
    assert torch.isfinite(actual).all(), f"{context} {name} contains nonfinite values"
    assert_accurate(expected, actual, name=f"{context} {name}", pcc_threshold=0.999)
    if name == "t_inv":
        _assert_t_inv_strict_lower_accurate(
            expected,
            actual,
            context=context,
            max_abs_threshold=t_inv_max_abs_threshold,
        )


def _assert_outputs_accurate(
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    context: str,
    t_inv_max_abs_threshold: float = _T_INV_MAX_ABS,
) -> None:
    for name, expected_output, actual_output in zip(OUTPUT_NAMES, expected, actual, strict=True):
        _assert_output_accurate(
            name,
            expected_output,
            actual_output,
            context=context,
            t_inv_max_abs_threshold=t_inv_max_abs_threshold,
        )


def _assert_t_inv_strict_lower_accurate(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    context: str,
    max_abs_threshold: float,
) -> None:
    lower_rows, lower_columns = torch.tril_indices(CHUNK_SIZE, CHUNK_SIZE, offset=-1)
    expected_strict_lower = expected[..., lower_rows, lower_columns]
    actual_strict_lower = actual[..., lower_rows, lower_columns]

    assert_accurate(
        expected_strict_lower,
        actual_strict_lower,
        name=f"{context} t_inv strictly-lower",
        pcc_threshold=0.999,
    )
    max_abs = float((expected_strict_lower - actual_strict_lower).abs().max())
    assert (
        max_abs <= max_abs_threshold
    ), f"{context} t_inv strictly-lower max abs error {max_abs:.6f} exceeds {max_abs_threshold:.6f}"


@pytest.mark.parametrize(
    "case",
    [_UNIT_TEST_CASE, _PRODUCTION_CASE],
    ids=lambda case: case.case_id,
)
def test_prepare_chunk_recurrence_contract_accuracy_and_determinism(
    device: ttnn.Device,
    case: _TestCase,
) -> None:
    output_bf16_mask = _PRODUCTION_OUTPUT_BF16_MASK
    compute_kernel_config = _production_compute_config(device)
    host_inputs = _case_host_inputs(case, seed=52797)
    expected = _oracle(host_inputs, case.num_heads, output_bf16_mask)
    inputs = _device_inputs(host_inputs, device)

    def run() -> tuple[ttnn.Tensor, ...]:
        return tuple(
            _run(
                inputs,
                case.num_heads,
                output_bf16_mask=output_bf16_mask,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel_config,
            )
        )

    reference_outputs, actual, mismatch_marker = collect_accuracy_and_determinism_results(device, run)
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name=f"{case.case_id} outputs device-side exact-value determinism marker",
    )
    assert len(reference_outputs) == 7
    expected_shapes = (
        (case.num_heads, case.num_chunks, CHUNK_SIZE, case.value_dim),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, case.key_dim),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, case.key_dim),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (case.num_heads, case.num_chunks, case.key_dim, CHUNK_SIZE),
        (case.num_heads, case.num_chunks, case.key_dim, 1),
        (case.num_heads, case.num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    input_addresses = {tensor.buffer_address() for tensor in inputs}
    output_addresses = set()
    for index, (output, shape) in enumerate(zip(reference_outputs, expected_shapes, strict=True)):
        expected_dtype = ttnn.bfloat16 if output_bf16_mask & (1 << index) else ttnn.float32
        assert output.dtype == expected_dtype
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(output.shape) == shape
        assert output.buffer_address() not in input_addresses
        output_addresses.add(output.buffer_address())
    assert len(output_addresses) == 7

    _assert_outputs_accurate(expected, actual, context=f"{case.case_id} invocation 0")
    for output in reference_outputs:
        ttnn.deallocate(output)


def test_prepare_chunk_recurrence_t_inv_is_stable_for_correlated_keys(device: ttnn.Device) -> None:
    host_inputs = _t_inv_numerical_stress_inputs()
    num_heads = host_inputs[-1].shape[0]
    expected = _oracle(host_inputs, num_heads, 0)
    device_inputs = _device_inputs(host_inputs, device)

    reference, outputs, mismatch_marker = collect_accuracy_and_determinism_results(
        device,
        lambda: tuple(_run(device_inputs, num_heads)),
    )
    assert_equal(
        torch.zeros_like(mismatch_marker),
        mismatch_marker,
        name="correlated keys outputs device-side exact-value determinism marker",
    )
    _assert_output_accurate(
        "t_inv",
        expected[-1],
        outputs[-1],
        context="correlated keys",
    )
    for output in reference:
        ttnn.deallocate(output)


def _strong_decay_inputs(chunk_log_decay: float) -> tuple[torch.Tensor, ...]:
    """GDN-style scalar decay broadcast over K, constant per token, summing to -chunk_log_decay per chunk.

    Every per-token value used below (|G_last| / 32) is exact in BF16, so the device and the
    oracle see the same cumulative decay and the case ID names the realized |G_last|.
    """
    num_heads, num_chunks, key_dim, value_dim = 2, 2, 128, 128
    q, k, v, g, beta = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=2207)
    per_token = torch.tensor(-chunk_log_decay / CHUNK_SIZE).to(torch.bfloat16)
    assert float(per_token) * CHUNK_SIZE == -chunk_log_decay, "per-token decay must be exact in BF16"
    g = torch.full_like(g, float(per_token))
    return q, k, v, g, beta


# Peak-error gates for the outputs that the anchored factors corrupt first: single entries
# (diagonal of intra, last token of k_dec_t) go wrong while aggregate PCC stays above 0.999.
# Clean device runs at |G_last| <= 144 give intra 7.2e-5 and k_dec_t (BF16) 1.24e-3 max abs error.
_STRONG_DECAY_PEAK_ERROR = {"intra": 2.5e-4, "k_dec_t": 3.0e-3}
_ANCHORED_RANGE_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="KDA prep anchors exp(G - G_last/2) and exp(G_last/2 - G); beyond |G_last| ~ 150 per chunk "
    "these factors leave the exponent range and intra/k_dec_t entries are silently wrong (finite)",
)


# GDN scalar decay on the KDA prep. The prep factors exp(G_i - G_j) into anchored separable terms
# and masks after the matmul; FP32 emulation (tt-work U2, check 3b) gives non-finite values above
# |G_last| ~ 100, while the device stays finite and accurate through 144 and is wrong from 160.
# Real Qwen GDN gates exceed this: 3008 (94 per token) matches the input-independent
# full-forgetting heads of Qwen3.6-35B-A3B layer 0 (~91.6 per token).
@pytest.mark.parametrize(
    "chunk_log_decay",
    [
        pytest.param(16.0, id="mild-g16"),
        pytest.param(90.0, id="strong-g90"),
        pytest.param(110.0, id="strong-g110"),
        pytest.param(144.0, id="strong-g144"),
        pytest.param(160.0, id="strong-g160", marks=_ANCHORED_RANGE_XFAIL),
        pytest.param(176.0, id="strong-g176", marks=_ANCHORED_RANGE_XFAIL),
        pytest.param(200.0, id="strong-g200", marks=_ANCHORED_RANGE_XFAIL),
        pytest.param(3008.0, id="qwen-g3008", marks=_ANCHORED_RANGE_XFAIL),
    ],
)
def test_prepare_chunk_recurrence_strong_scalar_decay(device: ttnn.Device, chunk_log_decay: float) -> None:
    host_inputs = _strong_decay_inputs(chunk_log_decay)
    num_heads = host_inputs[-1].shape[0]
    output_bf16_mask = _PRODUCTION_OUTPUT_BF16_MASK
    expected = _oracle(host_inputs, num_heads, output_bf16_mask)
    assert all(torch.isfinite(output).all() for output in expected), "oracle must stay finite at any decay"
    outputs = _run(
        _device_inputs(host_inputs, device),
        num_heads,
        output_bf16_mask=output_bf16_mask,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_production_compute_config(device),
    )
    actual = tuple(ttnn.to_torch(output) for output in outputs)
    non_finite = {name: int((~torch.isfinite(output.float())).sum()) for name, output in zip(OUTPUT_NAMES, actual)}
    logger.info(f"|G_last|={chunk_log_decay}: non-finite output entries {non_finite}")
    for name, expected_output, actual_output in zip(OUTPUT_NAMES, expected, actual):
        error = (expected_output.float() - actual_output.float()).abs()
        scale = float(expected_output.float().abs().max())
        wrong = torch.nonzero(error > 0.05 * max(scale, 1e-30))
        positions = sorted({tuple(index[-2:].tolist()) for index in wrong})
        logger.info(
            f"|G_last|={chunk_log_decay} {name}: max_abs_err={float(error.max()):.3e} (|expected|max={scale:.3e}); "
            f"{len(wrong)} entries off by >5% of |expected|max at (row, col) {positions[:12]}"
        )
    _assert_outputs_accurate(expected, actual, context=f"|G_last|={chunk_log_decay}")
    for name, threshold in _STRONG_DECAY_PEAK_ERROR.items():
        index = OUTPUT_NAMES.index(name)
        max_abs = float((expected[index].float() - actual[index].float()).abs().max())
        assert max_abs <= threshold, f"|G_last|={chunk_log_decay} {name} max abs error {max_abs:.3e} > {threshold:.1e}"
    for output in outputs:
        ttnn.deallocate(output)


# Weak decay (tt_metal_tracker-g1b.7, T3): per-chunk |G_last| from 1e-6 to 1. final_decay carries expm1(G_last), so a
# long-memory channel's forgetting survives BF16 storage; exp(G_last) would round to exactly 1.0 below 2^-9.
_WEAK_DECAY_CHUNK_LOG_DECAYS = (1e-6, 1e-4, 2.0**-10, 1e-2, 1.0)


def test_prepare_chunk_recurrence_weak_final_decay(device: ttnn.Device) -> None:
    num_heads, num_chunks, key_dim, value_dim = 1, 2, 128, 128
    q, k, v, g, beta = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=7301)
    rows = key_dim // len(_WEAK_DECAY_CHUNK_LOG_DECAYS)
    per_token = torch.zeros(key_dim)
    for index, chunk_log_decay in enumerate(_WEAK_DECAY_CHUNK_LOG_DECAYS):
        per_token[index * rows : (index + 1) * rows] = -chunk_log_decay / CHUNK_SIZE
    g = per_token.expand_as(g).to(torch.bfloat16).float()
    inputs = (q, k, v, g, beta)
    outputs = _run(
        _device_inputs(inputs, device),
        num_heads,
        output_bf16_mask=_PRODUCTION_OUTPUT_BF16_MASK,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_production_compute_config(device),
    )
    actual = ttnn.to_torch(outputs[OUTPUT_NAMES.index("final_decay")]).double()[..., 0]  # [H, N, K]
    g_last = _reshape_flat(g, num_heads, num_chunks, key_dim).double().sum(dim=2)  # [H, N, K], exact BF16 sums
    expected = torch.expm1(g_last)
    legacy = torch.exp(g_last).to(torch.bfloat16).double() - 1.0  # what an exp(G_last) BF16 output carries
    failures = []
    for index, chunk_log_decay in enumerate(_WEAK_DECAY_CHUNK_LOG_DECAYS):
        channels = slice(index * rows, (index + 1) * rows)
        reference = expected[..., channels]
        error = float(((actual[..., channels] - reference).abs() / reference.abs()).max())
        legacy_error = float(((legacy[..., channels] - reference).abs() / reference.abs()).max())
        logger.info(f"|G_last|={chunk_log_decay:.1e}: final_decay rel error {error:.3e} (exp form: {legacy_error:.3e})")
        if not error <= 2.0**-7:
            failures.append(f"|G_last|={chunk_log_decay:.1e} rel error {error:.3e}")
    assert not failures, "; ".join(failures)
    # Negative control: the gate is sensitive to the exp form at the weak end.
    assert float((legacy[..., :rows] - expected[..., :rows]).abs().max() / expected[..., :rows].abs().max()) > 0.5
    for output in outputs:
        ttnn.deallocate(output)


# Real-weight gate sources: (checkpoint env var, layer key prefix). Hidden states are synthetic (unit-RMS Gaussian
# through the layer's input RMSNorm); A_log, dt_bias, f_a and f_b are the checkpoint's, so the per-channel gate
# spread (strong and weak channels side by side) is the model's own.
_REAL_GATE_LAYERS = {
    "k3-layer1": ("KIMI_K3_CKPT", "language_model.model.layers.1."),
    "glm-layer0": ("GLM_5_3_FLASH_CKPT", "model.language_model.layers.0."),
}
_GATE_LOWER_BOUND = -5.0  # Kimi K3 and GLM-5.3-Flash gate_lower_bound


def _real_weight_gates(source: str, num_heads: int, sequence: int, *, seed: int) -> torch.Tensor:
    """Bounded KDA gates [1, sequence, num_heads*128] (BF16 values) of the num_heads most strongly decaying heads."""
    import json
    import os
    from pathlib import Path

    from safetensors import safe_open

    from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_gate_reference

    env, prefix = _REAL_GATE_LAYERS[source]
    root = os.getenv(env)
    if not root:
        pytest.skip(f"set {env} to the pinned checkpoint subset for real-weight gates")
    root = Path(root)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = ("self_attn.A_log", "self_attn.dt_bias", "self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight")
    names += ("input_layernorm.weight",)
    tensors = {}
    for name in names:
        with safe_open(root / weight_map[prefix + name], "pt") as shard:
            tensors[name] = shard.get_tensor(prefix + name)
    eps = json.loads((root / "config.json").read_text()).get("text_config", {}).get("rms_norm_eps", 1e-5)
    hidden = tensors["input_layernorm.weight"].numel()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(sequence, hidden, generator=generator)
    x = (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * tensors["input_layernorm.weight"].float()
    ).bfloat16()
    raw = (x @ tensors["self_attn.f_a_proj.weight"].T).bfloat16() @ tensors["self_attn.f_b_proj.weight"].T
    key_dim = 128
    heads = tensors["self_attn.dt_bias"].numel() // key_dim
    a_log = tensors["self_attn.A_log"].reshape(-1)[:heads]  # K3 pads A_log to 128 heads
    gate = kda_gate_reference(
        raw.float().reshape(1, sequence, heads, key_dim), a_log, tensors["self_attn.dt_bias"], _GATE_LOWER_BOUND
    )
    strongest = gate.mean(dim=(0, 1, 3)).argsort()[:num_heads]
    return gate[:, :, strongest].reshape(1, sequence, num_heads * key_dim).to(torch.bfloat16).float()


def _per_channel_gate_inputs(gate_case: str) -> tuple[torch.Tensor, ...]:
    """K3/GLM-like per-channel gates in [-5, 0] with fractional BF16 values (H=2, N=2, K=V=128)."""
    num_heads, num_chunks, key_dim, value_dim = 2, 2, 128, 128
    q, k, v, g, beta = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=4242)
    generator = torch.Generator().manual_seed(4243)
    if gate_case == "uniform-5":
        g = _GATE_LOWER_BOUND * torch.rand(g.shape, generator=generator)
    elif gate_case == "const-5":  # TF32-exact control: the strong-decay test's input class at |G_last| = 160
        g = torch.full_like(g, _GATE_LOWER_BOUND)
    elif gate_case == "mixed-5-weak":  # saturating and long-memory channels in the same key dot product
        weak = -(10 ** (-4 + 2 * torch.rand(g.shape, generator=generator)))
        g = torch.where(torch.arange(g.shape[-1]) % 2 == 0, torch.full_like(g, _GATE_LOWER_BOUND), weak)
    else:
        g = _real_weight_gates(gate_case, num_heads, num_chunks * CHUNK_SIZE, seed=4244)
    g = g.to(torch.bfloat16).float()  # KDA_GATE_DTYPE
    assert float(g.min()) >= _GATE_LOWER_BOUND and float(g.max()) <= 0.0
    return q, k, v, g, beta


# K3/GLM per-channel gates inside the supported range [-5, 0] (tt_metal_tracker-g1b.7, test T2). Unlike the
# constant gates of the strong-decay test (exact in TF32), fractional gates exercise the precision of the prep's
# exponent arguments. Peak gates are the strong-decay test's. The real-weight cases use the checkpoint's gate
# weights with synthetic hidden states and the two most strongly decaying heads of the layer. uniform-5 and
# glm-layer0 own k_dec_t's suffix-sum exponent: the anchored form k*exp(G_last/2 - G)*exp(G_last/2) failed them with
# last-token peak errors 9.0e-3 / 8.3e-3 > 3e-3; the suffix form gives 3.9e-4 / 3.6e-4 (tt_metal_tracker-g1b.4.18).
@pytest.mark.parametrize(
    "gate_case",
    [pytest.param(case, id=case) for case in ("uniform-5", "const-5", "mixed-5-weak", "k3-layer1", "glm-layer0")],
)
def test_prepare_chunk_recurrence_per_channel_gate_range(device: ttnn.Device, gate_case: str) -> None:
    host_inputs = _per_channel_gate_inputs(gate_case)
    num_heads = host_inputs[-1].shape[0]
    output_bf16_mask = _PRODUCTION_OUTPUT_BF16_MASK
    expected = _oracle(host_inputs, num_heads, output_bf16_mask)
    outputs = _run(
        _device_inputs(host_inputs, device),
        num_heads,
        output_bf16_mask=output_bf16_mask,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_production_compute_config(device),
    )
    actual = tuple(ttnn.to_torch(output) for output in outputs)
    g = _reshape_flat(host_inputs[3], num_heads, 2, 128)
    chunk_decay = -g.sum(dim=2)
    logger.info(
        f"{gate_case}: per-chunk |G_last| over (head, chunk, channel) max={float(chunk_decay.max()):.2f} "
        f"median={float(chunk_decay.median()):.3g} min={float(chunk_decay.min()):.3g}"
    )
    for name, expected_output, actual_output in zip(OUTPUT_NAMES, expected, actual):
        error = (expected_output.float() - actual_output.float()).abs()
        index = tuple(int(i) for i in torch.nonzero(error == error.max())[0])
        logger.info(
            f"{gate_case} {name}: max_abs_err={float(error.max()):.3e} at {index} "
            f"(|expected|max={float(expected_output.float().abs().max()):.3e})"
        )
    for output in outputs:
        ttnn.deallocate(output)
    _assert_outputs_accurate(expected, actual, context=gate_case)
    for name, threshold in _STRONG_DECAY_PEAK_ERROR.items():
        index = OUTPUT_NAMES.index(name)
        max_abs = float((expected[index].float() - actual[index].float()).abs().max())
        assert max_abs <= threshold, f"{gate_case} {name} max abs error {max_abs:.3e} > {threshold:.1e}"


@pytest.mark.parametrize("output_bf16_mask", [0, 0x26])
def test_prepare_chunk_recurrence_unbounded_legacy_call_matches_explicit_bounds(device, output_bf16_mask):
    case = _UNIT_TEST_CASE
    inputs = _device_inputs(_case_host_inputs(case, seed=1912), device)
    start = make_actual_start(device, 0)
    end = make_actual_start(device, case.num_chunks * CHUNK_SIZE)
    legacy = _run(inputs, case.num_heads, output_bf16_mask=output_bf16_mask)
    for bound in (None, end):
        explicit = ttnn.experimental.kda.prepare_chunk_recurrence(
            *inputs, case.num_heads, actual_start=start, actual_end=bound, output_bf16_mask=output_bf16_mask
        )
        for name, unbounded, bounded in zip(OUTPUT_NAMES, legacy, explicit, strict=True):
            assert_bit_identical(ttnn.to_torch(unbounded), ttnn.to_torch(bounded), name=name)
        for tensor in explicit:
            ttnn.deallocate(tensor)
    for tensor in (*legacy, *inputs, start, end):
        ttnn.deallocate(tensor)


def test_prepare_chunk_recurrence_rejects_end_without_start(device, expect_error):
    inputs = _device_inputs(_host_inputs(2, 2, 32, 32), device)
    end = make_actual_start(device, 64)
    with expect_error(RuntimeError, "actual_end requires actual_start"):
        ttnn.experimental.kda.prepare_chunk_recurrence(*inputs, 2, actual_end=end)
    for tensor in (*inputs, end):
        ttnn.deallocate(tensor)


def test_prepare_chunk_recurrence_cache_hit_rebinds_fresh_tensors(device: ttnn.Device) -> None:
    case = _UNIT_TEST_CASE
    host_a = _case_host_inputs(case, seed=1911)
    host_b = _case_host_inputs(case, seed=1912)
    inputs_a = _device_inputs(host_a, device)
    inputs_b = _device_inputs(host_b, device)

    outputs_a = _run(inputs_a, case.num_heads)
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    outputs_b = _run(inputs_b, case.num_heads)
    ttnn.synchronize_device(device)

    assert device.num_program_cache_entries() == entries
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(inputs_a, inputs_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(outputs_a, outputs_b, strict=True))
    _assert_outputs_accurate(
        _oracle(host_a, case.num_heads, 0),
        tuple(ttnn.to_torch(output) for output in outputs_a),
        context="cache miss tensors",
    )
    _assert_outputs_accurate(
        _oracle(host_b, case.num_heads, 0),
        tuple(ttnn.to_torch(output) for output in outputs_b),
        context="cache hit fresh tensors",
    )
    assert not torch.equal(ttnn.to_torch(outputs_a[0]), ttnn.to_torch(outputs_b[0]))


def test_prepare_chunk_recurrence_default_compute_config_matches_explicit_defaults(device: ttnn.Device) -> None:
    case = _UNIT_TEST_CASE
    inputs = _device_inputs(_case_host_inputs(case, seed=817), device)
    implicit = _run(inputs, case.num_heads)
    entries = device.num_program_cache_entries()
    explicit_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )
    explicit = _run(inputs, case.num_heads, compute_kernel_config=explicit_config)
    assert device.num_program_cache_entries() == entries
    for name, implicit_tt, explicit_tt in zip(OUTPUT_NAMES, implicit, explicit, strict=True):
        assert_bit_identical(
            ttnn.to_torch(implicit_tt), ttnn.to_torch(explicit_tt), name=f"{name} implicit vs explicit defaults"
        )


def test_prepare_chunk_recurrence_precise_math_uses_distinct_accurate_program(device: ttnn.Device) -> None:
    case = _UNIT_TEST_CASE
    host_inputs = _case_host_inputs(case, seed=818)
    inputs = _device_inputs(host_inputs, device)
    approximate = _run(inputs, case.num_heads)
    entries = device.num_program_cache_entries()
    precise_config = _production_compute_config(device)
    precise = _run(inputs, case.num_heads, compute_kernel_config=precise_config)
    assert device.num_program_cache_entries() == entries + 1
    expected = _oracle(host_inputs, case.num_heads, 0)
    _assert_outputs_accurate(
        expected,
        tuple(ttnn.to_torch(output) for output in approximate),
        context="default approximate math",
    )
    _assert_outputs_accurate(
        expected,
        tuple(ttnn.to_torch(output) for output in precise),
        context="explicit precise math",
    )


def test_prepare_chunk_recurrence_rejects_unsupported_compute_config(
    device: ttnn.Device, expect_error: Callable
) -> None:
    case = _UNIT_TEST_CASE
    inputs = _device_inputs(_case_host_inputs(case, seed=819), device)
    unsupported_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,
    )
    with expect_error(RuntimeError, "packer_l1_acc=true is unsupported"):
        _run(inputs, case.num_heads, compute_kernel_config=unsupported_config)


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_prepare_chunk_recurrence_production_performance(device: ttnn.Device) -> None:
    case = _PRODUCTION_CASE
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for chunk-recurrence preparation performance checks")

    host_inputs = _case_host_inputs(case, seed=117)
    inputs = _device_inputs(host_inputs, device)

    def run() -> list[ttnn.Tensor]:
        return _run(
            inputs,
            case.num_heads,
            output_bf16_mask=_PRODUCTION_OUTPUT_BF16_MASK,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=_production_compute_config(device),
        )

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert len(outputs) == 7
    assert tuple(outputs[0].shape) == (case.num_heads, case.num_chunks, CHUNK_SIZE, case.value_dim)
    performance = _prepare_chunk_recurrence_performance(
        inputs,
        outputs,
        measured_ns=duration_ns,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    logger.info(
        f"chunk-recurrence preparation {case.case_id}: measured_ns={duration_ns:.0f}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    upper = _PRODUCTION_EXPECTED_DURATION_NS * (1 + _PERFORMANCE_MARGIN)
    assert duration_ns <= upper, (
        f"{case.case_id} duration {duration_ns:.0f} ns exceeds {upper:.0f} ns "
        f"(reference {_PRODUCTION_EXPECTED_DURATION_NS} ns, margin {_PERFORMANCE_MARGIN * 100:.0f}%)"
    )


@pytest.mark.parametrize("host_index", range(5))
def test_prepare_chunk_recurrence_rejects_host_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    host_index: int,
) -> None:
    host_inputs = _host_inputs(2, 2, 32, 32)
    inputs = list(_device_inputs(host_inputs, device))
    dtype = ttnn.bfloat16 if host_index < 4 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host_inputs[host_index], dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(
        RuntimeError,
        f"{('q', 'k', 'v', 'g', 'beta')[host_index]} must be an allocated device tensor",
    ):
        _run(tuple(inputs), 2)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("g_dtype", "g must be BFLOAT16, got DataType::FLOAT32"),
        ("layout", "k must use TILE layout"),
        ("rank", "rank 3 production-flat"),
        ("leading", "leading dimension 1"),
        ("qk_shape", "q and k must have matching shapes"),
        ("g_shape", "g must be per V head: width num_heads times K"),
        ("sequence", "matching sequence lengths"),
        ("sequence_chunk", "sequence length must be positive and divisible by 32"),
        ("head_divisibility", "flat widths must be divisible by num_heads"),
        ("key_alignment", "K and V must be positive and tile aligned"),
        ("value_alignment", "K and V must be positive and tile aligned"),
        ("beta", "beta shape must be"),
        ("sharded", "q must use interleaved memory"),
    ],
)
def test_prepare_chunk_recurrence_rejects_invalid_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    host_inputs = list(_host_inputs(2, 2, 32, 32))
    inputs = list(_device_inputs(tuple(host_inputs), device))
    num_heads = 2
    if case == "g_dtype":
        inputs[3] = _to_device(host_inputs[3], device, ttnn.float32)
    elif case == "layout":
        inputs[1] = _to_device(host_inputs[1], device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "rank":
        inputs[1] = _to_device(host_inputs[1].reshape(1, 2, 32, 64), device, ttnn.bfloat16)
    elif case == "leading":
        inputs[0] = _to_device(host_inputs[0].expand(2, -1, -1).clone(), device, ttnn.bfloat16)
    elif case == "qk_shape":
        inputs[1] = _to_device(host_inputs[1][:, :, :32], device, ttnn.bfloat16)
    elif case == "g_shape":
        inputs[3] = _to_device(host_inputs[3][:, :, :32], device, ttnn.bfloat16)
    elif case == "sequence":
        inputs[2] = _to_device(host_inputs[2][:, :32], device, ttnn.bfloat16)
    elif case == "sequence_chunk":
        for index in range(4):
            inputs[index] = _to_device(host_inputs[index][:, :48], device, ttnn.bfloat16)
    elif case == "head_divisibility":
        num_heads = 3
    elif case == "key_alignment":
        num_heads = 2
        for index in (0, 1, 3):
            tensor = torch.randn(1, 64, 96).to(torch.bfloat16).float()
            inputs[index] = _to_device(tensor, device, ttnn.bfloat16)
    elif case == "value_alignment":
        inputs[2] = _to_device(torch.randn(1, 64, 96).to(torch.bfloat16).float(), device, ttnn.bfloat16)
    elif case == "beta":
        inputs[4] = _to_device(host_inputs[4][:, :1], device, ttnn.float32)
    elif case == "sharded":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [64, 64],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        inputs[0] = _to_device(host_inputs[0], device, ttnn.bfloat16, memory_config=sharded)
    with expect_error(RuntimeError, message):
        _run(tuple(inputs), num_heads)


def test_prepare_chunk_recurrence_rejects_invalid_options(device: ttnn.Device, expect_error: Callable) -> None:
    host_inputs = _host_inputs(2, 2, 32, 32)
    inputs = _device_inputs(host_inputs, device)
    with expect_error(RuntimeError, "num_heads must be positive"):
        _run(inputs, 0)
    for mask in (0x08, 0x40):
        with expect_error(RuntimeError, "unsupported KDA prep BF16 mask"):
            _run(inputs, 2, output_bf16_mask=mask)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [64, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory layout must be INTERLEAVED, got HEIGHT_SHARDED"):
        _run(inputs, 2, memory_config=sharded)


@pytest.mark.parametrize("output_bf16_mask", [0x00, 0x20, 0x26], ids=["all-fp32", "decay-bf16", "production"])
def test_prepare_chunk_recurrence_scratch_wrap(device: ttnn.Device, output_bf16_mask: int) -> None:
    """Keep row transactions contiguous after inverse scratch advances its DFB cursors."""
    grid = device.compute_with_storage_grid_size()
    work_items_per_core = 32
    chunks = grid.x * grid.y * work_items_per_core
    chunk = _t_inv_numerical_stress_inputs()
    inputs = tuple(x.repeat(1, chunks, 1) for x in chunk[:4]) + (chunk[4].repeat(1, chunks, 1, 1),)
    key_dim = chunk[3].shape[-1]
    expected = torch.expm1(chunk[3].sum(dim=1)).reshape(1, 1, key_dim, 1).repeat(1, chunks, 1, 1)
    if output_bf16_mask & (1 << 5):
        expected = expected.to(torch.bfloat16).float()
    outputs = _run(
        _device_inputs(inputs, device),
        1,
        output_bf16_mask=output_bf16_mask,
        compute_kernel_config=_production_compute_config(device),
    )
    actual = ttnn.to_torch(outputs[5]).float()
    # final_decay does not depend on the inverse. The old inverse used single-tile
    # transactions in row-sized DFBs, leaving their cursors offset for the next
    # work item; the subsequent row reservation crossed the ring end and corrupted
    # adjacent storage, which surfaced here as a final_decay mismatch.
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.004)


# K/V head mapping (GDN, num_key_heads < num_heads): V head hv uses K head hv // (HV / Hk), the Hugging Face
# repeat_interleave order. The defining contract is equality with the call on q/k expanded to HV heads.
@dataclass(frozen=True)
class _HeadMappingCase:
    case_id: str
    num_key_heads: int
    num_heads: int
    num_chunks: int
    key_dim: int
    value_dim: int


# Per-chip geometries at TP4 and 640 local rows (Qwen3.8-27B 16/48, Qwen3.6-35B-A3B 16/32, Qwen3.8-2.4T 16/128
# key/value heads, K = V = 128), plus toy shapes that cover a single K head and a non-power-of-two group.
_HEAD_MAPPING_CASES = (
    _HeadMappingCase("toy-hk1-hv2", 1, 2, 4, 32, 64),
    _HeadMappingCase("toy-hk2-hv6", 2, 6, 3, 64, 32),
    _HeadMappingCase("27b-chip-hk4-hv12", 4, 12, 20, 128, 128),
    _HeadMappingCase("35b-chip-hk4-hv8", 4, 8, 20, 128, 128),
    _HeadMappingCase("2p4t-chip-hk4-hv32", 4, 32, 20, 128, 128),
)


def _head_mapping_host_inputs(case: _HeadMappingCase, *, seed: int) -> tuple[torch.Tensor, ...]:
    """q, k with Hk heads; v, g, beta with HV heads (per-channel g is per V head)."""
    q, k, *_ = _host_inputs(case.num_key_heads, case.num_chunks, case.key_dim, case.value_dim, seed=seed)
    _, _, v, g, beta = _host_inputs(case.num_heads, case.num_chunks, case.key_dim, case.value_dim, seed=seed + 1)
    return q, k, v, g, beta


def _expand_key_heads(tensor: torch.Tensor, num_key_heads: int, num_heads: int) -> torch.Tensor:
    _, sequence, width = tensor.shape
    grouped = tensor.reshape(1, sequence, num_key_heads, width // num_key_heads)
    return grouped.repeat_interleave(num_heads // num_key_heads, dim=2).reshape(1, sequence, -1)


def _expanded_inputs(inputs: tuple[torch.Tensor, ...], num_key_heads: int, num_heads: int) -> tuple[torch.Tensor, ...]:
    q, k, v, g, beta = inputs
    return (_expand_key_heads(q, num_key_heads, num_heads), _expand_key_heads(k, num_key_heads, num_heads), v, g, beta)


@pytest.mark.parametrize("case", _HEAD_MAPPING_CASES, ids=lambda case: case.case_id)
def test_prepare_chunk_recurrence_head_mapping_equals_expansion(device: ttnn.Device, case: _HeadMappingCase) -> None:
    host_inputs = _head_mapping_host_inputs(case, seed=5401)
    options = dict(
        output_bf16_mask=_PRODUCTION_OUTPUT_BF16_MASK,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_production_compute_config(device),
    )
    mapped = _run(_device_inputs(host_inputs, device), case.num_heads, num_key_heads=case.num_key_heads, **options)
    expanded_inputs = _expanded_inputs(host_inputs, case.num_key_heads, case.num_heads)
    expanded = _run(_device_inputs(expanded_inputs, device), case.num_heads, **options)
    for name, mapped_output, expanded_output in zip(OUTPUT_NAMES, mapped, expanded, strict=True):
        assert_bit_identical(
            ttnn.to_torch(expanded_output), ttnn.to_torch(mapped_output), name=f"{case.case_id} {name}"
        )
    for tensor in (*mapped, *expanded):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize("num_heads", [8, 12, 32], ids=lambda heads: f"hk4-hv{heads}")
def test_prepare_chunk_recurrence_head_mapping_reference(device: ttnn.Device, num_heads: int) -> None:
    """Mapping with chronology: nonzero actual_start and actual_end mid-sequence, against the FP64 oracle."""
    case = _HeadMappingCase(f"hk4-hv{num_heads}", 4, num_heads, 20, 128, 128)
    start_row, valid_chunks = 64, 13
    host_inputs = _head_mapping_host_inputs(case, seed=5402)
    expanded = _expanded_inputs(host_inputs, case.num_key_heads, case.num_heads)
    valid_rows = valid_chunks * CHUNK_SIZE
    valid_inputs = tuple(tensor[:, :valid_rows] for tensor in expanded[:4]) + (expanded[4][:, :valid_chunks],)
    expected = _oracle(valid_inputs, case.num_heads, _PRODUCTION_OUTPUT_BF16_MASK)
    start = make_actual_start(device, start_row)
    end = make_actual_start(device, start_row + valid_rows)
    outputs = _run(
        _device_inputs(host_inputs, device),
        case.num_heads,
        num_key_heads=case.num_key_heads,
        output_bf16_mask=_PRODUCTION_OUTPUT_BF16_MASK,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_production_compute_config(device),
        actual_start=start,
        actual_end=end,
    )
    # Chunks past actual_end are unspecified; compare the valid prefix.
    actual = tuple(ttnn.to_torch(output)[:, :valid_chunks] for output in outputs)
    _assert_outputs_accurate(expected, actual, context=case.case_id)
    for tensor in (*outputs, start, end):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("zero_key_heads", "num_key_heads must be positive"),
        ("group_divisibility", "num_heads must be divisible by num_key_heads"),
        ("qk_shape", "q and k must have matching shapes"),
        ("g_key_heads", "g must be per V head: width num_heads times K"),
        ("key_alignment", "K and V must be positive and tile aligned"),
    ],
)
def test_prepare_chunk_recurrence_head_mapping_rejects_invalid_inputs(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    """Rejected before any program is built: the program cache does not grow."""
    mapping = _HeadMappingCase("invalid", 2, 4, 2, 32, 32)
    host_inputs = list(_head_mapping_host_inputs(mapping, seed=5403))
    inputs = list(_device_inputs(tuple(host_inputs), device))
    num_key_heads, num_heads = mapping.num_key_heads, mapping.num_heads
    if case == "zero_key_heads":
        num_key_heads = 0
    elif case == "group_divisibility":
        num_key_heads = 3
        for index in (0, 1):
            inputs[index] = _to_device(torch.randn(1, 64, 96).to(torch.bfloat16).float(), device, ttnn.bfloat16)
    elif case == "qk_shape":
        inputs[1] = _to_device(_expand_key_heads(host_inputs[1], 2, 4), device, ttnn.bfloat16)
    elif case == "g_key_heads":
        inputs[3] = _to_device(host_inputs[3][:, :, : 2 * 32], device, ttnn.bfloat16)
    elif case == "key_alignment":
        for index in (0, 1):
            inputs[index] = _to_device(torch.randn(1, 64, 96).to(torch.bfloat16).float(), device, ttnn.bfloat16)
    entries = device.num_program_cache_entries()
    with expect_error(RuntimeError, message):
        _run(tuple(inputs), num_heads, num_key_heads=num_key_heads)
    assert device.num_program_cache_entries() == entries


def test_prepare_chunk_recurrence_num_key_heads_is_program_identity(device: ttnn.Device) -> None:
    """Calls differing only in num_key_heads build distinct programs; an omitted num_key_heads is num_heads."""
    shared = _HeadMappingCase("cache-hk4", 4, 4, 2, 32, 32)
    grouped = _HeadMappingCase("cache-hk2", 2, 4, 2, 32, 32)
    shared_inputs = _device_inputs(_head_mapping_host_inputs(shared, seed=5404), device)
    grouped_inputs = _device_inputs(_head_mapping_host_inputs(grouped, seed=5404), device)
    implicit = _run(shared_inputs, 4)
    entries = device.num_program_cache_entries()
    explicit = _run(shared_inputs, 4, num_key_heads=4)
    assert device.num_program_cache_entries() == entries
    for name, implicit_output, explicit_output in zip(OUTPUT_NAMES, implicit, explicit, strict=True):
        assert_bit_identical(ttnn.to_torch(implicit_output), ttnn.to_torch(explicit_output), name=name)
    mapped = _run(grouped_inputs, 4, num_key_heads=2)
    assert device.num_program_cache_entries() == entries + 1
    repeated = _run(grouped_inputs, 4, num_key_heads=2)
    assert device.num_program_cache_entries() == entries + 1
    for tensor in (*implicit, *explicit, *mapped, *repeated):
        ttnn.deallocate(tensor)


# Scalar-decay mode (GDN, tt_metal_tracker-g1b.5.4.2): g holds one FP32 log decay per (V head, token), laid out like
# beta, [HV, N, 32, 1]; the mode is selected by that shape. The pairwise decay is formed in difference form, masked
# before the exp, so the mode is exact at any per-chunk decay. The strict xfails of
# test_prepare_chunk_recurrence_strong_scalar_decay above exercise the per-channel path with a broadcast gate and
# stay owned by that path.
def _scalar_decay_oracle(
    inputs: tuple[torch.Tensor, ...],
    num_heads: int,
    output_bf16_mask: int,
    *,
    num_key_heads: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """FP64 reference for scalar decay, from the FP32 gate as given (cumulative sums included)."""
    q, k, v, g, beta = inputs
    num_key_heads = num_heads if num_key_heads is None else num_key_heads
    num_chunks = beta.shape[1]
    key_dim = q.shape[-1] // num_key_heads
    value_dim = v.shape[-1] // num_heads
    q = _reshape_flat(_expand_key_heads(q, num_key_heads, num_heads), num_heads, num_chunks, key_dim).double()
    k = _reshape_flat(_expand_key_heads(k, num_key_heads, num_heads), num_heads, num_chunks, key_dim).double()
    v = _reshape_flat(v, num_heads, num_chunks, value_dim).double()
    beta = beta.double()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6) * (key_dim**-0.5)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    cumulative = torch.cumsum(g.double()[..., 0], dim=-1)  # [H, N, C]
    final = cumulative[..., -1:]
    causal = torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool).tril()
    pairwise = torch.exp((cumulative[..., :, None] - cumulative[..., None, :]).masked_fill(~causal, float("-inf")))
    v_beta = beta * v
    kd = beta * k * torch.exp(cumulative)[..., None]
    q_decay = q * torch.exp(cumulative)[..., None]
    k_dec_t = (k * torch.exp(final - cumulative)[..., None]).transpose(-1, -2)
    final_decay = torch.expm1(final)[..., None].expand(num_heads, num_chunks, key_dim, 1)
    akk = (beta * k) @ k.transpose(-1, -2) * pairwise
    intra = q @ k.transpose(-1, -2) * pairwise
    identity = torch.eye(CHUNK_SIZE, dtype=torch.float64)
    t_inv = torch.linalg.inv(identity + torch.tril(akk, diagonal=-1))
    outputs = (v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv)
    return tuple(
        output.to(torch.bfloat16) if output_bf16_mask & (1 << index) else output.float()
        for index, output in enumerate(outputs)
    )


def _scalar_gate_profile(profile: str, chunk_log_decay: float, generator: torch.Generator) -> torch.Tensor:
    """One chunk of FP32 per-token log decays summing to about -chunk_log_decay.

    spread: fractional, randomly distributed over the chunk. spike: one token carries nearly all of the decay and
    the others are small fractional gates, so later tokens pair large cumulative decays with small differences.
    constant: equal per-token decay (Qwen's input-independent full-forgetting heads are of this kind).
    """
    if profile == "spread":
        weights = -torch.log(torch.rand(CHUNK_SIZE, generator=generator, dtype=torch.float64))
        gate = -chunk_log_decay * weights / weights.sum()
    elif profile == "spike":
        gate = -torch.rand(CHUNK_SIZE, generator=generator, dtype=torch.float64) * (chunk_log_decay / (2 * CHUNK_SIZE))
        position = int(torch.randint(1, CHUNK_SIZE - 1, (1,), generator=generator))
        gate[position] = 0.0
        gate[position] = -(chunk_log_decay + float(gate.sum()))
    elif profile == "constant":
        gate = torch.full((CHUNK_SIZE,), -chunk_log_decay / CHUNK_SIZE, dtype=torch.float64)
    else:
        raise ValueError(f"unknown gate profile {profile}")
    assert float(gate.max()) <= 0.0
    return gate.float()


def _scalar_gates(profiles: Sequence[Sequence[tuple[str, float]]], *, seed: int) -> torch.Tensor:
    """Scalar gate [H, N, 32, 1] FP32 from a per-(head, chunk) table of (profile, |G_last|)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            torch.stack([_scalar_gate_profile(profile, decay, generator) for profile, decay in chunks])
            for chunks in profiles
        ]
    ).unsqueeze(-1)


def _scalar_device_inputs(inputs: tuple[torch.Tensor, ...], device: ttnn.Device) -> tuple[ttnn.Tensor, ...]:
    q, k, v, g, beta = inputs
    return (
        _to_device(q, device, ttnn.bfloat16),
        _to_device(k, device, ttnn.bfloat16),
        _to_device(v, device, ttnn.bfloat16),
        _to_device(g, device, ttnn.float32),
        _to_device(beta, device, ttnn.float32),
    )


def _per_channel_gate(g: torch.Tensor, key_dim: int) -> torch.Tensor:
    """[H, N, 32, 1] scalar gate -> the equivalent per-channel gate [1, T, H*K], broadcast over K per V head."""
    num_heads, num_chunks, chunk, _ = g.shape
    flat = g[..., 0].permute(1, 2, 0).reshape(1, num_chunks * chunk, num_heads, 1)
    return flat.expand(1, num_chunks * chunk, num_heads, key_dim).reshape(1, num_chunks * chunk, num_heads * key_dim)


_SCALAR_DECAY_PEAK_ERROR = _STRONG_DECAY_PEAK_ERROR  # intra 2.5e-4, k_dec_t 3e-3 (FP64 oracle)
_FINAL_DECAY_RELATIVE_ERROR = 2.0**-7


def _assert_scalar_decay_accurate(
    expected: Sequence[torch.Tensor], actual: Sequence[torch.Tensor], *, context: str
) -> dict[str, float]:
    """PCC on every output except final_decay, peak gates on intra/k_dec_t/t_inv, relative gate on final_decay.

    final_decay is constant across a chunk's K rows and can be constant across chunks, so its PCC is undefined;
    it is gated by the weak-end relative bound instead. Returns the peak absolute error per output.
    """
    peaks = {}
    for name, expected_output, actual_output in zip(OUTPUT_NAMES, expected, actual, strict=True):
        assert torch.isfinite(actual_output.float()).all(), f"{context} {name} contains nonfinite values"
        peaks[name] = float((expected_output.float() - actual_output.float()).abs().max())
        if name == "final_decay":
            reference = expected_output.double()
            relative = float(((actual_output.double() - reference).abs() / reference.abs().clamp_min(1e-30)).max())
            peaks["final_decay_rel"] = relative
            assert (
                relative <= _FINAL_DECAY_RELATIVE_ERROR
            ), f"{context} final_decay rel error {relative:.3e} > {_FINAL_DECAY_RELATIVE_ERROR:.1e}"
        else:
            _assert_output_accurate(name, expected_output, actual_output, context=context)
    logger.info(f"{context}: peak abs error per output {peaks}")
    for name, threshold in _SCALAR_DECAY_PEAK_ERROR.items():
        assert peaks[name] <= threshold, f"{context} {name} max abs error {peaks[name]:.3e} > {threshold:.1e}"
    return peaks


def _run_scalar(
    inputs: tuple[ttnn.Tensor, ...], num_heads: int, device: ttnn.Device, **options
) -> tuple[torch.Tensor, ...]:
    outputs = _run(
        inputs,
        num_heads,
        output_bf16_mask=_PRODUCTION_OUTPUT_BF16_MASK,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_production_compute_config(device),
        **options,
    )
    actual = tuple(ttnn.to_torch(output) for output in outputs)
    for output in outputs:
        ttnn.deallocate(output)
    return actual


# Per-chunk |G_last| from below BF16 resolution of exp(G_last) to far beyond the per-channel path's limit (~150):
# Qwen layer-0 per-chunk maxima 2931 (Qwen3.6-35B-A3B) and 8635 (Qwen3.8-2.4T), and constant full-forgetting heads at
# 94 and 500 per token. Each case runs spread and spike profiles (fractional FP32 gates) and the constant profile.
@pytest.mark.parametrize(
    "chunk_log_decay",
    [
        pytest.param(1e-6, id="g1e-6"),
        pytest.param(1e-3, id="g1e-3"),
        pytest.param(16.0, id="g16"),
        pytest.param(160.0, id="g160"),
        pytest.param(1e3, id="g1e3"),
        pytest.param(2931.0, id="qwen35b-g2931"),
        pytest.param(3008.0, id="full-forget94-g3008"),
        pytest.param(8635.0, id="qwen2p4t-g8635"),
        pytest.param(1e4, id="g1e4"),
        pytest.param(16000.0, id="full-forget500-g16000"),
    ],
)
def test_prepare_chunk_recurrence_scalar_decay_range(device: ttnn.Device, chunk_log_decay: float) -> None:
    num_heads, num_chunks, key_dim, value_dim = 2, 2, 128, 128
    q, k, v, _, beta = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=2209)
    g = _scalar_gates(
        [
            [("spread", chunk_log_decay), ("spike", chunk_log_decay)],
            [("spike", chunk_log_decay), ("constant", chunk_log_decay)],
        ],
        seed=2210,
    )
    host_inputs = (q, k, v, g, beta)
    expected = _scalar_decay_oracle(host_inputs, num_heads, _PRODUCTION_OUTPUT_BF16_MASK)
    assert all(torch.isfinite(output.float()).all() for output in expected), "oracle must stay finite at any decay"
    actual = _run_scalar(_scalar_device_inputs(host_inputs, device), num_heads, device)
    _assert_scalar_decay_accurate(expected, actual, context=f"scalar |G_last|={chunk_log_decay:g}")


# Weak end in scalar mode (g1b.7 T3): final_decay carries expm1(G_last), so forgetting below BF16's resolution of 1.0
# survives storage. One chunk per |G_last|, fractional spread gates.
def test_prepare_chunk_recurrence_scalar_weak_final_decay(device: ttnn.Device) -> None:
    num_heads, num_chunks, key_dim, value_dim = 1, len(_WEAK_DECAY_CHUNK_LOG_DECAYS), 128, 128
    q, k, v, _, beta = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=7302)
    g = _scalar_gates([[("spread", decay) for decay in _WEAK_DECAY_CHUNK_LOG_DECAYS]], seed=7303)
    actual = _run_scalar(_scalar_device_inputs((q, k, v, g, beta), device), num_heads, device)
    final_decay = actual[OUTPUT_NAMES.index("final_decay")].double()[..., 0]  # [H, N, K]
    expected = torch.expm1(g.double()[..., 0].sum(dim=-1))[..., None]  # [H, N, 1]
    legacy = torch.exp(g.double()[..., 0].sum(dim=-1)).to(torch.bfloat16).double()[..., None] - 1.0
    failures = []
    for chunk, chunk_log_decay in enumerate(_WEAK_DECAY_CHUNK_LOG_DECAYS):
        reference = expected[:, chunk]
        error = float(((final_decay[:, chunk] - reference).abs() / reference.abs()).max())
        legacy_error = float(((legacy[:, chunk] - reference).abs() / reference.abs()).max())
        logger.info(
            f"scalar |G_last|={chunk_log_decay:.1e}: final_decay rel error {error:.3e} (exp form {legacy_error:.3e})"
        )
        if not error <= _FINAL_DECAY_RELATIVE_ERROR:
            failures.append(f"|G_last|={chunk_log_decay:.1e} rel error {error:.3e}")
    assert not failures, "; ".join(failures)
    # Negative control: the gate is sensitive to the exp form at the weak end.
    assert float((legacy[:, 0] - expected[:, 0]).abs().max() / expected[:, 0].abs().max()) > 0.5


def _bf16_scalar_gates(gate_case: str, num_heads: int, num_chunks: int, *, seed: int) -> torch.Tensor:
    """Scalar gates whose values are exact in BF16, so both modes consume identical gates."""
    generator = torch.Generator().manual_seed(seed)
    shape = (num_heads, num_chunks, CHUNK_SIZE, 1)
    if gate_case == "fractional-0.05":  # the prep's default gate domain, fractional BF16 values
        g = -0.001 - 0.05 * torch.rand(shape, generator=generator)
    elif gate_case == "eighths-4.5":  # multiples of 1/8 in [-4.5, 0]: |G_last| <= 144, every exponent TF32-exact
        g = -torch.randint(0, 37, shape, generator=generator).float() / 8
    else:
        raise ValueError(gate_case)
    return g.to(torch.bfloat16).float()


# Scalar mode equals per-channel mode with g broadcast over K wherever the per-channel path is accurate (it loses
# precision for fractional gates at |G_last| ~ 80-150 and is wrong beyond ~150; see the xfails above).
@pytest.mark.parametrize("gate_case", ["fractional-0.05", "eighths-4.5"], ids=str)
def test_prepare_chunk_recurrence_scalar_matches_broadcast(device: ttnn.Device, gate_case: str) -> None:
    num_heads, num_chunks, key_dim, value_dim = 2, 4, 128, 128
    q, k, v, _, beta = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=2211)
    g = _bf16_scalar_gates(gate_case, num_heads, num_chunks, seed=2212)
    scalar = _run_scalar(_scalar_device_inputs((q, k, v, g, beta), device), num_heads, device)
    broadcast = _run_scalar(_device_inputs((q, k, v, _per_channel_gate(g, key_dim), beta), device), num_heads, device)
    _assert_outputs_accurate(broadcast, scalar, context=f"scalar vs broadcast {gate_case}")
    for name, threshold in _SCALAR_DECAY_PEAK_ERROR.items():
        index = OUTPUT_NAMES.index(name)
        max_abs = float((broadcast[index].float() - scalar[index].float()).abs().max())
        logger.info(f"scalar vs broadcast {gate_case} {name}: max abs difference {max_abs:.3e}")
        assert max_abs <= threshold, f"{gate_case} {name} scalar vs broadcast {max_abs:.3e} > {threshold:.1e}"


def _scalar_head_mapping_gates(case: _HeadMappingCase, *, seed: int) -> torch.Tensor:
    decays = (0.5, 24.0, 300.0)
    profiles = ("spread", "spike", "constant")
    return _scalar_gates(
        [
            [(profiles[(head + chunk) % 3], decays[(head * 7 + chunk) % 3]) for chunk in range(case.num_chunks)]
            for head in range(case.num_heads)
        ],
        seed=seed,
    )


@pytest.mark.parametrize("case", _HEAD_MAPPING_CASES, ids=lambda case: case.case_id)
def test_prepare_chunk_recurrence_scalar_decay_head_mapping_equals_expansion(
    device: ttnn.Device, case: _HeadMappingCase
) -> None:
    q, k, v, _, beta = _head_mapping_host_inputs(case, seed=5405)
    host_inputs = (q, k, v, _scalar_head_mapping_gates(case, seed=5406), beta)
    mapped = _run_scalar(
        _scalar_device_inputs(host_inputs, device), case.num_heads, device, num_key_heads=case.num_key_heads
    )
    expanded_inputs = _expanded_inputs(host_inputs, case.num_key_heads, case.num_heads)
    expanded = _run_scalar(_scalar_device_inputs(expanded_inputs, device), case.num_heads, device)
    for name, mapped_output, expanded_output in zip(OUTPUT_NAMES, mapped, expanded, strict=True):
        assert_bit_identical(expanded_output, mapped_output, name=f"{case.case_id} scalar {name}")


def test_prepare_chunk_recurrence_scalar_decay_bounded_reference(device: ttnn.Device) -> None:
    """Scalar mode with head mapping and chronology: nonzero actual_start, actual_end mid-sequence, FP64 oracle."""
    case = _HeadMappingCase("scalar-hk4-hv12", 4, 12, 20, 128, 128)
    start_row, valid_chunks = 64, 13
    q, k, v, _, beta = _head_mapping_host_inputs(case, seed=5407)
    host_inputs = (q, k, v, _scalar_head_mapping_gates(case, seed=5408), beta)
    valid_rows = valid_chunks * CHUNK_SIZE
    valid_inputs = (q[:, :valid_rows], k[:, :valid_rows], v[:, :valid_rows], host_inputs[3][:, :valid_chunks])
    valid_inputs += (beta[:, :valid_chunks],)
    expected = _scalar_decay_oracle(
        valid_inputs, case.num_heads, _PRODUCTION_OUTPUT_BF16_MASK, num_key_heads=case.num_key_heads
    )
    start = make_actual_start(device, start_row)
    end = make_actual_start(device, start_row + valid_rows)
    actual = _run_scalar(
        _scalar_device_inputs(host_inputs, device),
        case.num_heads,
        device,
        num_key_heads=case.num_key_heads,
        actual_start=start,
        actual_end=end,
    )
    # Chunks past actual_end are unspecified; compare the valid prefix.
    _assert_scalar_decay_accurate(expected, tuple(output[:, :valid_chunks] for output in actual), context=case.case_id)
    for tensor in (start, end):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("scalar_bf16", "scalar-decay g must be FLOAT32"),
        ("scalar_width", "scalar-decay g shape must be"),
        ("scalar_heads", "scalar-decay g shape must be"),
        ("scalar_chunks", "scalar-decay g shape must be"),
        ("gate_rank", "g must be either"),
    ],
    ids=["scalar-bf16", "scalar-width", "scalar-heads", "scalar-chunks", "gate-rank"],
)
def test_prepare_chunk_recurrence_scalar_decay_rejects_invalid_gates(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    """Rejected before any program is built: the program cache does not grow."""
    num_heads, num_chunks = 2, 2
    q, k, v, g, beta = _host_inputs(num_heads, num_chunks, 32, 32)
    scalar = torch.zeros(num_heads, num_chunks, CHUNK_SIZE, 1) - 0.25
    inputs = list(_scalar_device_inputs((q, k, v, scalar, beta), device))
    if case == "scalar_bf16":
        inputs[3] = _to_device(scalar, device, ttnn.bfloat16)
    elif case == "scalar_width":
        inputs[3] = _to_device(scalar.expand(-1, -1, -1, 2).contiguous(), device, ttnn.float32)
    elif case == "scalar_heads":
        inputs[3] = _to_device(scalar[:1], device, ttnn.float32)
    elif case == "scalar_chunks":
        inputs[3] = _to_device(scalar[:, :1], device, ttnn.float32)
    elif case == "gate_rank":
        inputs[3] = _to_device(g[0], device, ttnn.bfloat16)
    entries = device.num_program_cache_entries()
    with expect_error(RuntimeError, message):
        _run(tuple(inputs), num_heads)
    assert device.num_program_cache_entries() == entries


def test_prepare_chunk_recurrence_decay_mode_is_program_identity(device: ttnn.Device) -> None:
    """The decay mode selects a distinct program; a repeated scalar call with fresh tensors hits it and rebinds."""
    num_heads, num_chunks, key_dim, value_dim = 2, 3, 64, 32
    host_a = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=5409)
    host_b = _host_inputs(num_heads, num_chunks, key_dim, value_dim, seed=5410)
    gates = [
        [("spread", 40.0), ("spike", 400.0), ("constant", 4.0)],
        [("spike", 3.0), ("spread", 900.0), ("constant", 0.01)],
    ]
    scalar_a = (*host_a[:3], _scalar_gates(gates, seed=5411), host_a[4])
    scalar_b = (*host_b[:3], _scalar_gates(gates[::-1], seed=5412), host_b[4])
    per_channel = _run(_device_inputs(host_a, device), num_heads)
    entries = device.num_program_cache_entries()
    inputs_a = _scalar_device_inputs(scalar_a, device)
    outputs_a = _run(inputs_a, num_heads)
    assert device.num_program_cache_entries() == entries + 1
    inputs_b = _scalar_device_inputs(scalar_b, device)
    outputs_b = _run(inputs_b, num_heads)
    assert device.num_program_cache_entries() == entries + 1
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(inputs_a, inputs_b, strict=True))
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(outputs_a, outputs_b, strict=True))
    for host, outputs, context in ((scalar_a, outputs_a, "scalar miss"), (scalar_b, outputs_b, "scalar hit fresh")):
        _assert_scalar_decay_accurate(
            _scalar_decay_oracle(host, num_heads, 0),
            tuple(ttnn.to_torch(output) for output in outputs),
            context=context,
        )
    for tensor in (*per_channel, *outputs_a, *outputs_b):
        ttnn.deallocate(tensor)


# Op-level device time of the two decay modes at the GDN per-chip shapes (TP4, 640 rows, Hk = 4): scalar mode against
# per-channel mode on the same gate broadcast over K. References are medians of three runs (2026-10-05, LoudBox p150b,
# tt_metal_tracker-g1b.5.4.2); margin as the production case.
_GDN_PERFORMANCE_CASES = tuple(case for case in _HEAD_MAPPING_CASES if "chip" in case.case_id)
_GDN_EXPECTED_DURATION_NS = {
    ("27b-chip-hk4-hv12", "scalar"): 102_333,
    ("27b-chip-hk4-hv12", "per-channel"): 172_070,
    ("35b-chip-hk4-hv8", "scalar"): 79_821,
    ("35b-chip-hk4-hv8", "per-channel"): 124_123,
    ("2p4t-chip-hk4-hv32", "scalar"): 264_507,
    ("2p4t-chip-hk4-hv32", "per-channel"): 312_969,
}


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("decay_mode", ["scalar", "per-channel"], ids=str)
@pytest.mark.parametrize("case", _GDN_PERFORMANCE_CASES, ids=lambda case: case.case_id)
def test_prepare_chunk_recurrence_gdn_decay_mode_performance(
    device: ttnn.Device, case: _HeadMappingCase, decay_mode: str
) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for chunk-recurrence preparation performance checks")
    q, k, v, _, beta = _head_mapping_host_inputs(case, seed=118)
    gate = _scalar_head_mapping_gates(case, seed=119)
    if decay_mode == "scalar":
        inputs = _scalar_device_inputs((q, k, v, gate, beta), device)
    else:
        inputs = _device_inputs((q, k, v, _per_channel_gate(gate, case.key_dim), beta), device)

    def run() -> list[ttnn.Tensor]:
        return _run(
            inputs,
            case.num_heads,
            num_key_heads=case.num_key_heads,
            output_bf16_mask=_PRODUCTION_OUTPUT_BF16_MASK,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=_production_compute_config(device),
        )

    outputs, perf_record = profile_realtime_program(device, run)
    duration_ns = perf_record["duration_ns"]
    assert len(outputs) == 7
    reference = _GDN_EXPECTED_DURATION_NS[(case.case_id, decay_mode)]
    logger.info(
        f"chunk-recurrence preparation {case.case_id} {decay_mode}: measured_ns={duration_ns:.0f}, "
        f"reference_ns={reference}, runtime_id={perf_record['runtime_id']}"
    )
    upper = reference * (1 + _PERFORMANCE_MARGIN)
    assert duration_ns <= upper, (
        f"{case.case_id} {decay_mode} duration {duration_ns:.0f} ns exceeds {upper:.0f} ns "
        f"(reference {reference} ns, margin {_PERFORMANCE_MARGIN * 100:.0f}%)"
    )
