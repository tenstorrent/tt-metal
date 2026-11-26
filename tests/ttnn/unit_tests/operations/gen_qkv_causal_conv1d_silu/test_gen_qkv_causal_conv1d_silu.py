# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for gen_qkv_causal_conv1d_silu (immutable spec — do not modify).

Every accuracy case is a Phase-0 cell of the golden suite
(eval/golden_tests/gen_qkv_causal_conv1d_silu/feature_spec.py, LOOSE_CASES), so this file stays
inside the golden cells (BB-22); the remaining tests are the cases the op spec's rules require:
chunk-size invariance, bit-identical repeats (BB-06), the plan report and regime-pinned plans
(BB-08 / blocking-model regime pinning), the default compute config factory, and the host
refusals of ## Validation.

Shape coverage: one tile-row (T 32), multi tile-row (64..4096), non-square / unequal widths, both
input layouts. Batch is fixed at 1 by the op's contract, so there is no multi-batch shape.

Accuracy gates: PCC and relative RMS from the golden helpers' TOLERANCES (PCC only below HiFi4,
eval/metrics.LOWER_FIDELITY_PCC), plus the per-element fp32-accumulation bound whenever
fp32_dest_acc_en is on at HiFi4.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import (
    default_compute_kernel_config,
    gen_qkv_causal_conv1d_silu,
    get_last_plan,
    plan_limits,
)
from eval.metrics import LOWER_FIDELITY_PCC, Tolerance, check_output
from eval.golden_tests.gen_qkv_causal_conv1d_silu.helpers import FP32_DEST_BOUND, TOLERANCES

TAP_COUNT = 4
HISTORY_ROWS = TAP_COUNT - 1

_BUFFER = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
_OUTPUT_MEMORY = {"default": None, "dram_interleaved": ttnn.DRAM_MEMORY_CONFIG, "l1_interleaved": ttnn.L1_MEMORY_CONFIG}
_EXPECTED_OUTPUT_MEMORY = {
    "default": ttnn.DRAM_MEMORY_CONFIG,
    "dram_interleaved": ttnn.DRAM_MEMORY_CONFIG,
    "l1_interleaved": ttnn.L1_MEMORY_CONFIG,
}
_RM, _TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
_MF = ttnn.MathFidelity


def _case(
    case_id,
    inputs,
    layout,
    *,
    in_buffer="dram",
    history_buffer="dram",
    taps_buffer="dram",
    output_memory="default",
    fp32=False,
    fidelity=_MF.HiFi4,
    config_none=False,
    cancelling=False,
):
    return pytest.param(
        dict(
            inputs=inputs,
            layout=layout,
            in_buffer=in_buffer,
            history_buffer=history_buffer,
            taps_buffer=taps_buffer,
            output_memory=output_memory,
            fp32=fp32,
            fidelity=fidelity,
            config_none=config_none,
            cancelling=cancelling,
        ),
        id=case_id,
    )


# Phase-0 cells of the golden suite (same inputs tuple, placements and compute config).
PHASE0_CASES = [
    _case("conv1d-001_decode1_T32", ((1, 32, 2560), (512, 512, 1536), 256, None, 0), _RM, config_none=True),
    _case("conv1d-002_decode16_T64", ((1, 64, 2560), (512, 512, 1536), 256, None, 0), _RM, config_none=True),
    _case("conv1d-003_prefill_T128", ((1, 128, 2560), (512, 512, 1536), 256, None, 0), _RM, config_none=True),
    _case("conv1d-004_prefill_T4096", ((1, 4096, 2560), (512, 512, 1536), 256, None, 0), _RM, config_none=True),
    _case("conv1d-005_tile_input", ((1, 64, 2560), (512, 512, 1536), 256, None, 0), _TILE),
    _case(
        "conv1d-006_input_l1", ((1, 96, 2048), (1024, 512, 512), 256, None, 0), _RM, in_buffer="l1", fidelity=_MF.LoFi
    ),
    _case("conv1d-007_history_l1", ((1, 128, 2560), (512, 512, 1536), 128, None, 0), _RM, history_buffer="l1"),
    _case(
        "conv1d-008_taps_l1",
        ((1, 256, 2048), (1024, 512, 512), 128, None, 0),
        _RM,
        taps_buffer="l1",
        fidelity=_MF.HiFi2,
    ),
    _case(
        "conv1d-010_out_l1",
        ((1, 96, 2048), (1024, 512, 512), 512, None, 0),
        _RM,
        output_memory="l1_interleaved",
        fidelity=_MF.LoFi,
    ),
    _case(
        "conv1d-011_fp32_cancelling", ((1, 128, 2560), (512, 512, 1536), 64, None, 0), _RM, fp32=True, cancelling=True
    ),
    _case("conv1d-014_equal_widths", ((1, 96, 3072), (1024, 1024, 1024), 256, None, 0), _RM),
    _case("conv1d-015_narrow", ((1, 128, 896), (512, 256, 128), 128, None, 0), _RM, fidelity=_MF.LoFi),
    _case("conv1d-016_straddling_chunk", ((1, 256, 2560), (512, 512, 1536), 160, None, 0), _RM),
    _case("conv1d-017_single_chunk", ((1, 64, 896), (512, 256, 128), 896, None, 0), _RM, fidelity=_MF.HiFi2),
    _case(
        "conv1d-020_T2048", ((1, 2048, 2048), (1024, 512, 512), 1024, None, 0), _RM, output_memory="dram_interleaved"
    ),
    _case(
        "conv1d-021_tiny_out_l1",
        ((1, 32, 192), (96, 64, 32), 192, None, 0),
        _RM,
        output_memory="l1_interleaved",
        fidelity=_MF.LoFi,
    ),
]


# ---------------------------------------------------------------------------------------------
# Reference and operands
# ---------------------------------------------------------------------------------------------


def pytorch_reference(conv_input, history, taps, widths):
    """float64 (q, k, v) and the per-element Σ_j |x_j w_j| of the fp32 bound.

    y[t, c] = silu(Σ_j tap_j[c] · x[t-3+j, c]), x[-3..-1] = history rows 0..2. Equivalent to
    F.conv1d over cat(history, x) (channels-first) with weight[c, 0, j] = tap_j[c], groups = C.
    """
    seq = conv_input.shape[1]
    window = torch.cat((history, conv_input), dim=1).double()
    terms = torch.stack([window[:, j : j + seq] * taps[j].double() for j in range(TAP_COUNT)])
    y = F.silu(terms.sum(0))
    return y.split(list(widths), dim=-1), terms.abs().sum(0).split(list(widths), dim=-1)


def _random_operands(seq, width, channels, offset):
    torch.manual_seed(42)
    conv_input = torch.randn(1, seq, channels).bfloat16()
    history = torch.randn(1, HISTORY_ROWS, channels).bfloat16()
    taps = [torch.randn(1, 1, channels).bfloat16() for _ in range(TAP_COUNT)]
    full = torch.randn(1, seq, width).bfloat16()
    full[..., offset : offset + channels] = conv_input
    return full, conv_input, history, taps


def _cancelling_operands(seq, width, channels, offset):
    """Large activations under taps whose sum nearly cancels: a 16-bit accumulator breaches the
    fp32 bound, an fp32 accumulator meets it (same construction as the golden suite)."""
    torch.manual_seed(42)
    rows = HISTORY_ROWS + seq
    magnitude = 32 + 64 * torch.rand(channels)
    sign = torch.where(torch.rand(channels) < 0.5, -1.0, 1.0)
    window = (sign * magnitude * (1 + 0.01 * torch.randn(rows, channels))).bfloat16()[None]
    same_sign = (0.05 + 0.1 * torch.rand(3, channels)).bfloat16().float()
    delta = 0.02 * (2 * torch.rand(channels) - 1)
    tap_sign = torch.where(torch.rand(channels) < 0.5, -1.0, 1.0)
    taps = [(tap_sign * same_sign[j]).bfloat16()[None, None] for j in range(3)]
    taps.append((tap_sign * (delta - same_sign.sum(0))).bfloat16()[None, None])
    conv_input = window[:, HISTORY_ROWS:].contiguous()
    history = window[:, :HISTORY_ROWS].contiguous()
    full = torch.randn(1, seq, width).bfloat16()
    full[..., offset : offset + channels] = conv_input
    return full, conv_input, history, taps


def _compute_config(device, case):
    if case["config_none"]:
        return None
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=case["fidelity"],
        math_approx_mode=False,
        fp32_dest_acc_en=case["fp32"],
        packer_l1_acc=False,
    )


def _actual_start(device):
    return ttnn.from_torch(
        torch.tensor([0], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _run(device, case, *, chunk=None):
    """Build the operands, call the op, return everything the checks need."""
    (_, seq, width), widths, golden_chunk, page_channels, offset = case["inputs"]
    channels = sum(widths)
    make = _cancelling_operands if case["cancelling"] else _random_operands
    full, conv_input, history, taps = make(seq, width, channels, offset)

    input_tt = ttnn.from_torch(
        full, dtype=ttnn.bfloat16, layout=case["layout"], device=device, memory_config=_BUFFER[case["in_buffer"]]
    )
    history_tt = ttnn.from_torch(
        history,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_BUFFER[case["history_buffer"]],
    )
    taps_tt = [
        ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_BUFFER[case["taps_buffer"]]
        )
        for t in taps
    ]
    outputs = gen_qkv_causal_conv1d_silu(
        input_tt,
        history_tt,
        *taps_tt,
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(
            channel_chunk_size=golden_chunk if chunk is None else chunk
        ),
        actual_start=_actual_start(device),
        predecessor_carry=history_tt,
        memory_config=_OUTPUT_MEMORY[case["output_memory"]],
        compute_kernel_config=_compute_config(device, case),
        input_column_offset=offset,
        page_channels=page_channels,
    )
    return outputs, (full, conv_input, history, taps), (input_tt, history_tt, taps_tt)


def _tolerance(case):
    fp32 = case["fp32"] and not case["config_none"]
    pcc, rms = TOLERANCES[(ttnn.bfloat16, bool(fp32))]
    if case["fidelity"] != _MF.HiFi4:
        return Tolerance(pcc=min(pcc, LOWER_FIDELITY_PCC), rms=None)
    return Tolerance(pcc=pcc, rms=rms)


def _check_case(case, outputs, host, device_inputs):
    (_, seq, _), widths, _, _, _ = case["inputs"]
    full, conv_input, history, taps = host
    input_tt, history_tt, taps_tt = device_inputs
    assert isinstance(outputs, tuple) and len(outputs) == 3, "must return the tuple (q, k, v)"
    (q_ref, k_ref, v_ref), sums = pytorch_reference(conv_input, history, taps, widths)
    tolerance = _tolerance(case)
    expected_memory = _EXPECTED_OUTPUT_MEMORY[case["output_memory"]]
    for name, out, ref, width_, terms in zip("qkv", outputs, (q_ref, k_ref, v_ref), widths, sums):
        check_output(
            out, ref, shape=[1, seq, width_], dtype=ttnn.bfloat16, expected_layout=ttnn.TILE_LAYOUT, tolerance=tolerance
        )
        assert out.memory_config() == expected_memory, f"{name}: memory_config {out.memory_config()}"
        if case["fp32"] and case["fidelity"] == _MF.HiFi4:
            actual = ttnn.to_torch(out).double()
            relative, absolute = FP32_DEST_BOUND
            ratio = (actual - ref).abs() / (relative * ref.abs() + absolute * terms)
            assert (
                float(ratio.max()) <= 1.0
            ), f"{name}: fp32 accumulation bound exceeded, peak error/bound {float(ratio.max()):.3f}"
    # No input is modified.
    assert torch.equal(ttnn.to_torch(input_tt).reshape(full.shape), full), "input was modified"
    assert torch.equal(ttnn.to_torch(history_tt).reshape(history.shape), history), "history was modified"
    for j, (t_tt, t) in enumerate(zip(taps_tt, taps)):
        assert torch.equal(ttnn.to_torch(t_tt).reshape(t.shape), t), f"tap{j} was modified"


# ---------------------------------------------------------------------------------------------
# Accuracy on the Phase-0 golden cells
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", PHASE0_CASES)
def test_gen_qkv_causal_conv1d_silu(device, case):
    outputs, host, device_inputs = _run(device, case)
    _check_case(case, outputs, host, device_inputs)


# ---------------------------------------------------------------------------------------------
# Rules of the spec
# ---------------------------------------------------------------------------------------------


def test_chunk_size_invariance(device):
    """Same values for every valid chunk size: conv1d-002 (chunk 256) vs conv1d-009 (chunk 512)."""
    case = PHASE0_CASES[1].values[0]
    first, _, _ = _run(device, case, chunk=256)
    second, _, _ = _run(device, case, chunk=512)
    for a, b in zip(first, second):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), "result depends on channel_chunk_size"


@pytest.mark.parametrize("case", [PHASE0_CASES[0], PHASE0_CASES[4]], ids=["rm", "tile"])
def test_repeat_is_bit_identical(device, case):
    """BB-06: the same call on the same inputs returns bit-identical outputs."""
    first, _, _ = _run(device, case.values[0])
    second, _, _ = _run(device, case.values[0])
    for a, b in zip(first, second):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), "repeat call is not bit-identical"


# Regime-pinned plans: (golden case, plan limits, expected regime, expected block_rows, max cores/group).
# They force the multi-block / L1-halo, DRAM-halo, TILE extra-untilized-row and ragged-last-block paths
# on any grid, independent of what select_plan would pick for the device.
_PINNED = [
    pytest.param(1, dict(max_block_rows=1, max_cores_per_group=1), "rm_staged_window", 1, 1, id="rm_multiblock"),
    pytest.param(2, dict(max_block_rows=1, max_cores_per_group=2), "rm_staged_window", 1, 2, id="rm_dram_halo"),
    pytest.param(
        4, dict(max_block_rows=1, max_cores_per_group=1), "tile_untilize_staged_window", 1, 1, id="tile_multiblock"
    ),
    pytest.param(
        4, dict(max_block_rows=1, max_cores_per_group=2), "tile_untilize_staged_window", 1, 2, id="tile_prev_row_halo"
    ),
    pytest.param(
        10, dict(max_block_rows=2, max_cores_per_group=1), "rm_staged_window", 2, 1, id="rm_ragged_last_block"
    ),
]


@pytest.mark.parametrize("case_index,limits,regime,block_rows,max_cpg", _PINNED)
def test_regime_pinned(device, case_index, limits, regime, block_rows, max_cpg):
    case = PHASE0_CASES[case_index].values[0]
    with plan_limits(**limits):
        outputs, host, device_inputs = _run(device, case)
        plan = get_last_plan()
    assert plan.regime == regime
    assert plan.block_rows == block_rows
    assert 1 <= plan.cores_per_group <= max_cpg
    _check_case(case, outputs, host, device_inputs)


def test_plan_is_reported(device):
    """BB-08: the plan of the last call is reportable and uses the device grid."""
    case = PHASE0_CASES[3].values[0]  # conv1d-004
    outputs, host, device_inputs = _run(device, case)
    plan = get_last_plan()
    grid = device.compute_with_storage_grid_size()
    assert plan.regime == "rm_staged_window"
    assert plan.num_cores == plan.num_ch_groups * plan.cores_per_group
    assert plan.num_cores <= grid.x * grid.y
    assert plan.num_ch_groups * plan.block_ch_tiles == 2560 // 32
    _check_case(case, outputs, host, device_inputs)


def test_default_compute_kernel_config():
    config = default_compute_kernel_config()
    assert config.math_fidelity == ttnn.MathFidelity.HiFi4
    assert config.math_approx_mode is False
    assert config.fp32_dest_acc_en is False
    assert config.packer_l1_acc is False
    assert config.dst_full_sync_en is False


# ---------------------------------------------------------------------------------------------
# Host refusals (## Validation): ValueError (RuntimeError accepted where ttnn raises it)
# ---------------------------------------------------------------------------------------------


def _small_call(
    device,
    *,
    seq=32,
    widths=(32, 32, 32),
    chunk=96,
    history_shape=None,
    tap_channels=None,
    compute_kernel_config=None,
    memory_config=None,
    input_column_offset=0,
    carry_l1=False,
):
    channels = 96
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn(1, seq, channels).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    history = ttnn.from_torch(
        torch.randn(*(history_shape or (1, 3, channels))).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    carry = history
    if carry_l1:
        carry = ttnn.from_torch(
            torch.randn(1, 3, channels).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    taps = [
        ttnn.from_torch(
            torch.randn(1, 1, tap_channels or channels).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for _ in range(TAP_COUNT)
    ]
    return gen_qkv_causal_conv1d_silu(
        x,
        history,
        *taps,
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=chunk),
        actual_start=_actual_start(device),
        predecessor_carry=carry,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
        input_column_offset=input_column_offset,
    )


_REFUSALS = [
    pytest.param(dict(seq=48), id="T_not_multiple_of_32"),
    pytest.param(dict(chunk=64), id="chunk_not_dividing_C"),
    pytest.param(dict(chunk=128), id="chunk_above_C"),
    pytest.param(dict(widths=(32, 32, 16)), id="width_not_multiple_of_32"),
    pytest.param(dict(input_column_offset=16), id="offset_not_multiple_of_32"),
    pytest.param(dict(history_shape=(1, 2, 96)), id="history_shape"),
    pytest.param(dict(tap_channels=64), id="tap_last_dim"),
    pytest.param(dict(carry_l1=True), id="carry_spec_differs"),
    pytest.param(dict(memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG), id="sharded_output"),
    pytest.param(
        dict(
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True
            )
        ),
        id="math_approx_mode",
    ),
    pytest.param(
        dict(
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, packer_l1_acc=True
            )
        ),
        id="packer_l1_acc",
    ),
]


@pytest.mark.parametrize("kwargs", _REFUSALS)
def test_validation_refusals(device, expect_error, kwargs):
    # The unmodified small call ((1, 32, 96), widths 32/32/32, chunk 96) is a SUPPORTED cell, so the
    # refusal must come from the malformed argument — a ValueError, never an unsupported-cell
    # NotImplementedError (which is also a RuntimeError). The message text is the op's own choice,
    # so any message matches.
    with expect_error((ValueError, RuntimeError), "") as info:
        _small_call(device, **kwargs)
    assert not isinstance(
        info.value, NotImplementedError
    ), f"malformed call refused as an unsupported cell instead of a ValueError: {info.value!r}"
