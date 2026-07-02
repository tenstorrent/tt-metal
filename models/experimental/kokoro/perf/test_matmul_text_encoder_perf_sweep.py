# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Input × output × compute-config sweep for the TextEncoder/BiLSTM matmuls (``textencoder/new.log``).

Reuses the case-generation / program-config / memory-config / profiler harness from
:mod:`test_matmul_decoder_perf_sweep` and swaps in the matmul shapes lifted from the Kokoro
text-encoder Tracy report at ``textencoder/new.log``.

Shapes (all BF16 x BF16 => BF16, HiFi3, fp32_dest_acc_en=True — the production text-encoder default):

  | shape           | count | avg dev_us | role                                                    |
  |-----------------|-------|------------|---------------------------------------------------------|
  | 32 x 512 x 2048 |  144  |   ~6 us    | per-step recurrent ``h @ W_h_block`` (direction-fused)  |
  | 64 x 512 x 1024 |    6  |   18 us    | one-shot gate precompute ``x @ W_x``                    |
  | 64 x 64  x 512  |    3  |    6 us    | anti-identity reverse-input reorder ``anti @ x``        |
  | 64 x 64  x 256  |    3  |    3 us    | anti-identity reverse-output reorder ``anti @ hs_b_rev``|

What is swept (to find the best optimization strategy per shape):
  - INPUT layout   : in0 memory config (l1 / dram / width / height / block sharded)
  - OUTPUT layout  : out memory config (l1 / dram / width / height / block sharded)
  - CORE GRID      : every feasible (gx, gy) the shape tiles onto, per matmul family
  - COMPUTE CONFIG : math fidelity (LoFi / HiFi2 / HiFi3 / HiFi4), fp32_dest_acc_en, packer_l1_acc,
                     AND the in1 (weight) dtype (bf16 vs bfloat8_b — the half-DRAM-read lever)
  - FAMILY         : 1D_in0 (mcast in0) / 1D_in1 / 2D / dram-sharded

The compute/weight-dtype axis is the new dimension vs. the layout-only decoder harness: each
``ComputeStrategy`` rebuilds the ``compute_kernel_config`` (and the in1 dtype) and the full
input/output/grid sweep is re-run under it. PCC is still gated at 0.99 (bf16 out), so a
faster low-fidelity / bf8 strategy only wins if it also stays accurate enough.

IMPORTANT — read before acting on the winner:
  The text-encoder forward is host-dispatch bound, NOT compute bound: in ``textencoder/new.log``
  total device compute is a few ms but the op-to-op gap dwarfs it (device idle ~99%). These
  matmuls are 3-18 us of device time each. A config that halves device time saves microseconds
  against a ~hundreds-of-us/op host gap — invisible in wall-clock unless the model is traced.

  Worse, width/block-sharding the *in0* activation needs an ``InterleavedToSharded`` before and a
  ``ShardedToInterleaved`` after the matmul. Inside the per-step LSTM loop that is +2 host-dispatched
  ops/step — a large NET regression even when the isolated matmul is faster. So treat an in0-sharded
  (``ws/hs/bs``) winner as device-time-only; the loop should stay on ``dram``/``l1`` interleaved in0.
  The summary therefore also reports the fastest in0-INTERLEAVED config (loop-safe, no reshard).

REQUIRED ENV (test skips otherwise):
  - ``TT_METAL_DEVICE_PROFILER=1``
  - ``TT_METAL_PROFILER_MID_RUN_DUMP=1``

Run one shape (optionally one strategy):
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    KOKORO_MATMUL_SHAPE=32x512x2048 KOKORO_MATMUL_STRATEGY=hifi3 pytest -s \\
      models/experimental/kokoro/perf/test_matmul_text_encoder_perf_sweep.py -v

Run all text-encoder shapes × all strategies:
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    pytest -s models/experimental/kokoro/perf/test_matmul_text_encoder_perf_sweep.py -v
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

from . import test_matmul_decoder_perf_sweep as _dec
from .test_matmul_decoder_perf_sweep import (
    CaseResult,
    MatmulCase,
    ReportShape,
    _PCC_TARGETS,
    _device_kernel_us,
    _drain_profiler,
    _make_cases,
    _mem_in0,
    _mem_in1,
    _mem_out,
    _program_config,
)

# The decoder harness only pairs a sharded OUTPUT (ws/hs/bs) with a sharded or DRAM in0 — it never
# tries L1-interleaved in0 -> sharded out. That hides the fact that L1 in0 (cheaper to produce than a
# sharded in0, and the production layout) can feed the fast block-sharded-output configs. We add the
# missing ``("l1", <sharded>)`` rows so L1 inputs compete head-to-head. ``MatmulCase.feasible`` already
# admits l1->ws/hs/bs per family, so only the enumeration table needed widening.
_EXPANDED_LAYOUTS = {
    "1D_in0": [("dram", "l1"), ("l1", "l1"), ("dram", "ws"), ("l1", "ws"), ("ws", "ws")],
    "1D_in1": [("dram", "l1"), ("l1", "l1"), ("dram", "hs"), ("l1", "hs"), ("hs", "hs")],
    "2D": [("dram", "l1"), ("l1", "l1"), ("dram", "bs"), ("l1", "bs"), ("bs", "bs")],
    "dram": [("ws", "ws")],
}


def _make_cases_l1(M: int, K: int, N: int, gx_max: int, gy_max: int, max_cores: int) -> list[MatmulCase]:
    """``_make_cases`` with the L1-in0 -> sharded-out pairings enabled (restores ``_LAYOUTS`` after)."""
    orig = _dec._LAYOUTS
    _dec._LAYOUTS = _EXPANDED_LAYOUTS
    try:
        return _make_cases(M, K, N, gx_max, gy_max, max_cores)
    finally:
        _dec._LAYOUTS = orig


# (M, K, N, report_device_us, ...) from textencoder/new.log. report_us is the average device
# time per call (HiFi3, BF16 in/out, fp32_dest_acc_en). These are the must-beat baselines for
# an isolated matmul.
_REPORT_SHAPES: list[ReportShape] = [
    ReportShape(
        32, 512, 2048, 6.0, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi3, "recurrent_fused"
    ),
    ReportShape(
        64, 512, 1024, 18.0, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi3, "gate_precompute"
    ),
    ReportShape(
        64, 64, 512, 6.0, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi3, "rev_input_reorder"
    ),
    ReportShape(
        64, 64, 256, 3.0, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi3, "rev_output_reorder"
    ),
]


@dataclass(frozen=True)
class ComputeStrategy:
    """One point on the compute-config axis: math fidelity + dest-acc/packer flags + in1 dtype.

    ``in1_dtype`` is part of the *input* sweep — bfloat8_b weights halve the in1 DRAM read and
    speed the matmul up at a precision cost the PCC gate then accepts or rejects.
    """

    label: str
    in1_dtype: ttnn.DataType
    math_fidelity: ttnn.MathFidelity
    fp32_dest_acc_en: bool
    packer_l1_acc: bool


_F = ttnn.MathFidelity
_BF16 = ttnn.bfloat16
_BF8 = ttnn.bfloat8_b

# Compute-config + weight-dtype sweep. Fidelity descends LoFi -> HiFi4; bf16 and bfloat8_b weights.
# fp32_dest_acc_en=True/packer_l1_acc=False mirrors the production default for HiFi3/HiFi4; the
# low-fidelity rows drop fp32 dest acc (cheaper, the precision is already coarse there).
_COMPUTE_STRATEGIES: list[ComputeStrategy] = [
    ComputeStrategy("bf16/lofi", _BF16, _F.LoFi, False, True),
    ComputeStrategy("bf16/hifi2", _BF16, _F.HiFi2, False, True),
    ComputeStrategy("bf16/hifi3", _BF16, _F.HiFi3, True, False),  # production default
    ComputeStrategy("bf16/hifi4", _BF16, _F.HiFi4, True, False),
    ComputeStrategy("bf8b/lofi", _BF8, _F.LoFi, False, True),
    ComputeStrategy("bf8b/hifi2", _BF8, _F.HiFi2, False, True),
    ComputeStrategy("bf8b/hifi3", _BF8, _F.HiFi3, True, False),
]


def _filtered_shapes() -> list[ReportShape]:
    only = os.getenv("KOKORO_MATMUL_SHAPE")
    if not only:
        return _REPORT_SHAPES
    return [s for s in _REPORT_SHAPES if only in s.sweep_id or only == f"{s.M}x{s.K}x{s.N}"]


def _filtered_strategies() -> list[ComputeStrategy]:
    only = os.getenv("KOKORO_MATMUL_STRATEGY")
    if not only:
        return _COMPUTE_STRATEGIES
    return [s for s in _COMPUTE_STRATEGIES if only in s.label]


@dataclass
class StrategyResult:
    """A :class:`CaseResult` annotated with the compute strategy it was measured under."""

    strategy: ComputeStrategy
    result: CaseResult


def _measure_case(
    device: ttnn.Device,
    case: MatmulCase,
    strat: ComputeStrategy,
    shape: ReportShape,
    torch_input_a: torch.Tensor,
    torch_input_b: torch.Tensor,
    torch_output: torch.Tensor,
    pcc_target: float,
) -> CaseResult:
    M, K, N = shape.M, shape.K, shape.N
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=strat.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=strat.fp32_dest_acc_en,
        packer_l1_acc=strat.packer_l1_acc,
    )
    in0 = in1 = warm = timed = None
    try:
        in0 = ttnn.from_torch(
            torch_input_a,
            dtype=shape.in0_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_mem_in0(case, device),
        )
        in1 = ttnn.from_torch(
            torch_input_b,
            dtype=strat.in1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_mem_in1(case, device),
        )
        program_config = _program_config(case)
        out_mem_cfg = _mem_out(case, device)

        warm = ttnn.matmul(
            in0,
            in1,
            program_config=program_config,
            memory_config=out_mem_cfg,
            compute_kernel_config=compute_kernel_config,
            dtype=shape.out_dtype,
        )
        ttnn.synchronize_device(device)
        ttnn.deallocate(warm)
        warm = None
        _drain_profiler(device)
        ttnn.synchronize_device(device)

        timed = ttnn.matmul(
            in0,
            in1,
            program_config=program_config,
            memory_config=out_mem_cfg,
            compute_kernel_config=compute_kernel_config,
            dtype=shape.out_dtype,
        )
        ttnn.synchronize_device(device)
        dev_us = _device_kernel_us(device)
        if dev_us is None:
            raise RuntimeError("profiler produced no device data — is this a profiler build?")

        out = ttnn.to_torch(timed).float()
        assert out.shape == torch_output.shape, f"{out.shape} != {torch_output.shape}"
        pcc_pass, pcc = comp_pcc(torch_output.float(), out, pcc_target)
        tflops = 2 * M * K * N / 1e6 / dev_us
        return CaseResult(case, dev_us, tflops, float(pcc), bool(pcc_pass))
    except Exception as e:
        return CaseResult(case, float("inf"), 0.0, 0.0, False, str(e).strip().splitlines()[0][:90])
    finally:
        for t in (timed, warm, in1, in0):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        ttnn.synchronize_device(device)


@pytest.mark.parametrize("shape", _filtered_shapes(), ids=[s.sweep_id for s in _filtered_shapes()])
def test_matmul_text_encoder_perf_report_sweep(device, shape: ReportShape):
    if os.getenv("TT_METAL_DEVICE_PROFILER") is None:
        pytest.skip(
            "device-time sweep needs a profiler build: rebuild with --enable-profiler "
            "and set TT_METAL_DEVICE_PROFILER=1"
        )
    if os.getenv("TT_METAL_PROFILER_MID_RUN_DUMP") is None:
        pytest.skip(
            "set TT_METAL_PROFILER_MID_RUN_DUMP=1 so ReadDeviceProfiler flushes " "profile_log_device.csv mid-run"
        )

    M, K, N = shape.M, shape.K, shape.N
    grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = grid.x, grid.y
    max_cores = gx_max * gy_max
    cases = _make_cases_l1(M, K, N, gx_max, gy_max, max_cores)
    assert cases, f"no feasible configs generated for {M}x{K}x{N} on a {gx_max}x{gy_max} grid"

    strategies = _filtered_strategies()
    assert strategies, "no compute strategies selected (check KOKORO_MATMUL_STRATEGY)"

    pcc_target = _PCC_TARGETS[shape.out_dtype]

    torch.manual_seed(0)
    # in0 (activation) kept at the production bf16; in1 (weights) cast per-strategy by from_torch.
    # Reference is computed in fp32 from the same draws so every strategy is scored against the
    # exact-math result, exposing the fidelity/bf8 precision loss.
    torch_input_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, K, N), dtype=torch.float32)
    torch_output = torch.matmul(torch_input_a.float(), torch_input_b.float()).to(torch.float32)

    all_results: list[StrategyResult] = []
    for strat in strategies:
        for case in cases:
            res = _measure_case(device, case, strat, shape, torch_input_a, torch_input_b, torch_output, pcc_target)
            all_results.append(StrategyResult(strat, res))

    env_us = os.getenv("MATMUL_BASELINE_US")
    must_beat_us = float(env_us) if env_us is not None else shape.report_us
    passing = [sr for sr in all_results if sr.result.pcc_pass]
    winners = [sr for sr in passing if must_beat_us is None or sr.result.dev_us < must_beat_us]
    best = min(passing, key=lambda sr: sr.result.dev_us) if passing else None
    # in0 interleaved (dram/l1) winner — the only one usable inside the host-driven LSTM loop
    # without a per-step reshard penalty.
    best_no_reshard = min(
        (sr for sr in passing if sr.result.case.in0 in ("dram", "l1")),
        key=lambda sr: sr.result.dev_us,
        default=None,
    )

    logger.info(
        f"text-encoder matmul sweep {shape.sweep_id} in0={shape.in0_dtype} out={shape.out_dtype} "
        f"grid={gx_max}x{gy_max} | {len(strategies)} strategies x {len(cases)} layout/grid configs"
    )
    logger.info(
        f"{'strategy':>11} {'family':>7} {'in0/in1/out':>12} {'grid':>5} {'cores':>5} {'ibw':>4} "
        f"{'pcM':>4} {'pcN':>4} {'sub':>5} {'dev_us':>9} {'TFLOPs':>7} {'PCC':>8} {'result':>7}  note"
    )
    for sr in sorted(all_results, key=lambda sr: sr.result.dev_us):
        r = sr.result
        c = r.case
        sh, sw = c.subblock
        if r.err:
            metrics = f"{'-':>9} {'-':>7} {'-':>8} {'ERROR':>7}"
            note = r.err
        else:
            metrics = f"{r.dev_us:>9.2f} {r.tflops:>7.2f} {r.pcc:>8.4f} {('PASS' if r.pcc_pass else 'FAIL'):>7}"
            tags = []
            if sr is best:
                tags.append("best")
            if sr is best_no_reshard and sr is not best:
                tags.append("best-no-reshard")
            note = " ".join(tags)
        logger.info(
            f"{sr.strategy.label:>11} {c.family:>7} {c.layout:>12} {f'{c.grid_x}x{c.grid_y}':>5} "
            f"{c.num_cores:>5} {c.in0_block_w:>4} {c.per_core_M:>4} {c.per_core_N:>4} "
            f"{f'{sh}x{sw}':>5} {metrics}  {note}"
        )

    # Per-strategy best PCC-pass — shows how far each fidelity/dtype point can go on its own.
    logger.info("per-strategy fastest PCC-pass:")
    for strat in strategies:
        cand = [sr for sr in passing if sr.strategy is strat]
        if cand:
            b = min(cand, key=lambda sr: sr.result.dev_us)
            logger.info(
                f"  {strat.label:>11}: {b.result.dev_us:>7.2f}us PCC={b.result.pcc:.4f} " f"[{b.result.case.label}]"
            )
        else:
            logger.info(f"  {strat.label:>11}: no PCC>={pcc_target} config")

    must_beat_str = f"{must_beat_us:.2f}us" if must_beat_us is not None else "none (no baseline)"
    if best is not None:
        speedup = f" | speedup={must_beat_us / best.result.dev_us:.2f}x" if must_beat_us is not None else ""
        logger.info(
            f"FASTEST PCC-PASS: [{best.strategy.label}] {best.result.case.label} -> "
            f"{best.result.dev_us:.2f}us, {best.result.tflops:.2f} TFLOPs, "
            f"PCC={best.result.pcc:.4f} | must-beat={must_beat_str}{speedup}"
        )
        if best_no_reshard is not None:
            logger.info(
                f"FASTEST in0-INTERLEAVED (loop-safe, no per-step reshard): "
                f"[{best_no_reshard.strategy.label}] {best_no_reshard.result.case.label} -> "
                f"{best_no_reshard.result.dev_us:.2f}us, PCC={best_no_reshard.result.pcc:.4f}"
            )
        if best.result.case.in0 in ("ws", "hs", "bs"):
            logger.warning(
                "Fastest config sharded in0 — usable only if the tensor is ALREADY sharded. In the "
                "LSTM loop the InterleavedToSharded/ShardedToInterleaved reshards (+2 host ops/step) "
                "outweigh the device-time win on this host-bound model. Prefer the in0-interleaved row."
            )
    else:
        logger.info(f"FASTEST PCC-PASS: none (no config reached PCC>={pcc_target}) | must-beat={must_beat_str}")

    assert passing, f"No config reached PCC>={pcc_target} for {shape.sweep_id} — sweep harness or shape is broken."
    if must_beat_us is not None and not winners:
        logger.warning(
            f"No swept config beat the report baseline {must_beat_us:.2f}us for {shape.sweep_id}; "
            f"fastest PCC-pass = {best.result.dev_us:.2f}us ([{best.strategy.label}] {best.result.case.label}). "
            f"The production config may already be optimal, or the baseline includes layout/reshard "
            f"overhead not modeled in this isolated matmul."
        )
    if os.getenv("MATMUL_STRICT_BASELINE"):
        assert winners, (
            f"No config both passed PCC>={pcc_target} and beat the report time {must_beat_us:.2f}us for "
            f"{shape.sweep_id} (fastest PCC-pass = {best.result.dev_us:.2f}us)."
        )
