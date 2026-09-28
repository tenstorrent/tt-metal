# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device unit test for the Kimi-K3 KDA projection matmuls.

Shapes are the per-device slices of the production SP8xTP4 Galaxy layer at T=5120:
local T=640, hidden=7168, TP-local fused input-projection width 12440, TP-local
output-projection K=3072. Baseline and tuned schedules run with the production numerics,
are timed as traced replay loops, and are compared against an FP32 torch reference. The
latency bounds are calibrated for a single Galaxy Blackhole device.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config

pytestmark = [run_for_blackhole(), pytest.mark.perf]

_ROWS = 640
_HIDDEN = 7168
_INPUT_PROJECTION_WIDTH = 12440
_OUTPUT_PROJECTION_K = 3072
_REPLAYS = 20
_PCC_THRESHOLD = 0.99995
# Galaxy single-device traced wall time, 2026-09-28: baseline 1122/276 us, tuned 846/191 us.
_MAX_TUNED_US = {"input": 880.0, "output": 210.0}


def _compute_config(device, fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _traced_us(device, op: Callable[[], ttnn.Tensor]) -> tuple[float, ttnn.Tensor]:
    """Return the mean traced per-call wall time and one output."""
    ttnn.deallocate(op())
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = op()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(_REPLAYS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed_us = (time.perf_counter() - start) * 1e6 / _REPLAYS
    ttnn.release_trace(device, trace_id)
    return elapsed_us, output


def _linear(activation, weight, fidelity, program_config=None):
    compute_config = _compute_config(activation.device(), fidelity)
    return lambda: ttnn.linear(
        activation,
        weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        compute_kernel_config=compute_config,
    )


def _minimal(activation, weight, fidelity, config):
    compute_config = _compute_config(activation.device(), fidelity)
    return lambda: ttnn.experimental.minimal_matmul(
        activation,
        weight,
        compute_kernel_config=compute_config,
        config=config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
@pytest.mark.parametrize("projection", ["input", "output"])
def test_kda_projection_matmul_perf(device, projection) -> None:
    production = kimi_k3_program_config(active_seq_len_local=_ROWS, tp_ccl_topology=ttnn.Topology.Linear)
    if projection == "input":
        k, n = _HIDDEN, _INPUT_PROJECTION_WIDTH
        fidelity = ttnn.MathFidelity.HiFi4  # ttKDA.compute_config
        tuned_config = production.input_projection_minimal_matmul_config
    else:
        k, n = _OUTPUT_PROJECTION_K, _HIDDEN
        fidelity = production.output_projection_math_fidelity
        tuned_config = production.output_projection_program_config

    torch.manual_seed(0)
    activation = torch.randn(1, _ROWS, k)
    weight = torch.randn(k, n) / k**0.5
    golden = activation @ weight
    activation_tt = ttnn.from_torch(activation, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    weight_tt = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    assert tuned_config is not None
    tuned = _minimal if projection == "input" else _linear
    variants = {
        "baseline": _linear(activation_tt, weight_tt, fidelity),
        "tuned": tuned(activation_tt, weight_tt, fidelity, tuned_config),
    }
    results = {}
    for name, op in variants.items():
        elapsed_us, output = _traced_us(device, op)
        _, pcc = comp_pcc(golden, ttnn.to_torch(output).float())
        ttnn.deallocate(output)
        results[name] = (elapsed_us, pcc)
        logger.info(
            f"{projection} projection M={_ROWS} K={k} N={n} {fidelity.name} {name:8s} {elapsed_us:8.1f} us  PCC {pcc:.6f}"
        )

    for name, (_, pcc) in results.items():
        assert pcc >= _PCC_THRESHOLD, f"{name} PCC {pcc:.6f} < {_PCC_THRESHOLD}"
    tuned_us = results["tuned"][0]
    assert tuned_us <= _MAX_TUNED_US[projection], f"tuned {projection} projection {tuned_us:.1f} us regressed"
