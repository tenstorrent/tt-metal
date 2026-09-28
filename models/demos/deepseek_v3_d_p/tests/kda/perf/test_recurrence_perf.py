# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device unit test for the Kimi-K3 KDA recurrence ops at Galaxy SP8xTP4 shape.

Per device: 24 TP-local heads, 640 local tokens (20 chunks of 32), key and value dimension 128,
with the layer's compute configs (HiFi4 preparation and summary, HiFi2 scan, FP32 accumulation)
and preparation BF16 output mask. Each op is timed as a traced replay
loop. Set KDA_RECURRENCE_GOLDEN to a file prefix to save (first run) or bit-compare (later runs)
the outputs. The latency bounds are calibrated for a single Galaxy Blackhole device.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG,
    KDA_PREP_OUTPUT_BF16_MASK,
    KDA_RECURRENT_STATE_DTYPE,
    KDARecurrenceProgramConfig,
)
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import carry_matmul_program_config
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    device_protocol,
    group_summary_height_sharded,
    host_protocol,
    run_recurrent,
    run_summary,
    to_device,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf]

_HEADS = 24
_CHUNKS = 20
_DIM = 128
_BF16 = frozenset({"kd", "q_decay", "final_decay"})
_REPLAYS = 20
# Galaxy single-device traced wall time, 2026-09-28: summary 242 us before splitting value columns
# across cores, 229 us (227 us sharded) after; scan 299 us.
_MAX_US = {"prepare": 1000.0, "summary": 240.0, "summary_sharded": 240.0, "scan": 1000.0}
# Distributed-prefix carry matmul, 2026-09-28: 30.1 us with the default program, 16.3 us tuned.
_MAX_CARRY_MATMUL_US = 20.0


def _compute_config(device, fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _traced_us(device, op) -> tuple[float, list[ttnn.Tensor]]:
    for output in op():
        ttnn.deallocate(output)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = op()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(_REPLAYS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed_us = (time.perf_counter() - start) * 1e6 / _REPLAYS
    ttnn.release_trace(device, trace_id)
    return elapsed_us, outputs


def _prepare_inputs(device) -> tuple[ttnn.Tensor, ...]:
    generator = torch.Generator().manual_seed(1731)
    rows = _CHUNKS * 32
    q = 0.3 * torch.randn(1, rows, _HEADS * _DIM, generator=generator)
    k = 0.3 * torch.randn(1, rows, _HEADS * _DIM, generator=generator)
    v = 0.2 * torch.randn(1, rows, _HEADS * _DIM, generator=generator)
    gate = -0.001 - 0.05 * torch.rand(1, rows, _HEADS * _DIM, generator=generator)
    beta = torch.sigmoid(torch.randn(_HEADS, _CHUNKS, 32, 1, generator=generator))
    return tuple(
        ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        for tensor, dtype in (
            (q, ttnn.bfloat16),
            (k, ttnn.bfloat16),
            (v, ttnn.bfloat16),
            (gate, ttnn.bfloat16),
            (beta, ttnn.float32),
        )
    )


def _check_golden(name: str, outputs: list[torch.Tensor]) -> None:
    prefix = os.environ.get("KDA_RECURRENCE_GOLDEN")
    if not prefix:
        return
    path = Path(f"{prefix}.{name}.pt")
    if path.exists():
        for index, (expected, output) in enumerate(zip(torch.load(path), outputs)):
            assert torch.equal(expected, output), f"{name} output {index} differs from saved outputs"
        logger.info(f"{name} outputs bit-identical to {path}")
    else:
        torch.save(outputs, path)
        logger.info(f"saved {name} outputs to {path}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
@pytest.mark.parametrize("op_name", ["prepare", "summary", "summary_sharded", "scan"])
def test_kda_recurrence_perf(device, op_name) -> None:
    host_inputs = host_protocol(_HEADS, _CHUNKS, _DIM, _DIM, bf16_names=_BF16, seed=117)
    inputs = device_protocol(host_inputs, device)
    actual_start = make_actual_start(device, 0)
    compute_config = _compute_config(device)
    if op_name == "prepare":
        prepare_inputs = _prepare_inputs(device)

        def op():
            return ttnn.experimental.kda.prepare_chunk_recurrence(
                *prepare_inputs,
                _HEADS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_config,
                output_bf16_mask=KDA_PREP_OUTPUT_BF16_MASK,
                actual_start=actual_start,
            )

    elif op_name.startswith("summary"):
        # The layer height-shards summaries one head per core in L1.
        memory_config = group_summary_height_sharded(device, _HEADS, _DIM) if op_name == "summary_sharded" else None

        def op():
            return run_summary(
                inputs, actual_start=actual_start, compute_kernel_config=compute_config, memory_config=memory_config
            )

    else:
        generator = torch.Generator().manual_seed(118)
        state = to_device(0.05 * torch.randn(_HEADS, _DIM, _DIM, generator=generator), device)
        scan_config = _compute_config(device, KDARecurrenceProgramConfig().scan_math_fidelity)

        def op():
            return run_recurrent(inputs, state, actual_start=actual_start, compute_kernel_config=scan_config)

    elapsed_us, outputs = _traced_us(device, op)
    logger.info(f"KDA recurrence {op_name} heads={_HEADS} chunks={_CHUNKS} dim={_DIM}: {elapsed_us:.1f} us")
    _check_golden(op_name.removesuffix("_sharded"), [ttnn.to_torch(output) for output in outputs])
    assert elapsed_us <= _MAX_US[op_name], f"{op_name} {elapsed_us:.1f} us regressed past {_MAX_US[op_name]} us"


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_kda_prefix_carry_matmul_perf(device) -> None:
    """The tuned distributed-prefix matmul must stay bit-identical to the default program."""
    generator = torch.Generator().manual_seed(119)
    memory_config = KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG
    a, carry = (
        ttnn.from_torch(
            0.1 * torch.randn(1, _HEADS, _DIM, _DIM, generator=generator),
            dtype=KDA_RECURRENT_STATE_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        for _ in range(2)
    )
    compute_config = _compute_config(device, KDARecurrenceProgramConfig().affine_prefix_math_fidelity)
    program_config = carry_matmul_program_config(device, _HEADS, _DIM, _DIM)
    assert program_config is not None

    def matmul(config):
        return lambda: [
            ttnn.matmul(
                a,
                carry,
                memory_config=memory_config,
                dtype=KDA_RECURRENT_STATE_DTYPE,
                program_config=config,
                compute_kernel_config=compute_config,
            )
        ]

    default_us, (default,) = _traced_us(device, matmul(None))
    tuned_us, (tuned,) = _traced_us(device, matmul(program_config))
    logger.info(f"KDA prefix carry matmul: default {default_us:.1f} us, tuned {tuned_us:.1f} us")
    assert torch.equal(ttnn.to_torch(default), ttnn.to_torch(tuned)), "tuned carry matmul is not bit-identical"
    assert tuned_us <= _MAX_CARRY_MATMUL_US, f"carry matmul {tuned_us:.1f} us regressed"
