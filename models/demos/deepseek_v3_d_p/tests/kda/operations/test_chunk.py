# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole PCC tests for the chunk-parallel KDA operation."""

import time
from typing import Literal

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import compare_cpu_device
from models.demos.deepseek_v3_d_p.tt.kda import recurrence
from models.demos.deepseek_v3_d_p.tt.kda.config import KDARecurrenceProgramConfig
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    collect_accuracy_and_determinism_results,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device,
]


def _to_device(tensor: torch.Tensor, device: ttnn.Device, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_recurrence(
    device: ttnn.Device,
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    state: ttnn.Tensor,
    *,
    summary_group_chunks: int = 8,
    local_scan_strategy: Literal["direct", "grouped"] = "direct",
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    executor = recurrence.KDARecurrence(
        device,
        KDARecurrenceProgramConfig(
            summary_group_chunks=summary_group_chunks,
            local_scan_strategy=local_scan_strategy,
        ),
        sequence_parallel_axis=None,
    )
    return executor(q=q, k=k, v=v, gate=gate, beta=beta, initial_state=state)


@pytest.mark.parametrize(
    "sequence,heads,key_dim,value_dim,summary_group_chunks,local_scan_strategy",
    [
        (32, 2, 32, 32, 8, "direct"),
        (64, 32, 128, 128, 8, "direct"),
        (256, 4, 128, 128, 8, "direct"),
        (256, 2, 32, 32, 2, "grouped"),
        (512, 4, 128, 128, 8, "direct"),
        (2816, 12, 128, 128, 21, "grouped"),
        (5120, 2, 32, 32, 8, "grouped"),
        (5120, 1, 128, 128, 8, "grouped"),
        (5152, 2, 32, 32, 8, "grouped"),
        (5152, 12, 128, 128, 20, "direct"),
    ],
)
def test_chunk_recurrence_pcc(
    device: ttnn.Device,
    sequence: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    summary_group_chunks: int,
    local_scan_strategy: Literal["direct", "grouped"],
) -> None:
    generator = torch.Generator().manual_seed(401 + sequence + heads)
    shape = (1, sequence, heads)
    q = torch.randn(*shape, key_dim, generator=generator)
    k = torch.randn(*shape, key_dim, generator=generator)
    v = torch.randn(*shape, value_dim, generator=generator)
    gate = -0.02 * torch.rand(*shape, key_dim, generator=generator)
    beta = torch.sigmoid(torch.randn(*shape, generator=generator))
    state = 0.02 * torch.randn(1, heads, key_dim, value_dim, generator=generator)
    print("KDA_CPU_REFERENCE_CACHE=disabled; computing deterministic operation oracle", flush=True)
    reference_start = time.perf_counter()
    golden_output, golden_state = kda_recurrent_reference(q, k, v, gate, beta, state)
    golden_output = golden_output.to(torch.bfloat16)
    print(f"KDA_CPU_REFERENCE_SECONDS={time.perf_counter() - reference_start:.3f}", flush=True)

    q_tt = _to_device(q.reshape(1, sequence, heads * key_dim), device, ttnn.bfloat16)
    k_tt = _to_device(k.reshape(1, sequence, heads * key_dim), device, ttnn.bfloat16)
    v_tt = _to_device(v.reshape(1, sequence, heads * value_dim), device, ttnn.bfloat16)
    gate_tt = _to_device(gate.reshape(1, sequence, heads * key_dim), device, ttnn.bfloat16)
    beta_tt = _to_device(beta, device, ttnn.float32)
    state_tt = _to_device(state, device, ttnn.float32)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            final_state, output = _run_recurrence(
                device,
                q_tt,
                k_tt,
                v_tt,
                gate_tt,
                beta_tt,
                state_tt,
                summary_group_chunks=summary_group_chunks,
                local_scan_strategy=local_scan_strategy,
            )
        return output, final_state

    checks_determinism = (sequence, heads, key_dim, value_dim, summary_group_chunks, local_scan_strategy) == (
        256,
        2,
        32,
        32,
        2,
        "grouped",
    )
    if checks_determinism:
        (
            (output_tt, final_state_tt),
            (actual_output, actual_state),
            mismatch_marker,
        ) = collect_accuracy_and_determinism_results(device, run)
        assert mismatch_marker.item() == 0, "chunk recurrence output is not bit-identical across runs"
    else:
        output_tt, final_state_tt = run()
        actual_output = ttnn.to_torch(output_tt)
        actual_state = ttnn.to_torch(final_state_tt)

    assert final_state_tt.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    actual_output = actual_output.reshape(1, heads, sequence, value_dim).permute(0, 2, 1, 3)
    label = f"H={heads},K={key_dim},V={value_dim},T={sequence},group={summary_group_chunks}"
    _, output_failures = compare_cpu_device(
        f"{label} output",
        golden_output,
        actual_output,
        pcc_threshold=0.999,
    )
    _, state_failures = compare_cpu_device(
        f"{label} state",
        golden_state,
        actual_state,
        pcc_threshold=0.999,
    )
    failures = output_failures + state_failures
    assert not failures, "\n".join(failures)


def test_grouped_summary_preserves_weak_decay(device: ttnn.Device) -> None:
    sequence, heads, dim = 5120, 1, 32
    generator = torch.Generator().manual_seed(9401)
    shape = (1, sequence, heads, dim)
    q = torch.randn(*shape, generator=generator)
    k = torch.randn(*shape, generator=generator)
    v = torch.randn(*shape, generator=generator)
    gate = torch.full(shape, -1e-5)
    beta = torch.sigmoid(torch.randn(1, sequence, heads, generator=generator))
    state = 0.02 * torch.randn(1, heads, dim, dim, generator=generator)
    golden_output, golden_state = kda_recurrent_reference(q, k, v, gate, beta, state)
    golden_output = golden_output.to(torch.bfloat16)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        state_eight, output_eight_tt = _run_recurrence(
            device,
            _to_device(q.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(k.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(v.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(gate.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(beta, device, ttnn.float32),
            _to_device(state, device, ttnn.float32),
            summary_group_chunks=8,
        )

    ttnn.synchronize_device(device)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        state_two, output_two_tt = _run_recurrence(
            device,
            _to_device(q.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(k.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(v.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(gate.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(beta, device, ttnn.float32),
            _to_device(state, device, ttnn.float32),
            summary_group_chunks=2,
        )

    output_eight = ttnn.to_torch(output_eight_tt).reshape(1, sequence, heads, dim)
    output_two = ttnn.to_torch(output_two_tt).reshape(1, sequence, heads, dim)
    label = "weak-decay production contract"
    assert_accurate(golden_output, output_eight, name=f"{label} grouped output")
    assert_accurate(
        output_two,
        output_eight,
        name=f"{label} group-size output invariance",
        pcc_threshold=0.9999,
    )
    assert_accurate(
        ttnn.to_torch(state_two),
        ttnn.to_torch(state_eight),
        name=f"{label} group-size state invariance",
        pcc_threshold=0.9999,
    )
    assert_accurate(golden_state, ttnn.to_torch(state_eight), name=f"{label} grouped state")
