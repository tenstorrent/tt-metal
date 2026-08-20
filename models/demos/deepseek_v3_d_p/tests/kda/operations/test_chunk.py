# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole PCC tests for the chunk-parallel KDA operation."""

import time

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_accurate, assert_bit_identical, compare_cpu_device
from models.demos.deepseek_v3_d_p.tt.kda import ops
from models.demos.deepseek_v3_d_p.tt.kda.config import KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.const_tiles import build_kda_const_tiles

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def test_chunk_scan_strategy_policy(device: ttnn.Device) -> None:
    compute_config = ops._RecurrenceComputeConfig(None, None, None)
    cases = (
        (159, None, ops._DirectScan),
        (160, None, ops._LocalGroupedScan),
        (161, None, ops._DirectScan),
        (8, 0, ops._DistributedGroupedScan),
        (9, 0, ops._DistributedGroupedScan),
    )
    for num_chunks, sp_axis, expected in cases:
        actual = ops._select_scan(
            num_chunks=num_chunks,
            program_config=KDARecurrenceProgramConfig(summary_group_chunks=8),
            compute_config=compute_config,
            sequence_parallel_axis=sp_axis,
        )
        assert isinstance(actual, expected)


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
) -> ops._ScanResult:
    prefix_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
    )
    return ops._chunk_recurrence(
        ops._FlatRecurrenceInputs(q=q, k=k, v=v, gate=gate, beta=beta),
        state,
        ops._ChunkConstants(*build_kda_const_tiles(device)),
        program_config=KDARecurrenceProgramConfig(summary_group_chunks=summary_group_chunks),
        compute_config=ops._RecurrenceComputeConfig(
            preparation=None,
            affine_prefix=prefix_compute_config,
            grouped_scan=prefix_compute_config,
        ),
        sequence_parallel_axis=None,
    )


def test_chunk_recurrence_rejects_nonproduction_contract(device: ttnn.Device, expect_error) -> None:
    sequence, heads, dim = 32, 2, 32
    flat_shape = (1, sequence, heads * dim)
    beta_shape = (1, sequence, heads)
    state_shape = (1, heads, dim, dim)
    valid_inputs = {
        "q": _to_device(torch.randn(flat_shape), device, ttnn.bfloat16),
        "k": _to_device(torch.randn(flat_shape), device, ttnn.bfloat16),
        "v": _to_device(torch.randn(flat_shape), device, ttnn.bfloat16),
        "gate": _to_device(torch.randn(flat_shape), device, ttnn.bfloat16),
        "beta": _to_device(torch.randn(beta_shape), device, ttnn.float32),
    }
    valid_state = _to_device(torch.randn(state_shape), device, ttnn.float32)
    constants = ops._ChunkConstants(*build_kda_const_tiles(device))
    program_config = KDARecurrenceProgramConfig()
    compute_config = ops._RecurrenceComputeConfig(None, None, None)

    def run(
        inputs: dict[str, ttnn.Tensor],
        state: ttnn.Tensor = valid_state,
        chunk_constants: ops._ChunkConstants = constants,
    ) -> None:
        ops._chunk_recurrence(
            ops._FlatRecurrenceInputs(**inputs),
            state,
            chunk_constants,
            program_config=program_config,
            compute_config=compute_config,
            sequence_parallel_axis=None,
        )

    wrong_dtype_inputs = {
        "q": _to_device(torch.randn(flat_shape), device, ttnn.float32),
        "k": _to_device(torch.randn(flat_shape), device, ttnn.float32),
        "v": _to_device(torch.randn(flat_shape), device, ttnn.float32),
        "gate": _to_device(torch.randn(flat_shape), device, ttnn.float32),
        "beta": _to_device(torch.randn(beta_shape), device, ttnn.bfloat16),
    }
    for name, tensor in wrong_dtype_inputs.items():
        with expect_error(AssertionError, "^$"):
            run({**valid_inputs, name: tensor})

    with expect_error(AssertionError, "^$"):
        run(valid_inputs, _to_device(torch.randn(state_shape), device, ttnn.bfloat16))

    l1_state = ttnn.to_memory_config(valid_state, ttnn.L1_MEMORY_CONFIG)
    with expect_error(AssertionError, "^$"):
        run(valid_inputs, l1_state)

    rank_four_q = _to_device(torch.randn(1, 1, sequence, heads * dim), device, ttnn.bfloat16)
    with expect_error(ValueError, "flat rank-3"):
        run({**valid_inputs, "q": rank_four_q})

    invalid_constants = ops._ChunkConstants(
        constants.eye,
        constants.tril,
        _to_device(torch.ones(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE), device, ttnn.bfloat16),
    )
    with expect_error(AssertionError, "^$"):
        run(valid_inputs, chunk_constants=invalid_constants)


@pytest.mark.parametrize(
    "sequence,heads,key_dim,value_dim,summary_group_chunks",
    [
        (32, 2, 32, 32, 8),
        (64, 32, 128, 128, 8),
        (256, 4, 128, 128, 8),
        (512, 4, 128, 128, 8),
        (2816, 12, 128, 128, 21),
        (5120, 2, 32, 32, 8),
        (5120, 1, 128, 128, 8),
        (5152, 2, 32, 32, 8),
    ],
)
def test_chunk_recurrence_pcc(
    device: ttnn.Device,
    sequence: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    summary_group_chunks: int,
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
    print(f"KDA_CPU_REFERENCE_SECONDS={time.perf_counter() - reference_start:.3f}", flush=True)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        result = _run_recurrence(
            device,
            _to_device(q.reshape(1, sequence, heads * key_dim), device, ttnn.bfloat16),
            _to_device(k.reshape(1, sequence, heads * key_dim), device, ttnn.bfloat16),
            _to_device(v.reshape(1, sequence, heads * value_dim), device, ttnn.bfloat16),
            _to_device(gate.reshape(1, sequence, heads * key_dim), device, ttnn.bfloat16),
            _to_device(beta, device, ttnn.float32),
            _to_device(state, device, ttnn.float32),
            summary_group_chunks=summary_group_chunks,
        )

    assert result.final_state.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    actual_output = ttnn.to_torch(result.output)
    actual_output = actual_output.reshape(1, heads, sequence, value_dim).permute(0, 2, 1, 3)
    actual_state = ttnn.to_torch(result.final_state)
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

    with ttnn.manage_config("throw_exception_on_fallback", True):
        result_eight = _run_recurrence(
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
        result_two = _run_recurrence(
            device,
            _to_device(q.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(k.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(v.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(gate.reshape(1, sequence, heads * dim), device, ttnn.bfloat16),
            _to_device(beta, device, ttnn.float32),
            _to_device(state, device, ttnn.float32),
            summary_group_chunks=2,
        )

    output_eight = ttnn.to_torch(result_eight.output).reshape(1, sequence, heads, dim)
    output_two = ttnn.to_torch(result_two.output).reshape(1, sequence, heads, dim)
    label = "weak-decay production contract"
    assert_accurate(golden_output, output_eight, name=f"{label} grouped output")
    assert_accurate(
        output_two,
        output_eight,
        name=f"{label} group-size output invariance",
        pcc_threshold=0.9999,
    )
    assert_accurate(
        ttnn.to_torch(result_two.final_state),
        ttnn.to_torch(result_eight.final_state),
        name=f"{label} group-size state invariance",
        pcc_threshold=0.9999,
    )
    assert_accurate(golden_state, ttnn.to_torch(result_eight.final_state), name=f"{label} grouped state")


def test_chunk_recurrence_determinism(device: ttnn.Device) -> None:
    sequence, heads, dim = 64, 2, 32
    generator = torch.Generator().manual_seed(1401)
    shape = (1, sequence, heads)
    flat_shape = (1, sequence, heads * dim)
    q_tt = _to_device(torch.randn(*shape, dim, generator=generator).reshape(flat_shape), device, ttnn.bfloat16)
    k_tt = _to_device(torch.randn(*shape, dim, generator=generator).reshape(flat_shape), device, ttnn.bfloat16)
    v_tt = _to_device(torch.randn(*shape, dim, generator=generator).reshape(flat_shape), device, ttnn.bfloat16)
    gate_tt = _to_device(
        (-0.02 * torch.rand(*shape, dim, generator=generator)).reshape(flat_shape), device, ttnn.bfloat16
    )
    beta_tt = _to_device(torch.sigmoid(torch.randn(*shape, generator=generator)), device, ttnn.float32)
    state_tt = _to_device(0.02 * torch.randn(1, heads, dim, dim, generator=generator), device, ttnn.float32)

    results = []
    for _ in range(3):
        with ttnn.manage_config("throw_exception_on_fallback", True):
            result = _run_recurrence(device, q_tt, k_tt, v_tt, gate_tt, beta_tt, state_tt, summary_group_chunks=2)
        ttnn.synchronize_device(device)
        results.append((ttnn.to_torch(result.output), ttnn.to_torch(result.final_state)))

    first_output, first_state = results[0]
    for iteration, (output, final_state) in enumerate(results[1:], start=1):
        assert_bit_identical(first_output, output, name=f"chunk recurrence output iteration {iteration}")
        assert_bit_identical(first_state, final_state, name=f"chunk recurrence state iteration {iteration}")
