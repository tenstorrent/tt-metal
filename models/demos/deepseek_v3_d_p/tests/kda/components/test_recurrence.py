# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole contract tests for the KDA recurrence component."""

import time
from typing import Literal

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    collect_mesh_accuracy_and_determinism_results,
    compare_cpu_device,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tt.kda import recurrence
from models.demos.deepseek_v3_d_p.tt.kda.config import KDARecurrenceProgramConfig
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    collect_accuracy_and_determinism_results,
)

pytestmark = run_for_blackhole()


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
        pytest.param(32, 2, 32, 32, 8, "direct", id="direct-minimal"),
        pytest.param(64, 2, 32, 128, 8, "direct", id="direct-nonsquare-state"),
        pytest.param(256, 2, 32, 32, 2, "grouped", id="grouped-minimal"),
        pytest.param(2816, 12, 128, 128, 21, "grouped", id="grouped-divisor-fallback"),
        pytest.param(5120, 1, 128, 128, 8, "grouped", id="grouped-production-length"),
        pytest.param(5152, 2, 32, 32, 8, "grouped", id="grouped-tail-chunk"),
        pytest.param(5152, 12, 128, 128, 20, "direct", id="direct-long-sequence"),
    ],
)
@pytest.mark.use_module_device
def test_recurrence_matches_reference_and_is_deterministic(
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

    (
        (output_tt, final_state_tt),
        (actual_output, actual_state),
        mismatch_marker,
    ) = collect_accuracy_and_determinism_results(device, run)
    assert mismatch_marker.item() == 0, "recurrence results are not bit-identical across runs"

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


@pytest.mark.use_module_device
def test_grouped_scan_preserves_weak_decay_across_group_sizes(device: ttnn.Device) -> None:
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
            local_scan_strategy="grouped",
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
            local_scan_strategy="grouped",
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


def _to_mesh(
    tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    mesh_dims: tuple[int | None, int | None],
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=mesh_dims, mesh_shape=tuple(mesh_device.shape)),
    )


def _reconstruct_recurrence_output(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
) -> torch.Tensor:
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    rows, columns = tuple(mesh_device.shape)
    sp_size = (rows, columns)[sp_axis]
    tp_size = (rows, columns)[tp_axis]
    sequence_partitions = []
    for sp_rank in range(sp_size):
        head_partitions = []
        for tp_rank in range(tp_size):
            row, column = (sp_rank, tp_rank) if sp_axis == 0 else (tp_rank, sp_rank)
            head_partitions.append(shards[row * columns + column])
        sequence_partitions.append(torch.cat(head_partitions, dim=0).transpose(0, 1))
    return torch.cat(sequence_partitions, dim=0).unsqueeze(0)


def _distributed_recurrence_case(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> tuple[
    recurrence.KDARecurrence,
    tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
    torch.Tensor,
    torch.Tensor,
    int,
]:
    sp_axis = 1 - tensor_parallel_axis
    sequence, heads, dim = 128, 8, 32
    generator = torch.Generator().manual_seed(1621 + tensor_parallel_axis)
    shape = (1, sequence, heads)
    q = torch.randn(*shape, dim, generator=generator)
    k = torch.randn(*shape, dim, generator=generator)
    v = torch.randn(*shape, dim, generator=generator)
    gate = -0.02 * torch.rand(*shape, dim, generator=generator)
    beta = torch.sigmoid(torch.randn(*shape, generator=generator))
    initial_state = 0.02 * torch.randn(1, heads, dim, dim, generator=generator)
    expected_output, expected_state = kda_recurrent_reference(q, k, v, gate, beta, initial_state)

    activation_dims = [None, None]
    activation_dims[sp_axis] = 1
    activation_dims[tensor_parallel_axis] = 2
    state_dims = [None, None]
    state_dims[tensor_parallel_axis] = 1
    inputs = (
        _to_mesh(q.reshape(1, sequence, heads * dim), mesh_device, tuple(activation_dims), ttnn.bfloat16),
        _to_mesh(k.reshape(1, sequence, heads * dim), mesh_device, tuple(activation_dims), ttnn.bfloat16),
        _to_mesh(v.reshape(1, sequence, heads * dim), mesh_device, tuple(activation_dims), ttnn.bfloat16),
        _to_mesh(gate.reshape(1, sequence, heads * dim), mesh_device, tuple(activation_dims), ttnn.bfloat16),
        _to_mesh(beta, mesh_device, tuple(activation_dims), ttnn.float32),
        _to_mesh(initial_state, mesh_device, tuple(state_dims), ttnn.float32),
    )
    executor = recurrence.KDARecurrence(
        mesh_device,
        KDARecurrenceProgramConfig(summary_group_chunks=8),
        sequence_parallel_axis=sp_axis,
    )
    return executor, inputs, expected_output.to(torch.bfloat16), expected_state, sp_axis


def _run_distributed_recurrence(
    executor: recurrence.KDARecurrence,
    inputs: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    q, k, v, gate, beta, initial_state = inputs
    with ttnn.manage_config("throw_exception_on_fallback", True):
        new_state, output = executor(q=q, k=k, v=v, gate=gate, beta=beta, initial_state=initial_state)
    return output, new_state


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_distributed_recurrence_matches_serial_and_is_deterministic(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    executor, inputs, expected_output, expected_state, sp_axis = _distributed_recurrence_case(
        mesh_device, tensor_parallel_axis
    )

    (output_tt, state_tt), mismatch_markers = collect_mesh_accuracy_and_determinism_results(
        lambda: _run_distributed_recurrence(executor, inputs)
    )
    ttnn.synchronize_device(mesh_device)
    cache_entries = mesh_device.num_program_cache_entries()
    repeated_output, repeated_state = _run_distributed_recurrence(executor, inputs)
    ttnn.synchronize_device(mesh_device)
    assert mesh_device.num_program_cache_entries() == cache_entries
    ttnn.deallocate(repeated_output)
    ttnn.deallocate(repeated_state)
    assert output_tt.dtype == ttnn.bfloat16
    assert state_tt.dtype == ttnn.float32
    assert all(marker.item() == 0 for marker in mismatch_markers), "distributed recurrence is not bit-identical"

    actual_output = _reconstruct_recurrence_output(output_tt, mesh_device, sp_axis, tensor_parallel_axis)
    actual_state = reconstruct_state_at_sp_rank(state_tt, mesh_device, sp_axis, tensor_parallel_axis, sp_rank=0)
    label = f"tp_axis={tensor_parallel_axis}"
    assert_accurate(expected_output, actual_output, name=f"{label} output")
    assert_accurate(expected_state, actual_state, name=f"{label} state")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_distributed_recurrence_trace_replay_matches_eager(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    executor, inputs, _, _, sp_axis = _distributed_recurrence_case(mesh_device, tensor_parallel_axis)
    eager_output_tt, eager_state_tt = _run_distributed_recurrence(executor, inputs)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output_tt, traced_state_tt = _run_distributed_recurrence(executor, inputs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    eager_output = _reconstruct_recurrence_output(eager_output_tt, mesh_device, sp_axis, tensor_parallel_axis)
    traced_output = _reconstruct_recurrence_output(traced_output_tt, mesh_device, sp_axis, tensor_parallel_axis)
    eager_state = reconstruct_state_at_sp_rank(eager_state_tt, mesh_device, sp_axis, tensor_parallel_axis, sp_rank=0)
    traced_state = reconstruct_state_at_sp_rank(traced_state_tt, mesh_device, sp_axis, tensor_parallel_axis, sp_rank=0)
    ttnn.release_trace(mesh_device, trace_id)

    assert_bit_identical(eager_output, traced_output, name=f"tp_axis={tensor_parallel_axis} traced output")
    assert_bit_identical(eager_state, traced_state, name=f"tp_axis={tensor_parallel_axis} traced state")
