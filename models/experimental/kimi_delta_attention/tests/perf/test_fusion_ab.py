# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight Kimi-K3 fused-versus-composite PCC and performance comparisons."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.checkpoint import load_kda_layer_state_dict
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.kimi_k3_config import (
    KimiK3Config,
    kimi_k3_kda_config,
    kimi_k3_program_config,
)
from models.experimental.kimi_delta_attention.tests.utils import assert_all_finite
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention, _slice_width
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 256 * 1024 * 1024,
            }
        ],
        indirect=True,
    ),
]

_SEQUENCE = 5120
_REPETITIONS = 10
_PCC_THRESHOLD = 0.98


@dataclass(frozen=True)
class _OperationMeasurement:
    output: torch.Tensor
    wall_ms: float
    program_sum_us: float
    program_count: int


class _UnfusedConvolutionKimiDeltaAttention(KimiDeltaAttention):
    """Test-local unfused convolution alternative."""

    def _convolve_qkv(
        self,
        qkv: ttnn.Tensor,
        sequence: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        assert self.convolution_state is not None
        config = self.config
        channels = self._convolution_width
        qkv_row_major = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        state_row_major = ttnn.to_layout(
            self.convolution_state,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.sequence_parallel_size > 1:
            state_row_major, new_state = ttnn.transformer.kda_convolution_halo(
                qkv_row_major,
                state_row_major,
                sequence_parallel_axis=self.sequence_parallel_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            new_state = ttnn.slice(
                qkv_row_major,
                (0, sequence - (config.conv_kernel_size - 1), 0),
                (1, sequence, channels),
            )
        if self.convolution_state.layout != ttnn.ROW_MAJOR_LAYOUT:
            new_state = ttnn.to_layout(
                new_state,
                self.convolution_state.layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        conv_input = ttnn.concat(
            [state_row_major, qkv_row_major],
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conv_input = ttnn.to_layout(conv_input, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        terms = []
        for tap, weight in enumerate(self.weights.convolution_taps):
            shifted = ttnn.slice(
                conv_input,
                (0, tap, 0),
                (1, tap + sequence, channels),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            terms.append(ttnn.multiply(shifted, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        output = terms[0]
        for term in terms[1:]:
            output = ttnn.add(output, term, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.silu(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = _slice_width(output, 0, config.q_dim)
        k = _slice_width(output, config.q_dim, config.q_dim + config.k_dim)
        v = _slice_width(output, config.q_dim + config.k_dim, channels)
        return q, k, v, new_state


def _input_tensor(hidden: torch.Tensor, mesh_device: ttnn.MeshDevice, sequence_parallel_axis: int) -> ttnn.Tensor:
    if tuple(mesh_device.shape)[sequence_parallel_axis] == 1:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        mesh_dims: list[int | None] = [None, None]
        mesh_dims[sequence_parallel_axis] = 1
        mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=tuple(mesh_dims),
            mesh_shape=tuple(mesh_device.shape),
        )
    return ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _flatten_shards(tensor: ttnn.Tensor) -> torch.Tensor:
    return torch.cat([ttnn.to_torch(shard).float().reshape(-1) for shard in ttnn.get_device_tensors(tensor)])


def _collapse_records(records: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    per_program: dict[int, dict[str, object]] = {}
    for record in records:
        runtime_id = int(record["runtime_id"])
        if runtime_id == 0:
            continue
        duration_ns = float(record["duration_ns"])
        current = per_program.get(runtime_id)
        if current is None or duration_ns > float(current["duration_ns"]):
            per_program[runtime_id] = {
                "runtime_id": runtime_id,
                "duration_ns": duration_ns,
                "kernel_sources": tuple(record["kernel_sources"]),
            }
    assert per_program, "realtime profiler returned no non-sentinel program records"
    return tuple(per_program.values())


def _measure_operation(
    mesh_device: ttnn.MeshDevice,
    operation: Callable[[], ttnn.Tensor],
    repetitions: int,
) -> _OperationMeasurement:
    warm_output = operation()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)

    output, records = profile_realtime_program(
        mesh_device,
        operation,
        collect_all=True,
        record_timeout_seconds=10.0,
    )
    host_output = _flatten_shards(output)
    ttnn.deallocate(output)
    programs = _collapse_records(records)

    warm_output = operation()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = operation()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(repetitions):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(trace_output)
    return _OperationMeasurement(
        output=host_output,
        wall_ms=elapsed * 1e3 / repetitions,
        program_sum_us=sum(float(program["duration_ns"]) for program in programs) / 1e3,
        program_count=len(programs),
    )


def _make_layer(
    mesh_device: ttnn.MeshDevice,
    config: KDAConfig,
    state_dict: Mapping[str, torch.Tensor],
    checkpoint_dir: Path,
    tensor_parallel_axis: int,
    unfused_convolution: bool,
) -> KimiDeltaAttention:
    kwargs = {
        "tensor_cache_path": checkpoint_dir / "ttnn_cache" / "layer_1",
        "tt_ccl": TT_CCL(mesh_device),
        "tensor_parallel_axis": tensor_parallel_axis,
        "program_config": kimi_k3_program_config(
            tp_ccl_topology=(
                ttnn.Topology.Ring if tuple(mesh_device.shape)[1 - tensor_parallel_axis] == 1 else ttnn.Topology.Linear
            )
        ),
    }
    if not unfused_convolution:
        return KimiDeltaAttention(mesh_device, config, state_dict, **kwargs)
    return _UnfusedConvolutionKimiDeltaAttention(
        mesh_device,
        config,
        state_dict,
        **kwargs,
    )


def _pcc(name: str, expected: torch.Tensor, actual: torch.Tensor) -> float:
    assert_all_finite(f"{name} fused device result", expected)
    assert_all_finite(f"{name} composite device result", actual)
    passed, value = comp_pcc(expected, actual, pcc=_PCC_THRESHOLD)
    print(f"{name}: PCC={value:.6f}")
    assert passed, f"{name} PCC {value:.6f} < {_PCC_THRESHOLD}"
    return value


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
def test_kimi_k3_convolution_ab(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_checkpoint_dir: Path,
) -> None:
    """Isolate the fused convolution; recurrence amplifies its BF16-rounding differences."""
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "realtime profiler must be active for KDA fusion A/B"
    checkpoint_dir = kimi_k3_checkpoint_dir
    config = kimi_k3_kda_config()
    state_dict = load_kda_layer_state_dict(checkpoint_dir, KimiK3Config.FIRST_KDA_LAYER, config)
    hidden = torch.randn(
        1,
        _SEQUENCE,
        config.hidden_size,
        generator=torch.Generator().manual_seed(1607),
        dtype=torch.bfloat16,
    )
    sequence_parallel_axis = 1 - tensor_parallel_axis
    hidden_tt = _input_tensor(hidden, mesh_device, sequence_parallel_axis)
    fused_layer = _make_layer(mesh_device, config, state_dict, checkpoint_dir, tensor_parallel_axis, False)
    composite_layer = _make_layer(mesh_device, config, state_dict, checkpoint_dir, tensor_parallel_axis, True)
    fused_layer.reset_state(batch_size=1)
    composite_layer.reset_state(batch_size=1)
    projected = fused_layer._project_inputs(hidden_tt)
    sequence = projected.qkv.shape[1]

    with ttnn.manage_config("throw_exception_on_fallback", True):
        fused_outputs = fused_layer._convolve_qkv(projected.qkv, sequence)
        composite_outputs = composite_layer._convolve_qkv(projected.qkv, sequence)
    pcc = {
        name: _pcc(name, _flatten_shards(expected), _flatten_shards(actual))
        for name, expected, actual in zip(
            ("q", "k", "v", "convolution_state"),
            fused_outputs,
            composite_outputs,
            strict=True,
        )
    }

    repetitions = int(os.getenv("KDA_FUSION_AB_REPS", str(_REPETITIONS)))
    with ttnn.manage_config("throw_exception_on_fallback", True):
        fused_measurement = _measure_operation(
            mesh_device,
            lambda: fused_layer._convolve_qkv(projected.qkv, sequence)[0],
            repetitions,
        )
        composite_measurement = _measure_operation(
            mesh_device,
            lambda: composite_layer._convolve_qkv(projected.qkv, sequence)[0],
            repetitions,
        )
    mesh_shape = tuple(mesh_device.shape)
    result = {
        "fusion": "convolution",
        "scope": "isolated_operation",
        "layout": f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}",
        "sequence": _SEQUENCE,
        "local_sequence": sequence,
        "repetitions": repetitions,
        "pcc": pcc,
        "fused": {
            "wall_ms": fused_measurement.wall_ms,
            "program_sum_us": fused_measurement.program_sum_us,
            "program_count": fused_measurement.program_count,
        },
        "composite": {
            "wall_ms": composite_measurement.wall_ms,
            "program_sum_us": composite_measurement.program_sum_us,
            "program_count": composite_measurement.program_count,
        },
    }
    result["fused_wall_gain_pct"] = (
        100.0 * (composite_measurement.wall_ms - fused_measurement.wall_ms) / composite_measurement.wall_ms
    )
    result["fused_program_sum_gain_pct"] = (
        100.0
        * (composite_measurement.program_sum_us - fused_measurement.program_sum_us)
        / composite_measurement.program_sum_us
    )
    print("KDA_FUSION_AB=" + json.dumps(result, sort_keys=True))


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
def test_kimi_k3_gated_rms_ab(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_checkpoint_dir: Path,
) -> None:
    """Isolated A/B: chunk_kda's private output layout has no generic end-to-end adapter."""
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "realtime profiler must be active for KDA fusion A/B"
    checkpoint_dir = kimi_k3_checkpoint_dir
    config = kimi_k3_kda_config()
    state_dict = load_kda_layer_state_dict(checkpoint_dir, KimiK3Config.FIRST_KDA_LAYER, config)
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    sequence_parallel_size = mesh_shape[sequence_parallel_axis]
    tensor_parallel_size = mesh_shape[tensor_parallel_axis]
    local_sequence = _SEQUENCE // sequence_parallel_size
    local_heads = config.num_heads // tensor_parallel_size
    generator = torch.Generator().manual_seed(1607)
    host_input = torch.randn(
        local_heads,
        local_sequence,
        config.head_v_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    host_gate = torch.randn(
        1,
        local_sequence,
        local_heads * config.head_v_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(tensor: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

    fused_input = to_device(host_input, ttnn.float32)
    composite_input = to_device(
        host_input.reshape(1, local_heads, local_sequence, config.head_v_dim),
        ttnn.float32,
    )
    gate = to_device(host_gate)
    fused_weight = to_device(state_dict["o_norm.weight"])
    composite_weight = to_device(state_dict["o_norm.weight"].reshape(1, 1, -1))
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def fused() -> ttnn.Tensor:
        return ttnn.transformer.kda_gated_rms_norm(
            fused_input,
            gate,
            fused_weight,
            local_heads,
            epsilon=config.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )

    def composite() -> ttnn.Tensor:
        output = ttnn.rms_norm(
            composite_input,
            weight=composite_weight,
            epsilon=config.norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.permute(
            output,
            (0, 2, 1, 3),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )
        output = ttnn.reshape(output, (1, 1, local_sequence, local_heads * config.head_v_dim))
        output_gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.multiply(
            output_gate,
            output,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    repetitions = int(os.getenv("KDA_FUSION_AB_REPS", str(_REPETITIONS)))
    with ttnn.manage_config("throw_exception_on_fallback", True):
        fused_measurement = _measure_operation(mesh_device, fused, repetitions)
        composite_measurement = _measure_operation(mesh_device, composite, repetitions)
    pcc = _pcc("gated_rms isolated output", fused_measurement.output, composite_measurement.output)
    result = {
        "fusion": "gated_rms",
        "scope": "isolated_operation",
        "layout": f"SP{sequence_parallel_size}xTP{tensor_parallel_size}",
        "sequence": _SEQUENCE,
        "local_sequence": local_sequence,
        "repetitions": repetitions,
        "pcc": {"output": pcc, "recurrent": None, "convolution": None},
        "fused": {
            "wall_ms": fused_measurement.wall_ms,
            "program_sum_us": fused_measurement.program_sum_us,
            "program_count": fused_measurement.program_count,
        },
        "composite": {
            "wall_ms": composite_measurement.wall_ms,
            "program_sum_us": composite_measurement.program_sum_us,
            "program_count": composite_measurement.program_count,
        },
    }
    result["fused_wall_gain_pct"] = (
        100.0 * (composite_measurement.wall_ms - fused_measurement.wall_ms) / composite_measurement.wall_ms
    )
    result["fused_program_sum_gain_pct"] = (
        100.0
        * (composite_measurement.program_sum_us - fused_measurement.program_sum_us)
        / composite_measurement.program_sum_us
    )
    print("KDA_FUSION_AB=" + json.dumps(result, sort_keys=True))
