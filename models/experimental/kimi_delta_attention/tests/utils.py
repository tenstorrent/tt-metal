# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic KDA test cases and runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.kimi_delta_attention.config import KDAConfig, KDAProgramConfig
from models.experimental.kimi_delta_attention.checkpoint import load_kda_layer_state_dict
from models.experimental.kimi_delta_attention.reference import KDAReferenceState
from models.experimental.kimi_delta_attention.kimi_k3_config import (
    KimiK3Config,
    kimi_k3_kda_config,
    kimi_k3_program_config,
)
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


@dataclass(frozen=True)
class KimiK3TestCase:
    config: KDAConfig
    state_dict: dict[str, torch.Tensor]
    hidden: torch.Tensor
    checkpoint_dir: Path


def report_finiteness(name: str, tensor: torch.Tensor) -> tuple[bool, str]:
    """Report absolute/relative non-finite counts and return the verdict."""
    element_count = tensor.numel()
    nan_count = int(torch.isnan(tensor).sum().item())
    positive_inf_count = int(torch.isposinf(tensor).sum().item())
    negative_inf_count = int(torch.isneginf(tensor).sum().item())
    non_finite_count = nan_count + positive_inf_count + negative_inf_count

    def fraction(count: int) -> float:
        return count / element_count if element_count else 0.0

    summary = (
        f"{name} finiteness: "
        f"non_finite={non_finite_count}/{element_count} ({fraction(non_finite_count):.6e}), "
        f"nan={nan_count}/{element_count} ({fraction(nan_count):.6e}), "
        f"+inf={positive_inf_count}/{element_count} ({fraction(positive_inf_count):.6e}), "
        f"-inf={negative_inf_count}/{element_count} ({fraction(negative_inf_count):.6e})"
    )
    print(summary)
    return non_finite_count == 0, summary


def assert_all_finite(name: str, tensor: torch.Tensor) -> None:
    """Report non-finite counts/fractions and require an entirely finite tensor."""
    passed, summary = report_finiteness(name, tensor)
    assert passed, summary


def _mesh_coordinate(sp_rank: int, tp_rank: int, sp_axis: int) -> tuple[int, int]:
    return (sp_rank, tp_rank) if sp_axis == 0 else (tp_rank, sp_rank)


def _device_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def reconstruct_sp_tp_tensor(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    tp_dim: int,
    sp_dim: int,
) -> torch.Tensor:
    """Reconstruct a logical tensor from sequence- and tensor-parallel device shards."""
    shards = _device_shards(tensor)
    rows, columns = tuple(mesh_device.shape)
    sp_size, tp_size = (rows, columns)[sp_axis], (rows, columns)[tp_axis]
    partitions = []
    for sp_rank in range(sp_size):
        tp_shards = []
        for tp_rank in range(tp_size):
            row, column = _mesh_coordinate(sp_rank, tp_rank, sp_axis)
            shard = shards[row * columns + column]
            if shard.ndim == 4:
                shard = shard.reshape(shard.shape[0], shard.shape[-2], shard.shape[-1])
            tp_shards.append(shard)
        partitions.append(torch.cat(tp_shards, dim=tp_dim))
    return torch.cat(partitions, dim=sp_dim)


def reconstruct_state_at_sp_rank(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    sp_rank: int,
) -> torch.Tensor:
    """Reconstruct a tensor-parallel recurrent state at one sequence rank."""
    shards = _device_shards(tensor)
    columns = tuple(mesh_device.shape)[1]
    tp_size = tuple(mesh_device.shape)[tp_axis]
    tp_shards = []
    for tp_rank in range(tp_size):
        row, column = _mesh_coordinate(sp_rank, tp_rank, sp_axis)
        tp_shards.append(shards[row * columns + column])
    return torch.cat(tp_shards, dim=1)


def reconstruct_convolution_at_sp_rank(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    sp_rank: int,
    local_width: int,
) -> torch.Tensor:
    """Reconstruct grouped Q/K/V convolution state at one sequence rank."""
    shards = _device_shards(tensor)
    columns = tuple(mesh_device.shape)[1]
    tp_size = tuple(mesh_device.shape)[tp_axis]
    physical = []
    for tp_rank in range(tp_size):
        row, column = _mesh_coordinate(sp_rank, tp_rank, sp_axis)
        physical.append(shards[row * columns + column])
    return torch.cat(
        tuple(
            torch.cat([shard[..., index * local_width : (index + 1) * local_width] for shard in physical], dim=-1)
            for index in range(3)
        ),
        dim=-1,
    )


def compare_cpu_device(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    pcc_threshold: float,
) -> tuple[float, list[str]]:
    """Report CPU/device finiteness and PCC without hiding later tensor results."""
    cpu_finite, cpu_summary = report_finiteness(f"{name} CPU reference", expected)
    device_finite, device_summary = report_finiteness(f"{name} device result", actual)
    passed, pcc = comp_pcc(expected, actual, pcc=pcc_threshold)
    print(f"{name}: PCC={pcc:.6f}")
    failures = []
    if not cpu_finite:
        failures.append(cpu_summary)
    if not device_finite:
        failures.append(device_summary)
    if not passed:
        failures.append(f"{name} PCC {pcc:.6f} < {pcc_threshold}")
    return pcc, failures


def assert_kimi_k3_device_matches_reference(
    name: str,
    case: KimiK3TestCase,
    golden_output: torch.Tensor,
    golden_state: KDAReferenceState,
    layer: KimiDeltaAttention,
    output: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    *,
    pcc_threshold: float,
) -> dict[str, float]:
    """Reconstruct an SP/TP K3 result and compare every logical tensor with CPU."""
    sequence_parallel_axis = 1 - tensor_parallel_axis
    mesh_shape = tuple(mesh_device.shape)
    sp_size = mesh_shape[sequence_parallel_axis]
    tp_size = mesh_shape[tensor_parallel_axis]
    actual_output = reconstruct_sp_tp_tensor(
        output,
        mesh_device,
        sequence_parallel_axis,
        tensor_parallel_axis,
        tp_dim=2,
        sp_dim=1,
    )
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )
    local_width = case.config.num_heads // tp_size * case.config.head_k_dim
    output_pcc, failures = compare_cpu_device(
        f"{name} output", golden_output, actual_output, pcc_threshold=pcc_threshold
    )
    pcc = {"output": output_pcc}
    for sp_rank in range(sp_size):
        actual_recurrent = reconstruct_state_at_sp_rank(
            layer.recurrent_state, mesh_device, sequence_parallel_axis, tensor_parallel_axis, sp_rank
        )
        actual_convolution = reconstruct_convolution_at_sp_rank(
            layer.convolution_state,
            mesh_device,
            sequence_parallel_axis,
            tensor_parallel_axis,
            sp_rank,
            local_width,
        )
        recurrent_pcc, recurrent_failures = compare_cpu_device(
            f"{name} sp_rank={sp_rank} recurrent state",
            golden_state.recurrent,
            actual_recurrent,
            pcc_threshold=pcc_threshold,
        )
        pcc[f"sp_rank_{sp_rank}_recurrent"] = recurrent_pcc
        failures.extend(recurrent_failures)
        convolution_pcc, convolution_failures = compare_cpu_device(
            f"{name} sp_rank={sp_rank} convolution state",
            golden_convolution,
            actual_convolution,
            pcc_threshold=pcc_threshold,
        )
        pcc[f"sp_rank_{sp_rank}_convolution"] = convolution_pcc
        failures.extend(convolution_failures)
    assert not failures, "\n".join(failures)
    return pcc


def make_kimi_k3_test_case(checkpoint_dir: Path, *, sequence: int) -> KimiK3TestCase:
    """Load the pinned Kimi-K3 layer and deterministic input used by correctness and perf."""
    config = kimi_k3_kda_config()
    downloaded_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert KDAConfig.from_model_config(downloaded_config) == config
    state_dict = load_kda_layer_state_dict(checkpoint_dir, KimiK3Config.FIRST_KDA_LAYER, config)
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(1607),
        dtype=torch.bfloat16,
    )
    return KimiK3TestCase(config=config, state_dict=state_dict, hidden=hidden, checkpoint_dir=checkpoint_dir)


def make_kimi_k3_device_case(
    mesh_device: ttnn.MeshDevice,
    case: KimiK3TestCase,
    *,
    tensor_parallel_axis: int = 1,
    summary_group_chunks: int | None = None,
    weights: KDAWeights | None = None,
) -> tuple[KimiDeltaAttention, ttnn.Tensor]:
    """Construct the real-weight layer and sequence-parallel device input."""
    sequence_parallel_axis = 1 - tensor_parallel_axis
    mesh_dims: list[int | None] = [None, None]
    mesh_dims[sequence_parallel_axis] = 1
    hidden = ttnn.from_torch(
        case.hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=tuple(mesh_dims),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )
    layer = KimiDeltaAttention(
        mesh_device,
        case.config,
        case.state_dict if weights is None else None,
        tensor_cache_path=case.checkpoint_dir / "ttnn_cache",
        cache_name_prefix=f"layer_{KimiK3Config.FIRST_KDA_LAYER}.kda",
        weights=weights,
        tt_ccl=TT_CCL(mesh_device),
        tensor_parallel_axis=tensor_parallel_axis,
        summary_group_chunks=summary_group_chunks,
        program_config=kimi_k3_program_config(
            tp_ccl_topology=(
                ttnn.Topology.Ring if tuple(mesh_device.shape)[sequence_parallel_axis] == 1 else ttnn.Topology.Linear
            )
        ),
    )
    return layer, hidden


def run_profiled_forward(
    mesh_device: ttnn.MeshDevice,
    forward: Callable[[], ttnn.Tensor],
) -> tuple[ttnn.Tensor, list[dict[str, object]]]:
    """Run one correctness forward and require usable realtime-profiler records."""
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "realtime profiler must be active for KDA correctness"
    output, records = profile_realtime_program(
        mesh_device,
        forward,
        collect_all=True,
        record_timeout_seconds=10.0,
    )
    non_sentinel_records = [record for record in records if int(record["runtime_id"]) != 0]
    assert non_sentinel_records, "realtime profiler returned no program records"
    return output, non_sentinel_records


def make_config() -> KDAConfig:
    return KDAConfig(
        hidden_size=64,
        num_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )


def make_program_config(*, recurrent_state_dtype: ttnn.DataType = ttnn.float32) -> KDAProgramConfig:
    return KDAProgramConfig(recurrent_state_dtype=recurrent_state_dtype)


def random_weights(config: KDAConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260723)

    def normal(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return scale * torch.randn(*shape, generator=generator)

    hidden = config.hidden_size
    key_rank, value_rank = config.head_k_dim, config.head_v_dim
    weights = {
        "q_proj.weight": normal(config.q_dim, hidden),
        "k_proj.weight": normal(config.k_dim, hidden),
        "v_proj.weight": normal(config.v_dim, hidden),
        "q_conv1d.weight": normal(config.q_dim, 1, config.conv_kernel_size, scale=0.2),
        "k_conv1d.weight": normal(config.k_dim, 1, config.conv_kernel_size, scale=0.2),
        "v_conv1d.weight": normal(config.v_dim, 1, config.conv_kernel_size, scale=0.2),
        "A_log": torch.log(torch.linspace(1.0, 4.0, config.num_heads)).reshape(1, 1, config.num_heads, 1),
        "f_a_proj.weight": normal(key_rank, hidden),
        "f_b_proj.weight": normal(config.num_heads * key_rank, key_rank),
        "dt_bias": normal(config.num_heads * key_rank),
        "b_proj.weight": normal(config.num_heads, hidden),
        "o_norm.weight": 1.0 + normal(value_rank),
        "o_proj.weight": normal(hidden, config.num_heads * value_rank),
    }
    if config.use_full_rank_gate:
        weights["g_proj.weight"] = normal(config.v_dim, hidden)
    else:
        weights["g_a_proj.weight"] = normal(value_rank, hidden)
        weights["g_b_proj.weight"] = normal(config.num_heads * value_rank, value_rank)
    return weights
