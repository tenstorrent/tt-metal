# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic KDA test cases and runners."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import (
    KIMI_K3_FIRST_KDA_LAYER,
    KIMI_K3_HF_REVISION,
    KIMI_K3_LAYER_1_SHA256,
    kda_state_dict_sha256,
    load_kda_layer_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate


@dataclass(frozen=True)
class KimiK3TestCase:
    config: KDAConfig
    state_dict: dict[str, torch.Tensor]
    hidden: torch.Tensor
    weights_identity: str


def _mesh_coordinate(sp_rank: int, tp_rank: int, sp_axis: int) -> tuple[int, int]:
    return (sp_rank, tp_rank) if sp_axis == 0 else (tp_rank, sp_rank)


def _device_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def collect_mesh_accuracy_and_determinism_results(
    run: Callable[[], Sequence[ttnn.Tensor]],
    *,
    count: int = 3,
) -> tuple[tuple[ttnn.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Retain first mesh outputs and reduce exact repeat mismatches on device."""
    if count <= 1:
        raise ValueError("count must be greater than one")

    reference_outputs = tuple(run())
    if not reference_outputs:
        raise ValueError("run must return at least one output")

    mismatch_marker = None
    for _ in range(1, count):
        outputs = tuple(run())
        if len(outputs) != len(reference_outputs):
            for output in outputs:
                ttnn.deallocate(output)
            raise ValueError("run returned a different number of outputs")
        for reference, output in zip(reference_outputs, outputs, strict=True):
            if (
                output.shape != reference.shape
                or output.dtype != reference.dtype
                or output.layout != reference.layout
                or output.memory_config() != reference.memory_config()
            ):
                for repeat_output in outputs:
                    ttnn.deallocate(repeat_output)
                raise ValueError("run returned output with different metadata")
            mismatch = ttnn.ne(reference, output, dtype=ttnn.bfloat16)
            current_marker = ttnn.max(mismatch)
            ttnn.deallocate(mismatch)
            if mismatch_marker is None:
                mismatch_marker = current_marker
            else:
                updated_marker = ttnn.maximum(mismatch_marker, current_marker)
                ttnn.deallocate(mismatch_marker)
                ttnn.deallocate(current_marker)
                mismatch_marker = updated_marker
        for output in outputs:
            ttnn.deallocate(output)

    assert mismatch_marker is not None
    mismatch_marker_host = ttnn.from_device(mismatch_marker)
    mismatch_markers = tuple(ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(mismatch_marker_host))
    ttnn.deallocate(mismatch_marker)
    return reference_outputs, mismatch_markers


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
    """Run the shared accuracy contract without hiding later tensor results."""
    try:
        pcc = assert_accurate(expected, actual, name=name, pcc_threshold=pcc_threshold)
    except AssertionError as error:
        return float("nan"), [str(error)]
    return pcc, []


def check_kimi_k3_accuracy(
    name: str,
    case: KimiK3TestCase,
    golden_output: torch.Tensor,
    golden_state: KDAReferenceState,
    state: KdaState,
    output: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    *,
    pcc_threshold: float,
) -> dict[str, float]:
    """Reconstruct an SP/TP K3 result and run the shared accuracy contract on every endpoint."""
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
    golden_output = golden_output.to(torch.bfloat16)
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    ).to(torch.bfloat16)
    local_width = case.config.num_heads // tp_size * case.config.head_k_dim
    output_pcc, failures = compare_cpu_device(
        f"{name} output", golden_output, actual_output, pcc_threshold=pcc_threshold
    )
    pcc = {"output": output_pcc}
    for sp_rank in range(sp_size):
        actual_recurrent = reconstruct_state_at_sp_rank(
            state.recurrent, mesh_device, sequence_parallel_axis, tensor_parallel_axis, sp_rank
        )
        actual_convolution = reconstruct_convolution_at_sp_rank(
            state.convolution,
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


def _kda_config_from_kimi_k3_constants() -> KDAConfig:
    return KDAConfig(
        hidden_size=KimiK3Config.EMB_SIZE,
        num_heads=KimiK3Config.KDA_NUM_HEADS,
        head_k_dim=KimiK3Config.KDA_HEAD_DIM,
        head_v_dim=KimiK3Config.KDA_HEAD_DIM,
        conv_kernel_size=KimiK3Config.KDA_SHORT_CONV_KERNEL_SIZE,
        norm_eps=KimiK3Config.RMS_NORM_EPS,
        use_full_rank_gate=KimiK3Config.KDA_USE_FULL_RANK_GATE,
        gate_lower_bound=KimiK3Config.KDA_GATE_LOWER_BOUND,
    )


def make_kimi_k3_test_case(checkpoint_dir: Path, *, sequence: int) -> KimiK3TestCase:
    """Load the pinned Kimi-K3 layer and deterministic input used by correctness and perf."""
    config = _kda_config_from_kimi_k3_constants()
    downloaded_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert KDAConfig.from_model_config(downloaded_config) == config
    state_dict = load_kda_layer_state_dict(checkpoint_dir, KIMI_K3_FIRST_KDA_LAYER, config)
    checkpoint_identity = kda_state_dict_sha256(state_dict)
    assert checkpoint_identity == KIMI_K3_LAYER_1_SHA256, (
        f"Kimi-K3 layer {KIMI_K3_FIRST_KDA_LAYER} weights do not match pinned revision "
        f"{KIMI_K3_HF_REVISION}: {checkpoint_identity}"
    )
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(1607),
        dtype=torch.bfloat16,
    )
    return KimiK3TestCase(
        config=config,
        state_dict=state_dict,
        hidden=hidden,
        weights_identity=checkpoint_identity,
    )


def make_synthetic_kimi_k3_test_case(*, sequence: int) -> KimiK3TestCase:
    """Build deterministic production-dimension Kimi-K3 inputs without a checkpoint."""
    config = _kda_config_from_kimi_k3_constants()
    state_dict = random_weights(config)
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(1607),
        dtype=torch.bfloat16,
    )
    return KimiK3TestCase(
        config=config,
        state_dict=state_dict,
        hidden=hidden,
        weights_identity=kda_state_dict_sha256(state_dict),
    )


def make_kimi_k3_device_case(
    mesh_device: ttnn.MeshDevice,
    case: KimiK3TestCase,
    *,
    tensor_parallel_axis: int = 1,
    summary_group_chunks: int | None = None,
    program_config: KDAProgramConfig | None = None,
    weights: KDAWeights | None = None,
    cache_weights: bool = True,
) -> tuple[ttKDA, ttnn.Tensor]:
    """Construct a production-dimension Kimi-K3 layer and sequence-parallel input."""
    sequence_parallel_axis = 1 - tensor_parallel_axis
    tensor_cache_path = (
        kimi_k3_tensor_cache_path(case.weights_identity, mesh_device, tensor_parallel_axis) if cache_weights else None
    )
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
    default_program_config = kimi_k3_program_config(
        tp_ccl_topology=(
            ttnn.Topology.Ring if tuple(mesh_device.shape)[sequence_parallel_axis] == 1 else ttnn.Topology.Linear
        )
    )
    selected_program_config = program_config or default_program_config
    if summary_group_chunks is not None:
        selected_program_config = replace(
            selected_program_config,
            recurrence=replace(selected_program_config.recurrence, summary_group_chunks=summary_group_chunks),
        )
    layer = ttKDA(
        mesh_device,
        case.config,
        case.state_dict if weights is None else None,
        weight_cache_path=tensor_cache_path,
        layer_idx=KIMI_K3_FIRST_KDA_LAYER,
        weights=weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=sequence_parallel_axis,
        tp_axis=tensor_parallel_axis,
        program_config=selected_program_config,
    )
    return layer, hidden


def kimi_k3_tensor_cache_path(
    weights_identity: str,
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> Path:
    """Select TTNN model-cache storage for one weight content identity and mesh placement."""
    mesh_shape = tuple(mesh_device.shape)
    if len(weights_identity) != 64 or any(character not in "0123456789abcdef" for character in weights_identity):
        raise ValueError("weights_identity must be a lowercase SHA-256 hex digest")
    layout = f"mesh{mesh_shape[0]}x{mesh_shape[1]}_tpaxis{tensor_parallel_axis}"
    return Path(ttnn.CONFIG.model_cache_path) / "kimi_k3" / weights_identity / layout


def make_config(
    *,
    num_heads: int = 2,
    use_full_rank_gate: bool = False,
) -> KDAConfig:
    return KDAConfig(
        hidden_size=64,
        num_heads=num_heads,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
        use_full_rank_gate=use_full_rank_gate,
    )


def make_program_config() -> KDAProgramConfig:
    return KDAProgramConfig()


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
