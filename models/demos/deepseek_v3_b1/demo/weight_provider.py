# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight providers for the DeepSeek V3 B1 demo pipeline.
CacheWeightProvider loads from disk; SyntheticWeightProvider builds deterministic synthetic weights;
StateDictWeightProvider loads HuggingFace safetensors and runs the same prepare_* path as synthetic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

import ttnn
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions
from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, CacheContext, TensorCache
from models.demos.deepseek_v3_b1.weights.cache.overlapped_metadata import (
    overlapped_tensor_from_view_dict,
    overlapped_tensor_to_view_dict,
)
from models.demos.deepseek_v3_b1.weights.prepare import (
    _MTP_LAYER_IDX,
    NUM_ROUTED_EXPERTS,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    OverlappedTensor,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_gate_bias_weight,
    prepare_gate_up_weights,
    prepare_kv_b12_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_mtp_weights,
    prepare_o_proj_norms_weights,
    prepare_q_ab_kv_a_weights,
    prepare_routed_expert_weights,
    prepare_shared_down_proj_weight,
)


@dataclass(frozen=True)
class LoadPhase:
    name: str
    elapsed_s: float


@dataclass(frozen=True)
class LoadRecord:
    method: str
    layer_id: int | None
    total_s: float
    phases: list[LoadPhase]


@dataclass
class WeightProviderPerformanceReport:
    records: list[LoadRecord] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return sum(record.total_s for record in self.records)

    def summary(self) -> str:
        if not self.records:
            return "No weight load performance records."
        lines = [f"weight_loads={len(self.records)} total={self.total_s:.3f}s"]
        for record in self.records:
            layer_desc = "" if record.layer_id is None else f" layer={record.layer_id}"
            phase_desc = ", ".join(f"{phase.name}={phase.elapsed_s:.3f}s" for phase in record.phases)
            if phase_desc:
                lines.append(f"{record.method}{layer_desc}: total={record.total_s:.3f}s ({phase_desc})")
            else:
                lines.append(f"{record.method}{layer_desc}: total={record.total_s:.3f}s")
        return "\n".join(lines)

    def to_dict(
        self,
        *,
        stage_id: int | None = None,
        weight_provider_name: str | None = None,
        timestamp_utc: str | None = None,
        model_directory: str | None = None,
        cache_directory: str | None = None,
        hostname: str | None = None,
    ) -> dict[str, object]:
        report: dict[str, object] = {
            "total_s": self.total_s,
            "record_count": len(self.records),
            "records": [
                {
                    "method": record.method,
                    "layer_id": record.layer_id,
                    "total_s": record.total_s,
                    "phases": [{"name": phase.name, "elapsed_s": phase.elapsed_s} for phase in record.phases],
                }
                for record in self.records
            ],
        }
        if stage_id is not None:
            report["stage_id"] = stage_id
        if weight_provider_name is not None:
            report["weight_provider_name"] = weight_provider_name
        if timestamp_utc is not None:
            report["timestamp_utc"] = timestamp_utc
        if model_directory is not None:
            report["model_directory"] = model_directory
        if cache_directory is not None:
            report["cache_directory"] = cache_directory
        if hostname is not None:
            report["hostname"] = hostname
        return report

    def to_json(
        self,
        indent: int = 2,
        *,
        stage_id: int | None = None,
        weight_provider_name: str | None = None,
        timestamp_utc: str | None = None,
        model_directory: str | None = None,
        cache_directory: str | None = None,
        hostname: str | None = None,
    ) -> str:
        return json.dumps(
            self.to_dict(
                stage_id=stage_id,
                weight_provider_name=weight_provider_name,
                timestamp_utc=timestamp_utc,
                model_directory=model_directory,
                cache_directory=cache_directory,
                hostname=hostname,
            ),
            indent=indent,
        )


class PhaseTracker:
    """Accumulates sub-phase timings for one load_* call."""

    def __init__(self) -> None:
        self._start_s = time.perf_counter()
        self._current_phase_name: str | None = None
        self._current_phase_start_s: float | None = None
        self._phases: list[LoadPhase] = []

    def phase(self, name: str) -> None:
        now = time.perf_counter()
        if self._current_phase_name is not None and self._current_phase_start_s is not None:
            self._phases.append(LoadPhase(name=self._current_phase_name, elapsed_s=now - self._current_phase_start_s))
        self._current_phase_name = name
        self._current_phase_start_s = now

    def finish(self) -> tuple[float, list[LoadPhase]]:
        now = time.perf_counter()
        if self._current_phase_name is not None and self._current_phase_start_s is not None:
            self._phases.append(LoadPhase(name=self._current_phase_name, elapsed_s=now - self._current_phase_start_s))
        return now - self._start_s, list(self._phases)


_SLOW_DISPATCH_CORE_X_START = 12


@dataclass(frozen=True)
class DeviceShard:
    host_tensor: ttnn.Tensor
    device_tensor: ttnn.Tensor
    core_subset: ttnn.CoreRangeSet | None


def write_shard(shard: DeviceShard) -> None:
    if shard.core_subset is None:
        ttnn.copy_host_to_device_tensor(shard.host_tensor, shard.device_tensor)
    else:
        ttnn.copy_host_to_device_tensor_partial(shard.host_tensor, shard.device_tensor, shard.core_subset)


def _core_subset_fast_dispatch(grid: ttnn.CoreRangeSet) -> ttnn.CoreRangeSet | None:
    parts: list[ttnn.CoreRange] = []
    for r in grid.ranges():
        sx, sy, ex, ey = r.start.x, r.start.y, r.end.x, r.end.y
        nx1 = min(ex, _SLOW_DISPATCH_CORE_X_START - 1)
        if sx <= nx1:
            parts.append(ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(nx1, ey)))
    if not parts:
        return None
    return ttnn.CoreRangeSet(parts)


def _core_subset_slow_dispatch(grid: ttnn.CoreRangeSet) -> ttnn.CoreRangeSet | None:
    parts: list[ttnn.CoreRange] = []
    for r in grid.ranges():
        sx, sy, ex, ey = r.start.x, r.start.y, r.end.x, r.end.y
        nx0 = max(sx, _SLOW_DISPATCH_CORE_X_START)
        if nx0 <= ex:
            parts.append(ttnn.CoreRange(ttnn.CoreCoord(nx0, sy), ttnn.CoreCoord(ex, ey)))
    if not parts:
        return None
    return ttnn.CoreRangeSet(parts)


def _rebind_overlapped_views(
    host_views: dict[str, OverlappedTensor],
    device_fused: ttnn.Tensor,
) -> dict[str, OverlappedTensor]:
    return {
        name: overlapped_tensor_from_view_dict(device_fused, overlapped_tensor_to_view_dict(ot))
        for name, ot in host_views.items()
    }


def _enqueue_shards_for_fused_views(
    host_views: dict[str, OverlappedTensor],
    device: ttnn.MeshDevice,
    fd_shards: list[DeviceShard],
    sd_shards: list[DeviceShard],
) -> dict[str, OverlappedTensor]:
    fused_host = next(iter(host_views.values())).fused_tensor
    device_fused = ttnn.allocate_tensor_on_device(fused_host.spec, device)
    shard_spec = fused_host.memory_config().shard_spec
    if shard_spec is None:
        fd_shards.append(DeviceShard(fused_host, device_fused, None))
        return _rebind_overlapped_views(host_views, device_fused)
    grid = shard_spec.grid
    fd = _core_subset_fast_dispatch(grid)
    sd = _core_subset_slow_dispatch(grid)
    if fd is not None:
        fd_shards.append(DeviceShard(fused_host, device_fused, fd))
    if sd is not None:
        sd_shards.append(DeviceShard(fused_host, device_fused, sd))
    if fd is None and sd is None:
        raise RuntimeError("empty FD/SD core split for fused weight buffer")
    return _rebind_overlapped_views(host_views, device_fused)


def _enqueue_shards_for_plain_tensor(
    host_tensor: ttnn.Tensor,
    device: ttnn.MeshDevice,
    fd_shards: list[DeviceShard],
    sd_shards: list[DeviceShard],
) -> ttnn.Tensor:
    device_tensor = ttnn.allocate_tensor_on_device(host_tensor.spec, device)
    shard_spec = host_tensor.memory_config().shard_spec
    if shard_spec is None:
        fd_shards.append(DeviceShard(host_tensor, device_tensor, None))
        return device_tensor
    grid = shard_spec.grid
    fd = _core_subset_fast_dispatch(grid)
    sd = _core_subset_slow_dispatch(grid)
    if fd is not None:
        fd_shards.append(DeviceShard(host_tensor, device_tensor, fd))
    if sd is not None:
        sd_shards.append(DeviceShard(host_tensor, device_tensor, sd))
    if fd is None and sd is None:
        raise RuntimeError("empty FD/SD core split for weight tensor")
    return device_tensor


class WeightProvider(Protocol):
    """Provides embedding and LM head weights on demand; each host loads only what its stage needs."""

    def get_performance_report(self) -> WeightProviderPerformanceReport:
        ...

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        ...

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        ...

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        ...

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        ...

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        ...


def _layer_key(layer_id: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_id}."""
    return f"model.layers.{layer_id}.{suffix}"


def _build_synthetic_moe_state_dict(
    layer_id: int,
    *,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> dict[str, torch.Tensor]:
    """Build a synthetic MoE layer state dict with HF tensor shapes (randn for weights, ones for norms)."""
    state_dict: dict[str, torch.Tensor] = {}
    dtype = torch.bfloat16

    # Attention weights (HF shapes)
    state_dict[_layer_key(layer_id, "self_attn.q_a_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_B_OUT, LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_proj_with_mqa.weight")] = torch.randn(
        LogicalModelDimensions.KV_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.KV_B_PROJ_OUT, LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.o_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.O_PROJ_OUT, dtype=dtype
    )

    # Norms (ones per plan)
    state_dict[_layer_key(layer_id, "input_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "post_attention_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )

    # MoE gate
    state_dict[_layer_key(layer_id, "mlp.gate.weight")] = torch.randn(
        LogicalModelDimensions.GATE_NUM_INDICES, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.gate.e_score_correction_bias")] = torch.randn(
        LogicalModelDimensions.GATE_NUM_INDICES, dtype=dtype
    )

    # Shared experts
    state_dict[_layer_key(layer_id, "mlp.shared_experts.gate_proj.weight")] = torch.randn(
        LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.shared_experts.up_proj.weight")] = torch.randn(
        LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.shared_experts.down_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, dtype=dtype
    )

    # Routed experts
    for e in range(num_routed_experts):
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.gate_proj.weight")] = torch.randn(
            LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
        )
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.up_proj.weight")] = torch.randn(
            LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
        )
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.down_proj.weight")] = torch.randn(
            LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, dtype=dtype
        )

    return state_dict


def _build_synthetic_dense_state_dict(layer_id: int) -> dict[str, torch.Tensor]:
    """Build a synthetic dense layer state dict with HF tensor shapes (randn for weights, ones for norms)."""
    state_dict: dict[str, torch.Tensor] = {}
    dtype = torch.bfloat16

    # Attention weights (HF shapes)
    state_dict[_layer_key(layer_id, "self_attn.q_a_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_B_OUT, LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_proj_with_mqa.weight")] = torch.randn(
        LogicalModelDimensions.KV_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.KV_B_PROJ_OUT, LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.o_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.O_PROJ_OUT, dtype=dtype
    )

    # Norms (ones per plan)
    state_dict[_layer_key(layer_id, "input_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "post_attention_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )

    # Single MLP (used for both shared and routed in dense)
    state_dict[_layer_key(layer_id, "mlp.gate_proj.weight")] = torch.randn(
        LogicalModelDimensions.INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.up_proj.weight")] = torch.randn(
        LogicalModelDimensions.INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.down_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.INTERMEDIATE_SIZE, dtype=dtype
    )

    return state_dict


def _build_synthetic_mtp_state_dict(mtp_layer_idx: int = _MTP_LAYER_IDX) -> dict[str, torch.Tensor]:
    """Build a synthetic MTP state dict with only the lightweight MTP projection/norm tensors."""
    dtype = torch.bfloat16
    H = LogicalModelDimensions.HIDDEN_SIZE

    return {
        _layer_key(mtp_layer_idx, "hnorm.weight"): torch.ones(H, dtype=dtype),
        _layer_key(mtp_layer_idx, "enorm.weight"): torch.ones(H, dtype=dtype),
        _layer_key(mtp_layer_idx, "eh_proj.weight"): torch.randn(H, 2 * H, dtype=dtype),
    }


class PerformanceTrackingWeightProvider:
    def __init__(self) -> None:
        self._perf = WeightProviderPerformanceReport()

    def get_performance_report(self) -> WeightProviderPerformanceReport:
        return self._perf

    def create_tracker(self) -> PhaseTracker:
        return PhaseTracker()

    def record(self, method: str, layer_id: int | None, total_s: float, phases: list[LoadPhase]) -> None:
        self._perf.records.append(LoadRecord(method=method, layer_id=layer_id, total_s=total_s, phases=phases))


class CacheWeightProvider(PerformanceTrackingWeightProvider):
    """Load weights through TensorCache-backed ``prepare_*`` calls with LazyStateDict miss source.

    The cache directory is created on first use if it does not already exist.
    """

    def __init__(
        self,
        cache_path: Path,
        model_path: Path,
        *,
        hf_model_id: str | None = None,
        hf_revision: str = "local",
        schema_version: int = 1,
    ) -> None:
        super().__init__()
        cache_path = Path(cache_path)
        model_path = Path(model_path)
        assert model_path.exists(), f"Model path does not exist: {model_path}"
        assert model_path.is_dir(), f"Model path is not a directory: {model_path}"
        self._cache = TensorCache(cache_path)
        self._state_dict = LazyStateDict(model_path)
        self._schema_version = schema_version
        self._hf_model_id = hf_model_id or model_path.name
        self._hf_revision = hf_revision

    def _cache_config(self, device: ttnn.MeshDevice, *, host_only: bool = False) -> CacheConfig:
        context = CacheContext(
            schema_version=self._schema_version,
            hf_model_id=self._hf_model_id,
            hf_revision=self._hf_revision,
            mesh_shape=(device.shape[0], device.shape[1]),
        )
        return CacheConfig(cache=self._cache, context=context, host_only=host_only)

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        tracker = self.create_tracker()
        tracker.phase("setup")
        cache_config = self._cache_config(device)
        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            tracker.phase("prepare_embedding_weights")
            result = prepare_embedding_weights(self._state_dict, device, cache_config=cache_config)
            tracker.phase("fast_dispatch_exit")
        assert isinstance(result, DeepSeekV3EmbeddingLayerWeights)
        total_s, phases = tracker.finish()
        self.record(method="load_embedding", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        tracker = self.create_tracker()
        tracker.phase("setup")
        cache_config = self._cache_config(device)
        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            tracker.phase("prepare_lm_head_weights")
            result = prepare_lm_head_weights(self._state_dict, device, cache_config=cache_config)
            tracker.phase("fast_dispatch_exit")
        assert isinstance(result, DeepSeekV3LMHeadWeights)
        total_s, phases = tracker.finish()
        self.record(method="load_lm_head", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        tracker = self.create_tracker()
        tracker.phase("setup")
        host_cfg = self._cache_config(device, host_only=True)
        dev_cfg = self._cache_config(device, host_only=False)
        tracker.phase("prepare")
        q_ab_h = prepare_q_ab_kv_a_weights(
            device,
            self._state_dict,
            layer_id,
            move_to_device=False,
            cache_config=host_cfg,
        )
        kv_h = prepare_kv_b12_weights(
            device,
            self._state_dict,
            layer_id,
            move_to_device=False,
            cache_config=host_cfg,
        )
        o_h = prepare_o_proj_norms_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=True,
            move_to_device=False,
            cache_config=host_cfg,
        )
        gate_bias_h = prepare_gate_bias_weight(
            device,
            self._state_dict,
            layer_id,
            move_to_device=False,
            cache_config=host_cfg,
        )
        gu_h = prepare_gate_up_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=True,
            move_to_device=False,
            cache_config=host_cfg,
        )
        shared_down_h = prepare_shared_down_proj_weight(
            device,
            self._state_dict,
            layer_id,
            is_moe=True,
            move_to_device=False,
            cache_config=host_cfg,
        )
        tracker.phase("lower")
        fd_shards: list[DeviceShard] = []
        sd_shards: list[DeviceShard] = []
        q_ab = _enqueue_shards_for_fused_views(q_ab_h, device, fd_shards, sd_shards)
        kv_views = _enqueue_shards_for_fused_views(kv_h, device, fd_shards, sd_shards)
        o_views = _enqueue_shards_for_fused_views(o_h, device, fd_shards, sd_shards)
        gate_bias_tt = _enqueue_shards_for_plain_tensor(gate_bias_h, device, fd_shards, sd_shards)
        gu_views = _enqueue_shards_for_fused_views(gu_h, device, fd_shards, sd_shards)
        shared_down_proj = _enqueue_shards_for_plain_tensor(shared_down_h, device, fd_shards, sd_shards)
        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            routed = prepare_routed_expert_weights(
                device,
                self._state_dict,
                layer_id,
                is_moe=True,
                num_routed_experts=NUM_ROUTED_EXPERTS,
                move_to_device=True,
                cache_config=dev_cfg,
            )
            for shard in fd_shards:
                write_shard(shard)
            tracker.phase("fast_dispatch_exit")
        for shard in sd_shards:
            write_shard(shard)
        tracker.phase("assemble")

        total_s, phases = tracker.finish()
        self.record(method="load_moe_layer", layer_id=layer_id, total_s=total_s, phases=phases)

        assert isinstance(routed, MoERoutedExpertWeights)
        assert isinstance(o_views["gate_mm"], OverlappedTensor)
        assert isinstance(gate_bias_tt, ttnn.Tensor)

        return DeepSeekV3MoELayerWeights(
            q_a_proj=q_ab["q_a_proj"],
            q_b_proj=q_ab["q_b_proj"],
            kv_a_proj=q_ab["kv_a_proj"],
            o_proj=o_views["o_proj"],
            gate_mm=o_views["gate_mm"],
            attn_norm=o_views["attn_norm"],
            q_norm=o_views["q_norm"],
            kv_norm=o_views["kv_norm"],
            ffn_norm=o_views["ffn_norm"],
            gate_bias=gate_bias_tt,
            kv_b1_proj=kv_views["kv_b1_proj"],
            kv_b2_proj=kv_views["kv_b2_proj"],
            shared_gate_proj=gu_views["shared_gate_proj"],
            shared_up_proj=gu_views["shared_up_proj"],
            shared_down_proj=shared_down_proj,
            routed_gate_proj=routed.routed_gate_proj,
            routed_up_proj=routed.routed_up_proj,
            routed_down_proj=routed.routed_down_proj,
        )

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        tracker = self.create_tracker()
        tracker.phase("setup")
        host_cfg = self._cache_config(device, host_only=True)
        dev_cfg = self._cache_config(device, host_only=False)
        tracker.phase("prepare")
        q_ab_h = prepare_q_ab_kv_a_weights(
            device,
            self._state_dict,
            layer_id,
            move_to_device=False,
            cache_config=host_cfg,
        )
        kv_h = prepare_kv_b12_weights(
            device,
            self._state_dict,
            layer_id,
            move_to_device=False,
            cache_config=host_cfg,
        )
        o_h = prepare_o_proj_norms_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=False,
            move_to_device=False,
            cache_config=host_cfg,
        )
        gu_h = prepare_gate_up_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=False,
            move_to_device=False,
            cache_config=host_cfg,
        )
        shared_down_h = prepare_shared_down_proj_weight(
            device,
            self._state_dict,
            layer_id,
            is_moe=False,
            move_to_device=False,
            cache_config=host_cfg,
        )
        tracker.phase("lower")
        fd_shards: list[DeviceShard] = []
        sd_shards: list[DeviceShard] = []
        q_ab = _enqueue_shards_for_fused_views(q_ab_h, device, fd_shards, sd_shards)
        kv_views = _enqueue_shards_for_fused_views(kv_h, device, fd_shards, sd_shards)
        o_views = _enqueue_shards_for_fused_views(o_h, device, fd_shards, sd_shards)
        gu_views = _enqueue_shards_for_fused_views(gu_h, device, fd_shards, sd_shards)
        shared_down_proj = _enqueue_shards_for_plain_tensor(shared_down_h, device, fd_shards, sd_shards)
        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            routed = prepare_routed_expert_weights(
                device,
                self._state_dict,
                layer_id,
                is_moe=False,
                move_to_device=True,
                cache_config=dev_cfg,
            )
            for shard in fd_shards:
                write_shard(shard)
            tracker.phase("fast_dispatch_exit")
        for shard in sd_shards:
            write_shard(shard)
        tracker.phase("assemble")
        assert isinstance(routed, DenseRoutedExpertWeights)
        total_s, phases = tracker.finish()
        self.record(method="load_dense_layer", layer_id=layer_id, total_s=total_s, phases=phases)
        return DeepSeekV3DenseLayerWeights(
            q_a_proj=q_ab["q_a_proj"],
            q_b_proj=q_ab["q_b_proj"],
            kv_a_proj=q_ab["kv_a_proj"],
            o_proj=o_views["o_proj"],
            attn_norm=o_views["attn_norm"],
            q_norm=o_views["q_norm"],
            kv_norm=o_views["kv_norm"],
            ffn_norm=o_views["ffn_norm"],
            kv_b1_proj=kv_views["kv_b1_proj"],
            kv_b2_proj=kv_views["kv_b2_proj"],
            shared_gate_proj=gu_views["shared_gate_proj"],
            shared_up_proj=gu_views["shared_up_proj"],
            shared_down_proj=shared_down_proj,
            routed_gate_proj=routed.routed_gate_proj,
            routed_up_proj=routed.routed_up_proj,
            routed_down_proj=routed.routed_down_proj,
        )

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_mtp_weights")
        result = prepare_mtp_weights(self._state_dict, device, cache_config=self._cache_config(device))
        total_s, phases = tracker.finish()
        self.record(method="load_mtp", layer_id=None, total_s=total_s, phases=phases)
        return result


class SyntheticWeightProvider(PerformanceTrackingWeightProvider):
    """Create deterministic synthetic embedding and LM head weights in place (no cache)."""

    def __init__(self) -> None:
        super().__init__()

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_embedding_weights")
        emb_w = torch.zeros(
            (LogicalModelDimensions.VOCAB_SIZE, LogicalModelDimensions.HIDDEN_SIZE), dtype=torch.bfloat16
        )
        emb_w[
            torch.arange(LogicalModelDimensions.VOCAB_SIZE),
            torch.arange(LogicalModelDimensions.VOCAB_SIZE, dtype=torch.int64) % LogicalModelDimensions.HIDDEN_SIZE,
        ] = 1
        result = prepare_embedding_weights({"model.embed_tokens.weight": emb_w}, device, move_to_device=True)
        total_s, phases = tracker.finish()
        self.record(method="load_embedding", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_lm_head_weights")
        # Stride for synthetic one-hot pattern: 101 matmul cores × 160 per core (matches LM head sampling op layout).
        _lm_head_n_synthetic = 101 * 160
        lm_w = torch.full(
            (LogicalModelDimensions.VOCAB_SIZE, LogicalModelDimensions.HIDDEN_SIZE), -1.0, dtype=torch.bfloat16
        )
        lm_w[
            torch.arange(LogicalModelDimensions.HIDDEN_SIZE, dtype=torch.int64) % _lm_head_n_synthetic,
            torch.arange(LogicalModelDimensions.HIDDEN_SIZE),
        ] = 1
        result = prepare_lm_head_weights(
            {
                "lm_head.weight": lm_w,
                "model.norm.weight": torch.ones(LogicalModelDimensions.HIDDEN_SIZE, dtype=torch.bfloat16),
            },
            device,
            move_to_device=True,
        )
        total_s, phases = tracker.finish()
        self.record(method="load_lm_head", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_moe_layer_weights")
        sd = _build_synthetic_moe_state_dict(layer_id, num_routed_experts=NUM_ROUTED_EXPERTS)
        result = prepare_moe_layer_weights(
            device, sd, layer_id, num_routed_experts=NUM_ROUTED_EXPERTS, move_to_device=True
        )
        total_s, phases = tracker.finish()
        self.record(method="load_moe_layer", layer_id=layer_id, total_s=total_s, phases=phases)
        return result

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_dense_layer_weights")
        sd = _build_synthetic_dense_state_dict(layer_id)
        result = prepare_dense_layer_weights(device, sd, layer_id, move_to_device=True)
        total_s, phases = tracker.finish()
        self.record(method="load_dense_layer", layer_id=layer_id, total_s=total_s, phases=phases)
        return result

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_mtp_weights")
        sd = _build_synthetic_mtp_state_dict()
        result = prepare_mtp_weights(sd, device, move_to_device=True)
        total_s, phases = tracker.finish()
        self.record(method="load_mtp", layer_id=None, total_s=total_s, phases=phases)
        return result


class StateDictWeightProvider(PerformanceTrackingWeightProvider):
    """Load real HF safetensors via LazyStateDict and prepare weights at runtime (no tensorbin cache)."""

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        model_path = Path(model_path)
        assert model_path.exists(), f"Model path does not exist: {model_path}"
        assert model_path.is_dir(), f"Model path is not a directory: {model_path}"
        self._state_dict = LazyStateDict(model_path)

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_embedding_weights")
        result = prepare_embedding_weights(self._state_dict, device, move_to_device=True)
        total_s, phases = tracker.finish()
        self.record(method="load_embedding", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_lm_head_weights")
        result = prepare_lm_head_weights(self._state_dict, device, move_to_device=True)
        total_s, phases = tracker.finish()
        self.record(method="load_lm_head", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_moe_layer_weights")
        result = prepare_moe_layer_weights(
            device,
            self._state_dict,
            layer_id,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            move_to_device=True,
        )
        total_s, phases = tracker.finish()
        self.record(method="load_moe_layer", layer_id=layer_id, total_s=total_s, phases=phases)
        return result

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_dense_layer_weights")
        result = prepare_dense_layer_weights(device, self._state_dict, layer_id, move_to_device=True)
        total_s, phases = tracker.finish()
        self.record(method="load_dense_layer", layer_id=layer_id, total_s=total_s, phases=phases)
        return result

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        tracker = self.create_tracker()
        tracker.phase("prepare_mtp_weights")
        result = prepare_mtp_weights(self._state_dict, device, move_to_device=True)
        total_s, phases = tracker.finish()
        self.record(method="load_mtp", layer_id=None, total_s=total_s, phases=phases)
        return result
