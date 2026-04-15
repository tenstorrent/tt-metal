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
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

import ttnn
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions
from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, CacheContext, TensorCache
from models.demos.deepseek_v3_b1.weights.cache.types import FusionGroupSpec
from models.demos.deepseek_v3_b1.weights.prepare import (
    _GATE_BIAS_SENDER_CORE_GRID,
    _MTP_LAYER_IDX,
    DOWN_PROJ_SINGLE_DEVICE_SPEC,
    GATE_UP_SPEC,
    KV_B12_SPEC,
    NUM_ROUTED_EXPERTS,
    O_PROJ_GATE_MM_NORMS_SPEC,
    Q_AB_KV_A_SPEC,
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


class DispatchMode(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class WeightWorkItem:
    name: str
    dispatch_mode: DispatchMode
    prepare_fn: Callable[[], Any]
    result: Any = field(default=None, init=False)

    @property
    def phase_name(self) -> str:
        prefix = "fd" if self.dispatch_mode == DispatchMode.FAST else "sd"
        return f"{prefix}:{self.name}"

    def execute(
        self,
        tracker: PhaseTracker | None = None,
    ) -> None:
        if tracker is not None:
            tracker.phase(self.phase_name)
        self.result = self.prepare_fn()


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


def _core_range_set_uses_final_column(crs: ttnn.CoreRangeSet, grid_width: int = 13) -> bool:
    final_col = grid_width - 1
    for core_range in crs.ranges():
        if core_range.end.x >= final_col:
            return True
    return False


def _classify_fusion_group(spec: FusionGroupSpec) -> DispatchMode:
    for region in spec.regions:
        if _core_range_set_uses_final_column(region.core_range_set):
            return DispatchMode.SLOW
    return DispatchMode.FAST


def _classify_core_range_set(crs: ttnn.CoreRangeSet) -> DispatchMode:
    return DispatchMode.SLOW if _core_range_set_uses_final_column(crs) else DispatchMode.FAST


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

    def _cache_config(self, device: ttnn.MeshDevice) -> CacheConfig:
        context = CacheContext(
            schema_version=self._schema_version,
            hf_model_id=self._hf_model_id,
            hf_revision=self._hf_revision,
            mesh_shape=(device.shape[0], device.shape[1]),
        )
        return CacheConfig(cache=self._cache, context=context)

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        tracker = self.create_tracker()
        tracker.phase("setup")
        cache_config = self._cache_config(device)
        work_item = WeightWorkItem(
            name="embedding",
            dispatch_mode=DispatchMode.FAST,
            prepare_fn=lambda: prepare_embedding_weights(self._state_dict, device, cache_config=cache_config),
        )
        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            work_item.execute(tracker)
            tracker.phase("fast_dispatch_exit")
        result = work_item.result
        assert isinstance(result, DeepSeekV3EmbeddingLayerWeights)
        total_s, phases = tracker.finish()
        self.record(method="load_embedding", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        tracker = self.create_tracker()
        tracker.phase("setup")
        cache_config = self._cache_config(device)
        work_item = WeightWorkItem(
            name="lm_head",
            dispatch_mode=DispatchMode.FAST,
            prepare_fn=lambda: prepare_lm_head_weights(self._state_dict, device, cache_config=cache_config),
        )
        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            work_item.execute(tracker)
            tracker.phase("fast_dispatch_exit")
        result = work_item.result
        assert isinstance(result, DeepSeekV3LMHeadWeights)
        total_s, phases = tracker.finish()
        self.record(method="load_lm_head", layer_id=None, total_s=total_s, phases=phases)
        return result

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        tracker = self.create_tracker()
        tracker.phase("setup")
        cache_config = self._cache_config(device)
        tracker.phase("classify")
        work_items: list[WeightWorkItem] = [
            WeightWorkItem(
                name="routed_experts",
                dispatch_mode=DispatchMode.FAST,  # Always use fast dispatch for DRAM tensors
                prepare_fn=lambda: prepare_routed_expert_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=True,
                    num_routed_experts=NUM_ROUTED_EXPERTS,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="q_ab_kv_a",
                dispatch_mode=_classify_fusion_group(Q_AB_KV_A_SPEC),
                prepare_fn=lambda: prepare_q_ab_kv_a_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="kv_b12",
                dispatch_mode=_classify_fusion_group(KV_B12_SPEC),
                prepare_fn=lambda: prepare_kv_b12_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="o_proj_gate_mm_norms",
                dispatch_mode=_classify_fusion_group(O_PROJ_GATE_MM_NORMS_SPEC),
                prepare_fn=lambda: prepare_o_proj_norms_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=True,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="gate_bias",
                dispatch_mode=_classify_core_range_set(_GATE_BIAS_SENDER_CORE_GRID),
                prepare_fn=lambda: prepare_gate_bias_weight(
                    device,
                    self._state_dict,
                    layer_id,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="gate_up",
                dispatch_mode=_classify_fusion_group(GATE_UP_SPEC),
                prepare_fn=lambda: prepare_gate_up_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=True,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="shared_down_proj",
                dispatch_mode=_classify_core_range_set(DOWN_PROJ_SINGLE_DEVICE_SPEC.build_matmul_core_grid()),
                prepare_fn=lambda: prepare_shared_down_proj_weight(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=True,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
        ]
        fd_queue = [item for item in work_items if item.dispatch_mode == DispatchMode.FAST]
        sd_queue = [item for item in work_items if item.dispatch_mode == DispatchMode.SLOW]

        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            for item in fd_queue:
                item.execute(tracker)
            tracker.phase("fast_dispatch_exit")

        for item in sd_queue:
            item.execute(tracker)

        tracker.phase("assemble")
        results = {item.name: item.result for item in work_items}
        routed = results["routed_experts"]
        q_ab = results["q_ab_kv_a"]
        kv_views = results["kv_b12"]
        o_views = results["o_proj_gate_mm_norms"]
        gate_bias_tt = results["gate_bias"]
        gu_views = results["gate_up"]
        shared_down_proj = results["shared_down_proj"]

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
        cache_config = self._cache_config(device)
        tracker.phase("classify")
        work_items: list[WeightWorkItem] = [
            WeightWorkItem(
                name="routed_experts",
                dispatch_mode=DispatchMode.FAST,
                prepare_fn=lambda: prepare_routed_expert_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=False,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="q_ab_kv_a",
                dispatch_mode=_classify_fusion_group(Q_AB_KV_A_SPEC),
                prepare_fn=lambda: prepare_q_ab_kv_a_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="kv_b12",
                dispatch_mode=_classify_fusion_group(KV_B12_SPEC),
                prepare_fn=lambda: prepare_kv_b12_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="o_proj_gate_mm_norms",
                dispatch_mode=_classify_fusion_group(O_PROJ_GATE_MM_NORMS_SPEC),
                prepare_fn=lambda: prepare_o_proj_norms_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=False,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="gate_up",
                dispatch_mode=_classify_fusion_group(GATE_UP_SPEC),
                prepare_fn=lambda: prepare_gate_up_weights(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=False,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
            WeightWorkItem(
                name="shared_down_proj",
                dispatch_mode=_classify_core_range_set(DOWN_PROJ_SINGLE_DEVICE_SPEC.build_matmul_core_grid()),
                prepare_fn=lambda: prepare_shared_down_proj_weight(
                    device,
                    self._state_dict,
                    layer_id,
                    is_moe=False,
                    move_to_device=True,
                    cache_config=cache_config,
                ),
            ),
        ]
        fd_queue = [item for item in work_items if item.dispatch_mode == DispatchMode.FAST]
        sd_queue = [item for item in work_items if item.dispatch_mode == DispatchMode.SLOW]
        tracker.phase("fast_dispatch_enter")
        with ttnn.device.setup_fast_dispatch(device):
            for item in fd_queue:
                item.execute(tracker)
            tracker.phase("fast_dispatch_exit")
        for item in sd_queue:
            item.execute(tracker)
        tracker.phase("assemble")
        results = {item.name: item.result for item in work_items}
        routed = results["routed_experts"]
        q_ab = results["q_ab_kv_a"]
        kv_views = results["kv_b12"]
        o_views = results["o_proj_gate_mm_norms"]
        gu_views = results["gate_up"]
        shared_down_proj = results["shared_down_proj"]
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
