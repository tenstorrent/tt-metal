# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""DeepSeek-V3-family adapter for the model-agnostic prefill runner.

Implements `models.common.prefill_runner.adapter.PrefillModelAdapter` for both the deepseek_v3_d_p
and kimi_k2_6 variants. It is a thin wrapper: every method delegates to the existing
`runner_utils` / `integration_setup` helpers and the `TtDeepSeekPrefillPipeline`, so the new common
runner and the old in-package prefill_runner.py exercise the SAME underlying code (easy to compare).

The variant differences (config class, cache/trace paths, gate-mode default) come from the
`RunnerVariant` registry in runner_utils; the only behavioural branch is whether routing semaphores
go to L1_SMALL (Kimi).
"""

import os
from pathlib import Path
from typing import Optional

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.integration_setup import build_and_serialize_kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    get_variant,
    kv_cache_pcc_check,
    load_hf_config,
    load_trace_token_ids,
    prepare_prefill_input_tensor,
    resolve_trace_dir,
    resolve_weight_cache_path,
)
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
)

# Per-iter mesh distribution for the token input. Used by the H2D service's
# internal mapper; the producer process builds an equivalent mapper from
# MeshShape on its side.
# `Shard(0)` shards the leading axis across mesh rows (SP); `Replicate()`
# duplicates across mesh cols (TP).
_H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
)


class DeepSeekPrefillAdapter:
    """PrefillModelAdapter implementation backed by a RunnerVariant + TtDeepSeekPrefillPipeline."""

    supports_migration = True
    h2d_mapper_config = _H2D_MAPPER_CONFIG

    def __init__(self, variant):
        self._variant = variant

    # --- static knobs ---
    @property
    def name(self) -> str:
        return self._variant.name

    @property
    def default_gate_mode(self) -> str:
        return self._variant.default_gate_mode

    @property
    def uses_l1_small_semaphores(self) -> bool:
        # Kimi (single expert group, device gate) routes the MoE routing all-gather's global
        # semaphores to L1_SMALL; DeepSeek keeps them in main L1.
        return self._variant.name == "kimi_k2_6"

    @property
    def fabric_payload_size(self) -> int:
        return self._variant.model_config.FABRIC_PAYLOAD_SIZE

    # --- resource resolution ---
    def load_hf_config(self, max_seq_len: int):
        hf_config = load_hf_config(self._variant)
        hf_config.max_seq_len = max_seq_len
        return hf_config

    def resolve_weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        return resolve_weight_cache_path(self._variant, mesh_shape)

    def resolve_trace_dir(self) -> Path:
        return resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", self._variant.prefill_trace_default))

    def load_trace_token_ids(self, trace_dir, total_len=None) -> list:
        return load_trace_token_ids(trace_dir, total_len)

    # --- pipeline build (owns KV cache creation) ---
    def build_pipeline(
        self,
        *,
        mesh_device,
        hf_config,
        mesh_shape: tuple,
        num_layers: int,
        max_seq_len: int,
        chunk_size: int,
        num_users: int,
        capacity_factor: int,
        gate_fallback_mode: str,
        weight_cache_path: Optional[Path],
        kv_only_last_layer: bool,
    ):
        pipeline_config = TtPrefillPipelineConfig(
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            mesh_shape=mesh_shape,
            chunk_size=chunk_size,
            num_users=num_users,
            num_links=2,
            capacity_factor=capacity_factor,
            gate_fallback_mode=GateComputeMode[gate_fallback_mode],
            weight_cache_path=weight_cache_path,
            model_cfg=self._variant.model_config,
            kv_only_last_layer=kv_only_last_layer,
            routing_use_l1_small_for_semaphores=self.uses_l1_small_semaphores,
        )
        return TtDeepSeekPrefillPipeline(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            config=pipeline_config,
        )

    # --- token input layout ---
    def prepare_prefill_input_tensor(self, token_ids, mesh_device, sp_factor, is_balanced, mesh_shape, sp_axis):
        return prepare_prefill_input_tensor(token_ids, mesh_device, sp_factor, is_balanced, mesh_shape, sp_axis)

    # --- validation (reaches into the pipeline's own KV cache) ---
    def kv_cache_pcc_check(self, pipeline, slot_id: int, n_chunks: int, trace_dir=None) -> float:
        return kv_cache_pcc_check(pipeline, slot_id=slot_id, n_chunks=n_chunks, trace_dir=trace_dir)

    # --- disaggregation / migration ---
    def build_and_serialize_kv_chunk_table(self, pipeline, path: str) -> str:
        cfg = pipeline.config
        return build_and_serialize_kv_chunk_table(
            mesh_device=pipeline.mesh_device,
            kvpe_cache=pipeline.kvpe_cache,
            seq_len=cfg.max_seq_len,
            num_layers=cfg.num_layers,
            mesh_shape=cfg.mesh_shape,
            sp_axis=cfg.sp_axis,  # GLOBAL_MESH_SHAPE = (sp, tp) — SP is axis 0
            num_users=cfg.num_users,
            path=path,
        )


def make_adapter(name: str) -> DeepSeekPrefillAdapter:
    """Registry factory: resolve the RunnerVariant and return its adapter.

    Mirrors the old prefill_runner.py module-level setdefault so PREFILL_TTNN_CACHE is honoured and
    visible in logs even when only the variant default applies."""
    variant = get_variant(name)
    os.environ.setdefault("PREFILL_TTNN_CACHE", variant.ttnn_cache_default)
    return DeepSeekPrefillAdapter(variant)
