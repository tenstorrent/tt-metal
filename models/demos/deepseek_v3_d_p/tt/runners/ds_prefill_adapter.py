# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""DeepSeek-V3-family adapters for the model-agnostic prefill runner.

Two concrete `PrefillModelAdapter`s — `DeepSeekPrefillAdapter` and `KimiPrefillAdapter` — share their
logic via `_DeepSeekFamilyPrefillAdapter` and differ only in a handful of per-model constants (config
class, cache/trace paths, gate-mode default) plus whether routing semaphores go to L1_SMALL (Kimi).
Every method is a thin wrapper over the existing `runner_utils` / `integration_setup` helpers and the
`TtDeepSeekPrefillRuntime`, so the runner exercises the same underlying model code for both.
"""

import os
from pathlib import Path
from typing import Optional

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.integration_setup import build_and_serialize_kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    kv_cache_pcc_check,
    load_hf_config,
    load_trace_token_ids,
    prepare_prefill_input_tensor,
    resolve_trace_dir,
    resolve_weight_cache_path,
)
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_runtime import TtDeepSeekPrefillRuntime, TtPrefillRuntimeConfig

# Per-iter mesh distribution for the token input. Used by the H2D service's
# internal mapper; the producer process builds an equivalent mapper from
# MeshShape on its side.
# `Shard(0)` shards the leading axis across mesh rows (SP); `Replicate()`
# duplicates across mesh cols (TP).
_H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
)


class _DeepSeekFamilyPrefillAdapter:
    """Shared PrefillModelAdapter logic for the DeepSeek-V3 family. The concrete adapters below set the
    per-model constants; everything here is identical between DeepSeek and Kimi."""

    # --- per-model constants (set by the concrete subclasses) ---
    name: str = ""  # matches the weight-cache dir prefix: {name}_{arch}_{N}dev
    model_config: type = None  # DeepSeekV3Config | KimiK26Config (static dimension constants)
    hf_model_default: str = ""  # HF model dir for config.json; PREFILL_HF_MODEL overrides
    ttnn_cache_default: str = ""  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides
    default_gate_mode: str = ""  # GateComputeMode name; PREFILL_GATE_FALLBACK_MODE overrides
    prefill_trace_default: str = ""  # golden trace dir; resolve_trace_dir descends one level if needed
    uses_l1_small_semaphores: bool = False  # carve an L1_SMALL region when opening the mesh

    # --- shared knobs ---
    supports_migration = True
    h2d_mapper_config = _H2D_MAPPER_CONFIG

    def __init__(self):
        # Honour + surface PREFILL_TTNN_CACHE even when only the model default applies (mirrors the
        # old prefill_runner.py module-level setdefault).
        os.environ.setdefault("PREFILL_TTNN_CACHE", self.ttnn_cache_default)

    @property
    def fabric_payload_size(self) -> int:
        return self.model_config.FABRIC_PAYLOAD_SIZE

    # --- resource resolution ---
    def load_hf_config(self, max_seq_len: int):
        hf_config = load_hf_config(self.hf_model_default)
        hf_config.max_seq_len = max_seq_len
        return hf_config

    def resolve_weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        return resolve_weight_cache_path(self.name, self.ttnn_cache_default, mesh_shape)

    def resolve_trace_dir(self) -> Path:
        return resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", self.prefill_trace_default))

    def load_trace_token_ids(self, trace_dir, total_len=None) -> list:
        return load_trace_token_ids(trace_dir, total_len)

    # --- runtime build (owns KV cache creation) ---
    def build_runtime(
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
        runtime_config = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            mesh_shape=mesh_shape,
            chunk_size=chunk_size,
            num_users=num_users,
            num_links=2,
            capacity_factor=capacity_factor,
            gate_fallback_mode=GateComputeMode[gate_fallback_mode],
            weight_cache_path=weight_cache_path,
            model_cfg=self.model_config,
            kv_only_last_layer=kv_only_last_layer,
            routing_use_l1_small_for_semaphores=self.uses_l1_small_semaphores,
        )
        return TtDeepSeekPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            config=runtime_config,
        )

    # --- token input layout ---
    def prepare_prefill_input_tensor(self, token_ids, mesh_device, sp_factor, is_balanced, mesh_shape, sp_axis):
        return prepare_prefill_input_tensor(token_ids, mesh_device, sp_factor, is_balanced, mesh_shape, sp_axis)

    # --- validation (reaches into the runtime's own KV cache) ---
    def kv_cache_pcc_check(self, runtime, slot_id: int, n_chunks: int, trace_dir=None) -> float:
        return kv_cache_pcc_check(runtime, slot_id=slot_id, n_chunks=n_chunks, trace_dir=trace_dir)

    # --- disaggregation / migration ---
    def build_and_serialize_kv_chunk_table(self, runtime, path: str) -> str:
        cfg = runtime.config
        return build_and_serialize_kv_chunk_table(
            mesh_device=runtime.mesh_device,
            kvpe_cache=runtime.kvpe_cache,
            seq_len=cfg.max_seq_len,
            num_layers=cfg.num_layers,
            mesh_shape=cfg.mesh_shape,
            sp_axis=cfg.sp_axis,  # GLOBAL_MESH_SHAPE = (sp, tp) — SP is axis 0
            num_users=cfg.num_users,
            path=path,
        )


class DeepSeekPrefillAdapter(_DeepSeekFamilyPrefillAdapter):
    name = "deepseek_v3_d_p"
    model_config = DeepSeekV3Config
    hf_model_default = "models/demos/deepseek_v3/reference"
    ttnn_cache_default = "/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure"
    default_gate_mode = "DEVICE_FP32"
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad"
    uses_l1_small_semaphores = False  # semaphores stay in main L1


class KimiPrefillAdapter(_DeepSeekFamilyPrefillAdapter):
    name = "kimi_k2_6"
    model_config = KimiK26Config
    # Repo-local config (dot-free, in-tree). The runner only needs config dims; real weights come
    # from the TTNN cache. To use a different checkpoint, set PREFILL_HF_MODEL to a dot-free path
    # (transformers' trust_remote_code import chokes on the "." in the canonical
    # /mnt/models/moonshotai/Kimi-K2.6-dequantized dir name).
    hf_model_default = "models/demos/deepseek_v3_d_p/reference/kimi_k2_6"
    ttnn_cache_default = "/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill"
    default_gate_mode = "DEVICE_FP32"  # Kimi (1 expert group)
    # vllm-traced golden: metadata.json + kv_cache live under a single run-hash subdir, and the
    # per-layer KV is row-sharded into layer_N/rows_*.safetensors. resolve_trace_dir descends to
    # the subdir; kv_cache_pcc_check reassembles the shards.
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320"
    # Kimi (single expert group, device gate) routes the MoE routing all-gather's global semaphores to
    # L1_SMALL so they don't pin the main-L1 floor and clash with the next layer's MLA static CBs.
    uses_l1_small_semaphores = True
