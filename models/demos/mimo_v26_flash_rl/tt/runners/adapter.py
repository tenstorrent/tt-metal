# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash-RL in tt-metal's prefill service (ADAPTER_PATHS "mimo_v26_flash_rl").

The adapter is light at import (no ttnn / torch / model import at module load). The served KV cache is the device
model's own (kv_caches.py), its address table and the producer's read-back are kv_chunk_table.py, the runtime is
runtime.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from models.demos.common.prefill.adapter import PrefillModelAdapter


class MimoServiceConfig:
    NUM_LAYERS = 48
    # one bf16 hidden row (4096 x 2 B) + the fabric routing header, as the DeepSeek-family MoE dispatch needs
    FABRIC_PAYLOAD_SIZE = 4096 * 2 + 64
    MESH_SHAPE = (8, 4)
    CHUNK_SIZE = 5120
    MAX_SEQ_LEN = 56320


def _hf_dir() -> str:
    from models.demos.mimo_v26_flash_rl.tt.settings import settings

    return settings.get("SERVE_HF_DIR")


def validate_params(params) -> None:
    if tuple(params.mesh_shape) != MimoServiceConfig.MESH_SHAPE or (params.sp_axis, params.tp_axis) != (0, 1):
        raise ValueError(f"MiMo prefill runs on an SP 8 x TP 4 mesh, got {params.mesh_shape}")
    if params.chunk_size != MimoServiceConfig.CHUNK_SIZE:
        raise ValueError(f"MiMo prefill runs {MimoServiceConfig.CHUNK_SIZE}-token chunks, got {params.chunk_size}")
    if params.max_seq_len % params.chunk_size or params.max_seq_len > MimoServiceConfig.MAX_SEQ_LEN:
        raise ValueError(f"max_seq_len {params.max_seq_len}: a multiple of the chunk, at most 56320")
    if not (params.is_first_rank and params.is_last_rank):
        raise ValueError("MiMo prefill serves its layers on one rank")
    if params.use_trace or params.dflash_enabled or params.mtp_levels:
        raise ValueError("MiMo prefill is eager, without DFlash or MTP")


class MimoPrefillAdapter(PrefillModelAdapter):
    name = "mimo_v26_flash_rl"
    model_config = MimoServiceConfig
    hf_model_default = ""  # SERVE_HF_DIR in tt/settings.py
    ttnn_cache_default = ""
    prefill_trace_default = ""
    pipeline_activation_emb_tp_sharded = True
    supports_mtp = False

    def load_hf_config(self):
        cfg = json.loads((Path(_hf_dir()) / "config.json").read_text())
        return SimpleNamespace(**cfg, raw=cfg)

    def weight_cache_path(self, mesh_shape):
        return None  # weights load from the checkpoint (bfloat8_b on the device at load)

    def cache_kind(self, config_id: int) -> str:
        return "kv"

    def layer_split_boundaries(self, num_layers):
        return {0}

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        validate_params(params)
        from models.demos.mimo_v26_flash_rl.tt.runners.kv_caches import allocate_kv_caches

        return allocate_kv_caches(
            mesh_device,
            hf_config.raw,
            layers=range(params.first_layer_idx, params.first_layer_idx + params.num_layers),
            num_users=params.num_users,
            max_seq_len=params.max_seq_len,
            chunk_size=params.chunk_size,
        )

    def build_runtime(self, *, mesh_device, hf_config, params):
        validate_params(params)
        from models.demos.mimo_v26_flash_rl.tt.runners.runtime import MimoPrefillRuntime

        return MimoPrefillRuntime(mesh_device=mesh_device, hf_dir=_hf_dir(), config=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers=None):
        from models.demos.mimo_v26_flash_rl.tt.runners.kv_chunk_table import read_slot_kv_and_check_pcc

        return read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir, num_layers)
