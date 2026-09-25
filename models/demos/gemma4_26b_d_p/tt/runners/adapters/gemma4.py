# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``Gemma4PrefillAdapter`` — common/prefill engine (and tt-d-gen prefill worker) <-> Gemma-4 26B-A4B.

Import-light: the model stack is imported inside the methods.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

DEFAULT_CKPT = "/localdev/mstaletovic/hf_models/gemma-4-26B-A4B-it"


class Gemma4Config:
    """Static dims the engine / producer read. Two KV geometries (see tt/runners/kv_chunk_table.py)."""

    NUM_LAYERS = 30
    EMB_SIZE = 2816
    SLIDING_KV = (8, 256)  # (kv heads, head_dim)
    FULL_KV = (2, 512)
    SLIDING_WINDOW = 1024
    FABRIC_PAYLOAD_SIZE = 14 * 1024 + 64  # BH max payload + Fabric2D routing info (== init_helpers.get_max_payload_size() on BH)


class Gemma4PrefillAdapter(PrefillModelAdapter):
    name = "gemma4_26b_d_p"
    model_config = Gemma4Config
    hf_model_default = os.environ.get("GEMMA4_D_P_CKPT", DEFAULT_CKPT)
    ttnn_cache_default = ""
    prefill_trace_default = ""
    pipeline_activation_emb_tp_sharded = False  # hidden replicated over TP

    hf_repo_id = "google/gemma-4-26B-A4B-it"

    def load_hf_config(self):
        from transformers import AutoConfig

        path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
        logger.info(f"Loading HF config for {self.name} from {path}")
        cfg = AutoConfig.from_pretrained(path)
        return getattr(cfg, "text_config", cfg)

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        return None  # weights are converted from safetensors at build time (no TTNN cache yet)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.gemma4_26b_d_p.tt.tt_prefill_runtime import Gemma4PrefillRuntime, Gemma4RuntimeConfig

        cfg = Gemma4RuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            chunk_size=params.chunk_size,
            mesh_shape=tuple(params.mesh_shape),
            num_users=params.num_users,
            first_layer_idx=params.first_layer_idx,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            use_trace=False,
            ckpt_dir=os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default,
        )
        self._runtime = Gemma4PrefillRuntime(mesh_device, cfg)
        return self._runtime

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        # The runner calls build_runtime first; the runtime knows the per-type layer split.
        return self._runtime.allocate_kv_caches()
