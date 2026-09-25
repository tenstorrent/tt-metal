# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``MiMoV2PrefillAdapter`` — common/prefill engine (and tt-d-gen prefill worker) <-> MiMo-V2.6-Flash.

Import-light: the model stack is imported inside the methods. Weights are range-fetched from the HF repo
and dequantized per layer on first use (reference/weights.py), so a rank only pulls its own layers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

DEFAULT_CKPT = "/localdev/mstaletovic/hf_models/MiMo-V2.6-Flash-RL"


class MiMoV2Config:
    """Static dims the engine / producer read. Two KV geometries (see tt/runners/kv_chunk_table.py)."""

    NUM_LAYERS = 48
    EMB_SIZE = 4096
    GA_KV = (4, 192, 128)  # (kv heads, k dim, v dim)
    SWA_KV = (8, 192, 128)
    SLIDING_WINDOW = 128
    FABRIC_PAYLOAD_SIZE = 14 * 1024 + 64  # BH max payload + Fabric2D routing info


class MiMoV2PrefillAdapter(PrefillModelAdapter):
    name = "mimo_v2_d_p"
    model_config = MiMoV2Config
    hf_model_default = os.environ.get("MIMO_V2_CKPT", DEFAULT_CKPT)
    ttnn_cache_default = ""
    prefill_trace_default = ""
    pipeline_activation_emb_tp_sharded = False  # hidden replicated over TP

    hf_repo_id = "XiaomiMiMo/MiMo-V2.6-Flash-RL"

    def load_hf_config(self):
        from models.demos.mimo_v2_d_p.reference.hf import hf_config

        logger.info(f"Loading HF config for {self.name}")
        return hf_config()

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        return None  # dequantized from the fp8/mxfp4 checkpoint at build time (no TTNN cache yet)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.mimo_v2_d_p.tt.tt_prefill_runtime import MiMoPrefillRuntime, MiMoRuntimeConfig

        cfg = MiMoRuntimeConfig(
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
        self._runtime = MiMoPrefillRuntime(mesh_device, cfg)
        return self._runtime

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        # The runner calls build_runtime first; the runtime knows the per-type layer split.
        return self._runtime.allocate_kv_caches()
