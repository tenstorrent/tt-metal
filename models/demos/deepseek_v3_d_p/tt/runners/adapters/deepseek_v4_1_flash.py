# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""DeepSeek-V4.1-Flash prefill adapter (the encoder half of the disaggregated split, tt-blaze DS41F-0037 M2).

One rank (one galaxy, sp 8 x tp 4) runs the ENCODER (layers 0..19, Engram before 1 and 14) and layer 20 KV-only
(``tt/v41/prefill.py``) and exports the decode ring's contract (``tt/v41/kv_export.py``). The ring (tt-blaze
``deepseek_v4_1_flash``) replays the prompt's tail itself, so this side produces no first token and no LM head.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from loguru import logger

from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams


class DeepSeekV41FlashConfig:
    """The dimensions the generic runner reads (``runner_utils.open_mesh_device``: the fabric router payload). 14 KB = the
    Blackhole maximum (``moe.init_helpers.MAX_PAYLOAD_SIZE_BH``), what every V4.1 prefill test opened its mesh with."""

    EMB_SIZE = 5120
    FABRIC_PAYLOAD_SIZE = 14 * 1024


class DeepSeekV41FlashAdapter(PrefillModelAdapter):
    name = "deepseek_v4_1_flash"
    model_config = DeepSeekV41FlashConfig
    hf_model_default = "/mnt/tt-data/sdawle/models/DeepSeek-V4.1-Flash"
    ttnn_cache_default = "/mnt/tt-data/sdawle/v41p_weight_cache"
    prefill_trace_default = "/mnt/tt-data/sdawle/dsv41_traces"
    default_max_seq_len = 4096
    default_gate_mode = "DEVICE_FP32"
    l1_small_size = 1152
    routing_use_l1_small_for_semaphores = True
    supports_dflash = False

    def load_hf_config(self):
        """The checkpoint's own inference/config.json (``V41Config``) plus the few HF-style fields the runner reads."""
        from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config

        cfg = V41Config.load(os.environ.get("PREFILL_HF_MODEL", self.hf_model_default))
        hf = SimpleNamespace(
            v41=cfg,
            # the prefill rank's layers: the encoder + layer 20 KV-only
            num_hidden_layers=cfg.first_decoder_layer + 1,
            hidden_size=cfg.dim,
            vocab_size=cfg.vocab_size,
            n_routed_experts=cfg.n_routed_experts,
            num_experts_per_tok=cfg.n_activated_experts,
            moe_intermediate_size=cfg.moe_inter_dim,
            max_seq_len=int(os.environ.get("PREFILL_MAX_SEQ_LEN", self.default_max_seq_len)),
        )
        return hf

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.deepseek_v3_d_p.tt.v41.kv_export import allocate_v41_kv_export

        assert (
            params.first_layer_idx == 0 and params.is_first_rank and params.is_last_rank
        ), "V4.1 prefill is single-rank"
        exp = allocate_v41_kv_export(
            mesh_device,
            hf_config.v41,
            max_seq_len=params.max_seq_len,
            num_users=params.num_users,
            mesh_shape=tuple(params.mesh_shape),
            sp_axis=params.sp_axis,
        )
        logger.info(f"[dsv4.1-flash] export caches for max_seq_len {params.max_seq_len}, {params.num_users} user(s)")
        return exp

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.deepseek_v3_d_p.tt.v41.runtime import V41PrefillRuntime
        from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

        model_dir = os.environ.get("PREFILL_HF_MODEL", self.hf_model_default)
        return V41PrefillRuntime(
            mesh_device,
            hf_config.v41,
            checkpoint(model_dir),
            chunk_size=params.chunk_size,
            max_seq_len=params.max_seq_len,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            weight_cache_path=params.weight_cache_path,
            model_dir=model_dir,
        )

    hf_repo_id = "deepseek-ai/DeepSeek-V4.1-Flash"
    env_var = "DEEPSEEK_V4_V41_HF_MODEL"
    ttnn_cache_env = "TT_DSV41_FLASH_PREFILL_TTNN_CACHE"
