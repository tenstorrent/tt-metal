# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``MLAPrefillAdapter`` — shared adapter base for the DeepSeek-V3 model family.

DeepSeek-V3 and Kimi-K2.6 share one architecture (MLA attention + MoE) and one TT
implementation (``TtPrefillTransformer`` driven by ``TtPrefillRuntime``); they
differ only in static dimensions, default paths, HF-config quirks, and a couple of
device knobs. The adapter is pure glue: it tells the engine where this model's
config / weights / trace live and how to build its runtime. All operational behavior
(running a chunk, the KV layout, the migration table, PCC) lives on the runtime
(``TtPrefillRuntime``) the build returns; the engine drives those + owns all comms.

Concrete models (``deepseek_v3.py``, ``kimi_k2_6.py``) subclass this and set the
identity / default-path attributes, overriding a method only where they genuinely
diverge (e.g. Kimi's L1-small routing semaphores). A non-MLA architecture subclasses
``PrefillModelAdapter`` directly and builds its own runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches

# NOTE: the heavy model stack (TtPrefillRuntime / TtPrefillTransformer, the MoE gate, transformers'
# AutoConfig) is imported lazily inside the methods that need it. This keeps `import ...adapters` cheap
# enough for the H2D producers, which only read an adapter's path/trace attributes and must NOT pull in
# the device/model modules.


def unwrap_multimodal_config(cfg):
    """Unwrap a multimodal wrapper config (Kimi K2.5/K2.6) to its inner text_config.

    The LM fields the rest of the code reads (hidden_size, n_routed_experts, ...)
    live under ``text_config``. Also stubs ``quantization_config.weight_block_size``
    when missing so the dequant helper's eager read doesn't fail on a pre-dequantized
    checkpoint. A no-op for a plain (non-wrapped) config such as DeepSeek's.
    """
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        logger.info(f"Unwrapping multimodal wrapper config (inner model_type={cfg.text_config.model_type})")
        cfg = cfg.text_config
    qc = getattr(cfg, "quantization_config", None)
    if isinstance(qc, dict) and not qc.get("weight_block_size"):
        qc["weight_block_size"] = [128, 128]
        logger.info("Stubbed quantization_config.weight_block_size for pre-dequantized checkpoint")
    return cfg


class MLAPrefillAdapter(PrefillModelAdapter):
    """DeepSeek-V3-family prefill adapter (MLA + MoE over TtPrefillRuntime)."""

    # Weight-cache dir prefix; defaults to `name`. A model whose weights were tilized under a
    # different model's name overrides this to ride on that populated cache (e.g. Kimi-K2.7 reuses
    # the K2.6 cache — same architecture, only the checkpoint differs).
    weight_cache_name: Optional[str] = None

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        from transformers import AutoConfig

        model_path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
        logger.info(f"Loading HF config for model={self.name!r} from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return unwrap_multimodal_config(cfg)

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """Mirror the pytest weight_cache_path layout so the runner reads the same
        files the cache-populate run wrote:
          $PREFILL_TTNN_CACHE / {name}_{arch}_{N}dev / {sp}x{tp}
        Defaults to ``ttnn_cache_default``; None only if explicitly empty."""
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        arch = "bh" if is_blackhole() else "wh"
        num_devices = ttnn.get_num_devices()
        sp, tp = mesh_shape
        cache_name = self.weight_cache_name or self.name
        path = Path(env_cache) / f"{cache_name}_{arch}_{num_devices}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # KV cache + runtime build
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> MlaKvCaches:
        """Allocate the MLA kvpe KV cache (qk_rope_head_dim + kv_lora_rank per token; one shared cache of
        num_users * num_layers user-major slots). Dense MLA has no indexer cache, so ``index`` stays None.
        The engine owns the returned cache and passes it into every runtime call."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache

        return MlaKvCaches(
            kvpe=allocate_mla_kvpe_cache(
                mesh_device=mesh_device,
                hf_config=hf_config,
                max_seq_len=params.max_seq_len,
                mesh_shape=params.mesh_shape,
                sp_axis=params.sp_axis,
                num_layers=params.num_layers,
                num_users=params.num_users,
            )
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Construct the MLA model and return the runtime. The runtime (TtPrefillRuntime)
        is stateless w.r.t. the KV cache — the engine allocates it (allocate_kv_cache) and
        passes it into each runtime call; the runtime knows the layout to read/write it."""
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
        from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            num_links=params.num_links,
            capacity_factor=params.capacity_factor,
            gate_fallback_mode=GateComputeMode[params.gate_mode_name],
            weight_cache_path=params.weight_cache_path,
            model_cfg=self.model_config,
            first_layer_idx=params.first_layer_idx,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            kv_only_last_layer=params.kv_only_last_layer,
            routing_use_l1_small_for_semaphores=self.routing_use_l1_small_for_semaphores,
        )
        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            config=runtime_config,
        )
