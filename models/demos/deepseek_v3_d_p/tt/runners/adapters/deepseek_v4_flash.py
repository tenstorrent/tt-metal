# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""DeepSeek-V4-Flash prefill adapter.

A genuinely different architecture from the MLA family, so it subclasses ``PrefillModelAdapter`` directly:
mHC hyper-connections (4 residual streams mixed around every sublayer), three attention kinds by layer
(sliding-window / compressed-sparse with the lightning indexer / heavily-compressed), shared-KV MQA with
head_dim 512, a 256-expert MoE with the sqrtsoftplus gate and 3 hash-routed layers. The V4 HCA prefill
(``tt/mla/heavily_compressed_attention.py``), the V4 gate and the V4 references already live in this package.

What this adapter does today (plan M1): config, layer-kind schedule, the per-kind KV geometry and caches
(``tt/v4/kv_contract.py`` is the prefill <-> decode contract), registry + manifest. ``build_runtime`` lands with
the block/transformer/runtime (plan M2..M6, tt-blaze ``docs/plans/deepseek_v4_flash_prefill_ttnn_plan.md``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import (
    DeepSeekV4FlashConfig,
    deepseek_v4_flash_hf_config,
)


class DeepSeekV4FlashAdapter(PrefillModelAdapter):
    # --- identity & runner defaults ---
    name = "deepseek_v4_flash"
    model_config = DeepSeekV4FlashConfig
    # Pod defaults (NFS, shared by all four hosts); override with PREFILL_HF_MODEL / PREFILL_TTNN_CACHE /
    # PREFILL_TRACE_DIR in a rank binding's global_env (a shell export does not reach a tt-run rank).
    hf_model_default = "/mnt/tt-data/sdawle/models/DeepSeek-V4-Flash-0731"
    ttnn_cache_default = "/mnt/tt-data/sdawle/dsv4_flash_ttnn_cache"
    prefill_trace_default = "/mnt/tt-data/sdawle/dsv4_flash_traces"
    # The engine requires max_seq_len % chunk_size == 0 (chunk 5120), so "128k" is 26 chunks = 133,120 tokens.
    default_max_seq_len = 10240
    default_gate_mode = "DEVICE_FP32"
    # Single expert group + device gate like GLM/Kimi: the routing all-gather's semaphores go to L1_SMALL.
    l1_small_size = 1152
    routing_use_l1_small_for_semaphores = True
    supports_dflash = False

    def load_hf_config(self):
        """``deepseek_v4`` is not AutoConfig-loadable outside the reference package, so return the hand-built
        ``DeepseekV4Config`` (Flash dims + the checkpoint's compress_ratios schedule). ``PREFILL_HF_MODEL`` is
        honoured only for the field the runner needs from it later (weights); the schedule is data here."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", self.default_max_seq_len))
        return deepseek_v4_flash_hf_config(max_seq=max_seq)

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """Mirror the MLA family: ``$PREFILL_TTNN_CACHE / {name}_{arch}_{N}dev / {sp}x{tp}``."""
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        import ttnn
        from models.common.utility_functions import is_blackhole

        arch = "bh" if is_blackhole() else "wh"
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_{arch}_{ttnn.get_num_devices()}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """The KV groups for this rank's layer slice (``tt/v4/kv_cache.py``); the engine owns them."""
        from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches

        caches = allocate_v4_flash_kv_caches(mesh_device=mesh_device, hf_config=hf_config, params=params)
        logger.info(
            f"[dsv4-flash] KV groups for layers {params.first_layer_idx}..{params.first_layer_idx + params.num_layers - 1}: "
            f"{ {g: tuple(t.shape) for g, t in caches.group_tensors().items()} }"
        )
        return caches

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """The pure-ttnn V4 runtime over this rank's layer slice, weights from the checkpoint (``PREFILL_HF_MODEL``,
        dequantised per layer through ``tt/v4/weights``) or from the .tensorbin cache once it exists."""
        from models.demos.deepseek_v3_d_p.tt.v4.runtime import TtV4PrefillRuntime, TtV4PrefillRuntimeConfig
        from models.demos.deepseek_v3_d_p.tt.v4.weights import hf_names

        model_dir = os.environ.get("PREFILL_HF_MODEL", self.hf_model_default)
        weight_map = hf_names.read_weight_map(model_dir)

        def layer_weights(layer_idx: int) -> dict:
            w = hf_names.layer_torch_dict(model_dir, layer_idx, weight_map=weight_map)
            w.pop("__kind__", None)
            w["__experts__"] = [e for _, e in hf_names.iter_layer_experts(model_dir, layer_idx, weight_map=weight_map)]
            return w

        top = hf_names.top_level_torch_dict(model_dir, weight_map=weight_map)
        cfg = TtV4PrefillRuntimeConfig(
            chunk_size=params.chunk_size,
            max_seq_len=params.max_seq_len,
            first_layer_idx=params.first_layer_idx,
            num_layers=params.num_layers,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            num_users=params.num_users,
            mesh_shape=tuple(params.mesh_shape),
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            kv_only_last_layer=params.kv_only_last_layer,
            weight_cache_path=params.weight_cache_path,
        )
        return TtV4PrefillRuntime(
            mesh_device,
            hf_config,
            cfg,
            layer_weights=layer_weights,
            top_level_weights=top,
            num_links=params.num_links,
            dispatch_buffer_capacity_factor=params.capacity_factor,
        )

    # --- test metadata ---
    hf_repo_id = "deepseek-ai/DeepSeek-V4-Flash"
    env_var = "DEEPSEEK_V4_FLASH_HF_MODEL"
    ttnn_cache_env = "TT_DSV4_FLASH_PREFILL_TTNN_CACHE"
    ref_cache_env = "TT_DSV4_FLASH_PREFILL_HOST_REF_CACHE"
    supports_pretrained = True
    tokenizer_trust_remote_code = False
    needs_flat_config_dir = False

    @property
    def config_builder(self) -> Callable:
        return deepseek_v4_flash_hf_config

    @property
    def reference_model_cls(self) -> type:
        from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

        return DeepseekV4ForCausalLM

    @property
    def reference_attention_cls(self) -> Optional[type]:
        from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention

        return DeepseekV4Attention

    @property
    def reference_moe_cls(self) -> Optional[type]:
        from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

        return DeepseekV4SparseMoeBlock
