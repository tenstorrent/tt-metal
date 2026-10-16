# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Llama 3.1 8B prefill adapter — the common/prefill engine <-> Llama boundary.

Llama 3.1 8B is GQA (not MLA), so this subclasses ``PrefillModelAdapter`` directly.
Pure glue: says where config/weights/golden live, allocates the GQA KV cache with
n_kv_local=2, and builds the ``TtPrefillRuntime``.

Import-light: heavy model imports are deferred to the method bodies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.llama_3_1_8b.config import Llama31ModelConfig


@dataclass
class LlamaKvCachesWrapper(KvCaches):
    """Concrete KvCaches for Llama: wraps a single LlamaKVCache, indexable so the runtime's
    _resolve_kv pulls the cache back via [0]."""

    caches: list

    def __getitem__(self, idx):
        return self.caches[idx]


class Llama31PrefillAdapter(PrefillModelAdapter):
    """Llama 3.1 8B Instruct prefill adapter (GQA + Llama3 RoPE + dense SwiGLU MLP)."""

    name = "llama_3_1_8b"
    model_config = Llama31ModelConfig
    hf_model_default = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"
    ttnn_cache_default = ""
    prefill_trace_default = ""
    default_gate_mode = "DEVICE_FP32"

    hf_repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    env_var = "LLAMA_HF_MODEL"
    num_layers_to_download = 32
    tokenizer_trust_remote_code = False
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"

    def load_hf_config(self):
        from transformers import AutoConfig

        model_path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
        logger.info(f"Loading HF config for model={self.name!r} from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path)
        return getattr(cfg, "text_config", cfg)

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        arch = "bh" if is_blackhole() else "wh"
        sp, tp = mesh_shape
        num_devices = int(sp) * int(tp)
        path = Path(env_cache) / f"{self.name}_{arch}_{num_devices}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        from models.demos.llama_3_1_8b.tt.attention.kv_cache import allocate_kv_cache

        return LlamaKvCachesWrapper(
            [
                allocate_kv_cache(
                    mesh_device,
                    num_layers=params.num_layers,
                    max_seq_len=params.max_seq_len,
                    sp_axis=params.sp_axis,
                    num_users=params.num_users,
                    head_dim=hf_config.head_dim,
                )
            ]
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        import ttnn
        from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            default_chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            weight_cache_path=params.weight_cache_path,
            owns_kv_cache=False,
            topology=(
                ttnn.Topology.Linear if os.getenv("PREFILL_TOPOLOGY", "ring") == "linear" else ttnn.Topology.Ring
            ),
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
        )

        if os.getenv("LLAMA_WEIGHTS_FROM_CACHE") == "1":
            state_dict = {}
        else:
            state_dict = _load_safetensors_state_dict(os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default)

        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            config=runtime_config,
        )

    @property
    def reference_model_cls(self):
        from models.demos.llama_3_1_8b.reference.model import LlamaForCausalLM

        return LlamaForCausalLM


def _load_safetensors_state_dict(model_path: str) -> dict:
    """Load state_dict from safetensors files in the model directory."""
    from safetensors.torch import load_file

    model_dir = Path(model_path)
    state_dict = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        logger.info(f"Loading weights from {f.name}...")
        state_dict.update(load_file(str(f)))
    assert state_dict, f"No safetensors files found in {model_dir}"
    return state_dict
