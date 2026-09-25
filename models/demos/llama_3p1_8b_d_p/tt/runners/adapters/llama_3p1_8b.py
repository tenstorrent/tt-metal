# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``Llama31PrefillAdapter`` — the common/prefill engine <-> Llama-3.1-8B boundary.

Llama-3.1-8B is dense GQA, so this subclasses ``PrefillModelAdapter`` DIRECTLY rather than the
DeepSeek-family ``MLAPrefillAdapter`` (which is MLA + MoE). It is pure glue: it says where the
config / weights / golden trace live, allocates the GQA K/V cache, and builds the prefill runtime.
All operational behavior (running a chunk, reading/writing the cache, PCC) lives on the runtime; the
engine drives it and owns all comms + the cache lifetime.

**Import-light by contract.** The H2D producers import this module just to read the registry, so
nothing heavy may be imported at module load: no torch, no ttnn, no transformers, no safetensors, no
reference modeling. Everything heavy is imported lazily inside the methods. ``Llama31_8BConfig`` is
safe to name at class-definition time because it is a pure constants class with no imports of its own.

Scaffold status (tt-blaze#4143): the two self-contained methods are live; ``allocate_kv_cache`` and
``build_runtime`` raise until their dependencies land. Registration in ``ADAPTER_PATHS`` is
deliberately NOT part of this change — it lands with #4149, so a half-built model is not reachable
by name from the runner.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig


class Llama31PrefillAdapter(PrefillModelAdapter):
    """Llama-3.1-8B prefill adapter (dense GQA, group 4; no MoE, no sinks, no sliding window)."""

    # --- identity & runner defaults ---
    name = "llama_3p1_8b"
    model_config = Llama31_8BConfig
    # Repo-bundled config.json (no mount, no network needed for the config path), as gpt_oss_d_p
    # and kimi_k2_6 do. The WEIGHTS path is separate -- see tt/model_config.py.
    # PREFILL_HF_MODEL overrides, and when it does it supplies both config and weights.
    hf_model_default = "models/ttt_compat/model_params/Llama-3.1-8B-Instruct"
    ttnn_cache_default = ""  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides (empty => no cache)
    prefill_trace_default = ""  # golden trace dir (token_ids + KV); PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"  # dense model — no gate; the engine reads this unconditionally

    # --- test metadata ---
    hf_repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    env_var = "LLAMA31_8B_HF_MODEL"
    num_layers_to_download = 32
    tokenizer_trust_remote_code = True
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"

    # The DFlash drafter targets one architecture (Kimi-K2.6/K2.7) and no Llama drafter exists.
    supports_dflash = False

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        """Load and normalize the HF config. ``max_seq_len`` is set on the result by the engine."""
        from transformers import AutoConfig

        from models.demos.llama_3p1_8b_d_p.tt.model_config import cross_check_hf_config

        model_path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
        logger.info(f"Loading HF config for model={self.name!r} from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Defensive unwrap (Llama-3.1-8B is not a VL wrapper, but keep the pattern for variants).
        cfg = getattr(cfg, "text_config", cfg)
        cross_check_hf_config(cfg)
        return cfg

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """$PREFILL_TTNN_CACHE / {name}_{arch}_{N}dev / {sp}x{tp}. None if explicitly empty.

        Use ``sp * tp`` for N — not ``ttnn.get_num_devices()``. The runner calls this from
        ``_print_config`` before ``open_mesh_device``; with co-located migration workers
        ``GetNumAvailableDevices`` can throw ``unordered_map::at`` and abort Gate 2.

        ``is_blackhole`` is imported lazily rather than at module level (gpt-oss imports it at the
        top): it pulls in the device stack, which this module's import-light contract forbids.
        """
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None

        from models.common.utility_functions import is_blackhole

        arch = "bh" if is_blackhole() else "wh"
        sp, tp = mesh_shape
        num_devices = int(sp) * int(tp)
        path = Path(env_cache) / f"{self.name}_{arch}_{num_devices}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # KV cache + runtime build
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        """Allocate the GQA K/V cache and return it as this model's ``KvCaches`` subclass.

        Target layout (tt-blaze#4141): one packed cache per tensor, per chip
        ``[num_users * num_layers, 1, max_seq_len / sp, head_dim]`` bfloat8_b, user-major
        (``slot = user_id * num_layers + layer_idx``), one KV head per chip at TP=8, sequence
        block-cyclic over the SP rows, 32 contiguous tokens per DRAM bank.

        Llama needs only the single packed cache — no bounded sliding-window split, because all 32
        layers are full-causal.
        """
        raise NotImplementedError("Llama-3.1-8B prefill KV cache lands with tt-blaze#4141 (prefill: KV cache).")

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Build the model + runtime for this rank.

        The runtime is stateless w.r.t. the KV cache (``owns_kv_cache=False``): the engine allocated
        it via ``allocate_kv_cache`` and passes it into every call that touches it.
        """
        raise NotImplementedError(
            "Llama-3.1-8B prefill runtime lands with tt-blaze#4148 (prefill: Prefill model) and "
            "#4149 (prefill: Runner integration)."
        )

    # ------------------------------------------------------------------
    # Test-only reference handles (lazy by contract)
    # ------------------------------------------------------------------
    @property
    def reference_model_cls(self):
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM
