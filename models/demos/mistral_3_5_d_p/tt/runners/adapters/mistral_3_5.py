# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``MistralPrefillAdapter`` — the common/prefill engine <-> Mistral-Medium-3.5 boundary.

Adapted from ``gpt_oss_d_p/tt/runners/adapters/gpt_oss.py``. Mistral is dense GQA (not MLA), so this
subclasses ``PrefillModelAdapter`` DIRECTLY rather than the DeepSeek-family ``MLAPrefillAdapter``.
Pure glue: it says where the config / weights / golden trace live, allocates the GQA K/V cache, and
builds the ``TtPrefillRuntime``. All operational behaviour (running a chunk, reading/writing the
cache, PCC) lives on the runtime; the engine drives it and owns the comms and the cache lifetime.

Import-light by contract: the model stack (runtime, transformers, the KV cache) is imported LAZILY
inside the methods, because the H2D producers import this module too and must not pull a device
stack in.

Dropped from the donor: ``default_gate_mode`` (an MoE knob — Mistral is dense) and the expert dtype.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.mistral_3_5_d_p.reference.mistral_config import CONFIG_DIR, MistralMedium35Config


@dataclass
class MistralKvCaches(KvCaches):
    """Concrete ``KvCaches`` for Mistral: a list holding the single ``MistralKVCache`` (the k+v
    struct), indexable so the runtime's ``_resolve_kv`` pulls the cache back via ``[0]`` while the
    engine keeps treating it as opaque.

    NOTE the two similarly-named types, kept distinct exactly as in the donor:
    ``MistralKVCache`` (``tt/attention/kv_cache.py``) is the TENSOR struct; ``MistralKvCaches`` here
    is the ENGINE handle.
    """

    caches: list

    def __getitem__(self, idx):
        return self.caches[idx]


class MistralPrefillAdapter(PrefillModelAdapter):
    """Mistral-Medium-3.5-128B prefill adapter (dense GQA + dense SwiGLU, 88 layers)."""

    # --- identity & runner defaults ---
    name = "mistral_3_5_d_p"
    model_config = MistralMedium35Config
    hf_model_default = str(CONFIG_DIR)  # config.json dir; PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides (empty => none)
    prefill_trace_default = ""  # golden trace dir; PREFILL_TRACE_DIR overrides

    # --- test metadata ---
    hf_repo_id = MistralMedium35Config.HF_REPO_ID
    env_var = "MISTRAL_35_HF_MODEL"
    num_layers_to_download = MistralMedium35Config.NUM_LAYERS
    tokenizer_trust_remote_code = True
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"
    ref_cache_env = "MISTRAL_35_HOST_REF_CACHE"

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        """The text backbone's config, unwrapped from the ``Mistral3ForConditionalGeneration``
        wrapper (the vision tower is out of scope for prefill)."""
        from transformers import AutoConfig

        model_path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
        logger.info(f"Loading HF config for model={self.name!r} from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return getattr(cfg, "text_config", cfg)

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """``$PREFILL_TTNN_CACHE / {name}_{arch}_{N}dev / {sp}x{tp}``; None when explicitly empty.

        ``N`` is ``sp * tp``, NOT ``ttnn.get_num_devices()``: the runner calls this from
        ``_print_config`` before ``open_mesh_device``, and with co-located migration workers
        ``GetNumAvailableDevices`` can throw and abort the run (donor note, kept).
        """
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        arch = "bh" if is_blackhole() else "wh"
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_{arch}_{int(sp) * int(tp)}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # KV cache + runtime build
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        """Allocate the dense-GQA K/V cache: one shared cache of ``num_users * num_layers``
        user-major slots, K/V heads on the TP cols, sequence SP-sharded block-cyclic.

        Dense GQA has no secondary cache, so the returned handle holds just the one
        ``MistralKVCache`` at index 0. Migration uses the multi-config table in
        ``tt/runners/kv_chunk_table.py`` (config 0..N-1 = k heads, N..2N-1 = v heads).

        ``params.max_seq_len`` is the capacity the ENGINE asked for. It must be a whole number of
        chunks, because the whole-cache indexed rope and the address table both tile by chunk; the
        engine's own max_seq_len is already a multiple of its chunk_size, so this rounds up only in
        the degenerate case and says so.
        """
        from models.demos.mistral_3_5_d_p.tt.attention import allocate_kv_cache as allocate

        capacity = params.max_seq_len
        if capacity % params.chunk_size:
            capacity = -(-capacity // params.chunk_size) * params.chunk_size
            logger.warning(
                f"[{self.name}] PREFILL_MAX_SEQ_LEN={params.max_seq_len} is not a whole number of "
                f"chunk_size={params.chunk_size}; allocating {capacity} so the block-cyclic rope and "
                f"the KV address table tile the cache"
            )
        return MistralKvCaches(
            [
                allocate(
                    mesh_device,
                    num_layers=params.num_layers,
                    max_seq_len=capacity,
                    sp_axis=params.sp_axis,
                    num_users=params.num_users,
                    head_dim=hf_config.head_dim,
                )
            ]
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Build the model + runtime for this rank.

        The runtime is stateless w.r.t. the KV cache (``owns_kv_cache=False``): the engine allocated
        it via :meth:`allocate_kv_cache` and passes it into every call.
        """
        import ttnn

        from models.demos.mistral_3_5_d_p.tt.model_config import ModelArgs
        from models.demos.mistral_3_5_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        capacity = params.max_seq_len
        if capacity % params.chunk_size:
            capacity = -(-capacity // params.chunk_size) * params.chunk_size

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=capacity,
            # The engine's max_seq_len is the servable context; the capacity above may exceed it.
            servable_seq_len=min(params.max_seq_len, capacity),
            mesh_shape=params.mesh_shape,
            default_chunk_size=params.chunk_size,  # the engine's contract names it chunk_size
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            weight_cache_path=params.weight_cache_path,
            owns_kv_cache=False,  # the engine owns the cache and passes it into every call
            # This pod is a plain grid (no torus wrap-around), so Linear is the default; a
            # torus-wired galaxy sets PREFILL_TOPOLOGY=ring.
            topology=(
                ttnn.Topology.Ring if os.getenv("PREFILL_TOPOLOGY", "linear") == "ring" else ttnn.Topology.Linear
            ),
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
        )

        if os.getenv("MISTRAL_WEIGHTS_FROM_CACHE") == "1":
            logger.info("MISTRAL_WEIGHTS_FROM_CACHE=1 -> empty state_dict (weights load from the tilized cache)")
            state_dict = {}
        else:
            model_args = ModelArgs(mesh_device=mesh_device, max_seq_len=capacity)
            logger.info("Loading real weights (safetensors read + per-tensor fp8 dequantization) ...")
            state_dict = ModelArgs.load_state_dict(model_args.weights_path, num_layers=params.num_layers)

        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            config=runtime_config,
        )

    # ------------------------------------------------------------------
    # test-only reference handles (lazy)
    # ------------------------------------------------------------------
    @property
    def reference_model_cls(self):
        from transformers.models.ministral3 import modeling_ministral3

        return modeling_ministral3.Ministral3ForCausalLM

    @property
    def reference_attention_cls(self):
        from transformers.models.ministral3 import modeling_ministral3

        return modeling_ministral3.Ministral3Attention

    @property
    def reference_rotary_cls(self):
        """Ministral3 computes rope at the MODEL level and passes ``position_embeddings`` down, so a
        standalone attention has to be handed them (transformers >= 5 shape)."""
        from transformers.models.ministral3 import modeling_ministral3

        return modeling_ministral3.Ministral3RotaryEmbedding
