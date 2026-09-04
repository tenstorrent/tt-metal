# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``Llama31_8BPrefillAdapter`` — the common/prefill engine <-> Llama 3.1 8B boundary.

Copied from ``gpt_oss_d_p/tt/runners/adapters/gpt_oss.py``. Llama is GQA (not MLA), so this
subclasses ``PrefillModelAdapter`` directly rather than the DeepSeek-family ``MLAPrefillAdapter``.
Pure glue: it says where the config / weights / golden trace live, allocates the K/V cache, and
builds the ``TtPrefillRuntime``. All operational behavior lives on the runtime; the engine drives it
and owns both the comms and the cache lifetime.

Import-light on purpose: the model stack, ttnn and transformers are imported lazily INSIDE the
methods, because the H2D producers import this module too and must not pull a device stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.llama3_1_8b_d_p.reference.config import CONFIG_DIR, Llama31_8BConfig


@dataclass
class LlamaKvCaches(KvCaches):
    """Concrete ``KvCaches`` for Llama: a list holding the single ``LlamaKVCache`` (k+v struct).

    Indexable so the runtime's ``_resolve_kv`` pulls the cache back via ``[0]`` while the engine keeps
    treating it as an opaque handle.
    """

    caches: list

    def __getitem__(self, idx):
        return self.caches[idx]


class Llama31_8BPrefillAdapter(PrefillModelAdapter):
    """Llama 3.1 8B prefill adapter (GQA + dense SwiGLU, single pipeline stage)."""

    # --- identity & runner defaults ---
    name = "llama3_1_8b_d_p"
    model_config = Llama31_8BConfig
    hf_model_default = str(CONFIG_DIR)  # vendored config.json dir; PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides (empty => none)
    prefill_trace_default = ""  # golden trace dir; PREFILL_TRACE_DIR overrides
    # Dense model: no MoE gate, so the gate mode is inert here. Left at the base default rather than
    # invented, since the engine reads the attribute unconditionally.
    default_gate_mode = "DEVICE_FP32"

    # Dense: any layer split is valid, so layer_split_boundaries stays None (the base default).
    # Single stage anyway (spec topology.pipeline_stages == 1).

    # --- test metadata ---
    hf_repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    env_var = "LLAMA_3_1_8B_HF_MODEL"
    num_layers_to_download = Llama31_8BConfig.NUM_LAYERS
    tokenizer_trust_remote_code = False  # stock Llama tokenizer; no repo-shipped tokenizer code
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"  # spec acceptance.trace_layout

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        """Load the config from PREFILL_HF_MODEL, falling back to the vendored config.json dir.

        Uses this package's own ``LlamaConfig`` rather than ``AutoConfig`` so the config path needs
        neither ``transformers`` nor a checkpoint, and so every consumer reads the same object the
        constants test pins.
        """
        from models.demos.llama3_1_8b_d_p.reference.config import LlamaConfig

        model_path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
        logger.info(f"Loading config for model={self.name!r} from {model_path}")
        return LlamaConfig.from_json(Path(model_path) / "config.json")

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """``$PREFILL_TTNN_CACHE / {name}_{arch}_{N}dev / {sp}x{tp}``; None if explicitly empty.

        ``N`` is ``sp * tp``, NOT ``ttnn.get_num_devices()``: the runner calls this before
        ``open_mesh_device``, and with co-located migration workers the device count query can throw.
        The layout must mirror what the cache-populate run wrote — the two do not share a code path,
        only this convention, and a mismatch silently misses the cache (the model still runs, slowly).
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
        """Allocate the GQA K/V cache: ``num_users * num_layers`` user-major slots, KV heads on the TP
        cols (2 per chip at TP=4), sequence SP block-cyclic on the rows.

        Dense GQA has no auxiliary cache, so the returned handle holds exactly one ``LlamaKVCache`` at
        index 0. Migration describes it with the multi-config table in ``tt/runners/kv_chunk_table.py``
        (configs 0..7 = K heads, 8..15 = V heads).
        """
        from models.demos.llama3_1_8b_d_p.tt.attention import allocate_kv_cache

        tp = params.mesh_shape[params.tp_axis]
        n_kv_local = hf_config.num_key_value_heads // tp
        assert n_kv_local >= 1 and hf_config.num_key_value_heads % tp == 0, (
            f"TP={tp} must divide num_key_value_heads={hf_config.num_key_value_heads}; "
            f"KV-head replication is not implemented (spec known_risks)"
        )
        return LlamaKvCaches(
            [
                allocate_kv_cache(
                    mesh_device,
                    num_layers=params.num_layers,
                    max_seq_len=params.max_seq_len,
                    sp_axis=params.sp_axis,
                    num_users=params.num_users,
                    head_dim=hf_config.head_dim,
                    num_kv_heads_local=n_kv_local,
                )
            ]
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Build the model + runtime for this rank.

        ``owns_kv_cache=False``: the engine already allocated the cache via ``allocate_kv_cache`` and
        passes it into every call, so the runtime stays stateless with respect to it.
        """
        import ttnn
        from models.demos.llama3_1_8b_d_p.tt.model_config import ModelArgs
        from models.demos.llama3_1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

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
            # Default LINEAR: a plain-MESH Galaxy has no wraparound links. PREFILL_TOPOLOGY=ring on a
            # torus pod. This must match the fabric config the mesh was opened with.
            topology=(
                ttnn.Topology.Ring if os.getenv("PREFILL_TOPOLOGY", "linear") == "ring" else ttnn.Topology.Linear
            ),
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
        )

        if os.getenv("LLAMA_WEIGHTS_FROM_CACHE") == "1":
            state_dict = {}
        else:
            model_args = ModelArgs(mesh_device=mesh_device, max_seq_len=params.max_seq_len)
            logger.info("Loading real bf16 Llama 3.1 8B weights (safetensors read)...")
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            config=runtime_config,
        )

    # ------------------------------------------------------------------
    # Test-only reference handles (lazy — never imported at module load)
    # ------------------------------------------------------------------
    @property
    def reference_model_cls(self):
        from models.demos.llama3_1_8b_d_p.reference.model import LlamaModel

        return LlamaModel

    @property
    def reference_attention_cls(self):
        from models.demos.llama3_1_8b_d_p.reference.model import LlamaAttention

        return LlamaAttention
