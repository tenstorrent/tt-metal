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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig


@dataclass
class Llama31KvCaches(KvCaches):
    """The engine's opaque handle to this model's KV cache.

    Llama-3.1-8B is full-causal in all 32 layers, so unlike GPT-OSS there is no second,
    sliding-window cache: the list holds exactly one ``Llama31KVCache`` (a ``.k`` / ``.v`` pair) and
    the engine only ever reaches it through ``[0]``.
    """

    caches: list

    def __getitem__(self, idx):
        return self.caches[idx]


class Llama31PrefillAdapter(PrefillModelAdapter):
    """Llama-3.1-8B prefill adapter (dense GQA, group 4; no MoE, no sinks, no sliding window)."""

    # --- identity & runner defaults ---
    name = "llama_3p1_8b"
    model_config = Llama31_8BConfig
    # Repo-bundled config.json (no mount, no network needed for the config path), as gpt_oss_d_p
    # and kimi_k2_6 do. The WEIGHTS path is separate -- see tt/model_config.py.
    # PREFILL_HF_MODEL overrides, and when it does it supplies both config and weights.
    hf_model_default = "models/tt_transformers/model_params/Llama-3.1-8B-Instruct"
    ttnn_cache_default = ""  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides (empty => no cache)
    # Golden trace (metadata.json + kv_cache/layer_*.safetensors) for the #4150 per-layer KV check,
    # staged beside the other models' traces; PREFILL_TRACE_DIR overrides. 2048 tokens x 32 layers,
    # K in the **meta** frame (what blaze decode reads) and stored bf16 -- a consumer must round-trip
    # it through bfloat8_b before PCC, which tt_prefill_runtime.kv_cache_pcc_check does.
    prefill_trace_default = "/mnt/models/llama-3.1-8b-prefill-cache/golden/llama31_8b_kv_2048_32L"
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

        Sized by ``params.num_layers``, this **rank's** share, not the 32-layer global count: the
        model addresses its cache rank-locally (``tt/decoder.py``'s ``cache_layer_idx``), so a rank
        holding 8 layers allocates 8 slots per user rather than 32 slots to fill 8 of.
        """
        from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache

        num_kv_heads_per_chip = self._num_kv_heads_per_chip(params)
        logger.info(
            f"Allocating Llama-3.1-8B KV cache: users={params.num_users} layers={params.num_layers} "
            f"(global {params.first_layer_idx}..{params.first_layer_idx + params.num_layers - 1}) "
            f"max_seq_len={params.max_seq_len} sp={params.sp_factor} chunk={params.chunk_size} "
            f"kv_heads/chip={num_kv_heads_per_chip}"
        )
        cache = allocate_kv_cache(
            mesh_device,
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            sp_axis=params.sp_axis,
            num_users=params.num_users,
            chunk_size=params.chunk_size,
            num_kv_heads_per_chip=num_kv_heads_per_chip,
        )
        return Llama31KvCaches(caches=[cache])

    @staticmethod
    def _num_kv_heads_per_chip(params: PrefillRunParams) -> int:
        """KV heads each chip holds: ``8 / tp``, which is 1 at the production TP=8.

        Llama-3.1-8B has 8 KV heads, so at TP=8 every chip owns exactly one and the head dim needs
        no padding. A narrower TP (bring-up on an eight-chip box at TP=2) gives each chip several,
        which the cache supports; a TP wider than 8 would have to replicate heads and is refused
        rather than silently mis-sharded.
        """
        num_kv_heads = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
        if num_kv_heads % params.tp_factor:
            raise ValueError(
                f"tp={params.tp_factor} does not divide the {num_kv_heads} KV heads; Llama-3.1-8B "
                f"supports tp in {sorted(d for d in range(1, num_kv_heads + 1) if num_kv_heads % d == 0)}"
            )
        return num_kv_heads // params.tp_factor

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Build the model + runtime for this rank.

        The runtime is stateless w.r.t. the KV cache (``owns_kv_cache=False``): the engine allocated
        it via ``allocate_kv_cache`` and passes it into every call that touches it.
        """
        import ttnn
        from models.demos.llama_3p1_8b_d_p.tt.model_config import load_llama_state_dict
        from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        runtime_config = TtPrefillRuntimeConfig(
            max_seq_len=params.max_seq_len,
            chunk_size=params.chunk_size,
            mesh_shape=params.mesh_shape,
            num_layers=params.num_layers,
            num_users=params.num_users,
            tp_axis=params.tp_axis,
            num_links=params.num_links,
            # PREFILL_TOPOLOGY=linear runs pods without torus wraparound (same knob as the harness).
            topology=(
                ttnn.Topology.Linear if os.getenv("PREFILL_TOPOLOGY", "ring") == "linear" else ttnn.Topology.Ring
            ),
            weight_cache_path=params.weight_cache_path,
            owns_kv_cache=False,  # the engine owns it (from allocate_kv_cache) and passes it in
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
            vocab_size=int(getattr(hf_config, "vocab_size", Llama31_8BConfig.VOCAB_SIZE)),
        )

        # With a populated TTNN weight cache the safetensors read is pure cost — the cached device
        # tensors are read back by name and the torch weights are never touched.
        if os.getenv("LLAMA31_8B_WEIGHTS_FROM_CACHE") == "1":
            if params.weight_cache_path is None:
                raise ValueError("LLAMA31_8B_WEIGHTS_FROM_CACHE=1 needs a weight cache; set PREFILL_TTNN_CACHE.")
            state_dict = {}
        else:
            logger.info("Loading real bf16 Llama-3.1-8B weights (slow: safetensors read)...")
            state_dict = load_llama_state_dict(num_layers=params.num_layers, first_layer_idx=params.first_layer_idx)

        return TtPrefillRuntime(mesh_device=mesh_device, config=runtime_config, state_dict=state_dict)

    # ------------------------------------------------------------------
    # Test-only reference handles (lazy by contract)
    # ------------------------------------------------------------------
    @property
    def reference_model_cls(self):
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM
