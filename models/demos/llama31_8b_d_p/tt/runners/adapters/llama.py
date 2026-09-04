# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``LlamaPrefillAdapter`` — the common/prefill engine <-> Llama-3.1-8B boundary.

Implements §1 of ``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md``: a factory +
descriptor that says where the config / weights / golden trace live, allocates the packed GQA K/V
cache, and builds the ``TtPrefillRuntime``. It performs no device work, issues no comms, and does
**not** hold the cache — the engine owns all three (``adapter.py:96``).

Template: ``models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:41``. Llama-3.1-8B is dense GQA,
so this subclasses ``PrefillModelAdapter`` directly rather than the DeepSeek family's
``MLAPrefillAdapter``. Deletions vs the template: every MoE knob (``default_gate_mode`` is left at
the base-class value because the base declares it, but nothing in this package reads it — Llama-3.1
has no router), and ``supports_dflash`` stays ``False`` (the drafter is a Kimi-only checkpoint,
``adapter.py:133``).

**Import-light, and measured** (``DEC-102``). Module scope pulls stdlib + ``loguru`` + the engine's
own light ``adapter`` module + this package's zero-import ``tt/model_dims.py``, and **nothing else** —
no ``ttnn``, no ``torch``, no ``transformers``, no reference model, no ``tt/`` device module. That is
not a style preference:

* the H2D producer imports the selected adapter in a process that never opens a device;
* ``models/demos/deepseek_v3_d_p/tests/conftest.py:33`` builds ``TEST_VARIANTS`` by calling
  ``get_adapter(name)`` for **every** entry in ``ADAPTER_PATHS`` at collection time, so registering
  this adapter puts its import cost — and any import-time failure — into that whole suite.

Every heavy import is therefore inside the method that needs it. ``G-ADAPTER`` measures the
difference: this module imports in ~0.05 s against the template's ~2.4 s.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.llama31_8b_d_p.tt.model_dims import Llama31_8BConfig

# Package root: .../models/demos/llama31_8b_d_p (this file is tt/runners/adapters/llama.py).
_PKG_ROOT = Path(__file__).resolve().parents[3]

# The cache dtype the whole package is gated at (`DEC-017`): `tt/attention/dense_sp.py` asserts a
# bf8_b cache on the chunked ring path, so bfloat16 is a measurement-only mode that cannot ship.
# Named here as a string so this module still imports without ttnn; resolved in build_runtime.
_CACHE_DTYPE_NAME = "bfloat8_b"
_WEIGHT_DTYPE_NAME = "bfloat8_b"
# `ModelArgs.weight_cache_path` tags the cache directory with this (`_DTYPE_TAG`, `model_config.py`).
_WEIGHT_DTYPE_TAG = "bfp8"


@dataclass
class LlamaKvCaches(KvCaches):
    """Concrete ``KvCaches`` for Llama-3.1-8B: a one-element list holding the single
    :class:`~models.demos.llama31_8b_d_p.tt.attention.kv_cache.LlamaKVCache` (``.k`` / ``.v``).

    A list, not the bare cache, because ``TtPrefillRuntime._resolve_kv`` accepts either and the
    engine treats the handle as opaque — index 0 is the whole contract. Dense GQA has exactly one
    migratable cache pair, so there is no index 1 (contrast MiniMax-M3's k/v/index_k).
    """

    caches: list

    def __getitem__(self, idx):
        return self.caches[idx]

    def __len__(self):
        return len(self.caches)


class LlamaPrefillAdapter(PrefillModelAdapter):
    """Llama-3.1-8B-Instruct prefill adapter (dense GQA, full RoPE with llama3 scaling, no MoE)."""

    # --- identity & runner defaults ---
    name = "llama31_8b_d_p"
    model_config = Llama31_8BConfig
    # config.json dir; PREFILL_HF_MODEL overrides. Defaults to the config BUNDLED IN THIS PACKAGE
    # rather than to a checkpoint path: dimensions must not depend on a machine-local directory, and
    # `DEC-005` already asserts this copy is byte-identical to the tt_transformers one. The
    # checkpoint (weights) is a separate input, read from HF_MODEL by `ModelArgs`.
    hf_model_default = str(_PKG_ROOT / "configs" / "Llama-3.1-8B-Instruct")
    # "" => no override; `weight_cache_path` then falls back to the same root order `ModelArgs`
    # uses, so the runner reads the tree P8's cache-populate run wrote (`DEC-048`).
    ttnn_cache_default = ""
    # "" => the golden trace must come from PREFILL_TRACE_DIR (`DEC-057`: no package-local env var,
    # and no default path baked into the repo pointing at one developer's home directory).
    prefill_trace_default = ""

    # --- device / mesh knobs ---
    # No op in this package routes its semaphores to L1_SMALL (`tt/ccl.py` allocates global
    # semaphores on the default L1 grid), so nothing is carved out at mesh-open time.
    l1_small_size = 0
    routing_use_l1_small_for_semaphores = False
    # The cross-rank D2D hidden state: emb TP-sharded. Matches `DEC-018` scheme A, in which the
    # decoder layer's residual stays full-width but the activation handed across a pipeline boundary
    # is the TP-sharded one. Single-rank runs never build a D2D socket, so this is untested here —
    # it is the multi-rank assumption recorded in `R-040`.
    pipeline_activation_emb_tp_sharded = True
    # The DFlash drafter is a separate checkpoint targeting one architecture (Kimi only, today), so
    # PREFILL_DFLASH=1 must not attach it to Llama (`adapter.py:133`).
    supports_dflash = False

    # --- test metadata (consumed by the registry-fed `variant` fixture) ---
    hf_repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    env_var = "HF_MODEL"
    num_layers_to_download = 32
    # Llama-3.1 ships a stock fast tokenizer and a stock `llama` model_type, so neither the
    # trust_remote_code path nor the flat-config copy is needed (both exist for Kimi's BBPE and for
    # model types AutoConfig cannot load; `adapter.py:217`, `:222`).
    tokenizer_trust_remote_code = False
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"
    # P7/P8 gate thresholds for this package's KV product, for anyone wiring pytest coverage off the
    # adapter. Sources: `06_GATES.md` G-KV-TP8 / G-MESH-KV (min K 0.99789 one-shot, 0.99646 chunked).
    mla_pcc_threshold = 0.99
    moe_pcc_threshold = 0.99

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        """Load + normalise the Llama config into a :class:`RuntimeLlamaHFConfig`.

        Two things this must get right, both silent-wrongness traps:

        1. **It returns the normalised object, never a raw dict or a ``transformers``
           ``LlamaConfig``** (``DEC-009``). On transformers 5.12.1 a ``LlamaConfig`` has no
           ``.rope_theta`` at all — the value moved into ``rope_parameters`` — so the
           ``getattr(cfg, "rope_theta", DEFAULT)`` idiom the template uses silently substitutes the
           wrong theta, giving a RoPE that is wrong at every position with no exception anywhere
           (``R-014``, Appendix F.2). ``llama_hf_config`` resolves theta and the llama3 scaling in
           one place and asserts the limb factors.
        2. **The returned object must accept ``max_seq_len``.** The runner stamps it on the very
           next line (``prefill_runner.py:475``) and ``LlamaHFConfig`` is frozen, so this returns
           the ``RuntimeLlamaHFConfig`` subclass (``DEC-100``).

        Reads ``config.json`` straight off disk rather than through ``AutoConfig`` — no
        ``transformers`` import, no ``trust_remote_code``, and the file the package's own gates read.
        """
        import json

        from models.demos.llama31_8b_d_p.tt.model_config import runtime_llama_hf_config

        model_path = Path(os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default)
        config_json = model_path / "config.json"
        assert config_json.is_file(), (
            f"no config.json at {config_json}. PREFILL_HF_MODEL must name a directory holding the "
            f"model's config.json (the checkpoint dir works); unset it to use the config bundled "
            f"with this package ({self.hf_model_default})."
        )
        logger.info(f"Loading HF config for model={self.name!r} from {config_json}")
        with open(config_json) as fh:
            cfg = runtime_llama_hf_config(json.load(fh))
        logger.info(
            f"[{self.name}] config: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
            f"heads={cfg.num_attention_heads}q/{cfg.num_key_value_heads}kv head_dim={cfg.head_dim} "
            f"rope_theta={cfg.rope_theta} rope_type={cfg.rope_type} scaling={cfg.rope_scaling_factor}"
        )
        return cfg

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """The TTNN weight-cache dir for this model + mesh, mirroring what the populate run wrote.

        Must return **exactly** what ``ModelArgs.weight_cache_path(ttnn.bfloat8_b)`` returns, or the
        runner rebuilds a 7.9 GB cache the package already has:

            ``<root>/llama31_8b_d_p_<arch>_<N>dev/<rows>x<cols>/tensor_cache_bfp8``

        Root order: ``PREFILL_TTNN_CACHE`` (the engine's knob, first so a deployment can redirect it)
        then ``ModelArgs``' own order — ``LLAMA31_8B_TTNN_CACHE``, ``TT_CACHE_PATH``,
        ``$HF_MODEL/ttnn_cache``. ``tests/unit/test_prefill_adapter.py`` asserts the two agree.

        Both path segments below the root are mandatory and neither is cosmetic (``DEC-048``,
        ``R-017``): ``ttnn.as_tensor`` caches the **already-sharded per-device** tensor, so a cache
        written at ``(1,1)`` and replayed at ``(4,8)`` hands every chip the full unsharded weight and
        nothing downstream notices; and bf8_b/bf16 tensors get distinct ttnn filename suffixes, so
        sharing one directory would silently rebuild rather than conflict.

        ``N`` is ``sp * tp``, **not** ``ttnn.get_num_devices()``: the runner calls this from
        ``_print_config`` before ``open_mesh_device``, and with co-located migration workers
        ``GetNumAvailableDevices`` can throw (the template's note at ``gpt_oss.py:80``).
        """
        # Lazy: `is_blackhole` lives behind `models.common.utility_functions`, which imports ttnn.
        from models.common.utility_functions import is_blackhole

        root = (
            os.environ.get("PREFILL_TTNN_CACHE")
            or self.ttnn_cache_default
            or os.environ.get("LLAMA31_8B_TTNN_CACHE")
            or os.environ.get("TT_CACHE_PATH")
        )
        if not root:
            hf_model = os.environ.get("HF_MODEL")
            if not hf_model:
                logger.warning(
                    f"[{self.name}] no weight-cache root (PREFILL_TTNN_CACHE / LLAMA31_8B_TTNN_CACHE / "
                    f"TT_CACHE_PATH / HF_MODEL all unset): weights will be tilized from the checkpoint "
                    f"on every run and nothing will be cached."
                )
                return None
            root = Path(hf_model) / "ttnn_cache"

        arch = "bh" if is_blackhole() else "wh"
        rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
        path = (
            Path(root) / f"{self.name}_{arch}_{rows * cols}dev" / f"{rows}x{cols}" / f"tensor_cache_{_WEIGHT_DTYPE_TAG}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        """Allocate + zero the packed GQA K/V cache and hand the engine an opaque handle.

        One shared cache pair of ``num_users * num_layers`` user-major slots, per-chip shape
        ``[num_users*num_layers, 1, seq_local, head_dim]``: **one KV head per chip** (the 8 KV heads
        land on the 8 TP columns at write time) with the sequence SP-sharded block-cyclic on the
        rows. Dense GQA has no second cache, so index 0 is the whole handle.

        ``TP`` must equal ``num_key_value_heads`` (8). That is a property of *this* layout, not a
        preference — the per-chip cache is allocated with a head dim of exactly 1
        (``tt/attention/kv_cache.py:130``), so at TP<8 attention produces ``8/TP`` local heads and
        ``update_padded_kv_cache`` dies with a C++ ``TT_FATAL``. Checked here, before any allocation,
        so the message names the knob (``R-027``); ``TtPrefillRuntime._resolve_kv`` re-checks it at
        every use.
        """
        from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache

        n_kv = hf_config.num_key_value_heads
        assert params.tp_factor == n_kv, (
            f"[{self.name}] the packed KV cache holds exactly ONE KV head per chip, so the TP axis "
            f"must be {n_kv} wide (num_key_value_heads). Got TP={params.tp_factor} from "
            f"mesh_shape={tuple(params.mesh_shape)} (tp_axis={params.tp_axis}) — set PREFILL_TP={n_kv}. "
            f"Left to run, update_padded_kv_cache would abort with 'cache and input num-heads dim "
            f"must match' from C++ (R-027)."
        )
        cache_dtype = self._cache_dtype()
        logger.info(
            f"[{self.name}] allocating KV cache: layers={params.num_layers} users={params.num_users} "
            f"max_seq_len={params.max_seq_len} head_dim={hf_config.head_dim} sp_axis={params.sp_axis} "
            f"sp={params.sp_factor} dtype={cache_dtype}"
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
                    cache_dtype=cache_dtype,
                )
            ]
        )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Build this rank's model and return the ``TtPrefillRuntime`` (contract §2).

        Every per-run knob comes off ``params``; nothing here re-reads an engine ``PREFILL_*``
        variable. Two inputs genuinely are not on ``PrefillRunParams`` and are read from the
        environment, both already this package's documented variables rather than new ones:
        ``HF_MODEL`` (the checkpoint directory, read by ``ModelArgs``) and
        ``LLAMA_WEIGHTS_FROM_CACHE`` (skip the safetensors read and rebuild from the tilized weight
        cache — proven byte-identical at ``(4,8)`` by ``G-WEIGHTS``/``R-017``).

        Three config values are decided here rather than read, each with a reason:

        * ``sequence_parallel = params.sp_factor > 1`` — derived, not a knob. SP prefill and the
          chunked cache-read path are the same switch (``DEC-056``), and the runtime refuses
          ``sequence_parallel=True`` at ``sp == 1`` outright, so there is exactly one correct value
          per mesh and a knob could only be set wrong.
        * ``owns_kv_cache=False`` — the engine allocated the cache via
          :meth:`allocate_kv_cache` and passes it into every call (``DEC-055``).
        * ``cache_dtype``/``weight_dtype`` ``bfloat8_b`` — ``DEC-017``; the ring path asserts a bf8_b
          cache, so this is forced rather than chosen.
        """
        import ttnn
        from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs
        from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        # PREFILL_TOPOLOGY is this package's existing knob (tests/galaxy_prefill_kv_pcc.py) and the
        # template's (gpt_oss.py:140); `linear` runs a pod without torus wraparound. It must agree
        # with PREFILL_FABRIC_MODE — Ring collectives need FABRIC_1D_RING plus the torus mesh-graph
        # descriptor (`DEC-020` / `DEC-081`), and a Ring topology on a FABRIC_1D fabric hangs rather
        # than erroring.
        linear = os.getenv("PREFILL_TOPOLOGY", "ring").strip().lower() == "linear"
        topology = ttnn.Topology.Linear if linear else ttnn.Topology.Ring

        sequence_parallel = params.sp_factor > 1
        if params.max_seq_len <= params.chunk_size and sequence_parallel:
            # Not an error: it selects the SP bootstrap instead of the ring cache-read (`DEC-021`),
            # which is a correct-but-different attention core. Said out loud because it silently
            # changes which code path every gate below is measuring.
            logger.warning(
                f"[{self.name}] max_seq_len ({params.max_seq_len}) <= chunk_size "
                f"({params.chunk_size}): the per-chip cache shard leaves the ring op no room, so "
                f"attention runs the one-shot SP bootstrap, not the chunked cache-read (DEC-021). "
                f"Set PREFILL_MAX_SEQ_LEN strictly greater than PREFILL_CHUNK_SIZE to exercise the "
                f"served path."
            )
        if params.kv_only_last_layer:
            # PREFILL_KV_ONLY_LAST_LAYER defaults to 1 in the engine. It is a DeepSeek-family
            # optimisation (build the last block KV-only, skipping its MLP and the tail) that this
            # package does not implement. Prefill's product is the KV cache either way — the LM head
            # is already never built (`with_lm_head=False`) — so ignoring it costs one layer's MLP
            # per chunk and changes no output. Logged, not asserted, so the engine default runs.
            logger.info(
                f"[{self.name}] params.kv_only_last_layer=True is NOT implemented by this runtime: "
                f"the last layer runs its MLP as usual. Correctness is unaffected (the LM head is "
                f"never built); the cost is one MLP per chunk. See DEC-104."
            )

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=tuple(params.mesh_shape),
            # The engine's contract name is `chunk_size`; ours names the DEFAULT of a multi-size
            # runtime and exposes `.chunk_size` as a property alias (`DEC-054`).
            default_chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            topology=topology,
            cache_dtype=self._cache_dtype(),
            weight_dtype=getattr(ttnn, _WEIGHT_DTYPE_NAME),
            weight_cache_path=params.weight_cache_path,
            owns_kv_cache=False,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
            sequence_parallel=sequence_parallel,
        )

        if os.getenv("LLAMA_WEIGHTS_FROM_CACHE") == "1":
            assert params.weight_cache_path is not None, (
                "LLAMA_WEIGHTS_FROM_CACHE=1 but no weight cache path resolved; the model would be "
                "built from an empty state_dict with nothing to read. Set PREFILL_TTNN_CACHE / "
                "LLAMA31_8B_TTNN_CACHE / TT_CACHE_PATH, or unset LLAMA_WEIGHTS_FROM_CACHE."
            )
            logger.info(
                f"[{self.name}] LLAMA_WEIGHTS_FROM_CACHE=1 -> empty state_dict; loading the tilized "
                f"weight cache at {params.weight_cache_path}"
            )
            state_dict = {}
        else:
            model_args = ModelArgs(mesh_device=mesh_device, hf_config=hf_config)  # reads HF_MODEL
            logger.info(f"[{self.name}] loading the real bf16 checkpoint from {model_args.weights_path} (slow)...")
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        runtime = TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            config=runtime_config,
        )
        del state_dict
        return runtime

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cache_dtype(self):
        """``ttnn.bfloat8_b`` — resolved lazily so importing this module never imports ttnn."""
        import ttnn

        return getattr(ttnn, _CACHE_DTYPE_NAME)

    @property
    def reference_model_cls(self):
        """The torch reference the pytest fixtures compare against. Imported lazily and never at
        module scope — this is exactly the import the contract's checklist item 3 forbids eagerly."""
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM

    @property
    def reference_attention_cls(self):
        from transformers.models.llama.modeling_llama import LlamaAttention

        return LlamaAttention

    @property
    def reference_rotary_cls(self):
        """transformers >= 5 computes rope at the MODEL level and passes ``position_embeddings``
        down, so a standalone reference attention has to be handed ``(cos, sin)`` (``adapter.py:259``).
        This package runs transformers 5.12.1, so it is required, not optional."""
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        return LlamaRotaryEmbedding
