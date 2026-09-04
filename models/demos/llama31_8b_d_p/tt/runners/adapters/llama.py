# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`LlamaPrefillAdapter` — the disaggregated-prefill engine <-> this package's boundary.

**HF anchor:** none — this is the engine's glue, not model math. It says where the config, the
weights and the golden trace live, allocates the KV cache, and builds `tt/tt_prefill_runtime.py`'s
`TtPrefillRuntime`. Every operational behaviour (a chunk, the cache layout, the migration table)
lives on the runtime; the engine drives it and owns the cache's lifetime and all comms
(`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md:31-35`).

**Template:** `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:41`
(`GptOssPrefillAdapter`), whose attribute block is `:45-49`. Llama is GQA, not MLA, so this
subclasses `PrefillModelAdapter` (`models/demos/common/prefill/adapter.py:104`) directly rather
than the DeepSeek family's `MLAPrefillAdapter`.

**This module is import-light, and that is load-bearing rather than tidy.** Two importers pull it
without wanting a device stack: the H2D producer (`models/demos/common/prefill/runners/prefill_producer.py:89`)
and, eagerly for *every* registered model, the DeepSeek test conftest
(`models/demos/deepseek_v3_d_p/tests/conftest.py:33`). So `torch`, `ttnn`, `transformers`,
`safetensors` and this package's own `tt/` modules are imported **inside** the methods, and
`G-ADAPTER` measures it in a subprocess rather than trusting the convention.

**Knobs come from `params`** (`PrefillRunParams`, `models/demos/common/prefill/adapter.py:46`),
never from `os.environ` (`BRINGUP_RECIPE.md:1897-1899`). The two exceptions are the two the engine's own
contract defines as env overrides of *this class's* attributes — `PREFILL_HF_MODEL` over
`hf_model_default` and `PREFILL_TTNN_CACHE` over `ttnn_cache_default`
(`models/demos/common/prefill/adapter.py:116-117`) — and both are read exactly once, here.

Three things the engine does that no doc states, all measured by `G-RUNTIME`'s AST walk and relied
on below:

1. The engine **mutates** what `load_hf_config` returns — it assigns `max_seq_len` on the next line
   (`models/demos/common/prefill/runners/prefill_runner.py:477`). `LlamaHfConfig` is therefore a
   plain mutable object; a frozen dataclass raises `FrozenInstanceError` at runner startup.
2. `weight_cache_path` is called **before** the mesh is open (`prefill_runner.py:377` inside
   `_print_config`, called at `:454`; `open_mesh_device` is `:472`), so it must not touch a device.
3. `model_config` is read by the engine for `FABRIC_PAYLOAD_SIZE`
   (`models/demos/common/prefill/runners/runner_utils.py:41`) **and** by the producer's read-back for
   `NUM_KEY_VALUE_HEADS` / `HEAD_DIM` (`prefill_producer.py:550`) — `Llama31_8BConfig` below exists
   for those two callers and for nothing else.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

# The package's own config.json, relative to the repo root. `PREFILL_HF_MODEL` overrides it.
_BUNDLED_CONFIG_DIR = "models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct"


class Llama31_8BConfig:
    """Static model dimensions, for the two callers that need them **before** any device exists.

    Every value is `config.json`'s, verbatim, and `G-ADAPTER` asserts each one against the bundled
    file — so this class cannot drift into a second source of truth
    (`bringup_log/00_MODEL_CARD.md` §2). It is deliberately *not* the package's config reader:
    `tt/model_config.py::ModelArgs` is, and everything on a device path goes through it.

    `HEAD_DIM` is the one **derived** entry: Llama's `config.json` has no `head_dim` key, so it is
    `hidden_size // num_attention_heads = 4096 // 32` (`tt/config.py::derive_head_dim`, `DEC-020`).
    """

    # Core dimensions (config.json)
    EMB_SIZE = 4096  # hidden_size
    # Max fabric packet payload. The engine turns this into `FabricRouterConfig.max_packet_payload_size_bytes`
    # (`models/demos/common/prefill/runners/runner_utils.py:41`); the template sets it to the
    # embedding width (`models/demos/deepseek_v3_d_p/reference/gpt_oss_120b_config.py:18`).
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    INTERMEDIATE_SIZE = 14336
    HEAD_DIM = 128  # derived: hidden_size // num_attention_heads

    # Model architecture (config.json)
    NUM_LAYERS = 32  # num_hidden_layers
    VOCAB_SIZE = 128256
    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 8

    # Other (config.json)
    RMS_NORM_EPS = 1e-05
    ROPE_THETA = 500000.0
    MAX_POSITION_EMBEDDINGS = 131072

    # `config.json` keys these constants mirror, for `G-ADAPTER`'s equality check. `HEAD_DIM` is
    # absent on purpose — it is derived, and the gate checks the derivation separately.
    CONFIG_JSON_KEYS = {
        "EMB_SIZE": "hidden_size",
        "INTERMEDIATE_SIZE": "intermediate_size",
        "NUM_LAYERS": "num_hidden_layers",
        "VOCAB_SIZE": "vocab_size",
        "NUM_ATTENTION_HEADS": "num_attention_heads",
        "NUM_KEY_VALUE_HEADS": "num_key_value_heads",
        "RMS_NORM_EPS": "rms_norm_eps",
        "ROPE_THETA": "rope_theta",
        "MAX_POSITION_EMBEDDINGS": "max_position_embeddings",
    }


class LlamaHfConfig:
    """The engine's `hf_config` handle: an **attribute** view of the raw `config.json` **dict**.

    The engine reads `.hidden_size` (`prefill_runner.py:516`) and assigns `.max_seq_len`
    (`:477`), so it needs attributes and mutability. This package's modules take the **dict**
    (recipe P1 trap 2, `bringup_log/03_OUTLINE.md` §2.3), which is `.dims` — so `build_runtime`
    passes `hf_config.dims` and nothing downstream ever attribute-reads a config.

    `head_dim` is added because `config.json` has none and `allocate_kv_cache` needs it; it comes
    from `tt/config.py::derive_head_dim`, the package's single derivation (`DEC-032`).
    """

    def __init__(self, dims: dict, *, head_dim: int):
        self.dims = dims
        self.__dict__.update(dims)
        self.head_dim = head_dim

    def __repr__(self):
        return (
            f"LlamaHfConfig(hidden_size={self.hidden_size}, layers={self.num_hidden_layers}, "
            f"kv_heads={self.num_key_value_heads}, head_dim={self.head_dim}, "
            f"max_seq_len={getattr(self, 'max_seq_len', None)})"
        )


class LlamaPrefillAdapter(PrefillModelAdapter):
    """Llama-3.1-8B prefill adapter: dense GQA, 32 layers, 8 KV heads, packed block-cyclic cache."""

    # --- identity & runner defaults ---
    name = "llama31_8b_d_p"
    model_config = Llama31_8BConfig
    hf_model_default = _BUNDLED_CONFIG_DIR  # config.json dir; PREFILL_HF_MODEL overrides
    # Empty => the caller must supply one. `""` means "no cache" to the engine
    # (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:48`); here it means the *root* is
    # unset, and `weight_cache_path` falls back to this package's own `$TT_CACHE_PATH` before
    # giving up, because the cache this deployment reads was populated under that variable
    # (`tt/model_config.py:240`, `DEC-048`).
    ttnn_cache_default = ""
    prefill_trace_default = ""  # golden trace dir; PREFILL_TRACE_DIR supplies it (`DEC-066`)
    # `default_gate_mode` is inherited unused: it is the MoE gate's fallback mode and this model is
    # dense. The engine still prints it (`prefill_runner.py:393`), so it is left at the base default
    # rather than pinned to a value a reader could think this model consults.

    # --- device / mesh knobs ---
    # No L1_SMALL carve-out: nothing in this package routes semaphores to L1_SMALL
    # (`bringup_log/04_CCL_PLAN.md` §2 allocates every CCL semaphore as a global semaphore).
    l1_small_size = 0
    # The decoder layer consumes and produces a TP-sharded-emb activation (`tt/layer.py`), which is
    # the engine's default; it is stated here because the cross-rank D2D layout depends on it and
    # multi-rank is out of scope, so nothing else would catch a wrong value.
    pipeline_activation_emb_tp_sharded = True
    # No DFlash drafter checkpoint targets Llama-3.1-8B (`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md:58-60`).
    supports_dflash = False

    # --- test metadata (consumed by the registry-fed `variant` fixture) ---
    hf_repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    env_var = "HF_MODEL"
    num_layers_to_download = 32
    tokenizer_trust_remote_code = False
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self) -> LlamaHfConfig:
        """The raw `config.json` dict, wrapped in the mutable attribute view the engine needs.

        **`AutoConfig` is deliberately not used**, unlike the template
        (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:68`). On `transformers` 5.12.1 a
        `LlamaConfig` object has no `rope_theta` attribute — it moved into `rope_parameters` — so
        `getattr(cfg, "rope_theta", DEFAULT)` returns the default and produces a RoPE that is wrong
        at every position with nothing raised (recipe P1 trap 1, `07_RISKS.md` R-005). This package
        reads dimensions from the dict, once, and `ModelArgs` refuses a config object outright
        (`tt/model_config.py:101-107`).

        A `PREFILL_HF_MODEL` pointing somewhere other than the bundled directory is read **and
        compared**: a checkpoint whose `config.json` disagrees with the bundled copy is refused
        rather than silently becoming a second source of dimensions (`DEC-001`, `DEC-094`).
        """
        from models.demos.llama31_8b_d_p.tt.config import derive_head_dim
        from models.demos.llama31_8b_d_p.tt.model_config import BUNDLED_CONFIG_PATH, ModelArgs

        bundled = ModelArgs.load_bundled_config()
        # `hf_model_default` is a repo-relative string, because the engine prints it
        # (`prefill_runner.py:375`) and the template's is relative too. So the **default** path is
        # taken without any path arithmetic at all: resolving a relative default against the process
        # CWD made the comparison below fail whenever the runner was started from anywhere but the
        # repo root, and then reported a `PREFILL_HF_MODEL` the user had never set (`DEC-113`).
        config_dir = os.environ.get("PREFILL_HF_MODEL")
        if not config_dir:
            logger.info(f"[{self.name}] config.json from the bundled copy ({len(bundled)} keys)")
            return LlamaHfConfig(bundled, head_dim=derive_head_dim(bundled))
        config_path = os.path.join(config_dir, "config.json")
        if os.path.realpath(config_path) != os.path.realpath(BUNDLED_CONFIG_PATH):
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"PREFILL_HF_MODEL={config_dir!r} has no config.json. It must name a directory "
                    f"holding one (the engine's contract, adapter.py:116); the bundled default is "
                    f"{self.hf_model_default}."
                )
            with open(config_path) as f:
                dims = json.load(f)
            if dims != bundled:
                differing = sorted(key for key in set(dims) | set(bundled) if dims.get(key) != bundled.get(key))
                raise ValueError(
                    f"{config_path} disagrees with the bundled config on {differing}. Every "
                    f"dimension in this package, every gate threshold and the whole weight cache "
                    f"were built against the bundled copy (DEC-001), so a differing checkpoint is "
                    f"refused rather than run."
                )
        logger.info(f"[{self.name}] config.json from {config_dir} ({len(bundled)} keys, bundled copy asserted equal)")
        return LlamaHfConfig(bundled, head_dim=derive_head_dim(bundled))

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """`<root>/tensor_cache_bfp8_<sp>x<tp>` — the layout the cache-populate run wrote.

        **Mirrors `tt/model_config.py::ModelArgs.weight_cache_path`** (`:250`) rather than the
        engine's `{name}_{arch}_{N}dev/{sp}x{tp}` convention
        (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:89`), because the cache that
        exists on this box was written by P8's own runs under `$TT_CACHE_PATH` and a runner that
        invented a different path would silently re-derive 15 GB of weights (`DEC-095`).
        `G-ADAPTER` asserts the two agree, so they cannot drift.

        The dtype and the mesh shape are both in the path because `ttnn.as_tensor` persists the
        tilized, already-sharded, already-cast tensor (`tt/model_config.py:232-235`).

        Called **before** the mesh is open (`prefill_runner.py:377`), so nothing here touches a
        device — not even `ttnn.get_num_devices()`, which can abort with co-located migration
        workers (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:78-81`).
        """
        root = os.environ.get("PREFILL_TTNN_CACHE") or self.ttnn_cache_default or os.environ.get("TT_CACHE_PATH")
        if not root:
            raise ValueError(
                "the weight cache root is unset: set PREFILL_TTNN_CACHE (the engine's variable) or "
                "TT_CACHE_PATH (this package's). Returning None would make the runner rebuild every "
                "weight from the safetensors shards on every start, and this package refuses to "
                "fall back to the checkpoint directory (07_RISKS.md R-003, DEC-048)."
            )
        sp, tp = mesh_shape
        path = Path(root) / f"tensor_cache_bfp8_{int(sp)}x{int(tp)}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # KV cache + runtime build
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        """The packed block-cyclic K/V cache: one KV head per chip, sequence SP-sharded.

        Returns this package's `LlamaKVCache`, which **is** a `KvCaches`
        (`tt/attention/kv_cache.py:56`), so the engine gets its opaque handle without a wrapper —
        unlike the template, which boxes its cache in a one-element list
        (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:107-118`). The runtime's
        `_resolve_kv` accepts both (`DEC-062`).

        `params.num_layers` is **this rank's** slice; single-rank means the whole model.
        """
        import ttnn
        from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache

        cache = allocate_kv_cache(
            mesh_device,
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            sp_axis=params.sp_axis,
            num_users=params.num_users,
            head_dim=hf_config.head_dim,
            cache_dtype=ttnn.bfloat8_b,  # `DEC-021`
        )
        logger.info(
            f"[{self.name}] KV cache allocated: {params.num_users} user(s) x {params.num_layers} layer(s), "
            f"capacity {params.max_seq_len} tokens, per-chip rows {params.max_seq_len // params.sp_factor}, "
            f"head_dim {hf_config.head_dim}, bfloat8_b"
        )
        return cache

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """`TtPrefillRuntime` for this rank. Holds no KV cache — the engine passes it in per call."""
        import ttnn
        from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs
        from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        if params.use_trace:
            raise NotImplementedError(
                "PREFILL_USE_TRACE=1 needs capture_trace() and a metadata_msg path, neither of "
                "which this runtime implements; trace/2CQ is an explicit non-goal for this "
                "iteration (BRINGUP_RECIPE.md line 16, risk R-024). Run with PREFILL_USE_TRACE=0. "
                "Refused here, at build time, rather than on the first served chunk."
            )
        if params.dflash_enabled:
            raise NotImplementedError(
                "PREFILL_DFLASH=1 needs a DFlash drafter checkpoint targeting this architecture and "
                "none exists; supports_dflash is False, so reaching this means the engine's gate "
                "changed."
            )
        if params.kv_only_last_layer:
            # The engine defaults PREFILL_KV_ONLY_LAST_LAYER to 1 (`prefill_runner.py:80`). This
            # runtime writes EVERY layer's KV, which is what the producer's per-layer PCC reads
            # (`prefill_producer.py:565`) and what migration needs; the flag is therefore a knob
            # this model does not honour, and saying so out loud beats dropping it (`DEC-096`).
            logger.warning(
                f"[{self.name}] params.kv_only_last_layer is set, and this runtime ignores it: "
                f"every layer writes KV. The cache is sized for all {params.num_layers} layers, so "
                f"the flag buys nothing here — set PREFILL_KV_ONLY_LAST_LAYER=0 to stop asking."
            )

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            chunk_size=params.chunk_size,
            mesh_shape=params.mesh_shape,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            weight_cache_path=params.weight_cache_path,
            sequence_parallel=True,  # the SP attention cores; `tt/attention/prefill.py`
            # **`Linear`, not `Ring`, and it is measured rather than chosen** (`DEC-097`). This
            # galaxy cannot initialise `FABRIC_1D_RING` at all, and
            # `ttnn.transformer.ring_joint_scaled_dot_product_attention` under `Topology.Ring` asks
            # the fabric for the SP axis's wrap route and aborts at
            # `tt_metal/fabric/fabric.cpp:171` (`bringup_log/06_GATES.md` G-FABRIC-MATRIX,
            # `07_RISKS.md` R-030/R-031). `PrefillRunParams` carries no topology field, so this is
            # pinned here rather than read from an env var this package would then have to document.
            topology=ttnn.Topology.Linear,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
            use_trace=False,
        )

        model_path = os.environ.get("HF_MODEL")
        if not model_path:
            raise ValueError(
                "HF_MODEL is unset: this runtime loads the real bf16 checkpoint through "
                "ModelArgs.load_state_dict (tt/model_config.py:145) and there is no cache-only "
                "path here (DEC-098). Export HF_MODEL=<checkpoint dir>."
            )
        args = ModelArgs(mesh_device, hf_config=hf_config.dims, model_path=model_path, max_seq_len=params.max_seq_len)
        # `safetensors` memory-maps, so this returns in ~46 ms and a tensor's bytes are faulted in
        # only when something touches them — which, with the weight cache populated, nothing does
        # (`DEC-098` measured 290 cache loads and 0 regenerations).
        logger.info(f"[{self.name}] mapping the bf16 checkpoint at {args.model_path} (safetensors, lazy) ...")
        state_dict = ModelArgs.load_state_dict(args.model_path)

        runtime = TtPrefillRuntime(mesh_device, hf_config.dims, state_dict, runtime_config)
        del state_dict
        return runtime

    # ------------------------------------------------------------------
    # Test-only reference handle (imported lazily; never on the serving path)
    # ------------------------------------------------------------------
    @property
    def reference_model_cls(self):
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM
