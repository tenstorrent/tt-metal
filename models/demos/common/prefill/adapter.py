# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The prefill-runner ↔ model boundary: ``PrefillModelAdapter``.

The prefill runner (``prefill_runner.py``) is a model-agnostic orchestration
engine: it owns rank topology, the layer split, the H2D/D2D sockets, the
request/standalone loops, lease/reclaim, LayerAck, and shutdown. Everything that
differs per model lives behind a ``PrefillModelAdapter``.

To add a model you implement (or subclass) one adapter and register it; the
engine never imports a model class directly. See ``ADDING_A_PREFILL_MODEL.md``.

Two layers:
  * ``PrefillModelAdapter`` (here) — the abstract contract the engine calls, plus
    the model registry (``get_adapter`` / ``ADAPTER_PATHS``). Both are common.
  * A concrete adapter per model, living in that model's own package. The
    DeepSeek-V3 family ships a shared ``MLAPrefillAdapter`` base (MLA attention +
    MoE) with thin ``DeepSeekV3Adapter`` / ``KimiK26Adapter`` subclasses; a
    different architecture subclasses ``PrefillModelAdapter`` directly with its own
    KV layout. See ``runners/ADDING_A_PREFILL_MODEL.md``.

Import-safety: this module and every concrete adapter must stay light enough to
import in the serving process — NO reference-modeling / safetensors imports at
module load. The registry resolves a model name to its adapter class LAZILY (by
dotted path), so importing this common module never pulls any model's device /
runtime stack. The test-only reference-model handles are likewise exposed as
properties that import lazily.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

    import ttnn


@dataclass(frozen=True)
class PrefillRunParams:
    """The resolved (env-derived) knobs the engine hands an adapter to build a
    runtime. The runner fills this once from the environment + per-rank layer
    split; adapters read fields off it instead of re-reading os.environ, so the
    adapter method signatures stay stable as new knobs are added.

    Layer fields are THIS rank's slice: ``num_layers`` is the rank's layer count
    (== model total for single-rank) and ``first_layer_idx`` is the global index
    of its first layer.
    """

    mesh_shape: tuple  # (sp, tp)
    num_layers: int
    first_layer_idx: int
    is_first_rank: bool
    is_last_rank: bool
    max_seq_len: int
    chunk_size: int
    num_users: int
    capacity_factor: int
    num_links: int
    gate_mode_name: str  # GateComputeMode enum name
    kv_only_last_layer: bool
    weight_cache_path: Optional[Path]
    sp_axis: int = 0
    tp_axis: int = 1

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class KvCaches(tuple):
    """The device KV cache(s) a prefill model owns, returned by ``allocate_kv_cache`` as an ORDERED
    tuple of tensors. Callers index positionally:

      * ``kv_caches[0]`` — ALWAYS the primary KV cache (the MLA KVPE cache for the DeepSeek-V3 family).
      * ``kv_caches[1:]`` — any SECONDARY caches the model also owns. A sparse-attention (DSA) model
        like GLM-5.1 puts its lightning-indexer block-cyclic key cache at index 1, so the engine can
        build a single MERGED migration table over both (config 0 = KVPE, config 1 = index). A dense
        model returns just ``KvCaches([kvpe])`` (length 1).

    A tuple rather than a bare tensor plus optional allocate hooks: every model implements the same
    single ``allocate_kv_cache`` and just returns as many caches as it owns — a model that grows a
    third cache appends it at index 2, no new adapter method or engine argument. The engine allocates
    these once, OWNS their lifetime — it passes them into every runtime call that touches them
    (compile / prefill_chunk / build_kv_chunk_table / kv_cache_pcc_check) and frees them with the mesh
    at shutdown."""

    __slots__ = ()


class PrefillModelAdapter(ABC):
    """Per-model plumbing the prefill engine needs. One instance per model.

    Subclasses set the identity / default-path class attributes and implement the
    abstract methods. The DeepSeek-V3 family should subclass ``MLAPrefillAdapter``
    (which implements everything); only a genuinely different architecture needs
    to subclass this directly.
    """

    # --- identity & runner defaults (set on the concrete subclass) ---
    name: str  # registry key + weight-cache dir prefix ({name}_{arch}_{N}dev)
    model_config: type  # static model-dimension constants class (e.g. DeepSeekV3Config)
    hf_model_default: str  # config.json dir; PREFILL_HF_MODEL overrides
    ttnn_cache_default: str  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides
    prefill_trace_default: str  # golden trace dir (token_ids + KV); PREFILL_TRACE_DIR overrides
    default_gate_mode: str = "DEVICE_FP32"  # GateComputeMode name; PREFILL_GATE_FALLBACK_MODE overrides

    # --- device / mesh knobs ---
    # L1_SMALL carve-out at mesh-open time (0 = none). Needed when an op routes
    # its semaphores to L1_SMALL (see routing_use_l1_small_for_semaphores).
    l1_small_size: int = 0
    # Route the MoE routing all-gather's global semaphores to L1_SMALL instead of
    # pinning the main-L1 floor. Requires l1_small_size > 0.
    routing_use_l1_small_for_semaphores: bool = False

    # =====================================================================
    # Glue the engine calls. The adapter is a factory + descriptor only: it says
    # where this model's config / weights live and how to build its runtime. All
    # operational behavior (running a chunk, the KV layout, the migration table,
    # PCC) lives on the runtime that build_runtime returns — see the runtime
    # contract in runners/ADDING_A_PREFILL_MODEL.md. The engine owns all comms.
    # =====================================================================
    @abstractmethod
    def load_hf_config(self) -> "PretrainedConfig":
        """Load (and normalize) the HF config from PREFILL_HF_MODEL (falling back
        to ``hf_model_default``). The runner sets ``max_seq_len`` on the result."""

    @abstractmethod
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """The TTNN weight-cache dir for this model + mesh, mirroring the layout
        the cache-populate run wrote. None only if the cache is explicitly empty."""

    @abstractmethod
    def allocate_kv_cache(self, *, mesh_device: "ttnn.MeshDevice", hf_config, params: PrefillRunParams) -> KvCaches:
        """Allocate (and zero) this model's KV cache(s) on device and return them as a ``KvCaches``
        (ordered tuple): index 0 the primary KV cache, then any secondary caches the model owns. A
        dense model returns ``KvCaches([kvpe])``; a sparse-attention (DSA) model appends its index
        cache (``KvCaches([kvpe, index])``). This is the single place a model's KV layout is defined.
        The engine OWNS the returned caches' lifetime — see ``KvCaches``. ``params`` carries the
        per-rank knobs (max_seq_len, mesh_shape, this rank's num_layers, num_users, …)."""

    @abstractmethod
    def build_runtime(self, *, mesh_device: "ttnn.MeshDevice", hf_config, params: PrefillRunParams):
        """Construct the model for this rank and return a runtime handle. The runtime
        is stateless w.r.t. the KV cache — it receives the engine-owned ``KvCaches`` as an
        argument on each call. The engine then calls ``.compile(kv_caches)`` and drives
        it (make_chunk_input, prefill_chunk, and — when enabled — build_kv_chunk_table /
        kv_cache_pcc_check / set_layer_ack_channel). ``params`` carries the per-rank knobs."""

    # =====================================================================
    # Test-only metadata (HF download coordinates + reference modeling).
    #
    # Consumed by the pytest fixtures in tests/conftest.py. Scalars are plain
    # class attributes; the reference-model class handles are properties so they
    # import the torch reference model lazily — importing an adapter never pulls
    # reference modeling into the serving import path.
    # =====================================================================
    hf_repo_id: str = ""  # HuggingFace repo id for weight/config download
    env_var: str = ""  # env var pointing at on-disk weights (e.g. DEEPSEEK_V3_HF_MODEL)
    default_local_path: Optional[Path] = None
    shared_path: Optional[Path] = None
    num_layers_to_download: int = 24
    ttnn_cache_env: Optional[str] = None
    ref_cache_env: Optional[str] = None
    mla_ref_cache_env: Optional[str] = None
    moe_pcc_threshold: float = 0.999
    mla_pcc_threshold: float = 0.999
    supports_pretrained: bool = True
    # Whether the tokenizer needs trust_remote_code=True (custom tokenizer code shipped in the repo,
    # e.g. Kimi's tiktoken-backed BBPE). DeepSeek-V3 uses a stock fast tokenizer, so it turns this off
    # to avoid the flat-config trust_remote_code import path that otherwise breaks its load.
    tokenizer_trust_remote_code: bool = True
    # Whether config/tokenizer resolution must copy the HF snapshot into a flat dir of real files.
    # Only needed when loading with trust_remote_code=True (transformers resolves the remote module to
    # its blobs/ realpath and then can't find relative-import siblings by name). Variants that load
    # stock config/tokenizers (trust_remote_code=False) use the snapshot dir directly and set this off.
    needs_flat_config_dir: bool = True
    # Hand-built HF-attribute config factory (zero-arg callable) for models whose
    # ``model_type`` transformers can't load via AutoConfig (unregistered, e.g. DeepSeek-V3.2's
    # ``deepseek_v32`` / GLM's ``glm_moe_dsa``). None → resolve the config the normal way
    # (AutoConfig, or the runner's ``load_hf_config``). A sparse-attention variant overrides this
    # (typically as a lazily-importing property) — see the sparse-MLA adapters. It is the single
    # source the sparse-MLA CPU reference derives its ModelArgs from, so device + truth share dims.
    config_builder: Optional[Callable] = None
    # Golden-trace on-disk layout: "single_file" (one safetensors/layer) or
    # "chunked_group_a_v1" (per-tensor dir of row-sharded shards). Tests dispatch on it.
    prefill_trace_layout: str = "single_file"
    # Golden trace the pytest block/transformer chunked tests read. Usually the same
    # as the runner's ``prefill_trace_default`` (and defaults to it via __init_subclass__
    # when left None); override only when the test golden differs from the serving one.
    test_prefill_trace_default: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Default the test golden trace to the runner's, so a model that uses one
        # trace for both only sets ``prefill_trace_default``.
        if getattr(cls, "test_prefill_trace_default", None) is None:
            cls.test_prefill_trace_default = getattr(cls, "prefill_trace_default", None)

    @property
    def reference_model_cls(self) -> type:
        raise NotImplementedError(f"{type(self).__name__} has no reference model wired")

    @property
    def reference_attention_cls(self) -> Optional[type]:
        return None

    @property
    def reference_moe_cls(self) -> Optional[type]:
        return None


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# The single source of truth mapping a model name to its adapter class, as a
# ``"module.path:ClassName"`` string resolved lazily by ``get_adapter``. Adding a
# model is one line here (plus the adapter class in that model's package). Keeping
# these as strings means importing this common module never imports a model's
# device/runtime stack — only the selected model is imported, at get_adapter time.
DEFAULT_MODEL = "deepseek_v3_d_p"

ADAPTER_PATHS = {
    "deepseek_v3_d_p": "models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v3:DeepSeekV3Adapter",
    "kimi_k2_6": "models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_6:KimiK26Adapter",
    # GLM-5.1: sparse-attention (DSA) variant with a full prefill serving runtime (adapters/glm_5_1.py).
    "glm_5_1": "models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_1:GLM51Adapter",
    # DeepSeek-V3.2-Exp: DSA, still test-only (config + sparse-MLA reference parity; serving not wired).
    "deepseek_v32": "models.demos.deepseek_v3_d_p.tt.runners.adapters.sparse_mla:DeepSeekV32Adapter",
}

_ADAPTER_INSTANCES: dict = {}


def get_adapter(name: str) -> PrefillModelAdapter:
    """Resolve (and memoize) a registered adapter by name; raises KeyError listing
    the valid set. The adapter class is imported lazily from ``ADAPTER_PATHS``."""
    try:
        dotted = ADAPTER_PATHS[name]
    except KeyError:
        raise KeyError(f"Unknown PREFILL_MODEL={name!r}; valid: {sorted(ADAPTER_PATHS)}")
    if name not in _ADAPTER_INSTANCES:
        module_path, cls_name = dotted.split(":")
        adapter_cls = getattr(importlib.import_module(module_path), cls_name)
        _ADAPTER_INSTANCES[name] = adapter_cls()
    return _ADAPTER_INSTANCES[name]
