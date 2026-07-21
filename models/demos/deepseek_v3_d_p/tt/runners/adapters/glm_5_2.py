# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GLM-5.2 prefill adapter.

Same serving shape as GLM-5.1 (``adapters/glm_5_1.py``): a DSA (sparse-attention) MLA + MoE model, so
it subclasses ``MLAPrefillAdapter``, allocates the uncompressed bf16/ROW_MAJOR MLA KVPE cache plus the
block-cyclic lightning-indexer KEY cache, and inherits ``build_runtime`` / ``weight_cache_path``. GLM
diverges from the dense family the same two ways GLM-5.1 does — a DSA indexer (resolved from the
config's ``index_*`` attrs at model-build time) and a hand-built config (``glm_moe_dsa`` isn't
AutoConfig-loadable).

GLM-5.2 delta vs 5.1 = **cross-layer DSA indexer reuse**: only ``full`` layers run the indexer; the
following ``shared`` layers reuse the most recent full layer's top-k selection. That is entirely
CONFIG-DRIVEN — ``glm_5_2_hf_config`` carries the ``indexer_types`` full/shared map, and the transformer
/ ttMLA read it to bind ``TtIndexer`` (full) vs ``ReuseIndexer`` (shared) and to inject the reused
indices — so no reuse-specific wiring lives in the adapter.

Like GLM-5.1 (and Kimi) it has a single expert group with a device gate, so the MoE routing all-gather's
semaphores go to L1_SMALL (needs the L1_SMALL carve-out at mesh-open). Run with
``PREFILL_KV_ONLY_LAST_LAYER=0`` (the sparse path can't run a kv-only last layer); the glm52 manifest
sets it.
"""

from __future__ import annotations

import os
from typing import Callable

from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config, glm_5_2_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches


class GLM52Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "glm_5_2"
    model_config = GLM52Config
    hf_model_default = "/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8"
    ttnn_cache_default = "/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache"
    default_gate_mode = "DEVICE_FP32"
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/structured_traces/glm_52_55k_vllm"

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL.
    l1_small_size = 512
    routing_use_l1_small_for_semaphores = True

    def load_hf_config(self):
        """GLM's ``glm_moe_dsa`` isn't AutoConfig-loadable, so return the hand-built HF-attribute config
        (dims + the DSA ``index_*`` attrs + the ``indexer_types`` full/shared reuse map the sparse path
        resolves against). The runner overwrites ``max_seq_len`` after; seed the builder with it so rope
        config is consistent."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        return glm_5_2_hf_config(max_seq=max_seq)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params) -> MlaKvCaches:
        """GLM is sparse (DSA), so it owns TWO device caches, returned as a KvCaches tuple (the runner
        hands the whole tuple to every runtime call; the runtime pulls index 0 as the primary KV cache
        and index 1 as the secondary index cache). Mirrors glm_5_1.py:

          * index 0 — the MLA KVPE cache. sparse_sdpa reads it natively and requires it UNCOMPRESSED
            (bf16 ROW_MAJOR), not the dense bf8/TILE cache the base MLA adapter allocates. All layers.
          * index 1 — the lightning-indexer's per-user block-cyclic KEY cache (bfp8 TILE, ``index_head_dim``
            wide). GLM-5.2 cross-layer reuse: only ``full`` layers own an indexer and write this cache
            (``shared`` layers reuse a prior full layer's top-k and never write), so it is sized to the
            FULL-layer count (``num_full_indexer_layers``), not all layers — each full layer writes its
            compacted rank slot (see ``TtIndexer``), and the merged migration table's index config sizes
            itself from this tensor's shape. Falls back to ``num_layers`` when there is no ``indexer_types``
            map (GLM-5.1: every layer is full).

        The engine owns both, exactly like the dense KVPE cache."""
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
            MlaKvCacheFormat,
            init_kvpe_cache,
            init_mla_kv_cache,
        )

        kvpe_cache = init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=params.num_layers,
            num_users=params.num_users,
        )
        num_index_layers = num_full_indexer_layers(hf_config) or params.num_layers
        index_cache = init_kvpe_cache(
            kvpe_cache_head_dim=hf_config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=num_index_layers,
            num_users=params.num_users,
            dtype=ttnn.bfloat8_b,
        )
        return MlaKvCaches(kvpe=kvpe_cache, index=index_cache)

    def layer_split_boundaries(self, num_layers):
        """GLM-5.2 cross-layer reuse: a pipeline rank must start on a ``full`` layer (it seeds that
        rank's indexer-reuse chain; a ``shared`` first layer has no prior top-k to reuse). So the valid
        rank-start boundaries are the ``full`` layer indices. ``None`` absent an ``indexer_types`` map."""
        types = getattr(self.load_hf_config(), "indexer_types", None)
        return None if not types else {i for i in range(num_layers) if types[i] == "full"}

    # --- test metadata (HF download coordinates + PCC thresholds + golden trace) ---
    # FP8 repo, mirroring GLM-5.1 (a bf16 checkout would diverge from an FP8-derived trace).
    hf_repo_id = "zai-org/GLM-5.2-FP8"
    env_var = "GLM52_HF_MODEL"
    mla_ref_cache_env = "GLM52_MLA_REF_CACHE"
    ref_cache_env = "TT_GLM52_PREFILL_HOST_REF_CACHE"
    ttnn_cache_env = "TT_GLM52_PREFILL_TTNN_CACHE"
    supports_pretrained = True
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    prefill_trace_layout = "chunked_group_a_v1"
    test_prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/structured_traces/glm_52_55k_vllm"

    @property
    def config_builder(self) -> Callable:
        """The sparse-MLA reference tests resolve GLM's config through this (conftest
        _resolve_config_only); serving goes through load_hf_config. Both hand off to glm_5_2_hf_config."""
        return glm_5_2_hf_config
