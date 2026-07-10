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


class GLM52Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "glm_5_2"
    model_config = GLM52Config
    # Config is hand-built (see load_hf_config); this stays empty so an accidental AutoConfig read
    # yields an obvious empty path. Real weights come from the TTNN cache.
    hf_model_default = ""
    # GLM-5.2 staged TTNN weight cache. PLACEHOLDER — confirm/override with PREFILL_TTNN_CACHE /
    # TT_GLM52_PREFILL_TTNN_CACHE. (5.2 weights differ from 5.1, so this is NOT glm_5_1's cache.)
    ttnn_cache_default = "/data/nmilicevic/glm52_tp4_cache"
    default_gate_mode = "DEVICE_FP32"  # GLM: single expert group
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

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        """GLM is sparse (DSA): sparse_sdpa reads the KVPE cache natively and requires it UNCOMPRESSED —
        bf16 ROW_MAJOR — not the dense bf8/TILE cache the base MLA adapter allocates. One shared cache of
        num_users * num_layers user-major slots. Identical to GLM-5.1."""
        import ttnn
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

        return init_kvpe_cache(
            kvpe_cache_head_dim=hf_config.qk_rope_head_dim + hf_config.kv_lora_rank,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=params.num_layers,
            num_users=params.num_users,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def allocate_index_kv_cache(self, *, mesh_device, hf_config, params):
        """The lightning-indexer block-cyclic KEY cache (bfp8 TILE, ``index_head_dim`` wide), a folded
        per-user cache with the same block-cyclic layout as the KVPE cache.

        GLM-5.2 cross-layer reuse: only ``full`` layers own an indexer and write this cache (``shared``
        layers reuse a prior full layer's top-k and never write), so it is sized to the FULL-layer count
        (``num_full_indexer_layers``), not all layers — each full layer writes its compacted rank slot
        (see ``TtIndexer``). The merged migration table's index config sizes itself from this tensor's
        shape, so it too holds only full-layer entries. Falls back to ``num_layers`` when there is no
        ``indexer_types`` map (GLM-5.1: every layer is full)."""
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

        num_index_layers = num_full_indexer_layers(hf_config) or params.num_layers
        return init_kvpe_cache(
            kvpe_cache_head_dim=hf_config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=num_index_layers,
            num_users=params.num_users,
            dtype=ttnn.bfloat8_b,
        )

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
