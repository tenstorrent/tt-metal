# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GLM-5.3 prefill adapter.

Same serving shape as GLM-5.1 (``adapters/glm_5_1.py``): a DSA (sparse-attention) MLA + MoE model, so
it subclasses ``MLAPrefillAdapter``, allocates the uncompressed bf16/ROW_MAJOR MLA KVPE cache plus the
block-cyclic lightning-indexer KEY cache, and inherits ``build_runtime`` / ``weight_cache_path``. GLM
diverges from the dense family the same two ways GLM-5.1 does — a DSA indexer (resolved from the
config's ``index_*`` attrs at model-build time) and a hand-built config (``glm_moe_dsa`` isn't
AutoConfig-loadable).

GLM-5.3 delta vs 5.1 = **cross-layer DSA indexer reuse**: only ``full`` layers run the indexer; the
following ``shared`` layers reuse the most recent full layer's top-k selection. That is entirely
CONFIG-DRIVEN — ``glm_5_3_hf_config`` carries the ``indexer_types`` full/shared map, and the transformer
/ ttMLA read it to bind ``TtIndexer`` (full) vs ``ReuseIndexer`` (shared) and to inject the reused
indices — so no reuse-specific wiring lives in the adapter.

Like GLM-5.1 (and Kimi) it has a single expert group with a device gate, so the MoE routing all-gather's
semaphores go to L1_SMALL (needs the L1_SMALL carve-out at mesh-open).
"""

from __future__ import annotations

import os
from typing import Callable

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config, glm_5_3_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches


class GLM53Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "glm_5_3"
    model_config = GLM53Config
    hf_model_default = "/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4"
    ttnn_cache_default = "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/GLM-5.3-Cache-prefill"
    default_gate_mode = "DEVICE_FP32"
    prefill_trace_default = (
        "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/vllm-glm53-indexer-kcache-55k"
    )

    # Routing consumes 512 B; leave 256 B for sparse-MLA high-bandwidth-gather semaphores and rest for other needs.
    # 1216, not GLM-5.1's 1152: the tp_sharded fallback gather (used wherever the snake ring
    # cannot close) adds two high_bw_all_gather programs at two 16 B/bank semaphores each.
    l1_small_size = 1216
    routing_use_l1_small_for_semaphores = True

    supports_mtp = True

    def load_hf_config(self):
        """GLM's ``glm_moe_dsa`` isn't AutoConfig-loadable, so return the hand-built HF-attribute config
        (dims + the DSA ``index_*`` attrs + the ``indexer_types`` full/shared reuse map the sparse path
        resolves against). The runner overwrites ``max_seq_len`` after; seed the builder with it so rope
        config is consistent."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        return glm_5_3_hf_config(max_seq=max_seq)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params) -> MlaKvCaches:
        """GLM is sparse (DSA), so it owns TWO device caches, returned as a KvCaches tuple (the runner
        hands the whole tuple to every runtime call; the runtime pulls index 0 as the primary KV cache
        and index 1 as the secondary index cache). Mirrors glm_5_1.py:

          * index 0 — the MLA KVPE cache. sparse_sdpa reads it natively and requires it UNCOMPRESSED
            (bf16 ROW_MAJOR), not the dense bf8/TILE cache the base MLA adapter allocates. All layers.
          * index 1 — the lightning-indexer's per-user block-cyclic KEY cache (bfp8 TILE, ``index_head_dim``
            wide). GLM-5.3 cross-layer reuse: only ``full`` layers own an indexer and write this cache
            (``shared`` layers reuse a prior full layer's top-k and never write), so it is sized to the
            FULL-layer count, not all layers — each full layer writes its compacted rank slot (see
            ``TtIndexer``). Like the KVPE cache it holds THIS pipeline stage only, so its slots are
            numbered from this stage's first full layer and the migration table needs no extra stride.
            ``full_indexer_rank`` degenerates to the layer count without an ``indexer_types`` map
            (GLM-5.1: every layer is full).

        The engine owns both, exactly like the dense KVPE cache."""
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
        from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import enable_mtp_indexer_slot
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
            MlaKvCacheFormat,
            init_kvpe_cache,
            init_mla_kv_cache,
        )

        # KV dedup: seq_len/(sp*tp) rows per device instead of seq_len/sp. Both caches must use the same
        # tp_axis as the write op and the migration table.
        kv_tp_axis = params.tp_axis
        mtp_levels = params.mtp_levels if params.is_last_rank else 0
        if params.mtp_levels:
            enable_mtp_indexer_slot(hf_config)

        kvpe_cache = init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=hf_config,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=params.num_layers + mtp_levels,
            num_users=params.num_users,
            tp_axis=kv_tp_axis,
        )
        first_full = full_indexer_rank(hf_config, params.first_layer_idx)
        num_index_layers = (
            full_indexer_rank(hf_config, params.first_layer_idx + params.num_layers + mtp_levels) - first_full
        )
        index_cache = init_kvpe_cache(
            kvpe_cache_head_dim=hf_config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=num_index_layers,
            num_users=params.num_users,
            dtype=ttnn.bfloat8_b,
            tp_axis=kv_tp_axis,
        )
        return MlaKvCaches(kvpe=kvpe_cache, index=index_cache)

    def layer_split_boundaries(self, num_layers):
        """GLM-5.3 cross-layer reuse: a pipeline rank must start on a ``full`` layer (it seeds that
        rank's indexer-reuse chain; a ``shared`` first layer has no prior top-k to reuse). So the valid
        rank-start boundaries are the ``full`` layer indices. ``None`` absent an ``indexer_types`` map."""
        types = getattr(self.load_hf_config(), "indexer_types", None)
        return None if not types else {i for i in range(num_layers) if types[i] == "full"}

    # --- test metadata (HF download coordinates + PCC thresholds + golden trace) ---
    # FP8 repo, mirroring GLM-5.1 (a bf16 checkout would diverge from an FP8-derived trace). The bare
    # `zai-org/GLM-5.3` repo IS the FP8 one; hf_model_default above is a local copy of it.
    hf_repo_id = "zai-org/GLM-5.3"
    env_var = "GLM53_HF_MODEL"
    mla_ref_cache_env = "GLM53_MLA_REF_CACHE"
    ref_cache_env = "TT_GLM53_PREFILL_HOST_REF_CACHE"
    ttnn_cache_env = "TT_GLM53_PREFILL_TTNN_CACHE"
    supports_pretrained = True
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    prefill_trace_layout = "chunked_group_a_v1"
    # Default trace must carry the DSA indexer-K cache (dsa/indexer_k_layer_*), or the indexer-K PCC
    # checks silently skip. Same trace as prefill_trace_default, so serving and the tests agree.
    test_prefill_trace_default = (
        "/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/vllm-glm53-indexer-kcache-55k"
    )

    @property
    def config_builder(self) -> Callable:
        """The sparse-MLA reference tests resolve GLM's config through this (conftest
        _resolve_config_only); serving goes through load_hf_config. Both hand off to glm_5_3_hf_config."""
        return glm_5_3_hf_config
