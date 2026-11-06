# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K3.1 adapter: Kimi-K3 with a DSA lightning indexer on its 24 MLA layers.

Everything K3 has carries over -- the hybrid MLA/KDA schedule, AttnRes, the KDA state slabs, the
pipeline split rules -- so this subclasses ``KimiK3Adapter`` and changes only what the indexer changes:

  * the MLA layers read a **sparse** KV cache in GLM's scaled-FP8 format (the format K3.1's decode keeps,
    tt-blaze ``kimi_k3/cache_migration.py``), not K3's dense bf8 one;
  * each MLA layer also writes the indexer's **key cache** (bf8, ``INDEX_HEAD_DIM`` wide), slotted like the
    KV cache: one slot per MLA layer of this rank. Its keys stay in the model's own basis
    (``KimiK31Config.INDEX_HADAMARD``), as K3.1's decode stores them;
  * the migration table gains that cache as config "01", between the KV cache and the KDA state::

        "00"  kvpe              GLM scaled FP8, MLA layers
        "01"  index             bf8 [32, 128] per chunk, MLA layers
        "02"  kda_recurrent     as K3
        "03"  kda_convolution   as K3

    GLM's "00"/"01" stay where GLM's decode and the producer's index read-back expect them; decode pairs
    configs by name, so K3.1's decode table must publish the same order.
"""

from __future__ import annotations

from typing import Callable

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter


class KimiK31Adapter(KimiK3Adapter):
    # --- identity ---
    name = "kimi_k3_1"
    model_config = KimiK31Config
    hf_model_default = "/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized"
    ttnn_cache_default = "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3.1-Cache/Kimi-K3.1-Cache-prefill"
    # K3's DSpark drafter is keyed to K3, not K3.1.
    supports_dflash = False
    dflash_model_default = None
    dflash_golden_default = None
    # The K3.1 vLLM traces are raw chunked_group_a_v1 streams, not the structured layout the producer's
    # read-back walks (tests/sparse_mla/k31_trace.py reads them for the per-layer tests).
    prefill_trace_default = None
    mla_trace_defaults = ()
    kda_golden_default = None

    # --- test metadata ---
    hf_repo_id = "moonshotai/Kimi-K3.1-Pro"
    env_var = "KIMI_K31_HF_MODEL"
    mla_ref_cache_env = "KIMI_K31_MLA_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_K31_PREFILL_TTNN_CACHE"
    ref_cache_env = "TT_KIMI_K31_PREFILL_HOST_REF_CACHE"
    # None: conftest's pretrained_mla_layer_weights fixture would resolve the whole multi-TB checkpoint.
    pretrained_mla_layer = None

    @property
    def config_builder(self) -> Callable:
        from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import kimi_k3_1_hf_config

        return kimi_k3_1_hf_config

    def load_hf_config(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import kimi_k3_1_hf_config

        return kimi_k3_1_hf_config()

    @property
    def default_sparse_kv_cache_format(self):
        """Scaled-FP8 KV cache: GLM-5.3's, which K3.1's decode reads."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

        return MlaKvCacheFormat.SCALED_FP8

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """K3's caches with the KV cache made sparse and the indexer's key cache beside it.

        Both MLA caches are sized to this rank's MLA layers (``KimiK3LayerSchedule.num_mla_layers``), not
        its layer count, for the same reason as K3's (see ``KimiK3Adapter.allocate_kv_cache``): every slot
        stride downstream is computed against that count. The indexer's slot is the same rank-local MLA slot
        as the KV cache's, and its stride falls back to ``layer_num`` (= the MLA count) because K3.1 has no
        ``indexer_types`` map. Both are TP-deduped like every sparse cache, on the axis the writes use.
        """
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
        from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import KimiK3KvCaches
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache, init_mla_kv_cache

        schedule = KimiK3LayerSchedule.build(KimiK31Config, params.first_layer_idx, params.num_layers)
        kvpe = index = None
        if schedule.num_mla_layers > 0:
            kvpe = init_mla_kv_cache(
                cache_format=self.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
                hf_config=hf_config,
                mesh_device=mesh_device,
                seq_len=params.max_seq_len,
                mesh_shape=list(params.mesh_shape),
                sp_axis=params.sp_axis,
                num_kvpe_cache_layers=schedule.num_mla_layers,
                num_users=params.num_users,
                tp_axis=params.tp_axis,
                bank_aligned=True,
            )
            index = init_kvpe_cache(
                kvpe_cache_head_dim=KimiK31Config.INDEX_HEAD_DIM,
                mesh_device=mesh_device,
                seq_len=params.max_seq_len,
                mesh_shape=list(params.mesh_shape),
                sp_axis=params.sp_axis,
                num_kvpe_cache_layers=schedule.num_mla_layers,
                num_users=params.num_users,
                dtype=ttnn.bfloat8_b,
                tp_axis=params.tp_axis,
                bank_aligned=True,
            )

        kda_states = None
        kda_layer_ids = schedule.rank_kda_layer_ids()
        if kda_layer_ids:
            geometry = kimi_k3_kda_geometry(params.mesh_shape, params.sp_axis, params.tp_axis)
            kda_states = KdaStates.allocate(mesh_device, geometry, layer_ids=kda_layer_ids, num_slots=params.num_users)
        return KimiK3KvCaches(kvpe=kvpe, index=index, kda_states=kda_states)

    # --- migration table configs: 0 = kvpe, 1 = index, 2 = KDA recurrent, 3 = KDA convolution ---
    def cache_kind(self, config_id: int) -> str:
        kinds = ("kvpe", "index", "kda_recurrent", "kda_convolution")
        return kinds[config_id] if 0 <= config_id < len(kinds) else "other"

    def cache_layer_rows(self, config_id: int, num_layers: int) -> dict[int, int]:
        """Like K3's: every config sits on the model's layer axis; the index cache on the MLA layers."""
        kind = self.cache_kind(config_id)
        if kind in ("kvpe", "index"):
            ids = KimiK31Config.mla_layer_ids()
        elif kind in ("kda_recurrent", "kda_convolution"):
            ids = KimiK31Config.kda_layer_ids()
        else:
            return {}
        return {layer: layer for layer in ids if layer < num_layers}

    def cache_head_dim(self, config_id: int):
        kind = self.cache_kind(config_id)
        if kind == "kvpe":
            return KimiK31Config.KV_LORA_RANK + KimiK31Config.QK_ROPE_HEAD_DIM
        if kind == "index":
            return KimiK31Config.INDEX_HEAD_DIM
        return None
