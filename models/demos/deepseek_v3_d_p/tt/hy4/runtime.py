# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runtime for Hy4-preview.

Hy4 runs GLM-5.3's prefill stack unchanged: ``TtPrefillTransformer`` with sparse MLA, the DSA indexer
on Hy4's full/shared layer map, and the prefill MoE. Everything Hy4 changes there is configuration
(``reference/hy4_config.py``). The runtime differs from ``TtPrefillRuntime`` in two places:

* the weight-cache rebuild reads Hy4's checkpoint layout (fused routed experts, MXFP8 scales,
  ``linear_gate``) through ``tt/runners/hy4_checkpoint.py``;
* MLA / indexer caches built before the RoPE frame reorder (``hy4_checkpoint.HY4_MLA_CACHE_FORMAT``) are stale:
  their layers carry no format stamp, so their files are deleted and rebuilt from the checkpoint, and every layer
  is stamped once the model has built;
* there is no MTP predictor yet: the shared build would start GLM-5.3's, which reads the DeepSeek
  ``model.layers.78`` keys; Hy4 stores its MTP layer as ``model.mtp_layers.0`` (#65925).

The mesh layout is checked by the adapter, not here: ``Hy4Adapter`` resolves it (``tt/hy4/layout.py``) when the
runner asks for the weight-cache path, before the mesh opens, so the TP32 / DP1 layout fails there with its
blocking issues until the shared stack supports it; the GLM-5.3 dev / numerics layout builds this runtime.

Hooks for the sibling issues:

* iHC (#65891) changes the residual stream between blocks, so it lands as a Hy4 transformer here via
  the ``MODEL_CLS`` seam, the way Kimi-K3 swaps in ``TtKimiK3Transformer``.
* The KV hooks (``build_kv_chunk_table``, ``kv_migration_stages``, ``set_layer_completion_sink``) are
  inherited. They already follow ``indexer_types``, so the index-key table has Hy4's 21 full layers and
  completions carry global layer indices. ``build_kv_chunk_table`` first checks the chunk writes the 128-token
  stripe, and ``kv_layout`` describes the stripe and both caches for decode and serving (#65886).
"""

from __future__ import annotations

from loguru import logger

from models.demos.deepseek_v3_d_p.tt.hy4.kv_cache import KvBudget, KvStripe, describe_layers, hy4_stripe
from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import load_hy4_trunk_state_dict
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime


class TtHy4Runtime(TtPrefillRuntime):
    # MODEL_CLS stays TtPrefillTransformer until the iHC residual (#65891) needs its own transformer.

    _load_trunk_state_dict = staticmethod(load_hy4_trunk_state_dict)

    def _build_model(self, state_dict: dict) -> None:
        if self.config.mtp_levels:
            raise NotImplementedError(
                f"Hy4-preview prefill has no MTP predictor yet (#65925); got mtp_levels={self.config.mtp_levels}"
            )
        cache = self.config.weight_cache_path
        layers = range(self.config.first_layer_idx, self.config.first_layer_idx + self.config.num_layers)
        if cache:
            stale = hc.stale_mla_cache_layers(cache, layers)
            if stale:
                if not self._can_build_missing_cache():
                    raise RuntimeError(
                        f"Hy4 MLA/indexer cache at {cache} for layers {stale} predates format "
                        f"{hc.HY4_MLA_CACHE_FORMAT} (RoPE frame reorder) and there is no checkpoint at "
                        f"{self.config.hf_model_path} to rebuild it from"
                    )
                n = hc.purge_mla_cache(cache, stale)
                logger.warning(
                    f"Hy4: MLA/indexer cache of layers {stale[0]}..{stale[-1]} ({len(stale)} layers, {n} files) "
                    f"predates format {hc.HY4_MLA_CACHE_FORMAT}; deleted, rebuilding from the checkpoint"
                )
        super()._build_model(state_dict)
        if cache:
            hc.stamp_mla_cache(cache, layers)

    # --- context-parallel KV cache (#65886) ---
    @property
    def kv_stripe(self) -> KvStripe:
        """Where each token's KVPE and index-key rows live; raises on a 32-chip mesh if the chunk size would not
        write 128-token blocks."""
        return hy4_stripe(self.config.mesh_shape, self.config.sp_axis, self.config.chunk_size)

    def kv_layout(self, kv_caches=None) -> dict:
        """JSON-able description of both caches for decode (#65888 / #65889) and serving (#65924): the stripe,
        the slot formula, the layer -> index-key slot map and the per-token bytes. With ``kv_caches`` it also
        records the allocated shapes and dtypes."""
        types = list(getattr(self.hf_config, "indexer_types", None) or ["full"] * self.config.num_layers)
        layers = describe_layers(types)
        budget = KvBudget(num_layers=len(types), num_index_layers=len(layers["full_indexer_layers"]))
        extra = {
            "max_seq_len": self.config.max_seq_len,
            "num_users": self.config.num_users,
            "rows_per_chip": self.kv_stripe.rows_per_chip(self.config.max_seq_len),
            "values_per_token": budget.values_per_token,
            "bytes_per_token": budget.bytes_per_token,
            **layers,
        }
        if kv_caches is not None:
            extra["kvpe"] = {
                "format": str(kv_caches.kvpe.format.value),
                "shape": list(kv_caches.kvpe.storage.shape),
                "dtype": str(kv_caches.kvpe.storage.dtype),
            }
            if kv_caches.index is not None:
                extra["index"] = {"shape": list(kv_caches.index.shape), "dtype": str(kv_caches.index.dtype)}
        return self.kv_stripe.descriptor(**extra)

    def build_kv_chunk_table(self, kv_caches, path, **kwargs) -> str:
        # The table's block-cyclic period is config.chunk_size; it describes the 128-token stripe only at 4096.
        self.kv_stripe  # noqa: B018 -- raises on a chunk size decode could not address
        return super().build_kv_chunk_table(kv_caches, path, **kwargs)
