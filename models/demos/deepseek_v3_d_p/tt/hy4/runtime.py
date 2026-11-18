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

* iHC (#65891) is in the shared ``TtPrefillTransformer`` / ``TtPrefillBlock``, switched on by
  ``Hy4Config.ENABLE_IHC`` (tt/hy4/ihc.py): the residual between blocks is 4 fp32 streams, and the iHC parameters are
  the ``ihc`` weight category, read by ``hy4_checkpoint``.
* The KV hooks (``build_kv_chunk_table``, ``kv_migration_stages``, ``set_layer_completion_sink``) are
  inherited. They already follow ``indexer_types``, so the index-key table has Hy4's 21 full layers and
  completions carry global layer indices.
"""

from __future__ import annotations

from loguru import logger

from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import load_hy4_trunk_state_dict
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime


class TtHy4Runtime(TtPrefillRuntime):
    # MODEL_CLS stays TtPrefillTransformer: iHC (#65891) is gated inside it by ENABLE_IHC.

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
