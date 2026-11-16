# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runtime for Hy4-preview.

Hy4 runs GLM-5.3's prefill stack unchanged: ``TtPrefillTransformer`` with sparse MLA, the DSA indexer
on Hy4's full/shared layer map, and the prefill MoE. Everything Hy4 changes there is configuration
(``reference/hy4_config.py``). The runtime differs from ``TtPrefillRuntime`` in two places:

* the weight-cache rebuild reads Hy4's checkpoint layout (fused routed experts, MXFP8 scales,
  ``linear_gate``) through ``tt/runners/hy4_checkpoint.py``;
* there is no MTP predictor yet: the shared build would start GLM-5.3's, which reads the DeepSeek
  ``model.layers.78`` keys; Hy4 stores its MTP layer as ``model.mtp_layers.0`` (#65925).

Hooks for the sibling issues:

* iHC (#65891) changes the residual stream between blocks, so it lands as a Hy4 transformer here via
  the ``MODEL_CLS`` seam, the way Kimi-K3 swaps in ``TtKimiK3Transformer``.
* The KV hooks (``build_kv_chunk_table``, ``kv_migration_stages``, ``set_layer_completion_sink``) are
  inherited. They already follow ``indexer_types``, so the index-key table has Hy4's 21 full layers and
  completions carry global layer indices.
"""

from __future__ import annotations

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
        super()._build_model(state_dict)
