# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GLM-5.4 prefill adapter: GLM-5.3 with layers 27-54 (0-indexed 26-53) run twice (``reference/glm_5_4_config.py``).

* Weights: GLM-5.4's are not published, so every shared tensor is GLM-5.3's (tier 2, ``tests/glm54_weights.py``):
  the checkpoint and TTNN cache paths are GLM-5.3's, and the cache directory is named ``glm_5_3`` so no GLM-5.4
  directory is ever created (user decision: no GLM-5.4 cache on Weka before the weights are released). An incomplete
  cache is an error here, never a rebuild into GLM-5.3's cache. The damp gates are synthetic host bf16 weights
  (``reference.glm_5_4.synthetic_gates``, seed ``GLM54_SYNTHETIC_SEED``), not in any cache.
* Placement: the runner splits the 106 layer passes (``pipeline_schedule``), not the 78 layers. With four ranks the
  split is G0 = L1-26, G1 = L27-54 pass 1 + gate 1, G2 = L27-54 pass 2 + gate 2 (its own copy of the loop weights,
  read from the same cache files, and of pass 1's KVPE), G3 = L55-78 (+ MTP). A rank holds whole loop passes. G1
  sends G2 the loop handoff (``tt/loop_handoff.py``) after the activation.
* Caches: a stage running loop pass 2 (one rank, or G2) keeps pass 2's window in a ring past each slot's pass-1
  positions (``kv_cache_slot_rows``, ``tt.mla.two_bank``).
* Not supported yet: a packed forward (the loop runs per chunk) and KV migration (the decode handoff of a looped model
  is deferred, Q2.3; ``prefill_runner`` builds no chunk table for a looped model).
"""

from __future__ import annotations

import os

from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config, glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches


class GLM54Adapter(GLM53Adapter):
    # --- identity & runner defaults ---
    name = "glm_5_4"
    model_config = GLM54Config
    # GLM-5.3's checkpoint, cache and golden trace (inherited paths), read unchanged.
    weight_cache_model = "glm_5_3"
    builds_missing_weight_cache = False
    # GLM-5.3's budget plus the damp gate's two collectives (its norm's stats all_gather and its own all_gather of the
    # normed input), each an all_gather multicast program at two 16 B/bank semaphores; both gates share the programs.
    l1_small_size = GLM53Adapter.l1_small_size + 2 * 2 * 16
    # The loop runs per chunk: TtPrefillTransformer.forward_packed does not run it.
    supports_packed_forward = False
    supports_packed_forward_trace = False
    supports_packed_forward_d2h_acks = False
    supports_packed_forward_indexer_indices = False
    supports_packed_forward_mtp = False

    def load_hf_config(self):
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        return glm_5_4_hf_config(max_seq=max_seq)

    @property
    def config_builder(self):
        return glm_5_4_hf_config

    def pipeline_schedule(self):
        from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import pass_schedule

        return pass_schedule(glm_5_4_hf_config())

    def layer_split_boundaries(self, num_layers: int) -> set:
        """Rank starts in pass positions: any but inside a loop pass."""
        schedule = self.pipeline_schedule()
        assert num_layers <= len(schedule), f"{num_layers} positions, GLM-5.4 runs {len(schedule)} layer passes"
        return schedule.valid_starts()

    def loop_gate_weights(self, hf_config) -> list:
        from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import GLM54_SYNTHETIC_SEED, synthetic_gates

        return synthetic_gates(hf_config.hidden_size, hf_config.loop_count, seed=GLM54_SYNTHETIC_SEED)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params) -> MlaKvCaches:
        """GLM-5.3's caches, sized for the stage: a stage running the loop only in passes >= 2 (G2) has no indexer,
        so no index cache; its KVPE cache holds the loop layers' pass-1 KV copy (bank A), never migrated."""
        from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import (
            runs_loop_without_indexer,
            stage_indexer_types,
        )

        loop_passes = params.loop_passes or ()
        if not runs_loop_without_indexer(hf_config, loop_passes):
            return super().allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
        assert not (params.mtp_levels and params.is_last_rank), "MTP runs after the coda, not on a loop pass-2 stage"
        stage_config = _with_indexer_types(hf_config, stage_indexer_types(hf_config, loop_passes))
        return super().allocate_kv_cache(mesh_device=mesh_device, hf_config=stage_config, params=params)

    def kv_cache_slot_rows(self, hf_config, params) -> tuple[int, bool]:
        """A stage that runs loop pass 2 (one rank, or G2) keeps pass 2's window in a ring past each slot's pass-1
        positions (tt.mla.two_bank). Such a cache is never migrated (prefill_runner builds no chunk table for a looped
        model), so its slots need no bank alignment."""
        from models.demos.deepseek_v3_d_p.tt.mla.two_bank import pass2_kv_cache_seq_len

        if not any(p >= 2 for p in params.loop_passes or ()):
            return super().kv_cache_slot_rows(hf_config, params)
        return pass2_kv_cache_seq_len(params.max_seq_len, params.chunk_size), False


def _with_indexer_types(hf_config, indexer_types: list):
    import copy

    config = copy.copy(hf_config)
    config.indexer_types = indexer_types
    return config
