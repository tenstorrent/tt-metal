# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runtime for Kimi-K3.

`TtPrefillRuntime` is reused whole. Only two things differ, and both are consequences of Kimi-K3
carrying recurrent state that no other model in this package has:

* the transformer it drives is `TtKimiK3Transformer` (the `MODEL_CLS` seam), because only 24 of
  Kimi-K3's 93 layers write a KV slab and its residual is block-structured, so it cannot reuse the
  shared block; and
* the KDA carries must be zeroed at the head of a request. A carry summarises the whole prefix
  behind it, so leaving the previous request's carry in place is not a small error — it conditions
  every token of the new one on text it never saw.
"""

from __future__ import annotations

import inspect

from loguru import logger

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime


class TtKimiK3Runtime(TtPrefillRuntime):
    MODEL_CLS = TtKimiK3Transformer

    @property
    def activation_planes(self) -> int:
        """The live stream plus every AttnRes snapshot sealed before this rank's first layer.

        A read scores the live sum against the whole sealed set, and the snapshots for upstream
        layers exist only on the rank that produced them, so they arrive in the payload. Mirrors
        `KimiK3Adapter.pipeline_activation_planes`, which sizes the socket — the two are read at the
        same boundary and disagreeing shows up as a rendezvous failure, not a wrong answer.
        """
        return 1 + self.config.first_layer_idx // KimiK3Config.ATTN_RES_BLOCK_SIZE

    def _my_mla_layer_ids(self):
        """The GLOBAL model layers in this rank's slice that own a KV slab."""
        first = int(self.config.first_layer_idx)
        end = first + int(self.config.num_layers)
        return [layer for layer in KimiK3Config.mla_layer_ids() if first <= layer < end]

    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        """Number the KVPE stage in compacted MLA-slot space, not model-layer space.

        Kimi-K3 writes a KV slab on 24 of its 93 layers, so a 12-layer rank owns 3. The shared
        implementation reports the rank's LAYER span, which disagrees with the cache it describes
        (`cache batch dim 3 != num_users(1) * num_my_layers(12)`) and, past that assert, would step
        the DRAM bank round-robin once per layer instead of once per slab and address every slab
        wrongly. The DSA index cache already has this shape and is already numbered in compacted
        `full_indexer_rank` space; this is the same treatment for a hybrid KVPE cache.

        The compacted numbering is internal: `kv_table_layer_rows` maps it back to the model's layer
        axis before anything is published.
        """
        from models.demos.common.prefill.runners.migration import KvCacheStage

        first_layer_idx = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        num_my_layers = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        mla_ids = KimiK3Config.mla_layer_ids()
        first_slot = sum(1 for layer in mla_ids if layer < first_layer_idx)
        my_slots = [layer for layer in mla_ids if first_layer_idx <= layer < first_layer_idx + num_my_layers]

        slabs_per_user = kv_caches.kvpe.storage.shape[0] // self.config.num_users
        if slabs_per_user != len(my_slots):
            raise RuntimeError(
                f"KVPE cache holds {slabs_per_user} slabs per slot but layers "
                f"[{first_layer_idx}, {first_layer_idx + num_my_layers}) own {len(my_slots)} MLA layers "
                f"({my_slots}); the table cannot place them unless the cache is sized to the stage"
            )
        if kv_caches.index is not None:
            raise RuntimeError("Kimi-K3 has no DSA index cache; a merged table here is unexpected")
        return [KvCacheStage(self.kv_migration_base_address(kv_caches), first_slot, len(my_slots))]

    def kv_table_layer_rows(self, stage_layouts):
        """Publish slab i at its model layer, so table rows stay on the layer axis.

        Golden traces and every consumer are keyed by model layer, so widening here means the
        read-back side needs no knowledge of the compacted numbering.
        """
        mla_ids = KimiK3Config.mla_layer_ids()
        if stage_layouts:
            total_slabs = sum(stage["count"] for stage in stage_layouts[0])
        else:
            total_slabs = len(self._my_mla_layer_ids())
        return list(mla_ids[:total_slabs])

    def prefill_chunk(self, *args, **kwargs):
        """Reset the KDA carries at the start of a request, then defer to the shared runtime.

        `actual_start == 0` is the head of a request, and it is the only safe moment to zero: the
        reset is a device copy from a held zero tensor rather than a reallocation, so it must not
        land inside a captured region, which would re-zero on every replay and destroy the carry
        the trace exists to advance.
        """
        # Bound from the real signature rather than by counting positions: `actual_start` is the
        # fourth positional parameter, and reading the third instead would take `slot_id` — which is
        # 0 on every chunk of a single-user run, so the carries would be zeroed at every chunk
        # boundary and multi-chunk prefill would silently lose its recurrence.
        bound = inspect.signature(TtPrefillRuntime.prefill_chunk).bind_partial(self, *args, **kwargs)
        actual_start = bound.arguments.get("actual_start")
        if actual_start == 0:
            states = getattr(self.model, "kda_states", None)
            if states is not None:
                for slot in range(states.num_slots):
                    states.reset(slot)
                logger.debug(f"KDA carries reset for {states.num_slots} slot(s) at request head")
        return super().prefill_chunk(*args, **kwargs)
