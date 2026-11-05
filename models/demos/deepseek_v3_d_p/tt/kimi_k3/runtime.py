# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runtime for Kimi-K3.

`TtPrefillRuntime` is reused with Kimi-K3-specific transformer, carry and interval handling:

* the transformer it drives is `TtKimiK3Transformer` (the `MODEL_CLS` seam), because only 24 of
  Kimi-K3's 93 layers write a KV slab and its residual is block-structured, so it cannot reuse the
  shared block; and
* the KDA carries must be zeroed at the head of a request. A carry summarises the whole prefix
  behind it, so leaving the previous request's carry in place is not a small error — it conditions
  every token of the new one on text it never saw; and
* the chunk start must be 32-token aligned for KDA; the end may be any real length. Host bounds
  are checked before execution; device-only trace metadata must satisfy the same contract on
  every replay.
"""

from __future__ import annotations

import inspect

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import validate_kda_bounds
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime


class TtKimiK3Runtime(TtPrefillRuntime):
    MODEL_CLS = TtKimiK3Transformer

    def _build_model(self, state_dict: dict) -> None:
        # The shared build starts GLM-5.3's MTP predictor before MODEL_CLS runs, so reject here.
        if self.config.mtp_levels:
            raise ValueError(f"Kimi-K3 has no MTP predictor; got mtp_levels={self.config.mtp_levels}")
        super()._build_model(state_dict)

    @property
    def activation_planes(self) -> int:
        """The live stream plus every AttnRes snapshot sealed before this rank's first layer.

        A read scores the live sum against the whole sealed set, and the snapshots for upstream
        layers exist only on the rank that produced them, so they arrive in the payload. Mirrors
        `KimiK3Adapter.pipeline_activation_planes`, which sizes the socket — the two are read at the
        same boundary and disagreeing shows up as a rendezvous failure, not a wrong answer.
        """
        return 1 + self.config.first_layer_idx // KimiK3Config.ATTN_RES_BLOCK_SIZE

    def _pack_activation(self, hidden, partial):
        """The drafter partial rides the handoff as one more plane after the AttnRes ones: the payload is
        already [1, planes, rows, H], and widening it to 2H would double every sealed plane too."""
        packed = ttnn.concat([hidden, partial], dim=1)
        ttnn.deallocate(hidden)
        ttnn.deallocate(partial)
        return packed

    def _unpack_activation(self, packed):
        """Inverse of `_pack_activation`: every plane but the last is the AttnRes payload, the last is the partial."""
        hidden_planes = packed.shape[1] - 1
        hidden = ttnn.slice(packed, [0, 0, 0, 0], [1, hidden_planes, packed.shape[2], packed.shape[3]])
        partial = ttnn.slice(packed, [0, hidden_planes, 0, 0], list(packed.shape))
        return hidden, partial

    def make_placeholder_activation(self, dflash_packed=True):
        """Under DSpark the packed form is built the way it is sent: the AttnRes planes plus the partial's."""
        if not (self.config.dflash_enabled and dflash_packed):
            return super().make_placeholder_activation(dflash_packed)
        return self._pack_activation(super().make_placeholder_activation(False), self._make_partial_placeholder())

    def _make_partial_placeholder(self):
        """One plane at the hidden's width: the partial alone, without the AttnRes planes beside it."""
        hidden = super().make_placeholder_activation(dflash_packed=False)
        partial = ttnn.slice(hidden, [0, 0, 0, 0], [1, 1, hidden.shape[2], hidden.shape[3]])
        ttnn.deallocate(hidden)
        return partial

    def _dflash_tap_layer(self, target_layer_idx: int) -> int:
        """DSpark taps target t at layer t+1's attention input, so tap 23 belongs to the rank starting at 24."""
        return target_layer_idx + 1

    def _my_mla_layer_ids(self):
        """The GLOBAL model layers in this rank's slice that own a KV slab."""
        first = int(self.config.first_layer_idx)
        end = first + int(self.config.num_layers)
        return [layer for layer in KimiK3Config.mla_layer_ids() if first <= layer < end]

    def _schedule(self, first_layer_idx=None, num_my_layers=None):
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule

        first = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        count = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        return KimiK3LayerSchedule.build(KimiK3Config, first, count)

    def compile(self, kv_caches) -> None:
        """Hand the engine-owned KDA state slabs to the model, then warm up as usual.

        The slabs are the KDA carries' only store, so they must be on the model before the first
        forward: under trace `compile()` runs `_prepare_trace`, whose warm pass is the first time a
        KDA layer loads and stores, and the capture bakes the slab addresses in. The shared runtime
        passes the model only `kv_caches.kvpe`, which is why this is an attribute and not a forward
        argument.
        """
        slabs = getattr(kv_caches, "kda_states", None)
        has_kda_layers = self.model.kda_scratch is not None
        if (slabs is not None) != has_kda_layers:
            raise RuntimeError(
                "Kimi-K3 KDA state is split: the engine allocated "
                f"{'the' if slabs is not None else 'no'} slabs and the model has "
                f"{'its' if has_kda_layers else 'no'} KDA layers; allocate_kv_cache and build_runtime "
                "disagree on which layers are KDA"
            )
        if slabs is not None:
            self._check_kda_slabs(slabs, self.model.schedule.rank_kda_layer_ids())
        self.model.kda_states = slabs
        super().compile(kv_caches)

    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        """Three stages, always, in table-config order: kvpe, KDA recurrent, KDA convolution. Under DSpark
        the drafter's K and V follow, as in the shared runtime.

        Kimi-K3 writes a KV slab on 24 of its 93 layers, so a 12-layer rank owns 3. The shared
        implementation reports the rank's LAYER span, which disagrees with the cache it describes
        (`cache batch dim 3 != num_users(1) * num_my_layers(12)`) and, past that assert, would step
        the DRAM bank round-robin once per layer instead of once per slab and address every slab
        wrongly. So the kvpe stage is numbered in compacted MLA-slot space, and the two KDA stages in
        compacted KDA-slot space, each over one consolidated slab (`KdaStates`). The per-stage
        allgather is collective, so every rank returns exactly three, with `KvCacheStage(0, 0, 0)`
        standing in for a cache this rank does not hold.

        The compacted numberings are internal: `kv_table_layer_rows` and `kda_table_layer_rows` map
        them back to the model's layer axis before anything is published.
        """
        from models.demos.common.prefill.runners.migration import KvCacheStage

        schedule = self._schedule(first_layer_idx, num_my_layers)
        first_layer_idx, last = schedule.first_layer_idx, schedule.first_layer_idx + schedule.num_layers
        # Compacted slot spaces: how many slabs / carries the ranks before this one own.
        first_slot = sum(1 for layer in schedule.mla_layer_ids if layer < first_layer_idx)
        my_slots = [layer for layer in schedule.mla_layer_ids if first_layer_idx <= layer < last]
        first_kda = sum(1 for layer in KimiK3Config.kda_layer_ids() if layer < first_layer_idx)
        my_kda = list(schedule.rank_kda_layer_ids())

        if kv_caches.index is not None:
            raise RuntimeError("Kimi-K3 has no DSA index cache; a merged table here is unexpected")
        # An empty stage still sits at this rank's slot offset: the merged table requires the gathered
        # stages to tile [0, total), and one at 0 would break that on any later rank.
        null_kda = KvCacheStage(0, first_kda, 0)
        # `allocate_kv_cache` returns kvpe=None for a rank that owns no full-attention layer, which
        # is reachable: a 1-layer bring-up run is layer 0, and layer 0 is KDA. No slabs, no stage.
        if kv_caches.kvpe is None:
            if my_slots:
                raise RuntimeError(f"no KVPE cache allocated but layers {my_slots} own MLA slabs")
            kvpe_stage = KvCacheStage(0, first_slot, 0)
        else:
            slabs_per_user = kv_caches.kvpe.storage.shape[0] // self.config.num_users
            if slabs_per_user != len(my_slots):
                raise RuntimeError(
                    f"KVPE cache holds {slabs_per_user} slabs per slot but layers "
                    f"[{first_layer_idx}, {last}) own {len(my_slots)} MLA layers "
                    f"({my_slots}); the table cannot place them unless the cache is sized to the stage"
                )
            kvpe_stage = KvCacheStage(self._kvpe_base_address(kv_caches), first_slot, len(my_slots))

        kda = getattr(kv_caches, "kda_states", None)
        if kda is None:
            if my_kda:
                raise RuntimeError(f"no KDA state slabs allocated but layers {my_kda} are KDA layers")
            return [kvpe_stage, null_kda, null_kda, *self._dflash_stages(last)]
        self._check_kda_slabs(kda, my_kda)
        return [
            kvpe_stage,
            KvCacheStage(int(kda.recurrent.buffer_address()), first_kda, len(my_kda)),
            KvCacheStage(int(kda.convolution.buffer_address()), first_kda, len(my_kda)),
            *self._dflash_stages(last),
        ]

    def _check_kda_slabs(self, slabs, layer_ids) -> None:
        """The slabs must hold exactly `layer_ids` and every slot the runtime admits: the copies skip an
        out-of-range slot on device without an error."""
        if tuple(slabs.layer_ids) != tuple(layer_ids):
            raise RuntimeError(f"KDA slabs cover layers {slabs.layer_ids} but the runtime owns {tuple(layer_ids)}")
        if slabs.num_slots != self.config.num_users:
            raise RuntimeError(f"KDA slabs hold {slabs.num_slots} slots but the runtime serves {self.config.num_users}")

    def _dflash_stages(self, end_layer: int):
        """The drafter's K and V after the three stages. Their rows continue the model's layer axis (93..97 on
        the tail), where the drafter's layer completions are numbered too."""
        return self._dflash_migration_stages(end_layer) if self.config.dflash_enabled else []

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

    def kda_table_layer_rows(self, stage_layout):
        """The KDA analogue of `kv_table_layer_rows`: compacted KDA slot i -> model layer."""
        total = sum(stage["count"] for stage in stage_layout) if stage_layout else self._schedule().num_kda_layers
        return list(KimiK3Config.kda_layer_ids()[:total])

    def build_kv_chunk_table(
        self,
        kv_caches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers=None,
        stage_layouts=None,
    ) -> str:
        """One merged table over kvpe (config "00") and the KDA slabs (configs "01" and "02").

        `stage_layouts` is one gathered layout per stage of `kv_migration_stages`, in order; None
        (single-rank / tests) gathers them here. Only the kvpe layout goes to the shared builder as a
        block-cyclic stage; the KDA layouts travel inside `KdaTableSpec`, since their configs sit on
        the contract's synthetic position axis, not the token axis. Under pipeline parallelism rank 0
        builds the table and may hold different KDA layers than a stage it describes, which is why the
        spec carries the geometry and the gathered layouts rather than only this rank's tensors.
        """
        from models.demos.common.prefill.runners.migration import allgather_kv_stage_layout
        from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
        from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry
        from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import (
            KdaTableSpec,
            build_and_serialize_kv_chunk_table,
        )

        if kv_caches.kvpe is None:
            raise RuntimeError(
                "Kimi-K3 cannot publish a KV chunk table from a rank without an MLA layer; run at least "
                f"through layer {KimiK3Config.mla_layer_ids()[0]} or disable migration"
            )
        if stage_layouts is None:
            stage_layouts = [
                allgather_kv_stage_layout(
                    self.mesh_device, stage.base_addr, self.config.mesh_shape, stage.first_layer, stage.count
                )
                for stage in self.kv_migration_stages(kv_caches, first_layer_idx, num_my_layers)
            ]
        stage_layouts, dflash_caches, dflash_spec, dflash_first_layer = self._dflash_table_args(stage_layouts, 3)
        if len(stage_layouts) != 3:
            raise RuntimeError(
                f"Kimi-K3 declares three migration stages (kvpe, KDA recurrent, KDA convolution) but "
                f"{len(stage_layouts)} layouts were gathered; the caller must gather one per stage"
            )
        kvpe_layout, recurrent_layout, convolution_layout = stage_layouts

        # The gathered kvpe stage is authoritative for this rank's slab range (compacted MLA-slot space),
        # not the caller's model-layer span. Mirrors the shared implementation.
        my_rank = int(ttnn.distributed_context_get_rank())
        mine = [stage for stage in kvpe_layout if stage["rank"] == my_rank]
        if mine:
            first_layer_idx = int(mine[0]["first_layer"])
            num_my_layers = int(mine[0]["count"])

        kda = None
        if sum(stage["count"] for stage in recurrent_layout) > 0:
            slabs = getattr(kv_caches, "kda_states", None)
            geometry = KdaContractGeometry.from_kda_config(
                kimi_k3_kda_config(),
                mesh_shape=tuple(self.config.mesh_shape),
                sp_axis=self.config.sp_axis,
                tp_axis=self.config.tp_axis,
            )
            if slabs is not None and slabs.geometry != geometry:
                raise RuntimeError(f"KDA slab geometry {slabs.geometry} != runtime geometry {geometry}")
            kda = KdaTableSpec(
                geometry=geometry,
                recurrent=None if slabs is None else slabs.recurrent,
                convolution=None if slabs is None else slabs.convolution,
                recurrent_layout=recurrent_layout,
                convolution_layout=convolution_layout,
                layer_rows=self.kda_table_layer_rows(recurrent_layout),
            )

        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kvpe_cache=kv_caches.kvpe,
            seq_len=self.config.max_seq_len,
            num_layers=self.config.num_layers,
            mesh_shape=self.config.mesh_shape,
            sp_axis=self.config.sp_axis,
            tp_axis=self.config.tp_axis,
            num_users=self.config.num_users,
            chunk_size_global=self.config.chunk_size,
            path=path,
            first_layer_idx=first_layer_idx,
            num_my_layers=num_my_layers,
            stage_layouts=[kvpe_layout],
            layer_rows=self.kv_table_layer_rows([kvpe_layout]),
            kda=kda,
            dflash_caches=dflash_caches,
            dflash_spec=dflash_spec,
            dflash_first_layer=dflash_first_layer,
        )

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
        # Validate before resetting carries or replaying a trace. Device-only
        # metadata must satisfy the same aligned-start, nonempty-interval contract.
        validate_kda_bounds(actual_start, bound.arguments.get("actual_end"))
        if actual_start == 0:
            # This slot only. Zeroing every slot would wipe the in-flight carries of any OTHER
            # user mid-prefill, and a recurrent carry cannot be rebuilt from the chunk in front
            # of it -- the remaining chunks would be conditioned on nothing and the run would
            # return a plausible answer computed from the wrong history, with no error.
            slot_id = bound.arguments.get("slot_id", 0)
            self.model.reset_streams(slot_id)
            logger.debug(f"KDA carries reset for slot {slot_id} at request head")
        return super().prefill_chunk(*args, **kwargs)
