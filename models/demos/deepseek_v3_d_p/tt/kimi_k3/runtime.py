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

import ttnn
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

    def _my_kda_layer_ids(self):
        """The GLOBAL model layers in this rank's slice that carry a KDA state."""
        first = int(self.config.first_layer_idx)
        end = first + int(self.config.num_layers)
        return [layer for layer in KimiK3Config.kda_layer_ids() if first <= layer < end]

    def compile(self, kv_caches) -> None:
        """Bind the engine-owned KDA state slabs to the model's carries, then warm up as usual.

        The binding must precede the first forward: under trace `compile()` runs `_prepare_trace`,
        whose warm pass is the first time a commit executes, and every commit from then on also
        exports into the slabs. Binding later would leave the capture without the export.
        """
        self._bind_kda_states(kv_caches)
        super().compile(kv_caches)

    def _bind_kda_states(self, kv_caches) -> None:
        slabs = getattr(kv_caches, "kda_states", None)
        carries = getattr(self.model, "kda_states", None)
        if slabs is None and carries is None:
            return
        if slabs is None or carries is None:
            raise RuntimeError(
                "Kimi-K3 KDA state is split: the engine allocated "
                f"{'no' if slabs is None else 'the'} slabs and the model built "
                f"{'no' if carries is None else 'its'} carries; allocate_kv_cache and build_runtime "
                "disagree on which layers are KDA"
            )
        carries.bind_slabs(slabs)

    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        """Three stages, always, in table-config order: kvpe, KDA recurrent, KDA convolution.

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

        first_layer_idx = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        num_my_layers = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        last = first_layer_idx + num_my_layers
        mla_ids = KimiK3Config.mla_layer_ids()
        kda_ids = KimiK3Config.kda_layer_ids()
        first_slot = sum(1 for layer in mla_ids if layer < first_layer_idx)
        my_slots = [layer for layer in mla_ids if first_layer_idx <= layer < last]
        first_kda = sum(1 for layer in kda_ids if layer < first_layer_idx)
        my_kda = [layer for layer in kda_ids if first_layer_idx <= layer < last]

        if kv_caches.index is not None:
            raise RuntimeError("Kimi-K3 has no DSA index cache; a merged table here is unexpected")
        if self.config.dflash_enabled:
            raise RuntimeError("Kimi-K3 declares its own three migration stages; DFlash stages are not supported")

        null = KvCacheStage(0, 0, 0)
        # `allocate_kv_cache` returns kvpe=None for a rank that owns no full-attention layer, which
        # is reachable: a 1-layer bring-up run is layer 0, and layer 0 is KDA. No slabs, no stage.
        if kv_caches.kvpe is None:
            if my_slots:
                raise RuntimeError(f"no KVPE cache allocated but layers {my_slots} own MLA slabs")
            kvpe_stage = null
        else:
            slabs_per_user = kv_caches.kvpe.storage.shape[0] // self.config.num_users
            if slabs_per_user != len(my_slots):
                raise RuntimeError(
                    f"KVPE cache holds {slabs_per_user} slabs per slot but layers "
                    f"[{first_layer_idx}, {last}) own {len(my_slots)} MLA layers "
                    f"({my_slots}); the table cannot place them unless the cache is sized to the stage"
                )
            kvpe_stage = KvCacheStage(self.kv_migration_base_address(kv_caches), first_slot, len(my_slots))

        kda = getattr(kv_caches, "kda_states", None)
        if kda is None:
            if my_kda:
                raise RuntimeError(f"no KDA state slabs allocated but layers {my_kda} are KDA layers")
            return [kvpe_stage, null, null]
        if tuple(kda.layer_ids) != tuple(my_kda):
            raise RuntimeError(f"KDA slabs cover layers {kda.layer_ids} but this stage owns {my_kda}")
        if kda.num_slots != self.config.num_users:
            raise RuntimeError(f"KDA slabs hold {kda.num_slots} slots but the runtime serves {self.config.num_users}")
        return [
            kvpe_stage,
            KvCacheStage(int(kda.recurrent.buffer_address()), first_kda, len(my_kda)),
            KvCacheStage(int(kda.convolution.buffer_address()), first_kda, len(my_kda)),
        ]

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
        total = sum(stage["count"] for stage in stage_layout) if stage_layout else len(self._my_kda_layer_ids())
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
        """One merged table over kvpe (config "0") and the KDA slabs (configs "1" and "2").

        `stage_layouts` is one gathered layout per stage of `kv_migration_stages`, in order; None
        (single-rank / tests) gathers them here. Only the kvpe layout goes to the shared builder as a
        block-cyclic stage; the KDA layouts travel inside `KdaTableSpec`, since their configs are
        segment-addressed rather than token-addressed. Under pipeline parallelism rank 0 builds the
        table and may hold different KDA layers than a stage it describes, which is why the spec
        carries the geometry and the gathered layouts rather than only this rank's tensors.
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
        )

    def warmup_ack_count(self) -> int:
        """One record per KV-WRITING layer, not per layer.

        `capture_trace()`'s warm pass fires the ack from `zero_pad_and_ack`, which a block only calls
        when its attention wrote a KV slab. Kimi-K3 is hybrid: a 24-layer rank owns 6 MLA layers and
        69 of the model's 93 layers are KDA and ack nothing. The shared implementation reports the
        rank's LAYER count, so the runner's drain loop asked the FIFO for 24 records when 6 exist and
        blocked on the seventh -- every traced 4-rank run hung after capture, before the request loop,
        spinning at ~76% CPU with no log line after "layer-completion routing up".
        """
        if not self.config.use_trace or self._trace_d2h_service is None:
            return 0
        return len(self._my_mla_layer_ids())

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
                # This slot only. Zeroing every slot would wipe the in-flight carries of any OTHER
                # user mid-prefill, and a recurrent carry cannot be rebuilt from the chunk in front
                # of it -- the remaining chunks would be conditioned on nothing and the run would
                # return a plausible answer computed from the wrong history, with no error.
                slot_id = bound.arguments.get("slot_id", 0)
                states.reset(slot_id)
                logger.debug(f"KDA carries reset for slot {slot_id} at request head")
        return super().prefill_chunk(*args, **kwargs)
