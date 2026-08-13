# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Runtime lifecycle for one Kimi-K3 hybrid prefill pipeline rank."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import allocate_kda_state
from models.demos.deepseek_v3_d_p.tt.kimi_k3.segment import KimiK3SegmentLayout, TtKimiK3Segment
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import KimiK3Caches, MlaKvCaches
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache


def validate_kimi_k3_chunk_shape(params: PrefillRunParams) -> None:
    """Reject a physical chunk the distributed KDA recurrence cannot execute."""
    if params.chunk_size <= 0 or params.chunk_size % params.sp_factor:
        raise ValueError(
            f"Kimi-K3 chunk_size {params.chunk_size} must be positive and divisible by SP factor {params.sp_factor}"
        )
    recurrence = kimi_k3_program_config().recurrence
    local_tokens = params.chunk_size // params.sp_factor
    if local_tokens % recurrence.chunk_size:
        raise ValueError(f"Kimi-K3 local chunk {local_tokens} must be divisible by KDA tile {recurrence.chunk_size}")
    local_chunks = local_tokens // recurrence.chunk_size
    if params.sp_factor > 1 and local_chunks % recurrence.summary_group_chunks:
        required = params.sp_factor * recurrence.chunk_size * recurrence.summary_group_chunks
        raise ValueError(
            f"Kimi-K3 chunk_size {params.chunk_size} must be divisible by {required} for SP={params.sp_factor} "
            f"and KDA summary_group_chunks={recurrence.summary_group_chunks}"
        )


def allocate_kimi_k3_caches(
    *,
    mesh_device: ttnn.MeshDevice,
    hf_config,
    params: PrefillRunParams,
) -> KimiK3Caches:
    """Allocate the global compact 24-slot MLA schema and every rank-local KDA carry."""
    validate_kimi_k3_chunk_shape(params)
    layout = KimiK3SegmentLayout.build(params.first_layer_idx, params.num_layers)
    mla = None
    kda = []
    try:
        mla = MlaKvCaches(
            kvpe=allocate_mla_kvpe_cache(
                mesh_device=mesh_device,
                hf_config=hf_config,
                max_seq_len=params.max_seq_len,
                mesh_shape=params.mesh_shape,
                sp_axis=params.sp_axis,
                num_layers=len(KimiK3Config.mla_layer_ids()),
                num_users=params.num_users,
            )
        )
        kda_config = kimi_k3_kda_config()
        kda_program_config = kimi_k3_program_config()
        for _ in range(params.num_users):
            slot = {}
            kda.append(slot)
            for layer_idx in layout.kda_layer_indices:
                slot[layer_idx] = allocate_kda_state(
                    mesh_device,
                    kda_config,
                    kda_program_config,
                    tp_axis=params.tp_axis,
                )
    except Exception:
        for slot in kda:
            for state in slot.values():
                state.deallocate()
        if mla is not None:
            ttnn.deallocate(mla.kvpe.storage)
        raise
    return KimiK3Caches(mla=mla, kda=kda)


class KimiK3Runtime:
    """Drive one real K3 segment while the common runner owns transport and cache lifetime."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        hf_config,
        params: PrefillRunParams,
        query_cache_id: str = KimiK3Config.HF_REVISION,
    ) -> None:
        validate_kimi_k3_chunk_shape(params)
        if params.use_trace:
            raise ValueError("Kimi-K3 composed trace is not implemented; use_trace must be false")
        if params.kv_only_last_layer:
            raise ValueError("Kimi-K3 AttnRes requires the final MLA and MLP outputs; kv_only_last_layer must be false")
        if params.weight_cache_path is None:
            raise ValueError("Kimi-K3 composition requires an explicit TTNN weight/query cache path")
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.params = params
        # The common runner's runtime protocol reads these rank/trace fields through ``runtime.config``.
        # PrefillRunParams already is the immutable resolved config for this runtime, so expose that same
        # object rather than maintaining a second K3-specific copy that can drift.
        self.config = params
        self.layout = KimiK3SegmentLayout.build(params.first_layer_idx, params.num_layers)
        if params.is_first_rank != self.layout.is_first or params.is_last_rank != self.layout.is_last:
            raise ValueError(
                "Kimi-K3 rank role disagrees with its certified segment: "
                f"segment=[{params.first_layer_idx}, {params.first_layer_idx + params.num_layers}), "
                f"is_first_rank={params.is_first_rank}, is_last_rank={params.is_last_rank}"
            )
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode

        self.segment = TtKimiK3Segment(
            mesh_device=mesh_device,
            hf_config=hf_config,
            layout=self.layout,
            seq_len=params.chunk_size,
            max_seq_len=params.max_seq_len,
            num_users=params.num_users,
            weight_cache_path=Path(params.weight_cache_path),
            query_cache_id=query_cache_id,
            num_links=params.num_links,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            capacity_factor=params.capacity_factor,
            gate_fallback_mode=GateComputeMode[params.gate_mode_name],
            routing_use_l1_small_for_semaphores=True,
            overlap_shared_expert_with_dispatch=params.overlap_shared_expert_with_dispatch,
        )
        self.compiled = False

    def _allocate_kda_slot(self):
        config = kimi_k3_kda_config()
        program_config = kimi_k3_program_config()
        slot = {}
        try:
            for layer_idx in self.layout.kda_layer_indices:
                slot[layer_idx] = allocate_kda_state(
                    self.mesh_device,
                    config,
                    program_config,
                    tp_axis=self.params.tp_axis,
                )
        except Exception:
            self._deallocate_kda_slot(slot)
            raise
        return slot

    @staticmethod
    def _deallocate_kda_slot(slot) -> None:
        for state in slot.values():
            state.deallocate()

    def _reset_kda_slot(self, caches: KimiK3Caches, slot_id: int) -> None:
        replacement = self._allocate_kda_slot()
        old = caches.kda[slot_id]
        caches.kda[slot_id] = replacement
        self._deallocate_kda_slot(old)

    def make_placeholder_activation(self) -> ttnn.Tensor:
        candidates = 1 if self.layout.is_first else self.layout.num_sealed_before + 1
        local_tokens = self.params.chunk_size // self.params.sp_factor
        local_hidden = KimiK3Config.EMB_SIZE // self.params.tp_factor
        return ttnn.from_torch(
            torch.zeros(1, candidates, local_tokens, local_hidden, dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def make_chunk_input(self, token_ids: list[int]) -> ttnn.Tensor:
        if self.layout.is_first:
            if len(token_ids) != self.params.chunk_size:
                raise ValueError(
                    f"Kimi-K3 chunk input must contain exactly {self.params.chunk_size} padded tokens, "
                    f"got {len(token_ids)}"
                )
            return prepare_prefill_input_tensor(
                token_ids,
                self.mesh_device,
                self.params.sp_factor,
                False,
                self.params.mesh_shape,
                self.params.sp_axis,
            )
        return self.make_placeholder_activation()

    def compile(self, caches: KimiK3Caches) -> None:
        """Warm the exact segment without contaminating persistent KDA request state."""
        saved = caches.kda[0]
        caches.kda[0] = self._allocate_kda_slot()
        output = None
        try:
            output = self.segment.forward(
                self.make_chunk_input([0] * self.params.chunk_size),
                caches,
                slot_id=0,
                actual_start=0,
                actual_isl=self.params.chunk_size,
            )
            ttnn.synchronize_device(self.mesh_device)
        finally:
            if output is not None:
                ttnn.deallocate(output)
            self._deallocate_kda_slot(caches.kda[0])
            caches.kda[0] = saved
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        caches: KimiK3Caches,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        record_dev: Optional[ttnn.Tensor] = None,
    ) -> Optional[ttnn.Tensor]:
        del record_dev, request_id
        if not 0 <= slot_id < self.params.num_users:
            raise ValueError(f"slot_id {slot_id} is outside [0, {self.params.num_users})")
        if not 0 <= actual_start < actual_end <= self.params.max_seq_len:
            raise ValueError(
                f"invalid Kimi-K3 chunk [{actual_start}, {actual_end}) for max_seq_len={self.params.max_seq_len}"
            )
        if actual_start + self.params.chunk_size > self.params.max_seq_len:
            raise ValueError(
                f"Kimi-K3 physical chunk [{actual_start}, {actual_start + self.params.chunk_size}) exceeds "
                f"max_seq_len={self.params.max_seq_len}"
            )
        if actual_end - actual_start > self.params.chunk_size:
            raise ValueError(
                f"Kimi-K3 chunk length {actual_end - actual_start} exceeds physical chunk size "
                f"{self.params.chunk_size}"
            )
        if d2h_service is not None:
            raise NotImplementedError("Kimi-K3 device-side D2H layer acknowledgements are not composed yet")
        if actual_start == 0:
            self._reset_kda_slot(caches, slot_id)
        output = self.segment.forward(
            input_tensor,
            caches,
            slot_id=slot_id,
            actual_start=actual_start,
            actual_isl=actual_end - actual_start,
        )
        if self.layout.is_last:
            ttnn.deallocate(output)
            return None
        return output

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        del layer_ack_channel
        raise NotImplementedError("Kimi-K3 layer acknowledgements require KDA carry migration support")

    def set_layer_completion_sink(self, sink) -> None:
        del sink
        raise NotImplementedError("Kimi-K3 layer completions require KDA carry migration support")

    def capture_trace(self, _kv_cache) -> None:
        raise NotImplementedError("Kimi-K3 composed trace is not implemented")

    def release_trace(self) -> None:
        self.segment.release()

    def build_kv_chunk_table(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Kimi-K3 migration must describe compact MLA slots plus KDA recurrent/convolution carries"
        )
