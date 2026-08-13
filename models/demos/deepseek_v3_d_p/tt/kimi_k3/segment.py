# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""One contiguous Kimi-K3 pipeline segment over the real hybrid layer schedule."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResState, attn_res_segment, finalize_attn_res
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import KimiK3AttnResDeviceQueries, KimiK3AttnResQueryCache
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import KimiK3Caches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear


@dataclass(frozen=True)
class KimiK3SegmentLayout:
    """Global layer identities and the compact global MLA slots owned by one pipeline rank."""

    first_layer_idx: int
    num_layers: int
    layer_indices: tuple[int, ...]
    mla_layer_indices: tuple[int, ...]
    kda_layer_indices: tuple[int, ...]
    mla_slot_by_layer: Mapping[int, int]
    num_sealed_before: int

    @classmethod
    def build(cls, first_layer_idx: int, num_layers: int) -> "KimiK3SegmentLayout":
        stop = first_layer_idx + num_layers
        if first_layer_idx not in KimiK3Config.PIPELINE_RANK_STARTS:
            raise ValueError(
                f"Kimi-K3 segment must start at one of {sorted(KimiK3Config.PIPELINE_RANK_STARTS)}, "
                f"got {first_layer_idx}"
            )
        if num_layers <= 0 or stop > KimiK3Config.NUM_LAYERS:
            raise ValueError(
                f"Kimi-K3 segment [{first_layer_idx}, {stop}) is outside model depth " f"{KimiK3Config.NUM_LAYERS}"
            )
        starts = sorted(KimiK3Config.PIPELINE_RANK_STARTS)
        next_boundary = (
            starts[starts.index(first_layer_idx) + 1] if first_layer_idx != starts[-1] else KimiK3Config.NUM_LAYERS
        )
        if stop != next_boundary:
            raise ValueError(
                f"Kimi-K3 segment [{first_layer_idx}, {stop}) is not one of the certified 31-layer "
                f"production segments; expected [{first_layer_idx}, {next_boundary})"
            )
        layer_indices = tuple(range(first_layer_idx, stop))
        mla_set = set(KimiK3Config.mla_layer_ids())
        mla = tuple(layer_idx for layer_idx in layer_indices if layer_idx in mla_set)
        kda = tuple(layer_idx for layer_idx in layer_indices if layer_idx not in mla_set)
        if not mla:
            raise ValueError(f"Kimi-K3 segment [{first_layer_idx}, {stop}) owns no MLA cache slot")
        return cls(
            first_layer_idx=first_layer_idx,
            num_layers=num_layers,
            layer_indices=layer_indices,
            mla_layer_indices=mla,
            kda_layer_indices=kda,
            mla_slot_by_layer={layer_idx: KimiK3Config.mla_kv_slot(layer_idx) for layer_idx in mla},
            num_sealed_before=len(range(0, first_layer_idx, KimiK3Config.ATTN_RES_BLOCK_SIZE)),
        )

    @property
    def is_first(self) -> bool:
        return self.first_layer_idx == 0

    @property
    def is_last(self) -> bool:
        return self.first_layer_idx + self.num_layers == KimiK3Config.NUM_LAYERS


class TtKimiK3Segment:
    """Compose KDA/MLA attention, AttnRes, and dense/LatentMoE modules for one rank."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        hf_config,
        layout: KimiK3SegmentLayout,
        seq_len: int,
        max_seq_len: int,
        num_users: int,
        weight_cache_path: Path,
        query_cache_id: str,
        num_links: int = 1,
        sp_axis: int = 0,
        tp_axis: int = 1,
        capacity_factor: int = 2,
        gate_fallback_mode=None,
        routing_use_l1_small_for_semaphores: bool = True,
        overlap_shared_expert_with_dispatch: bool = True,
        state_dict_by_layer: Optional[Mapping[int, dict]] = None,
        embed_weight=None,
    ) -> None:
        self.mesh_device = mesh_device
        self.layout = layout
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.num_users = num_users
        self.weight_cache_path = Path(weight_cache_path)
        topology = per_axis_topology()
        state_dict_by_layer = state_dict_by_layer or {}
        self._require_weight_cache(
            mesh_device=mesh_device,
            layout=layout,
            cache_path=self.weight_cache_path,
            tp_axis=tp_axis,
            embed_weight=embed_weight,
            state_dict_by_layer=state_dict_by_layer,
        )
        self.attn_res = TtAttnRes(
            mesh_device,
            hidden_size=KimiK3Config.EMB_SIZE,
            eps=KimiK3Config.RMS_NORM_EPS,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            num_links=num_links,
        )
        candidate_queries = KimiK3AttnResQueryCache.load(
            self.attn_res,
            self.weight_cache_path,
            layout.layer_indices,
            cache_id=query_cache_id,
            include_output=layout.is_last,
        )
        if candidate_queries.layer_indices != layout.layer_indices:
            self._release_queries(candidate_queries)
            raise ValueError(
                f"AttnRes query layers {candidate_queries.layer_indices} do not match segment {layout.layer_indices}"
            )
        if layout.is_last and candidate_queries.output is None:
            self._release_queries(candidate_queries)
            raise ValueError("the final Kimi-K3 segment requires the model output AttnRes query")
        self.queries = candidate_queries

        self.blocks = []
        try:
            self.embed = (
                TtParallelEmbedding(
                    mesh_device=mesh_device,
                    vocab_size=KimiK3Config.VOCAB_SIZE,
                    emb_dim=KimiK3Config.EMB_SIZE,
                    torch_weight=embed_weight,
                    sp_axis=sp_axis,
                    tp_axis=tp_axis,
                    weight_cache_path=self.weight_cache_path,
                )
                if layout.is_first
                else None
            )
            kda_config = kimi_k3_kda_config()
            kda_program_config = kimi_k3_program_config()
            for layer_idx in layout.layer_indices:
                kwargs = {}
                if gate_fallback_mode is not None:
                    kwargs["gate_fallback_mode"] = gate_fallback_mode
                self.blocks.append(
                    TtPrefillBlock(
                        mesh_device=mesh_device,
                        config=hf_config,
                        model_cfg=KimiK3Config,
                        state_dict=state_dict_by_layer.get(layer_idx, {}),
                        layer_idx=layer_idx,
                        seq_len=seq_len,
                        dispatch_buffer_capacity_factor=capacity_factor,
                        num_links=num_links,
                        topology=topology,
                        sp_axis=sp_axis,
                        tp_axis=tp_axis,
                        is_balanced=False,
                        weight_cache_path=self.weight_cache_path,
                        is_chunked=True,
                        slot_num=num_users,
                        layer_num=len(KimiK3Config.mla_layer_ids()),
                        max_seq_len=max_seq_len,
                        kv_only=False,
                        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
                        overlap_shared_expert_with_dispatch=overlap_shared_expert_with_dispatch,
                        kda_config=kda_config,
                        kda_program_config=kda_program_config,
                        **kwargs,
                    )
                )
        except Exception:
            for block in self.blocks:
                block.release_sub_device_managers()
            self._release_queries(self.queries)
            raise

    @staticmethod
    def _require_weight_cache(
        *,
        mesh_device: ttnn.MeshDevice,
        layout: KimiK3SegmentLayout,
        cache_path: Path,
        tp_axis: int,
        embed_weight,
        state_dict_by_layer: Mapping[int, dict],
    ) -> None:
        """Reject incomplete cache-only construction before a module can persist placeholder weights."""
        missing = []
        init_checker(cache_path)
        try:
            if layout.is_first and embed_weight is None and not TtParallelEmbedding.check_cache_complete(cache_path):
                missing.append("embed")
            kda_config = kimi_k3_kda_config()
            experts_per_chip = KimiK3Config.NUM_ROUTED_EXPERTS // mesh_device.get_num_devices()
            for layer_idx in layout.layer_indices:
                if state_dict_by_layer.get(layer_idx):
                    continue
                if not TtPrefillBlock.check_cache_complete(
                    cache_path,
                    layer_idx,
                    is_dense=layer_idx < KimiK3Config.NUM_DENSE_LAYERS,
                    experts_per_chip=experts_per_chip,
                    model_cfg=KimiK3Config,
                    mesh_device=mesh_device,
                    tp_axis=tp_axis,
                    kda_config=kda_config,
                ):
                    missing.append(f"layer_{layer_idx}")
        finally:
            report_and_clear()
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing components)"
            raise FileNotFoundError(f"incomplete Kimi-K3 TTNN cache at {cache_path}: {preview}{suffix}")

    def _release_queries(self, queries: KimiK3AttnResDeviceQueries) -> None:
        for query in (*queries.pre, *queries.post):
            self.attn_res.release_query(query)
        if queries.output is not None:
            self.attn_res.release_query(queries.output)

    def forward(
        self,
        input_tensor: ttnn.Tensor,
        caches: KimiK3Caches,
        *,
        slot_id: int,
        actual_start: int,
        actual_isl: int,
    ) -> ttnn.Tensor:
        """Consume one rank input and return packed handoff state or the final AttnRes hidden."""
        if not 0 <= slot_id < self.num_users:
            raise ValueError(f"slot_id {slot_id} is outside [0, {self.num_users})")
        if self.layout.is_first:
            try:
                embedded = self.embed(input_tensor)
            finally:
                ttnn.deallocate(input_tensor)
            state = TtAttnResState(prefix_sum=embedded, block_residual=None)
        else:
            state = TtAttnResState.from_packed(input_tensor, num_sealed=self.layout.num_sealed_before)

        kda_states = caches.kda[slot_id]
        attn_fns = []
        mlp_fns = []
        for block in self.blocks:

            def attention(hidden, *, block=block):
                if block.is_kda:
                    old_state = kda_states[block.layer_idx]
                    output, next_state = block.forward_attention_module(
                        hidden,
                        kda_state=old_state,
                    )
                    old_state.deallocate()
                    kda_states[block.layer_idx] = next_state
                    return output
                output, _ = block.forward_attention_module(
                    hidden,
                    kvpe_cache=caches.mla.kvpe,
                    cache_layer_idx=self.layout.mla_slot_by_layer[block.layer_idx],
                    actual_start=actual_start,
                    cache_user_id=slot_id,
                )
                return output

            def mlp(hidden, *, block=block):
                return block.forward_mlp_module(
                    hidden,
                    actual_isl=actual_isl,
                    actual_start=actual_start,
                )

            attn_fns.append(attention)
            mlp_fns.append(mlp)

        state = attn_res_segment(
            self.attn_res,
            state,
            self.layout.layer_indices,
            self.queries.pre,
            self.queries.post,
            attn_fns,
            mlp_fns,
            block_size=KimiK3Config.ATTN_RES_BLOCK_SIZE,
        )
        if self.layout.is_last:
            return finalize_attn_res(self.attn_res, state, self.queries.output)
        return state.take_packed()

    def release(self) -> None:
        """Release query operands and per-block sub-device managers before mesh close."""
        if self.queries is not None:
            self._release_queries(self.queries)
            self.queries = None
        for block in self.blocks:
            block.release_sub_device_managers()
