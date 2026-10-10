# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build and drive one Kimi-K3 layer (or a short tail) through the production module stack.

``TtKimiK3Transformer`` is built exactly as the runtime builds a rank -- same block config, routing
semaphores in L1_SMALL, no weight cache -- at ``first_layer_idx = L``. A rank that does not hold
layer 0 receives the packed handoff ``[1, S+1, N, d]`` (plane 0 the live stream, then the sealed
set), which is how a synthetic inherited state gets in; ``allow_unaligned_entry`` lets L sit
anywhere in its block.

Rows: everything here is in NATURAL token order on the host. A chunk's device rows carry
``mla_row_permutation(start, sp, local_rows)`` (the identity at a chunk-aligned start); placing
and composing translate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import CHECKPOINT_PREFIX
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    mla_row_permutation,
    reconstruct_convolution_at_sp_rank,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.synthetic_weights import (
    make_attn_res_weights,
    make_embedding,
    make_final_norm,
    make_layer_state,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry, KdaStates
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_tp0, unrotate_cache_layer

SP_AXIS, TP_AXIS = 0, 1
LOCAL_ROWS = 640  # rows per chip: a legal KDA local T, and AttnRes's measured shape
H = KimiK3Config.EMB_SIZE


@dataclass(frozen=True)
class Case:
    """One test case: the global layers it runs and how the rank ends."""

    name: str
    layers: tuple[int, ...]
    kv_only: bool = False  # production last rank: KV write only
    full_tail: bool = False  # q_out read + final norm (non-pipelined model end)
    priority: str = "P0"

    @property
    def first(self) -> int:
        return self.layers[0]

    @property
    def is_last(self) -> bool:
        return self.layers[-1] == KimiK3Config.NUM_LAYERS - 1


CASES = (
    Case("L0", (0,)),
    Case("L12", (12,)),
    Case("L27", (27,)),
    Case("L36", (36,)),
    Case("L84", (84,)),
    Case("L87", (87,)),
    Case("L92kv", (92,), kv_only=True),
    Case("L92full", (92,), full_tail=True, priority="P1"),
    Case("T91_92kv", (91, 92), kv_only=True, priority="P1"),
    Case("L89", (89,), priority="P2"),
    Case("L1", (1,), priority="P2"),
)


@dataclass
class ChunkOutput:
    live: torch.Tensor | None = None  # [T, d] natural order
    sealed: torch.Tensor | None = None  # [T, S, d]
    model_out: torch.Tensor | None = None  # [T, d] (full tail)


@dataclass
class SingleLayer:
    mesh_device: object
    case: Case
    seed: int
    chunks: int = 2
    slot_num: int = 1
    model: TtKimiK3Transformer = field(init=False)
    state: dict = field(init=False)

    def __post_init__(self):
        self.sp = tuple(self.mesh_device.shape)[SP_AXIS]
        self.chunk = self.sp * LOCAL_ROWS
        self.max_seq_len = self.chunks * self.chunk
        self.config = kimi_k3_hf_config(max_seq=max(8192, self.max_seq_len))
        last = self.case.layers[-1]
        self.state = {
            "layers": [
                make_layer_state(layer, seed=self.seed, kv_only=self.case.kv_only and layer == last)
                for layer in self.case.layers
            ],
            "attn_res_weights": make_attn_res_weights(self.case.first, len(self.case.layers), seed=self.seed),
            "attn_res_prefix": CHECKPOINT_PREFIX,
        }
        if self.case.first == 0:
            self.state["embed_weight"] = make_embedding(self.seed)
        if self.case.full_tail:
            self.state["norm_weight"] = make_final_norm(self.seed)

        self.model = TtKimiK3Transformer(
            self.mesh_device,
            self.config,
            KimiK3Config,
            self.state,
            num_layers=len(self.case.layers),
            seq_len=self.chunk,
            first_layer_idx=self.case.first,
            is_first_rank=self.case.first == 0,
            is_last_rank=self.case.is_last,
            kv_only_last_layer=self.case.kv_only,
            allow_unaligned_entry=True,
            build_tail=self.case.full_tail,
            is_chunked=True,
            slot_num=self.slot_num,
            max_seq_len=self.max_seq_len,
            weight_cache_path=None,
            num_links=2,
            topology=per_axis_topology(),
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
            is_balanced=False,
            dispatch_buffer_capacity_factor=2,
            routing_use_l1_small_for_semaphores=True,
        )
        self.kvpe = None
        if self.model.schedule.num_mla_layers:
            self.kvpe = allocate_mla_kvpe_cache(
                mesh_device=self.mesh_device,
                hf_config=self.config,
                max_seq_len=self.max_seq_len,
                mesh_shape=tuple(self.mesh_device.shape),
                sp_axis=SP_AXIS,
                num_layers=self.model.schedule.num_mla_layers,
                num_users=self.slot_num,
            )
        self.slabs = None
        if self.model.kda_states is not None:
            geometry = KdaContractGeometry.from_kda_config(
                kimi_k3_kda_config(), mesh_shape=tuple(self.mesh_device.shape), sp_axis=SP_AXIS, tp_axis=TP_AXIS
            )
            self.slabs = KdaStates.allocate(
                self.mesh_device, geometry, layer_ids=self.model.kda_states.layer_ids, num_slots=self.slot_num
            )
            self.model.kda_states.bind_slabs(self.slabs)

    # ------------------------------------------------------------------------------------- rows

    def permutation(self, start: int) -> torch.Tensor:
        return mla_row_permutation(start, self.sp, LOCAL_ROWS)

    @property
    def attn_res(self):
        return self.model._attn_res

    def place_state(self, live: torch.Tensor, sealed: torch.Tensor, start: int) -> ttnn.Tensor:
        """Pack ``[T, d]`` live + ``[T, S, d]`` sealed (natural order) into the device handoff."""
        perm = self.permutation(start)
        planes = torch.cat([live.unsqueeze(1), sealed], dim=1)[perm]  # [T, S+1, d] device rows
        return ttnn.from_torch(
            planes.permute(1, 0, 2).unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=self.attn_res.stream_mapper,
        )

    def place_tokens(self, tokens: torch.Tensor, start: int) -> ttnn.Tensor:
        return prepare_prefill_input_tensor(
            tokens.tolist(),
            self.mesh_device,
            self.sp,
            False,
            tuple(self.mesh_device.shape),
            SP_AXIS,
            chunk_start=start,
        )

    def compose_planes(self, tensor: ttnn.Tensor, start: int) -> torch.Tensor:
        """``[1, P, N, d]`` on device -> ``[T, P, d]`` natural order, fp32."""
        planes = ttnn.to_torch(tensor, mesh_composer=self.attn_res.stream_composer).float()
        planes = planes[0].permute(1, 0, 2)  # [N, P, d]
        natural = torch.empty_like(planes)
        natural[self.permutation(start)] = planes
        return natural

    # ------------------------------------------------------------------------------------- run

    def run_chunk(self, inputs: ttnn.Tensor, start: int, end: int | None = None, slot: int = 0) -> ChunkOutput:
        end = start + self.chunk if end is None else end
        out = self.model.forward(inputs, kvpe_cache=self.kvpe, actual_start=start, actual_end=end, cache_user_id=slot)
        result = ChunkOutput()
        if out is None:
            return result
        if self.case.is_last:
            result.model_out = self.compose_planes(out, start)[:, 0]
        else:
            planes = self.compose_planes(out, start)
            result.live, result.sealed = planes[:, 0], planes[:, 1:]
        ttnn.deallocate(out)
        return result

    def kda_carry(self, layer: int, slot: int = 0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """A KDA layer's carry per SP rank (it is replicated across SP): ``([H, K, V]...)``, ``([3, 3*q]...)``.

        Keyed by GLOBAL layer index, as ``KdaStateCache`` is."""
        state = self.model.kda_states.read(layer, slot)
        mesh = self.mesh_device
        local_width = KimiK3Config.KDA_NUM_HEADS // tuple(mesh.shape)[TP_AXIS] * KimiK3Config.KDA_HEAD_DIM
        recurrent = [
            reconstruct_state_at_sp_rank(state.recurrent, mesh, SP_AXIS, TP_AXIS, rank)
            .float()
            .reshape(KimiK3Config.KDA_NUM_HEADS, KimiK3Config.KDA_HEAD_DIM, KimiK3Config.KDA_HEAD_DIM)
            for rank in range(self.sp)
        ]
        convolution = [
            reconstruct_convolution_at_sp_rank(state.convolution, mesh, SP_AXIS, TP_AXIS, rank, local_width)
            .float()
            .reshape(KimiK3Config.KDA_SHORT_CONV_KERNEL_SIZE - 1, -1)
            for rank in range(self.sp)
        ]
        return recurrent, convolution

    def mla_kv(self, local: int, tokens: int, slot: int = 0) -> torch.Tensor:
        """``[tokens, 576]`` KV rows of a local MLA layer, natural order (bf8 cache, as fp32)."""
        kv_slot = self.model.schedule.kv_slot(local) + slot * self.model.schedule.num_mla_layers
        gathered = gather_cache_tp0(self.kvpe.storage, self.mesh_device)
        positions = blockcyclic_positions(self.sp, self.chunk, gathered.shape[1])
        return unrotate_cache_layer(gathered[kv_slot], positions, tokens)

    def routed_experts(self, local: int = 0):
        provider = self.state["layers"][local].get("routed_expert_weights")
        return None if provider is None else provider.reference_experts()

    def close(self):
        if self.model.kda_states is not None:
            self.model.kda_states.deallocate()
        if self.slabs is not None:
            ttnn.deallocate(self.slabs.recurrent)
            ttnn.deallocate(self.slabs.convolution)
        self.model.release_sub_device_managers()
