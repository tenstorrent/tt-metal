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
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry, KdaStates, allocate_native_state
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
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
    local_rows: int = LOCAL_ROWS
    model: TtKimiK3Transformer = field(init=False)
    state: dict = field(init=False)

    def __post_init__(self):
        self.sp = tuple(self.mesh_device.shape)[SP_AXIS]
        self.chunk = self.sp * self.local_rows
        self.max_seq_len = self.chunks * self.chunk
        self.config = kimi_k3_hf_config(max_seq=max(8192, self.max_seq_len))
        last = self.case.layers[-1]
        self.state = {
            "layers": [
                make_layer_state(layer, seed=self.seed, variant=local, kv_only=self.case.kv_only and layer == last)
                for local, layer in enumerate(self.case.layers)
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
            self.geometry = KdaContractGeometry.from_kda_config(
                kimi_k3_kda_config(), mesh_shape=tuple(self.mesh_device.shape), sp_axis=SP_AXIS, tp_axis=TP_AXIS
            )
            self.slabs = KdaStates.allocate(
                self.mesh_device, self.geometry, layer_ids=self.model.kda_states.layer_ids, num_slots=self.slot_num
            )
            self.model.kda_states.bind_slabs(self.slabs)

    # ------------------------------------------------------------------------------------- rows

    def permutation(self, start: int) -> torch.Tensor:
        return mla_row_permutation(start, self.sp, self.local_rows)

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
        # Shard by shard and concatenated on host: the same tensor as `stream_composer` gives (checked
        # bit-equal), ~9x faster -- the composer path was most of the suite's readback time.
        # SP (tokens, dim 2) runs down mesh rows, TP (hidden, dim 3) across columns.
        rows, columns = tuple(self.mesh_device.shape)
        assert (SP_AXIS, TP_AXIS) == (0, 1), "shard order below assumes SP on mesh rows, TP on columns"
        shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
        planes = torch.cat(
            [torch.cat(shards[r * columns : (r + 1) * columns], dim=3) for r in range(rows)], dim=2
        ).float()
        planes = planes[0].permute(1, 0, 2)  # [N, P, d]
        natural = torch.empty_like(planes)
        natural[self.permutation(start)] = planes
        return natural

    # ------------------------------------------------------------------------------------- run

    def run_chunk(self, inputs: ttnn.Tensor, start: int, end: int | None = None, slot: int = 0) -> ChunkOutput:
        end = start + self.chunk if end is None else end
        # As `TtPrefillRuntime.prefill_chunk`'s eager branch calls it: `actual_isl` is what turns on
        # the MoE's padding-aware path for a chunk that ends mid-chunk.
        out = self.model.forward(
            inputs,
            kvpe_cache=self.kvpe,
            actual_isl=end - start,
            actual_start=start,
            actual_end=end,
            cache_user_id=slot,
        )
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

    def carry_addresses(self, slot: int = 0) -> dict:
        """Buffer addresses of every KDA carry: a capture bakes them in, so they must never move."""
        return {
            layer: (
                self.model.kda_states.read(layer, slot).recurrent.buffer_address(),
                self.model.kda_states.read(layer, slot).convolution.buffer_address(),
            )
            for layer in self.model.kda_states.layer_ids
        }

    def slab_round_trip(self, layer: int, slot: int = 0) -> bool:
        """Import the migration slab's copy of a carry into a fresh state; True iff it is bit-identical
        to the live carry on every device (the slab is written on every commit)."""
        imported = allocate_native_state(self.mesh_device, self.geometry)
        try:
            self.slabs.import_layer(imported, slot, layer)
            live = self.model.kda_states.read(layer, slot)
            pairs = ((live.recurrent, imported.recurrent), (live.convolution, imported.convolution))
            return all(
                torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
                for x, y in pairs
                for a, b in zip(ttnn.get_device_tensors(x), ttnn.get_device_tensors(y), strict=True)
            )
        finally:
            ttnn.deallocate(imported.recurrent)
            ttnn.deallocate(imported.convolution)

    def mla_kv(self, local: int, tokens: int, slot: int = 0) -> torch.Tensor:
        """``[tokens, 576]`` KV rows of a local MLA layer, natural order (bf8 cache, as fp32)."""
        kv_slot = self.model.schedule.kv_slot(local) + slot * self.model.schedule.num_mla_layers
        gathered = gather_cache_tp0(self.kvpe.storage, self.mesh_device)
        positions = blockcyclic_positions(self.sp, self.chunk, gathered.shape[1])
        return unrotate_cache_layer(gathered[kv_slot], positions, tokens)

    def moe_diagnostic(self, ffn_input: torch.Tensor, start: int, local: int = 0) -> dict:
        """Run this layer's ``TtMoe`` eagerly on a host-given FFN input (natural order ``[T, d]``) with
        intermediates on. Returns the device's top-k ``indices``/``weights`` ``[T, k]`` (natural
        order), the MoE ``output`` ``[T, d]``, and the dispatch accounting.

        The transformer's forward cannot return intermediates; this runs the same module object the
        layer uses, outside any capture, so the routing it reports is the device gate's on exactly
        the input the reference sees -- the basis of the flip-rate check and the device-routed MoE
        oracle (plan §5.2, S9).
        """
        ffn = self.model.layers[local].ffn
        perm = self.permutation(start)
        x = ttnn.from_torch(
            ffn_input[perm].bfloat16().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=self.attn_res.stream_mapper,
        )
        moe_input = ttnn.squeeze(x, dim=0)
        out, inter = ffn(moe_input, return_intermediates=True, actual_isl=ffn_input.shape[0], actual_start=start)
        ttnn.deallocate(x)

        def natural_rows(per_device: list[torch.Tensor]) -> torch.Tensor:
            # SP rank r's rows live on every chip of mesh row r; take TP column 0 and check the rest.
            columns = tuple(self.mesh_device.shape)[1]
            shards = []
            for rank in range(self.sp):
                row = per_device[rank * columns].reshape(-1, per_device[0].shape[-1])
                for column in range(1, columns):
                    replica = per_device[rank * columns + column].reshape(row.shape)
                    assert torch.equal(row, replica), f"routing differs across TP replicas on SP rank {rank}"
                shards.append(row)
            device_order = torch.cat(shards, dim=0)
            natural = torch.empty_like(device_order)
            natural[perm] = device_order
            return natural

        indices = natural_rows([ttnn.to_torch(t).to(torch.int64) for t in ttnn.get_device_tensors(inter.gate_indices)])
        weights = natural_rows([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(inter.gate_scores)])
        output = self.compose_planes(ttnn.unsqueeze(out, 0), start)[:, 0]

        counts = ttnn.to_torch(
            ttnn.unsqueeze_to_4D(inter.expert_token_counts),
            mesh_composer=ttnn.create_mesh_composer(self.mesh_device, ttnn.MeshComposerConfig(dims=[1, 0])),
        ).to(torch.int64)
        per_chip = counts.flatten().view(-1, ffn.experts_per_chip).sum(dim=1)
        capacity = ffn.dispatch_module.max_dispatch_buffer_token_size
        ttnn.deallocate(out)
        return {
            "indices": indices,
            "weights": weights,
            "output": output,
            "max_per_chip_dispatch": int(per_chip.max()),
            "dispatch_capacity": int(capacity),
        }

    # --------------------------------------------------------------------------------- trace (S6)
    #
    # The runtime's capture recipe (`TtPrefillRuntime._prepare_trace` / `capture_trace` /
    # `prefill_chunk`'s traced branch), minus D2D, acks and MTP: a persistent input and per-chunk
    # metadata at fixed addresses, a `SubDeviceTraceController` that segments the capture at the MoE
    # sub-device swaps, one warm traced forward, then capture once and replay per chunk. Applied to
    # the transformer directly because the runtime refuses an unaligned rank by design.

    def _meta_tensor(self, values) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.tensor(list(values), dtype=torch.int64).reshape(1, 1, 1, len(values)),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _forward_traced(self):
        # Full chunk as actual_isl, as the runtime does: on this path it only switches padding
        # awareness on; the real bounds come from the device metadata.
        return self.model.forward(
            self.trace_input,
            kvpe_cache=self.kvpe,
            actual_isl=self.chunk,
            metadata_msg=self.trace_meta_msg,
            actual_start=None,
            actual_end=None,
            cache_user_id=0,
            metadata=self.trace_meta,
        )

    def prepare_trace(self, template: ttnn.Tensor) -> None:
        """``template`` becomes the persistent input (its spec is what every replay copies into)."""
        if self.slot_num != 1:
            raise ValueError("traced K3 supports one KDA slot")
        self.trace_input = template
        self.trace_meta = ChunkMetadata(
            self._meta_tensor((0,)), self._meta_tensor((0,)), self._meta_tensor((self.chunk,)), None
        )
        self.trace_meta_msg = self._meta_tensor((0, 0, self.chunk))
        self.controller = SubDeviceTraceController(self.mesh_device)
        self.model.set_trace_controller(self.controller)
        warm = self._forward_traced()
        ttnn.synchronize_device(self.mesh_device)
        if warm is not None:
            ttnn.deallocate(warm)

    def capture_trace(self) -> None:
        self.controller.begin_capture()
        self.trace_output = self._forward_traced()
        self.controller.end_capture()
        ttnn.synchronize_device(self.mesh_device)
        get_tt_ccl(self.mesh_device).set_shared_rs_input_keepalive(None)

    def replay(self, inputs: ttnn.Tensor, start: int, end: int) -> ChunkOutput:
        ttnn.copy(inputs, self.trace_input)
        for value, destination in zip((0, start, end), self.trace_meta.scalars, strict=True):
            source = self._meta_tensor((value,))
            ttnn.copy(source, destination)
            ttnn.deallocate(source)
        message = self._meta_tensor((0, start, end))
        ttnn.copy(message, self.trace_meta_msg)
        ttnn.deallocate(message)
        self.controller.replay()
        ttnn.synchronize_device(self.mesh_device)
        result = ChunkOutput()
        if self.trace_output is None:
            return result
        planes = self.compose_planes(self.trace_output, start)
        if self.case.is_last:
            result.model_out = planes[:, 0]
        else:
            result.live, result.sealed = planes[:, 0], planes[:, 1:]
        return result

    def release_trace(self) -> None:
        self.controller.release()
        self.model.set_trace_controller(None)

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
