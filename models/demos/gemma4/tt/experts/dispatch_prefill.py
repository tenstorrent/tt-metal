# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert prefill on the DeepSeek dispatch/combine stack (token routing instead of masked dense compute).

Layout: the mesh rows (axis 0) form the dispatch group and the columns (axis 1) are the dispatch groups,
exactly the expert placement the lane engine already uses (ep over the lane axis = 32 experts per column,
tp over the rows = E/(rows*cols) per chip). Every chip holds all tokens of the call (the lane wrapper gathers
them); each row takes its slice of the tokens, dispatches them within its column to the chips owning the
chosen experts, the fused GeGLU routed-expert kernel runs only the selected experts, combine brings the
outputs back to the token's row, and the reduce sums the top-k and reduce-scatters over the columns. Two
all-gathers restore the replicated [tokens, hidden] layout the lane wrapper expects.

Enabled with GEMMA4_MOE_DISPATCH_PREFILL=1 (see experts/__init__.py). The direct ``dispatch`` / ``combine`` ops only
run along mesh axis 0 ("cluster_axis must be 0 ... not tested"); a single-row mesh (the 4-chip / 8-chip entries)
dispatches along its columns, which needs GEMMA4_MOE_DISPATCH_RING=1: the hop-forwarding ``dispatch_fabric2d`` /
``combine_fabric2d`` pair on a wrap-wired axis (open the mesh with FABRIC_1D_RING; the model's own collectives can
stay Linear). The 4-chip line can instead open as the column mesh MeshShape(4, 1) with GEMMA4_TP_AXIS=0 (TP over
the rows): the rows are then the one dispatch group (32 experts per chip, direct ops, FABRIC_1D) and no ring is
needed. The experts run on the model-local GeGLU op
(custom_ops/geglu_routed_expert, a ttnn.generic_op) over TtRoutedExpert's weights, buffers and index table;
TtRoutedExpert itself only loads and caches the weights, its fused op never runs here.
"""
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombine2dModule, TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import MoERoutingOutputs, TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.gemma4.custom_ops.geglu_routed_expert import GegluRoutedExpert
from models.demos.minimax_m3.tt.moe.tt_reduce import TtMiniMaxReduce

# tokens per dispatch chunk across the whole mesh (rows * seq_len_per_chip); every chip holds the chunk
CHUNK_TOKENS = int(os.environ.get("GEMMA4_MOE_DISPATCH_CHUNK", "4096"))


def dispatch_prefill_enabled(mesh_shape=None):
    """Effective switch: ``GEMMA4_MOE_DISPATCH_PREFILL`` when set, else the mesh-shape default (on for a Blackhole
    (4,1) column mesh; tt/mesh_orientation.py)."""
    from models.common.utility_functions import is_blackhole
    from models.demos.gemma4.tt.mesh_orientation import dispatch_prefill_default

    return dispatch_prefill_default(mesh_shape, blackhole=is_blackhole())


class _RoutingSetupOnAxis(TtMoERoutingSetup):
    """The reference routing setup shards its dispatch table over the mesh columns (one group per column) and runs
    the cross-chip offset cumsum along the rows. On a single-row mesh the dispatch group IS the row (4-chip / 8-chip
    entries): every column gets the one group's table row and the cumsum runs along ``cluster_axis`` = 1."""

    def __init__(self, mesh_device, expert_dispatch_table, cluster_axis, **kwargs):
        self.cluster_axis = cluster_axis
        if cluster_axis == 1:
            # the base class maps dims=(None, 0): dim 0 sharded over the columns -> replicate the single group row
            assert expert_dispatch_table.shape[0] == 1, "a single-row mesh has one dispatch group"
            expert_dispatch_table = expert_dispatch_table.repeat(mesh_device.shape[1], 1)
        super().__init__(mesh_device, expert_dispatch_table, **kwargs)

    def forward(self, ttnn_top_k_experts_indices, num_routed_experts, num_experts_per_tok):
        if self.cluster_axis == 0:
            return super().forward(ttnn_top_k_experts_indices, num_routed_experts, num_experts_per_tok)
        assert isinstance(ttnn_top_k_experts_indices, ttnn.Tensor), "device indices only (the gate's TILE output)"
        if ttnn_top_k_experts_indices.layout != ttnn.TILE_LAYOUT:
            ttnn_top_k_experts_indices = ttnn.to_layout(ttnn_top_k_experts_indices, ttnn.TILE_LAYOUT)
        if len(ttnn_top_k_experts_indices.shape) == 3:
            ttnn_top_k_experts_indices = ttnn.squeeze(ttnn_top_k_experts_indices, 0)
        expert_histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn_top_k_experts_indices, self.experts_in_dispatch_group, num_routed_experts, num_experts_per_tok
        )
        (
            global_dispatch_offsets,
            total_counts_per_expert,
            expert_region_offsets,
            all_global_dispatch_offsets,
        ) = ttnn.experimental.deepseek_prefill.offset_cumsum(
            expert_histograms,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            experts_per_chip=self.experts_per_chip,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_l1_small_for_semaphores=self.use_l1_small_for_semaphores,
        )
        return MoERoutingOutputs(
            global_dispatch_offsets,
            total_counts_per_expert,
            expert_region_offsets,
            expert_histograms,
            all_global_dispatch_offsets,
        )


class Gemma4DispatchExperts:
    """Routed experts for prefill via dispatch -> fused GeGLU experts -> combine -> reduce."""

    def __init__(
        self,
        mesh_device,
        config,
        expert_weights,  # list of E dicts {gate_proj [I,H], up_proj [I,H], down_proj [H,I]} (HF (out,in)), or None
        tensor_cache_path=None,
        layer_idx=0,
        weights_dtype=ttnn.bfloat8_b,
        num_links=1,
    ):
        self.mesh_device = mesh_device
        rows, cols = mesh_device.shape
        self.rows, self.cols = rows, cols
        self.num_experts = config.num_experts
        self.top_k = config.top_k  # Gemma4ExpertConfig
        self.hidden = config.hidden_size
        self.inter = config.moe_intermediate_size
        assert self.num_experts % (rows * cols) == 0
        self.experts_per_chip = self.num_experts // (rows * cols)
        # Dispatch group = the mesh rows (axis 0), one group per column. A single-row mesh (the 4-chip and 8-chip
        # entries) dispatches along its columns instead: one group, no replicated axis. The stack needs a group of
        # more than one chip.
        self.dispatch_axis = 0 if rows > 1 else 1
        self.rep_axis = 1 - self.dispatch_axis
        dgs, ndg = mesh_device.shape[self.dispatch_axis], mesh_device.shape[self.rep_axis]
        self.dgs, self.ndg = dgs, ndg
        self._padding_cache: dict = {}  # n_tokens -> per-device [real, pad_side] (see _padding_config)
        self._row_masks: dict = {}  # indices shape -> (row index, sentinel) TILE tensors
        assert dgs > 1, "the dispatch stack needs a dispatch group of more than one chip"
        # The stack's routing table is sharded with the groups over the mesh columns and its offset cumsum runs
        # along the rows; _RoutingSetupOnAxis carries the single-row mesh (one group over the columns).
        assert (
            CHUNK_TOKENS % (dgs * ttnn.TILE_SIZE) == 0
        ), f"chunk {CHUNK_TOKENS} must split into {dgs} tile-aligned rows"
        self.seq_per_chip = CHUNK_TOKENS // dgs
        topology = ttnn.Topology.Linear
        # Ring transport (dispatch_fabric2d / combine_fabric2d): the only dispatch/combine pair that runs along
        # mesh axis 1, so the single-row 4-chip and 8-chip entries need it; the axis must be a fabric ring.
        self.ring = os.environ.get("GEMMA4_MOE_DISPATCH_RING", "0") == "1"
        self.num_links = num_links
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
            self.seq_per_chip, self.num_experts, self.top_k, rows * cols, dgs, self.top_k
        )
        assert experts_per_chip == self.experts_per_chip
        assert max_buf >= dgs * self.seq_per_chip * self.top_k, "dispatch buffer below the drop-free worst case"
        self.metadata_len, self.max_buf = metadata_len, max_buf
        assert self.ring or self.dispatch_axis == 0, (
            "the direct dispatch/combine ops only run along mesh axis 0: a single-row mesh needs "
            "GEMMA4_MOE_DISPATCH_RING=1 (dispatch_fabric2d / combine_fabric2d on a FABRIC_1D_RING mesh)"
        )
        table = ExpertMapping.create_dispatch_table(self.num_experts, dgs, ndg)
        self.routing_setup = _RoutingSetupOnAxis(
            mesh_device, table, self.dispatch_axis, num_links=num_links, experts_per_chip=experts_per_chip
        )
        self.tt_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh_device, table, dispatch_axis=self.dispatch_axis
        )
        if self.ring:
            self.dispatch = None  # ttnn.experimental.deepseek_prefill.dispatch_fabric2d, called in _chunk
            self.combine = TtCombine2dModule(
                mesh_device=mesh_device,
                experts_per_chip=experts_per_chip,
                num_experts_per_tok=self.top_k,
                seq_len_per_chip=self.seq_per_chip,
                cluster_axis=self.dispatch_axis,
                num_links=num_links,
                topology=ttnn.Topology.Ring,
            )
        else:
            self.dispatch = TtDispatchModule(
                mesh_device=mesh_device,
                dispatch_group_size=dgs,
                experts_per_chip=experts_per_chip,
                num_routed_experts=self.num_experts,
                num_experts_per_tok=self.top_k,
                metadata_len=metadata_len,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=self.seq_per_chip,
                emb_dim=self.hidden,
                cluster_axis=self.dispatch_axis,
                num_links=num_links,
                topology=topology,
                subdevice_id=None,
            )
            self.combine = TtCombineModule(
                mesh_device=mesh_device,
                dispatch_group_size=dgs,
                num_dispatch_groups=ndg,
                experts_per_chip=experts_per_chip,
                num_experts_per_tok=self.top_k,
                seq_len_per_chip=self.seq_per_chip,
                cluster_axis=self.dispatch_axis,
                num_links=num_links,
                topology=topology,
                init_zeros=False,
            )
        # global expert index table [groups, chips-in-group, ...]: dim 1 over the dispatch axis, dim 0 over the other
        idx_dims = (1, 0) if self.dispatch_axis == 0 else (0, 1)
        global_idx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=experts_per_chip, dispatch_group_size=dgs, num_dispatch_groups=ndg
            ),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=idx_dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        global_idx = ttnn.squeeze(ttnn.squeeze(global_idx, 0), 0)
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            global_expert_idx_table=global_idx,
            emb_dim=self.hidden,
            hidden_dim=self.inter,
            max_tokens=max_tok,
            torch_weights=expert_weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            weight_cache_path=Path(tensor_cache_path) if tensor_cache_path else None,  # the loader joins with '/'
            cache_name_prefix=f"layer_{layer_idx}.dispatch_experts",
            # Weight loader and cache only: the GeGLU op below runs the experts, so TtRoutedExpert's own
            # fused op (and its activation) never executes; Silu is the value the shared op validates.
            activation=ttnn.RoutedExpertActivation.Silu,
            hybrid_token_threshold=None,
        )
        # The Gemma 4 model-local GeGLU routed-expert op (custom_ops/geglu_routed_expert) on TtRoutedExpert's
        # ND-sharded bf4 weights and index table.
        self.geglu_op = GegluRoutedExpert(
            mesh_device,
            self.routed_expert.gate_projs,
            self.routed_expert.up_projs,
            self.routed_expert.down_projs,
            global_idx,
            max_tokens=max_tok,
        )
        self.reduce = TtMiniMaxReduce(
            mesh_device=mesh_device, topk_dim=3, cluster_axis=self.rep_axis, num_links=num_links, topology=topology
        )
        logger.info(
            f"Gemma4DispatchExperts: mesh {rows}x{cols} axis {self.dispatch_axis} {'ring' if self.ring else 'direct'}, "
            f"{experts_per_chip} experts/chip, chunk {CHUNK_TOKENS} "
            f"({self.seq_per_chip}/chip), dispatch buffer {max_buf} rows, max {max_tok} tokens/expert"
        )

    @staticmethod
    def hf_expert_list(gate_up_proj, down_proj):
        """HF fused gate_up_proj [E, 2I, H] (gate rows first) + down_proj [E, H, I] -> list of E (out,in) dicts."""
        E, two_i, _ = gate_up_proj.shape
        I = two_i // 2
        return [
            {
                "gate_proj": gate_up_proj[e, :I, :].float(),
                "up_proj": gate_up_proj[e, I:, :].float(),
                "down_proj": down_proj[e].float(),
            }
            for e in range(E)
        ]

    # ── Padding awareness (reference TtMoe / MiniMax contract) ─────────────────────────────────────────────
    # A dispatch chunk is CHUNK_TOKENS rows; the rows past the chunk's real tokens are zero padding that the
    # routing setup, dispatch, experts and combine would otherwise process (a 128-token prompt pads to 4096:
    # 97% of the expert work wasted). With GEMMA4_DISPATCH_PADDING=1 the padded rows' expert ids are set to
    # the sentinel (num_experts, whose dispatch-table column is -1) and dispatch gets the per-device
    # [local_real_tokens, pad_side] config that bounds its token loop, exactly as the DeepSeek gate does.
    PADDING_ENABLED = os.environ.get("GEMMA4_DISPATCH_PADDING", "1") != "0"

    def _padding_config(self, n_tokens):
        """Per-device ``[local_real_tokens, 0]`` (right padding) for a chunk with ``n_tokens`` real rows, or
        None when every row is real. mesh_partition hands dispatch-axis chip r the rows [r*S, (r+1)*S)."""
        S = self.seq_per_chip
        if not self.PADDING_ENABLED or n_tokens >= CHUNK_TOKENS:
            return None
        if n_tokens not in self._padding_cache:
            rows = [[min(S, max(0, n_tokens - r * S)), 0] for r in range(self.dgs)]
            mesh_shape = tuple(self.mesh_device.shape)
            dims = (0, None) if self.dispatch_axis == 0 else (None, 0)
            self._padding_cache[n_tokens] = ttnn.from_torch(
                torch.tensor(rows, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=mesh_shape),
            )
        return self._padding_cache[n_tokens]

    def _sentinel_padded_rows(self, indices, padding_config):
        """Set the padded rows' expert ids (TILE, [S, K]) to the ``num_experts`` sentinel so the routing setup
        counts nothing for them and dispatch maps them to -1 (mirrors TtMoEGatePrefill._sentinel_padded_rows)."""
        key = tuple(indices.shape)
        if key not in self._row_masks:
            rows, k = indices.shape[-2], indices.shape[-1]
            row_index = torch.arange(rows, dtype=torch.int16).reshape(rows, 1).expand(rows, k).reshape(key)
            self._row_masks[key] = tuple(
                ttnn.from_torch(
                    t,
                    device=self.mesh_device,
                    dtype=indices.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for t in (row_index, torch.full_like(row_index, self.num_experts))
            )
        row_index, sentinel = self._row_masks[key]
        real = ttnn.slice(padding_config, [0, 0], [1, 1])
        real = ttnn.typecast(ttnn.to_layout(real, ttnn.TILE_LAYOUT), indices.dtype)
        is_real = ttnn.lt(row_index, real)
        out = ttnn.where(is_real, indices, sentinel)
        for t in (real, is_real, indices):
            ttnn.deallocate(t)
        return out

    def _chunk(self, hidden, routing, n_tokens):
        """One dispatch chunk. hidden [1,1,T,H] and routing [1,1,T,E] replicated on every chip, T = CHUNK_TOKENS
        (rows beyond n_tokens are zero padding). Returns [1,1,T,H] replicated (full top-k sum)."""
        # each chip of the dispatch group takes its slice of the chunk (tokens [r*S, (r+1)*S)); the other axis
        # keeps the same slice
        x = ttnn.mesh_partition(hidden, dim=2, cluster_axis=self.dispatch_axis)  # [1,1,S,H] per chip
        r = ttnn.mesh_partition(routing, dim=2, cluster_axis=self.dispatch_axis)  # [1,1,S,E]
        weights, indices = ttnn.topk(r, k=self.top_k, dim=-1)  # [1,1,S,K] bf16 / uint16, TILE
        ttnn.deallocate(r)
        indices = ttnn.reshape(indices, (self.seq_per_chip, self.top_k))
        padding_config = self._padding_config(n_tokens)
        if padding_config is not None:
            indices = self._sentinel_padded_rows(indices, padding_config)
        offsets, counts, region_offsets, _, all_offsets = self.routing_setup(
            ttnn_top_k_experts_indices=indices,
            num_routed_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
        )
        indices_rm = ttnn.to_layout(ttnn.reshape(indices, (1, self.seq_per_chip, self.top_k)), ttnn.ROW_MAJOR_LAYOUT)
        weights_rm = ttnn.to_layout(ttnn.reshape(weights, (1, self.seq_per_chip, self.top_k)), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(indices)
        ttnn.deallocate(weights)
        x_rm = ttnn.to_layout(ttnn.reshape(x, (1, self.seq_per_chip, self.hidden)), ttnn.ROW_MAJOR_LAYOUT)
        if x_rm.dtype != ttnn.bfloat16:
            x_rm = ttnn.typecast(x_rm, ttnn.bfloat16)
        ttnn.deallocate(x)
        if self.ring:
            # every source chip's offsets row ([extent, E], replicated along the axis): a forwarding chip sizes
            # what it relays from them
            buffer, metadata = ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
                x_rm,
                indices_rm,
                all_offsets,
                self.tt_dispatch_table,
                counts,
                region_offsets,
                padding_config=padding_config,
                experts_per_chip=self.experts_per_chip,
                num_routed_experts=self.num_experts,
                num_experts_per_tok=self.top_k,
                metadata_len=self.metadata_len,
                max_dispatch_buffer_token_size=self.max_buf,
                seq_len_per_chip=self.seq_per_chip,
                cluster_axis=self.dispatch_axis,
                num_links=self.num_links,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            buffer, metadata = self.dispatch(
                x_rm, weights_rm, indices_rm, offsets, self.tt_dispatch_table, padding_config=padding_config
            )
        ttnn.deallocate(x_rm)
        ttnn.deallocate(offsets)
        expert_out = self.geglu_op(ttnn.reshape(buffer, (self.max_buf, self.hidden)), counts, region_offsets)
        ttnn.deallocate(buffer)
        expert_out = ttnn.unsqueeze(ttnn.unsqueeze(expert_out, dim=0), dim=0)
        if self.ring:
            combined = self.combine(expert_out, metadata, counts, region_offsets, all_offsets)  # [1,1,S,K,H]
        else:
            combined = self.combine(expert_out, metadata, counts, region_offsets)  # [1,1,S,K,H] row-major
        ttnn.deallocate(all_offsets)
        ttnn.deallocate(expert_out)
        ttnn.deallocate(metadata)
        ttnn.deallocate(counts)
        ttnn.deallocate(region_offsets)
        weights_rm = ttnn.to_memory_config(weights_rm, ttnn.DRAM_MEMORY_CONFIG)
        indices_rm = ttnn.to_memory_config(indices_rm, ttnn.DRAM_MEMORY_CONFIG)
        # weighted top-k sum of the local groups' slots, then reduce-scatter over the columns -> [S, H/cols]
        part = self.reduce(
            combined, weights=weights_rm, indices=indices_rm, expert_dispatch_table=self.tt_dispatch_table
        )
        ttnn.deallocate(combined)
        ttnn.deallocate(weights_rm)
        ttnn.deallocate(indices_rm)
        part = ttnn.reshape(part, (1, 1, self.seq_per_chip, self.hidden // self.ndg))
        if self.ndg > 1:
            full = ttnn.all_gather(part, dim=3, cluster_axis=self.rep_axis, topology=ttnn.Topology.Linear)  # [1,1,S,H]
            ttnn.deallocate(part)
        else:
            full = part
        out = ttnn.all_gather(full, dim=2, cluster_axis=self.dispatch_axis, topology=ttnn.Topology.Linear)  # [1,1,T,H]
        ttnn.deallocate(full)
        return out

    def forward(self, hidden, routing):
        """hidden [1,1,N,H] and routing [1,1,N,E] (final per-expert weights, zeros off the top-k), both replicated on
        every chip. Returns [1,1,N,H] replicated: sum over the top-k experts of weight * expert(hidden)."""
        n = hidden.shape[2]
        outs = []
        for start in range(0, n, CHUNK_TOKENS):
            end = min(start + CHUNK_TOKENS, n)
            h = hidden if (start == 0 and end == n) else ttnn.slice(hidden, [0, 0, start, 0], [1, 1, end, self.hidden])
            r = (
                routing
                if (start == 0 and end == n)
                else ttnn.slice(routing, [0, 0, start, 0], [1, 1, end, self.num_experts])
            )
            if end - start < CHUNK_TOKENS:
                pad = CHUNK_TOKENS - (end - start)
                h = ttnn.pad(h, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
                r = ttnn.pad(r, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
            o = self._chunk(h, r, end - start)
            if end - start < CHUNK_TOKENS:
                o = ttnn.slice(o, [0, 0, 0, 0], [1, 1, end - start, self.hidden])
            outs.append(o)
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
