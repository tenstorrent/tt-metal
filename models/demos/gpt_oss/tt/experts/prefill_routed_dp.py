# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routed prefill experts for the data-parallel prefill (tt/dp_prefill.py): every device holds all E experts and runs
the routed MoE of its own user, with no collective.

Built from a layer's TP module (RoutedPrefillExperts, E / N experts per device) and reuses its _chunk /
_pair_positions / _chunk_constants unchanged. What differs:

  * experts_per_chip = E and every table is replicated: the dispatch table maps every expert to the local chip, the
    local mask is all ones and the global-index table is 0..E-1;
  * slot v holds expert tp.expert_order[v] (the TP module's calibrated order, identity when None), so the TP MLP's
    routed router (TopKRouter.with_expert_order) emits the slot ids for both paths and the two route identically;
  * the weights are the TP module's own cached bytes, regrouped on host (no dequantize / requantize and no new cache
    files): the identity-order cache holds expert d * E/N + e as device d's shard of local slot e, and slot v of
    every device gets the shard of expert order[v]. Without a cache directory the HF weights are placed directly;
  * _routing counts and offsets the local histogram with stock ops: offset_cumsum all-gathers over its cluster
    axis, which TT_FATALs on a one-device axis (mesh axis 0 of 1x4) and on axis 1 would sum the other users';
  * __call__ returns the local sum (no all-reduce); call_groups returns the 4 partials TP's devices would compute
    for this user ([4, 1, S, H], one combine per TP expert group over the experts computed once), bit for bit, for
    dp_prefill.tp_exact_sum.
"""

import types

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import routed_expert_weight_memory_config

from .prefill_routed import (
    ROUTED_COMBINE_BLOCK,
    ROUTED_PREFILL_WEIGHTS_VERSION,
    RoutedPrefillExperts,
    _bias_dims,
    _l1_if_fits,
)


class DPRoutedPrefillExperts(RoutedPrefillExperts):
    """All E routed experts on every device; one user per device (data-parallel prefill)."""

    def __init__(self, tp: RoutedPrefillExperts, state_dict=None):
        mesh_device = tp.mesh_device
        self.mesh_device = mesh_device
        self.ccl_manager = tp.ccl_manager
        self.mesh_config = None
        self.num_experts = E = tp.num_experts
        self.top_k = k = tp.top_k
        self.hidden_size = tp.hidden_size
        self.intermediate_size = tp.intermediate_size
        self.num_devices = 1  # per user, one device holds every expert
        self.dispatch_group_size = 1
        self.num_dispatch_groups = 1
        self.experts_per_chip = E
        self.chunk_size = tp.chunk_size
        self.min_seq = tp.min_seq
        self.num_links = tp.num_links
        self.weights_dtype = tp.weights_dtype
        # Slot order of the TP module; the router both paths share emits these slot ids.
        self.expert_order = tp.expert_order
        self.cache_dir = tp.cache_dir
        self._consts = {}
        self._prefix_tri = tp._prefix_tri
        self._prefix_tri_small = tp._prefix_tri_small
        self._prefix_compute_config = tp._prefix_compute_config
        self._select_compute_config = tp._select_compute_config
        rep = ttnn.ReplicateTensorToMesh(mesh_device)

        def dev(t, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        # The TP module's slot selector / fold / expert-id constants do not depend on the device split: reuse them.
        self.slot_select = tp.slot_select
        self.slot_fold = tp.slot_fold
        self.slot_expert_ids_f32 = tp.slot_expert_ids_f32
        self.expert_ids_f32 = tp.expert_ids_f32
        self.local_mask_f32 = dev(torch.ones(1, E), ttnn.float32)
        # Every expert on chip 0 of a one-device dispatch group; the padding sentinel column stays -1.
        table = torch.full((1, E + 1), -1, dtype=torch.int32)
        table[0, :E] = 0
        self.dispatch_table = dev(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.global_expert_idx_table = ttnn.from_torch(
            torch.arange(E, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        order = list(self.expert_order) if self.expert_order is not None else list(range(E))
        prefix = f"routed_v{ROUTED_PREFILL_WEIGHTS_VERSION}"
        has_weights = bool(state_dict) and "gate_up_proj" in state_dict
        if self.cache_dir is not None and tp._cache_complete(prefix):
            projs, biases = self._from_tp_cache(tp, order, prefix)
        else:
            assert has_weights, "DP routed experts need the TP routed-weight cache or HF expert weights"
            projs, biases = self._from_state_dict(state_dict, order)
        self.routed_expert = types.SimpleNamespace(
            gate_projs=projs["gate"], up_projs=projs["up"], down_projs=projs["down"]
        )
        self.gate_biases, self.up_biases, self.down_biases = biases["gate"], biases["up"], biases["down"]
        # TP-exact partial sums (call_groups): group d = the slots TP device d holds; its table keeps only those
        # (post_combine_reduce skips the -1 entries), so group d's partial is TP device d's partial bit for bit.
        self.num_tp_groups = tp.num_devices
        epc_tp = tp.experts_per_chip
        self.group_tables = []
        for d in range(self.num_tp_groups):
            t = torch.full((1, E + 1), -1, dtype=torch.int32)
            t[0, d * epc_tp : (d + 1) * epc_tp] = 0
            self.group_tables.append(dev(t, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))

    def _weight_mem(self, n_dim):
        return routed_expert_weight_memory_config(self.mesh_device, n_dim, dram_nd_sharded=True)

    def _from_tp_cache(self, tp, order, prefix):
        """Replicated slot tensors from the TP identity-order cache files (host load, shards moved as stored)."""
        mesh_device = self.mesh_device
        rows, cols = mesh_device.shape
        epc_tp = tp.experts_per_chip
        mapper = ExpertMapping.get_weights_mesh_mapper(mesh_device)

        def replicate(shard, memory_config):
            t = ttnn.from_host_shards([shard] * (rows * cols), mesh_device.shape)
            t = ttnn.squeeze(ttnn.squeeze(t, 0), 0)
            mem = memory_config(t) if callable(memory_config) else memory_config
            return ttnn.to_device(t, mesh_device, memory_config=mem)

        def regroup(name, dtype, memory_config):
            shards = []
            for e in range(epc_tp):
                host = ttnn.as_tensor(
                    None,
                    mesh_mapper=mapper,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    cache_file_name=str(tp.cache_dir / f"{prefix}.local_{e}_{name}"),
                )
                shards.append(ttnn.get_device_tensors(host))
            out = [replicate(shards[o % epc_tp][o // epc_tp], memory_config) for o in order]
            del shards
            return out

        projs = {
            p: regroup(p, self.weights_dtype, lambda t: self._weight_mem(t.shape[-1])) for p in ("gate", "up", "down")
        }
        biases = {p: regroup(f"{p}_bias", ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG) for p in ("gate", "up", "down")}
        return projs, biases

    def _from_state_dict(self, state_dict, order):
        """Replicated slot tensors from the HF expert weights (no cache)."""
        from models.demos.gpt_oss_d_p.tt.moe.weights import prepare_routed_expert_weights

        mesh_device = self.mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        torch_weights, _ = prepare_routed_expert_weights(
            state_dict, self.num_experts, self.hidden_size, self.intermediate_size
        )
        projs = {"gate": [], "up": [], "down": []}
        for o in order:
            for proj in ("gate", "up", "down"):
                w = torch_weights[o][f"{proj}_proj"].T.contiguous()  # [K, N]
                host = ttnn.from_torch(w, dtype=self.weights_dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
                projs[proj].append(ttnn.to_device(host, mesh_device, memory_config=self._weight_mem(w.shape[-1])))
        del torch_weights
        b = _bias_dims(self.num_experts, self.hidden_size, self.intermediate_size, state_dict)
        biases = {
            p: [
                ttnn.from_torch(
                    b[p][o].reshape(1, -1).to(torch.bfloat16),
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep,
                )
                for o in order
            ]
            for p in ("gate", "up", "down")
        }
        return projs, biases

    def _routing(self, indices):
        """Counts and tile-aligned region offsets of this device's own histogram, with local ops.

        counts = masked_bincount (every expert local); region[e] = sum over e' < e of ceil32(counts[e']) in fp32
        (exact for integers < 2^24); both [1, E] uint32 ROW_MAJOR like offset_cumsum's outputs."""
        E = self.num_experts
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(indices, self.dispatch_table, E, self.top_k)
        counts = ttnn.reshape(hist, (1, E))
        h = ttnn.typecast(ttnn.to_layout(counts, ttnn.TILE_LAYOUT), ttnn.float32)
        aligned = ttnn.mul(ttnn.ceil(ttnn.mul(h, 1.0 / ttnn.TILE_SIZE)), float(ttnn.TILE_SIZE))
        ttnn.deallocate(h)
        incl = ttnn.cumsum(aligned, dim=-1)
        excl = ttnn.sub(incl, aligned)
        ttnn.deallocate(aligned)
        region = ttnn.to_layout(ttnn.typecast(excl, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(excl)
        return incl, counts, region  # incl stands in for the unused offsets (_chunk frees it)

    def _chunk_groups(self, x, indices, weights):
        """RoutedPrefillExperts._chunk with the combine run once per TP group: returns G lists of [1, 1, Cs, H] bf8
        blocks, list d = the partial of the experts TP device d holds (dispatch / experts computed once for all).
        Everything up to the expert op is _chunk's code unchanged."""
        C = x.shape[-2]
        H = self.hidden_size
        k = self.top_k
        offsets, counts, region_offsets = self._routing(indices)
        ttnn.deallocate(offsets)
        idx_rm = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, C, k))
        w_rm = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT)
        if w_rm.dtype != ttnn.bfloat16:
            w_rm = ttnn.typecast(w_rm, ttnn.bfloat16)
        w_5d = ttnn.reshape(w_rm, (1, 1, C, k, 1))
        consts = self._chunk_constants(C)
        pos_tm, pos_sm = self._pair_positions(indices, region_offsets, consts)
        if consts["scatter_groups"] == 1:
            tok_of_row = ttnn.scatter(consts["row_init"], -1, pos_sm, consts["pair_token"])
            ttnn.deallocate(pos_sm)
            tok_of_row = ttnn.to_layout(tok_of_row, ttnn.ROW_MAJOR_LAYOUT)
        else:
            tok_g = ttnn.scatter(consts["row_init_g"], -1, pos_sm, consts["pair_token_g"])
            ttnn.deallocate(pos_sm)
            tok_g_tile = ttnn.to_layout(tok_g, ttnn.TILE_LAYOUT)
            ttnn.deallocate(tok_g)
            tok_max = ttnn.max(tok_g_tile, dim=0, keepdim=True)
            ttnn.deallocate(tok_g_tile)
            tok_u = ttnn.typecast(tok_max, ttnn.uint32)
            ttnn.deallocate(tok_max)
            tok_of_row = ttnn.to_layout(tok_u, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(tok_u)
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=_l1_if_fits(C * H * 2))
        x_rm = ttnn.reshape(x_rm, (C, H))
        dispatched = ttnn.embedding(tok_of_row, x_rm, padding_idx=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(tok_of_row)
        ttnn.deallocate(x_rm)
        dispatched = ttnn.reshape(dispatched, (consts["rows"], H))
        expert_out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            dispatched,
            region_offsets,
            counts,
            self.global_expert_idx_table,
            self.routed_expert.gate_projs,
            self.routed_expert.up_projs,
            self.routed_expert.down_projs,
            max_dispatched_tokens_per_expert=C,
            activation=ttnn.RoutedExpertActivation.SwiGluOai,
            gate_biases=self.gate_biases,
            up_biases=self.up_biases,
            down_biases=self.down_biases,
        )
        ttnn.deallocate(dispatched)
        ttnn.deallocate(counts)
        ttnn.deallocate(region_offsets)
        expert_out = ttnn.experimental.deepseek_prefill.insert(
            expert_out,
            consts["spare_zeros"],
            consts["spare_start"],
            consts["spare_count"],
            consts["spare_idx"],
            local_expert_id=0,
        )
        out_rm = ttnn.to_layout(expert_out, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        if out_rm is not expert_out:
            ttnn.deallocate(expert_out)
        Cs = min(C, ROUTED_COMBINE_BLOCK)
        combined_mem = _l1_if_fits(Cs * k * H * 2)
        part_mem = _l1_if_fits(Cs * H * 2)
        pieces = [[] for _ in self.group_tables]
        for s in range(0, C, Cs):
            e = min(s + Cs, C)
            if s == 0 and e == C:
                pos_s, w_s, idx_s = pos_tm, w_5d, idx_rm
            else:
                pos_s = ttnn.slice(pos_tm, (s, 0), (e, k))
                w_s = ttnn.slice(w_5d, (0, 0, s, 0, 0), (1, 1, e, k, 1))
                idx_s = ttnn.slice(idx_rm, (0, s, 0), (1, e, k))
            combined = ttnn.embedding(
                pos_s, out_rm, padding_idx=consts["spare"], layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=combined_mem
            )
            combined = ttnn.reshape(combined, (1, 1, e - s, k, H))
            for d, table in enumerate(self.group_tables):
                part = ttnn.experimental.deepseek_prefill.post_combine_reduce(
                    combined, w_s, idx_s, table, expert_dim=3, output_memory_config=part_mem
                )
                pieces[d].append(ttnn.typecast(part, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                ttnn.deallocate(part)
            ttnn.deallocate(combined)
            if pos_s is not pos_tm:
                ttnn.deallocate(pos_s)
                ttnn.deallocate(w_s)
                ttnn.deallocate(idx_s)
        ttnn.deallocate(out_rm)
        ttnn.deallocate(pos_tm)
        ttnn.deallocate(w_rm)
        ttnn.deallocate(idx_rm)
        return pieces

    def call_groups(self, hidden_states, topk_indices, topk_weights):
        """The TP partial sums of this device's user: [G, 1, S, H] bf8, row block d = what TP device d's routed MoE
        computes for that user before its all-reduce (bit for bit)."""
        seq_len = hidden_states.shape[-2]
        assert self.supports(hidden_states.shape), f"DP routed prefill: unsupported shape {hidden_states.shape}"
        C = min(self.chunk_size, seq_len)
        per_group = [[] for _ in self.group_tables]
        for start in range(0, seq_len, C):
            end = min(start + C, seq_len)
            if start == 0 and end == seq_len:
                x_c, i_c, w_c = hidden_states, topk_indices, topk_weights
            else:
                x_c = ttnn.slice(hidden_states, (0, 0, start, 0), (1, 1, end, self.hidden_size))
                i_c = ttnn.slice(topk_indices, (start, 0), (end, self.top_k))
                w_c = ttnn.slice(topk_weights, (start, 0), (end, self.top_k))
            for d, blocks in enumerate(self._chunk_groups(x_c, i_c, w_c)):
                per_group[d].extend(blocks)
            if x_c is not hidden_states:
                ttnn.deallocate(x_c)
                ttnn.deallocate(i_c)
                ttnn.deallocate(w_c)
        parts = []
        for blocks in per_group:
            if len(blocks) == 1:
                parts.append(blocks[0])
            else:
                parts.append(ttnn.concat(blocks, dim=2))
                for t in blocks:
                    ttnn.deallocate(t)
        out = ttnn.concat(parts, dim=0)
        for t in parts:
            ttnn.deallocate(t)
        return out

    def __call__(self, hidden_states, topk_indices, topk_weights):
        """hidden_states: [1, 1, S, H], this device's user; topk ids are slot ids (see module docstring).
        Returns the local MoE output [1, 1, S, H] bf8 (no collective)."""
        seq_len = hidden_states.shape[-2]
        assert self.supports(hidden_states.shape), f"DP routed prefill: unsupported shape {hidden_states.shape}"
        x = hidden_states
        C = min(self.chunk_size, seq_len)
        outputs = []
        for start in range(0, seq_len, C):
            end = min(start + C, seq_len)
            if start == 0 and end == seq_len:
                x_c, i_c, w_c = x, topk_indices, topk_weights
            else:
                x_c = ttnn.slice(x, (0, 0, start, 0), (1, 1, end, self.hidden_size))
                i_c = ttnn.slice(topk_indices, (start, 0), (end, self.top_k))
                w_c = ttnn.slice(topk_weights, (start, 0), (end, self.top_k))
            outputs.extend(self._chunk(x_c, i_c, w_c))
            if x_c is not x:
                ttnn.deallocate(x_c)
                ttnn.deallocate(i_c)
                ttnn.deallocate(w_c)
        if len(outputs) == 1:
            return outputs[0]
        out = ttnn.concat(outputs, dim=2)
        for t in outputs:
            ttnn.deallocate(t)
        return out
