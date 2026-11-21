# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Routed experts, expert-parallel over all 32 chips.

Adapted from models/demos/deepseek_v3_d_p/tt/moe/tt_moe.py (TtMoERoutingSetup, TtDispatchModule, TtRoutedExpert's
extract -> FFN -> insert loop, TtCombineModule, TtReduceModule). 4 dispatch groups, one per TP column c, each of the 8
chips of its column (SP axis 0); group c owns experts 32c..32c+31, chip r of it experts 32c+4r..32c+4r+3. Per chip
(SP row r, TP column c):

    x        [1, 1, 640, 1536] ffn_norm -> all_gather(axis 1) -> x_full [1, 640, 6144] ROW_MAJOR
    routing  dense [1, 1, 640, 128] fp32 (the router's boundary) -> topk(4): indices uint16, weights bf16
    setup    masked_bincount (this group's experts) -> offset_cumsum(axis 0): dispatch offsets, counts, regions
    dispatch x_full rows to the chips of this column holding their experts (axis 0) -> flat buffer [N, 6144]
    experts  per local expert: extract its region [5120, 6144] -> gate / up (HiFi4, fp32 out) -> silu(gate) * up
             (fp32) -> down (bf16) -> insert back into the buffer
    combine  the buffer back to the source chips (axis 0) -> [1, 1, 640, 4, 6144], this group's slots only
    reduce   post_combine_reduce: sum_k weight_k * slot_k over this group's slots -> [1, 1, 640, 6144]
             -> EXPERTS_PARTIAL_DTYPE -> reduce_scatter(axis 1) sums the 4 groups -> bf16 [1, 1, 640, 1536]

EXPERTS_IMPL picks the expert step. "unified": unified_routed_expert_moe, as the DeepSeek / GLM / Kimi prefill runs it,
every local expert in one op on its own token count, x packed to bfloat8_b inside. "loop": extract -> gate / up /
silu * up / down -> insert per local expert, each over the most tokens an expert can get (all 5120 of its group).
"""

import os
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import sub_device, tp_gather
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import SP_AXIS, TP_AXIS
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings


def _compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("EXPERTS_MATH_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("EXPERTS_FP32_ACC"),
        packer_l1_acc=False,
    )


def _unified_compute_config(mesh):
    """unified_routed_expert_moe runs LoFi with a bf16 DEST whatever it is given, and rejects any other config."""
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def cache_complete(cache_dir) -> bool:
    """Whether a layer's routed-expert weight cache was written whole (TtExperts writes ``done`` last)."""
    return cache_dir is not None and (Path(cache_dir) / "done").is_file()


def _dump(host: ttnn.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{os.getpid()}.{path.name}")  # renamed into place, so a reader never sees a partial file
    ttnn.dump_tensor(tmp, host)
    os.replace(tmp, path)


class TtExperts:
    # read at call time, not build time: settings may change between runs of one loaded model
    @property
    def cfg(self):
        return _compute_config(self.mesh)

    @property
    def mid_dtype(self):
        return getattr(ttnn, settings.get("EXPERTS_INTERMEDIATE_DTYPE"))

    @property
    def topology(self):
        # the dispatch / combine (axis 0)
        return getattr(ttnn.Topology, settings.get("EXPERTS_TOPOLOGY"))

    @property
    def tp_topology(self):
        # the TP (axis 1) gather and reduce_scatter; the dispatch / combine use EXPERTS_TOPOLOGY
        return getattr(ttnn.Topology, settings.get("EXPERTS_TP_TOPOLOGY"))

    @property
    def partial_dtype(self):
        return getattr(ttnn, settings.get("EXPERTS_PARTIAL_DTYPE"))

    def __init__(self, mesh, expert_weight, n_experts: int, top_k: int, rows_per_chip: int, cache_dir=None):
        """expert_weight(e, m) -> [out, in] host tensor (any float dtype) of routed expert e, m in w1 / w3 ([I, H])
        and w2 ([H, I]). rows_per_chip: the chunk's tokens per SP row (chunk / 8). cache_dir: this layer's weight
        cache; a complete one (cache_complete) is read instead of expert_weight, which may then be None, otherwise
        the converted weights are written there."""
        self.mesh = mesh
        self.groups = mesh.shape[TP_AXIS]  # dispatch groups, one per TP column
        self.group_size = mesh.shape[SP_AXIS]  # chips per group, along SP
        self.n_experts, self.top_k, self.rows = n_experts, top_k, rows_per_chip
        assert n_experts % (self.groups * self.group_size) == 0
        self.per_chip = n_experts // (self.groups * self.group_size)
        self.per_group = self.per_chip * self.group_size
        assert rows_per_chip % 64 == 0, "masked_bincount needs a multiple of 64 rows per chip"
        self.weight_dtype = getattr(ttnn, settings.get("EXPERTS_WEIGHT_DTYPE"))
        # an expert gets at most every token of its group; a chip at most top_k per token (all its experts), plus the
        # tile padding of each local expert's region (offset_cumsum aligns them to 32 rows)
        self.max_per_expert = self.group_size * rows_per_chip
        cap = self.max_per_expert * min(settings.get("EXPERTS_CAPACITY_FACTOR"), top_k)
        self.buffer_rows = cap + ttnn.TILE_SIZE * (min(cap, self.per_chip) - 1)
        self.metadata_len = 3  # source chip, token, top-k slot

        def ep(t, dims, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(
                t,
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
            )

        # expert id -> its chip within the group, -1 outside the group; a trailing -1 column for the pad sentinel id
        table = torch.full((self.groups, n_experts + 1), -1, dtype=torch.int32)
        for g in range(self.groups):
            for j in range(self.per_group):
                table[g, g * self.per_group + j] = j // self.per_chip
        self.dispatch_table = ep(table, (None, 0), ttnn.int32)  # [1, E + 1] per chip: its column's row
        # (group, chip, local slot) -> global expert id: [per_chip] uint32 per chip
        ids = torch.arange(n_experts, dtype=torch.int32).reshape(self.groups, self.group_size, self.per_chip)
        self.global_ids = ttnn.squeeze(ttnn.squeeze(ep(ids, (1, 0), ttnn.uint32), 0), 0)

        # each local slot's weights, chip (r, c) holding expert 32c + 4r + slot, stored [K, N] (the checkpoint's
        # transpose), the layout unified_routed_expert_moe reads; the per-expert loop reads the same tensors
        cached = cache_complete(cache_dir)
        self.w = []
        for slot in range(self.per_chip):
            mats = {}
            for m in ("w1", "w3", "w2"):
                path = Path(cache_dir) / f"slot{slot}_{m}.tensorbin" if cache_dir is not None else None
                if cached:
                    host = ttnn.load_tensor(path)
                else:
                    stack = torch.stack(
                        [
                            expert_weight(int(ids[c, r, slot]), m).to(torch.bfloat16).t().contiguous()
                            for r in range(self.group_size)
                            for c in range(self.groups)
                        ]
                    )
                    host = ttnn.from_torch(
                        stack.reshape(self.group_size, self.groups, *stack.shape[1:]),
                        dtype=self.weight_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)),
                    )
                    del stack
                    host = ttnn.squeeze(ttnn.squeeze(host, 0), 0)
                    if path is not None:
                        _dump(host, path)
                mats[m] = ttnn.to_device(host, mesh, ttnn.DRAM_MEMORY_CONFIG)
            self.w.append(mats)
        if cache_dir is not None and not cached:
            (Path(cache_dir) / "done").touch()
        self.gates = [w["w1"] for w in self.w]
        self.ups = [w["w3"] for w in self.w]
        self.downs = [w["w2"] for w in self.w]

    def _linear(self, x, w, dtype):
        return ttnn.matmul(
            x,
            w,
            dtype=dtype,
            compute_kernel_config=self.cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self, x: ttnn.Tensor, routing: ttnn.Tensor, reduce: bool = True, indices=None, overlap=None
    ) -> ttnn.Tensor:
        """x: ffn_norm [1, 1, S/8, H/4] bf16 per chip; routing: dense [1, 1, S/8, E] fp32 per chip (TP-replicated)
        -> experts_out [1, 1, S/8, H/4] bf16 per chip. reduce=False (MOE_REDUCE=fused): this group's partial
        [1, 1, S/8, H] (bf16), the TP sum left to the caller. indices (ROUTER_OUT=topk): the router's top-k expert ids
        [1, 1, S/8, k] (deallocated here); routing is then their weights [1, 1, S/8, k] fp32."""
        rows, k = self.rows, self.top_k
        assert x.shape[-2] == rows, f"{x.shape[-2]} rows per chip, built for {rows}"

        # top-k of the dense routing: its k nonzero (positive) weights and their expert ids
        if indices is None:
            values, indices = ttnn.topk(routing, k=k, dim=-1, largest=True, sorted=True)
        else:
            values = routing  # the router's step output: the caller's, not deallocated here
        if indices.dtype != ttnn.uint16:
            idx16 = ttnn.typecast(indices, ttnn.uint16)
            ttnn.deallocate(indices)
            indices = idx16
        weights = ttnn.typecast(values, ttnn.bfloat16)
        if values is not routing:
            ttnn.deallocate(values)
        weights_rm = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(weights)
        indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)

        # routing setup: this group's per-expert counts, gathered over the group (axis 0) into write offsets
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn.reshape(indices, (rows, k)), self.dispatch_table, self.n_experts, k
        )
        offsets, counts, regions, all_offsets = ttnn.experimental.deepseek_prefill.offset_cumsum(
            hist,
            cluster_axis=SP_AXIS,
            num_links=settings.get("NUM_LINKS"),
            experts_per_chip=self.per_chip,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hist)
        ttnn.deallocate(all_offsets)

        # dispatch the full-width rows along the group
        x_full = tp_gather.gather(x, self.tp_topology, self.w[0]["w1"].shape[-2])
        if settings.get("EXPERTS_DISPATCH_IN") == "tile":
            # dispatch untilizes a TILE input itself; same bf16 rows
            x_rm = x_full
        else:
            x_rm = ttnn.to_layout(x_full, ttnn.ROW_MAJOR_LAYOUT)
            tp_gather.done(x_full, x)
        hidden = x_rm.shape[-1]
        idx3 = ttnn.reshape(indices_rm, (1, rows, k))
        ov = None
        if overlap is not None and sub_device.enabled():
            # dispatch on the row-0 sub-device, overlap() (the shared expert) on the rest of the grid
            ov = sub_device.overlap(self.mesh)
            ov.load()
        buf, meta = ttnn.experimental.deepseek_prefill.dispatch(
            input_tensor=ttnn.reshape(x_rm, (1, rows, hidden)),
            indices_tensor=idx3,
            expert_offsets_tensor=offsets,
            expert_dispatch_table_tensor=self.dispatch_table,
            dispatch_group_size=self.group_size,
            experts_per_chip=self.per_chip,
            num_routed_experts=self.n_experts,
            num_experts_per_tok=k,
            metadata_len=self.metadata_len,
            max_dispatch_buffer_token_size=self.buffer_rows,
            cluster_axis=SP_AXIS,
            num_links=settings.get("NUM_LINKS"),
            topology=self.topology,
            **(dict(subdevice_id=ov.dispatch_id) if ov is not None else {}),
        )
        if ov is not None:
            # nothing dispatch reads is freed before the clear: a buffer freed while the other sub-device's op
            # still runs could be handed to the concurrent op
            overlap(ov)
            ov.clear()
        elif overlap is not None:
            overlap(None)
        if x_rm is x_full:
            tp_gather.done(x_full, x)
        else:
            ttnn.deallocate(x_rm)
        ttnn.deallocate(offsets)

        if settings.get("EXPERTS_IMPL") == "unified":
            flat = self._unified(buf, counts, regions, hidden)
        else:
            flat = self._loop(buf, counts, regions, hidden)

        # back to the source chips: [1, 1, rows, k, H], this group's slots written
        combined = self._combine(flat, meta, counts, regions, hidden)
        ttnn.deallocate(flat)
        ttnn.deallocate(meta)
        ttnn.deallocate(counts)
        ttnn.deallocate(regions)
        return self._reduce(combined, weights_rm, indices_rm, indices, idx3, reduce)

    def _unified(self, buf, counts, regions, hidden):
        """unified_routed_expert_moe: every local expert in one op, each on its own token count only (read on the
        device), not on the expert's whole 5120-row region; it reads the ROW_MAJOR bf16 dispatch buffer directly
        (tilized and packed to bfloat8_b inside) and writes a fresh bfloat8_b buffer."""
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            ttnn.reshape(buf, (self.buffer_rows, hidden)),
            regions,
            counts,
            self.global_ids,
            self.gates,
            self.ups,
            self.downs,
            max_dispatched_tokens_per_expert=self.max_per_expert,
            compute_kernel_config=_unified_compute_config(self.mesh),
            activation=ttnn.RoutedExpertActivation.Silu,
        )
        ttnn.deallocate(buf)
        return out

    def _loop(self, buf, counts, regions, hidden):
        """The local experts one at a time, each on its whole region of the flat buffer (written back in place)."""
        flat = ttnn.to_layout(ttnn.reshape(buf, (self.buffer_rows, hidden)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(buf)
        for slot, w in enumerate(self.w):
            xe = ttnn.experimental.deepseek_prefill.extract(
                flat,
                regions,
                counts,
                self.global_ids,
                local_expert_id=slot,
                max_dispatched_tokens_per_expert=self.max_per_expert,
            )
            gate = self._linear(xe, w["w1"], self.mid_dtype)
            up = self._linear(xe, w["w3"], self.mid_dtype)
            ttnn.deallocate(xe)
            h = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=self.mid_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            y = self._linear(h, w["w2"], ttnn.bfloat16)
            ttnn.deallocate(h)
            ttnn.experimental.deepseek_prefill.insert(flat, y, regions, counts, self.global_ids, local_expert_id=slot)
            ttnn.deallocate(y)
        return flat

    def _combine(self, flat, meta, counts, regions, hidden):
        rows, k = self.rows, self.top_k
        return ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(flat, (1, 1, self.buffer_rows, hidden)),
            meta,
            counts,
            regions,
            dispatch_group_size=self.group_size,
            experts_per_chip=self.per_chip,
            num_experts_per_tok=k,
            seq_len_per_chip=rows,
            cluster_axis=SP_AXIS,
            num_links=settings.get("NUM_LINKS"),
            topology=self.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            init_zeros=False,
        )

    def _reduce(self, combined, weights_rm, indices_rm, indices, idx3, reduce=True):
        rows, k = self.rows, self.top_k
        # weighted sum over this group's slots (the others skipped by the dispatch table), then the 4 groups over TP
        part = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined,
            ttnn.reshape(weights_rm, (1, 1, rows, k, 1)),
            idx3,
            self.dispatch_table,
            expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(combined)
        ttnn.deallocate(weights_rm)
        ttnn.deallocate(indices_rm)
        ttnn.deallocate(indices)
        if not reduce:
            return part
        if part.dtype != self.partial_dtype:
            p = ttnn.typecast(part, self.partial_dtype)
            ttnn.deallocate(part)
            part = p
        y = ttnn.reduce_scatter(
            part, dim=3, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=self.tp_topology
        )
        ttnn.deallocate(part)
        if y.dtype != ttnn.bfloat16 and not settings.get("RESIDUAL_FP32_IN"):
            yb = ttnn.typecast(y, ttnn.bfloat16)
            ttnn.deallocate(y)
            y = yb
        return y
