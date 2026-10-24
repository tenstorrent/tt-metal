# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Routed experts of the MiMo MoE layers on the [8, 4] mesh: expert parallel over the whole mesh.

Layout (plan.md, DeepSeek 2D dispatch groups, deepseek_v3_d_p/tt/moe/README.md): 4 dispatch groups, one per TP column
(axis 1), each an 8-chip group along SP (axis 0). Group c holds experts [64 c, 64 c + 64); chip (r, c) holds the 8
experts (8 c + r) * 8 + [0, 8). Every TP column sees the same 640 tokens of its SP row, so each column's combine yields
the partial sum over its 64 experts and a TP reduce_scatter sums the 4 groups.

Per chip, one chunk of 5120 tokens:
    x [1, 1, 160, 4096] bf16 (ffn_norm rows) -> all_gather over TP -> [640, 4096]
    routing [1, 1, 640, 256] fp32 (the router's dense output) -> bf16 -> topk 8 -> ids uint16, weights bf16
    -> masked_bincount + offset_cumsum (group-local offsets) -> dispatch over axis 0 -> flat buffer [B, 4096] bf16
    -> per local expert: extract (cap 5120) -> gate / up linear (fp32 out) -> silu(g) * u (fp32) -> down linear (bf16)
       -> insert
    -> combine over axis 0 -> [1, 1, 640, 8, 4096] -> post_combine_reduce (routing weights, other groups' slots skipped)
    -> reduce_scatter over TP (fp32) -> [1, 1, 160, 4096] bf16

Precision (known issues "bfp8 expert activations fail on a layer with outlier channels", "bf16 expert intermediates
and HiFi2 widen the per-token norm ratio", "The fused Silu expert kernel fails the norm-ratio check"): x stays bf16,
gate / up / h fp32, HiFi4 + fp32 dest; the fused routed-expert kernels are not used. Weights are the mxfp4 experts
(exact in bf16) as bfloat8_b in the checkpoint's [out, in] orientation with transpose_b, so the bfp8 blocks run along
K, where the 32-wide mxfp4 blocks make them exact.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

import ttnn

from . import ccl
from .settings import settings

METADATA_LEN = 3  # src chip, token, top-k slot


def dispatch_table(n_experts: int, group_size: int, groups: int) -> torch.Tensor:
    """[groups, n_experts + 1] int32: the chip (0..group_size-1) of each expert within its group, -1 elsewhere; the
    trailing column is the padding sentinel (always -1), as deepseek_v3_d_p ExpertMapping.create_dispatch_table."""
    per_group = n_experts // groups
    per_chip = per_group // group_size
    t = torch.full((groups, n_experts + 1), -1, dtype=torch.int32)
    for g in range(groups):
        for e in range(g * per_group, (g + 1) * per_group):
            t[g, e] = (e - g * per_group) // per_chip
    return t


def global_expert(group: int, chip: int, local: int, per_chip: int, group_size: int) -> int:
    """The global expert id of local expert ``local`` on chip ``chip`` of group ``group`` (column-major groups)."""
    return (group * group_size + chip) * per_chip + local


class Experts:
    def __init__(self, mesh, ck, layer_idx: int, rows_per_chip: int):
        cfg = ck.config
        self.mesh = mesh
        self.sp, self.tp = tuple(mesh.shape)  # dispatch group size, number of groups
        self.n_experts = cfg["n_routed_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.per_chip = self.n_experts // (self.sp * self.tp)
        self.rows = rows_per_chip  # tokens of one SP row (the dispatch source per chip)
        # an expert takes each token of its group at most once: per-expert cap = the group's tokens; a chip's flat
        # buffer holds up to min(top_k, per_chip) x that (every token picking several local experts) plus the tile
        # alignment of each local expert's region (deepseek_v3_d_p compute_constants). The dispatch kernel drops
        # rows past the buffer silently, so the factor defaults to the worst case.
        self.cap = self.sp * self.rows
        factor = settings.get("EXPERTS_BUFFER_FACTOR") or min(self.top_k, self.per_chip)
        self.buffer_rows = self.cap * factor + ttnn.TILE_SIZE * (self.per_chip - 1)
        self.num_links = settings.get("EXPERTS_NUM_LINKS")
        self.topology = getattr(ttnn.Topology, settings.get("EXPERTS_TOPOLOGY"))

        shape = tuple(mesh.shape)
        table = dispatch_table(self.n_experts, self.sp, self.tp)
        # each chip: its group's row [1, E + 1] (groups across columns, replicated down the rows)
        self.table = ttnn.from_torch(
            table,
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=(None, 0)),
        )
        idx = torch.tensor(
            [
                [[global_expert(c, r, l, self.per_chip, self.sp) for l in range(self.per_chip)] for r in range(self.sp)]
                for c in range(self.tp)
            ],
            dtype=torch.int32,
        )  # [groups, group chips, per_chip]: deepseek_v3_d_p create_global_expert_idx_table
        gidx = ttnn.from_torch(
            idx,
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=shape, dims=(1, 0)),
        )
        self.global_idx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)  # per chip [per_chip]

        p = f"model.layers.{layer_idx}.mlp.experts"

        def load(e):
            """Expert e's (gate, up, down) [N, K] as host bfloat8_b TILE tensors (bf16 holds mxfp4 exactly)."""
            q = f"{p}.{e}"
            return tuple(
                ttnn.from_torch(ck.mxfp4(f"{q}.{n}").to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
                for n in ("gate_proj", "up_proj", "down_proj")
            )

        # per local expert, one host shard per chip in row-major mesh order, converted in parallel (one 2-D [N, K]
        # tensor per chip: the extracted token rows are 2-D)
        self.w_gate, self.w_up, self.w_down = [], [], []
        with ThreadPoolExecutor(16) as ex:
            for l in range(self.per_chip):
                ids = [global_expert(c, r, l, self.per_chip, self.sp) for r in range(self.sp) for c in range(self.tp)]
                ws = list(ex.map(load, ids))
                for k, dst in enumerate((self.w_gate, self.w_up, self.w_down)):
                    host = ttnn.from_host_shards([x[k] for x in ws], mesh_shape=mesh.shape)
                    dst.append(ttnn.to_device(host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                del ws

        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("EXPERTS_FIDELITY")),
            math_approx_mode=False,
            fp32_dest_acc_en=settings.get("EXPERTS_FP32_ACC"),
            packer_l1_acc=True,
        )
        self.inter_dtype = ttnn.float32 if settings.get("EXPERTS_INTER_FP32") else ttnn.bfloat16
        self.out_fp32 = settings.get("EXPERTS_OUT_FP32")

    def _linear(self, x, w, dtype):
        return ttnn.linear(
            x,
            w,
            transpose_b=True,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_cfg,
        )

    def _ffn(self, buf, region, counts):
        """Every local expert's SwiGLU on its rows of the flat TILE buffer [B, H], written back in place."""
        for l in range(self.per_chip):
            xe = ttnn.experimental.deepseek_prefill.extract(
                buf, region, counts, self.global_idx, local_expert_id=l, max_dispatched_tokens_per_expert=self.cap
            )
            g = self._linear(xe, self.w_gate[l], self.inter_dtype)
            u = self._linear(xe, self.w_up[l], self.inter_dtype)
            xe.deallocate(True)
            h = ttnn.multiply(
                g,
                u,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)],
                dtype=self.inter_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            g.deallocate(True)
            u.deallocate(True)
            y = self._linear(h, self.w_down[l], ttnn.bfloat16)  # insert needs the buffer's dtype
            h.deallocate(True)
            buf = ttnn.experimental.deepseek_prefill.insert(buf, y, region, counts, self.global_idx, local_expert_id=l)
            y.deallocate(True)
        return buf

    def __call__(self, x, routing):
        """x [1, 1, 160, H] bf16 (this chip's ffn_norm rows), routing [1, 1, 640, E] fp32 (its SP row's dense
        routing weights, the same on every TP column) -> [1, 1, 160, H] bf16 (this chip's rows of the MoE output)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        R, K = self.rows, self.top_k
        H = x.shape[-1]

        # top-k ids and weights from the dense routing (the 8 nonzero entries of each row)
        r16 = ttnn.typecast(routing, ttnn.bfloat16, memory_config=dram)
        w, ids = ttnn.topk(r16, k=K, dim=-1, largest=True, sorted=True, memory_config=dram)
        r16.deallocate(True)
        ids_rm = ttnn.reshape(ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram), (1, R, K))
        w_rm = ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram), (1, 1, R, K, 1))
        w.deallocate(True)

        hist = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn.reshape(ids, (R, K)), self.table, self.n_experts, K
        )
        offsets, counts, region, all_offsets = ttnn.experimental.deepseek_prefill.offset_cumsum(
            hist, cluster_axis=ccl.SP_AXIS, num_links=self.num_links, experts_per_chip=self.per_chip, memory_config=dram
        )
        hist.deallocate(True)
        all_offsets.deallocate(True)
        ids.deallocate(True)

        xg = ccl.gather_rows(x)
        x_rm = ttnn.reshape(ttnn.to_layout(xg, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram), (1, R, H))
        xg.deallocate(True)
        disp, meta = ttnn.experimental.deepseek_prefill.dispatch(
            input_tensor=x_rm,
            indices_tensor=ids_rm,
            expert_offsets_tensor=offsets,
            expert_dispatch_table_tensor=self.table,
            dispatch_group_size=self.sp,
            experts_per_chip=self.per_chip,
            num_routed_experts=self.n_experts,
            num_experts_per_tok=K,
            metadata_len=METADATA_LEN,
            max_dispatch_buffer_token_size=self.buffer_rows,
            cluster_axis=ccl.SP_AXIS,
            num_links=self.num_links,
            topology=self.topology,
        )
        x_rm.deallocate(True)
        offsets.deallocate(True)

        buf = ttnn.to_layout(ttnn.reshape(disp, (self.buffer_rows, H)), ttnn.TILE_LAYOUT, memory_config=dram)
        disp.deallocate(True)
        buf = self._ffn(buf, region, counts)

        comb = ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(buf, (1, 1, self.buffer_rows, H)),
            meta,
            counts,
            region,
            dispatch_group_size=self.sp,
            experts_per_chip=self.per_chip,
            num_experts_per_tok=K,
            seq_len_per_chip=R,
            memory_config=dram,
            cluster_axis=ccl.SP_AXIS,
            num_links=self.num_links,
            topology=self.topology,
            init_zeros=True,
        )
        buf.deallocate(True)
        meta.deallocate(True)
        counts.deallocate(True)
        region.deallocate(True)

        # this group's weighted sum over the top-k slots (the other groups' experts skipped by the dispatch table)
        part = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb, w_rm, ids_rm, self.table, expert_dim=3, output_memory_config=dram
        )  # [1, 1, R, H] bf16 TILE
        comb.deallocate(True)
        w_rm.deallocate(True)
        ids_rm.deallocate(True)
        if self.out_fp32:  # the 4 group partials summed in fp32, rounded to bf16 once
            p32 = ttnn.typecast(part, ttnn.float32, memory_config=dram)
            part.deallocate(True)
            y32 = ccl.scatter_rows(p32)
            p32.deallocate(True)
            y = ttnn.typecast(y32, ttnn.bfloat16, memory_config=dram)
            y32.deallocate(True)
            return y
        y = ccl.scatter_rows(part)
        part.deallocate(True)
        return y
