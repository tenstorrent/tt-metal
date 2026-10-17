# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS routed experts (plan.md step 6), expert parallel over the [8, 4] mesh.

128 experts, top-4, 4 experts per chip: column-major ExpertMapping, so column c (a dispatch group of the 8 chips
along axis 0) holds experts 32c..32c+31 and chip (r, c) holds 4 (c * 8 + r) .. 4 (c * 8 + r) + 3. Per chip:

    routing [1, 1, S, 128] fp32 (dense, 0 off the chosen experts) -> ttnn.topk k 4 -> indices uint16, weights
    masked_bincount + offset_cumsum (axis 0): per-expert counts and offsets with GLOBAL expert ids ([1, 128])
    dispatch (axis 0): the column's tokens into this chip's flat buffer [capacity x 8 x S + pad, 2880] bf16
    per local expert: extract [8 S, 2880] -> ttnn.linear gate_up ([out, in] bfp8 + transpose_b, bias, fp32 out)
        -> clamp(gate, max 7), clamp(up, +-7), (up + 1) * gate * sigmoid(1.702 gate) (fp32)
        -> ttnn.linear down ([out, in] bfp8 + transpose_b, bias, bf16 out) -> insert back into the buffer
    combine (axis 0) -> [1, 1, S, 4, 2880] -> post_combine_reduce (routing weights, the dispatch table masks the
        other columns' slots) -> [1, 1, S, 2880]
    ttnn.all_reduce over TP (axis 1) in fp32: the sum over the 4 dispatch groups, bf16 out, replicated over columns.

The accurate path (rule 7): every expert matmul at HiFi4 with fp32 dest; the fused unified_routed_expert_moe kernel
hard-codes LoFi and bfp8 activations (known issues), so it is not used. Weights bfp8 stored [out, in] so the bfp8
blocks run along the in-features (mxfp4 blocks are exact there).
"""

import torch

import ttnn

from .ccl import SP_AXIS, TP_AXIS
from .settings import settings

SWIGLU_ALPHA = 1.702


def _compute_config(mesh, fidelity: str, fp32_acc: bool):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def global_expert(row: int, col: int, local: int, dgs: int, epc: int) -> int:
    """Column-major ExpertMapping: the global id of local expert ``local`` on chip (row, col)."""
    return (col * dgs + row) * epc + local


def dispatch_table(n_experts: int, dgs: int, ndg: int) -> torch.Tensor:
    """[ndg, n_experts + 1] int32: the chip (row) of each expert in dispatch group (column) g, -1 elsewhere; the last
    column is the padding sentinel (always -1)."""
    per_group, epc = n_experts // ndg, n_experts // ndg // dgs
    t = torch.full((ndg, n_experts + 1), -1, dtype=torch.int32)
    for g in range(ndg):
        for e in range(g * per_group, (g + 1) * per_group):
            t[g, e] = (e - g * per_group) // epc
    return t


class TtExperts:
    """One layer's routed experts. ``w_gu`` [E, 2I, H] ([gate rows | up rows], de-interleaved), ``b_gu`` [E, 2I],
    ``w_d`` [E, H, I], ``b_d`` [E, H]: the checkpoint's experts dequantized, [out, in]."""

    def __init__(self, mesh, ccl, w_gu, b_gu, w_d, b_d, top_k: int, limit: float, seq_local: int):
        self.mesh, self.ccl = mesh, ccl
        rows, cols = tuple(mesh.shape)
        self.dgs, self.ndg = rows, cols
        self.E, self.top_k, self.limit = int(w_gu.shape[0]), int(top_k), float(limit)
        self.I, self.H = int(w_d.shape[2]), int(w_d.shape[1])
        assert self.E % (rows * cols) == 0
        self.epc = self.E // (rows * cols)
        self.seq_local = seq_local
        assert seq_local % ttnn.TILE_SIZE == 0
        self.max_tok = rows * seq_local  # every token of the column to one expert
        cap = settings.get("EXPERTS_CAPACITY_FACTOR")
        # offset_cumsum aligns each local expert's region to a tile: reserve that padding (compute_constants)
        self.max_buf = self.max_tok * cap + ttnn.TILE_SIZE * (min(self.max_tok * cap, self.epc) - 1)
        self.metadata_len = 3

        mm = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1))

        def per_chip(t: torch.Tensor, local: int) -> torch.Tensor:
            """[E, ...] -> [rows, cols, ...]: chip (r, c) gets its local expert ``local``."""
            ids = [[global_expert(r, c, local, rows, self.epc) for c in range(cols)] for r in range(rows)]
            return t[torch.tensor(ids)]

        def upload(t, dtype):
            return ttnn.from_torch(
                t.float(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mm,
            )

        self.w_gu, self.b_gu, self.w_d, self.b_d = [], [], [], []
        for e in range(self.epc):
            self.w_gu.append(upload(per_chip(w_gu, e), ttnn.bfloat8_b))  # [out 2I, in H]
            self.b_gu.append(upload(per_chip(b_gu, e)[:, :, None, :], ttnn.float32))
            self.w_d.append(upload(per_chip(w_d, e), ttnn.bfloat8_b))  # [out H, in I]
            self.b_d.append(upload(per_chip(b_d, e)[:, :, None, :], ttnn.float32))

        table = dispatch_table(self.E, rows, cols)
        self.table = ttnn.from_torch(
            table,
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, 0)),
        )  # [1, E + 1] per chip: its column's row
        gidx = torch.tensor(
            [
                [[global_expert(r, c, e, rows, self.epc) for e in range(self.epc)] for r in range(rows)]
                for c in range(cols)
            ],
            dtype=torch.int32,
        )  # [cols, rows, epc]
        self.gidx = ttnn.from_torch(
            gidx.reshape(cols, rows, 1, self.epc),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(1, 0)),
        )
        self.gidx = ttnn.reshape(self.gidx, (1, self.epc))

        self.mm_config = _compute_config(mesh, settings.get("EXPERTS_FIDELITY"), settings.get("EXPERTS_FP32_ACC"))
        self.mid_dtype = getattr(ttnn, settings.get("EXPERTS_MID_DTYPE"))
        self.reduce_fp32 = settings.get("EXPERTS_REDUCE_FP32")

    def _expert(self, tokens: ttnn.Tensor, e: int) -> ttnn.Tensor:
        """tokens [max_tok, H] bf16 TILE -> the expert's output [max_tok, H] bf16 TILE."""
        rows = tokens.shape[-2]
        gu = ttnn.linear(
            ttnn.reshape(tokens, (1, 1, rows, self.H)),
            self.w_gu[e],
            bias=self.b_gu[e],
            transpose_b=True,
            dtype=self.mid_dtype,
            compute_kernel_config=self.mm_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, rows, self.I])
        up = ttnn.slice(gu, [0, 0, 0, self.I], [1, 1, rows, 2 * self.I])
        gu.deallocate(True)
        gate_c = ttnn.clamp(gate, max=self.limit)
        gate.deallocate(True)
        up_c = ttnn.clamp(up, min=-self.limit, max=self.limit)
        up.deallocate(True)
        scaled = ttnn.multiply(gate_c, SWIGLU_ALPHA)
        sig = ttnn.sigmoid(scaled)
        scaled.deallocate(True)
        glu = ttnn.multiply(gate_c, sig)
        sig.deallocate(True)
        gate_c.deallocate(True)
        up1 = ttnn.add(up_c, 1.0)
        up_c.deallocate(True)
        hdn = ttnn.multiply(up1, glu)
        up1.deallocate(True)
        glu.deallocate(True)
        out = ttnn.linear(
            hdn,
            self.w_d[e],
            bias=self.b_d[e],
            transpose_b=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.mm_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hdn.deallocate(True)
        return ttnn.reshape(out, (rows, self.H))

    def __call__(self, x: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        """x [1, 1, S, H] bf16 TILE (the chip's rows, replicated over TP), routing [1, 1, S, E] fp32 TILE dense
        routing weights -> [1, 1, S, H] bf16 TILE, replicated over TP."""
        S = x.shape[-2]
        assert S == self.seq_local, f"built for {self.seq_local} rows per chip, got {S}"
        vals, idx = ttnn.topk(routing, k=self.top_k, dim=-1, largest=True, sorted=True)
        if idx.dtype != ttnn.uint16:
            i0 = idx
            idx = ttnn.typecast(i0, ttnn.uint16)
            i0.deallocate(True)
        idx_t = ttnn.reshape(idx, (S, self.top_k))  # uint16 TILE [S, k] for masked_bincount
        counts = ttnn.experimental.deepseek_prefill.masked_bincount(idx_t, self.table, self.E, self.top_k)
        offsets, totals, regions, all_offsets = ttnn.experimental.deepseek_prefill.offset_cumsum(
            counts,
            cluster_axis=SP_AXIS,
            num_links=self.ccl.num_links,
            experts_per_chip=self.epc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        counts.deallocate(True)
        all_offsets.deallocate(True)
        idx_rm = ttnn.reshape(ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT), (1, S, self.top_k))
        idx.deallocate(True)
        w_rm = ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(vals, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT), (1, S, self.top_k)
        )
        vals.deallocate(True)

        x3 = ttnn.reshape(x, (1, S, self.H))
        buf, meta = ttnn.experimental.deepseek_prefill.dispatch(
            x3,
            idx_rm,
            offsets,
            self.table,
            dispatch_group_size=self.dgs,
            experts_per_chip=self.epc,
            num_routed_experts=self.E,
            num_experts_per_tok=self.top_k,
            metadata_len=self.metadata_len,
            max_dispatch_buffer_token_size=self.max_buf,
            cluster_axis=SP_AXIS,
            num_links=self.ccl.num_links,
            topology=ttnn.Topology.Linear,
        )
        offsets.deallocate(True)
        flat = ttnn.reshape(buf, (self.max_buf, self.H))
        if flat.layout == ttnn.TILE_LAYOUT and flat.dtype == ttnn.bfloat16:
            flat_t = flat  # to_layout would hand back the same buffer (known issue): keep it
        else:
            flat_t = ttnn.to_layout(flat, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            buf.deallocate(True)

        for e in range(self.epc):
            tokens = ttnn.experimental.deepseek_prefill.extract(
                flat_t, regions, totals, self.gidx, local_expert_id=e, max_dispatched_tokens_per_expert=self.max_tok
            )
            y = self._expert(tokens, e)
            tokens.deallocate(True)
            ttnn.experimental.deepseek_prefill.insert(flat_t, y, regions, totals, self.gidx, local_expert_id=e)
            y.deallocate(True)

        comb = ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(flat_t, (1, 1, self.max_buf, self.H)),
            meta,
            totals,
            regions,
            dispatch_group_size=self.dgs,
            experts_per_chip=self.epc,
            num_experts_per_tok=self.top_k,
            seq_len_per_chip=S,
            cluster_axis=SP_AXIS,
            num_links=self.ccl.num_links,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            init_zeros=True,
        )
        flat_t.deallocate(True)
        meta.deallocate(True)
        totals.deallocate(True)
        regions.deallocate(True)

        w5 = ttnn.reshape(w_rm, (1, 1, S, self.top_k, 1))
        red = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb,
            w5,
            idx_rm,
            self.table,
            expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        comb.deallocate(True)
        w_rm.deallocate(True)
        idx_rm.deallocate(True)
        red = ttnn.reshape(red, (1, 1, S, self.H))
        part = ttnn.typecast(red, ttnn.float32) if self.reduce_fp32 else red
        out = ttnn.all_reduce(part, cluster_axis=TP_AXIS, num_links=self.ccl.num_links, topology=self.ccl.topology)
        if part is not red:
            part.deallocate(True)
        red.deallocate(True)
        if out.dtype != ttnn.bfloat16:
            o32 = out
            out = ttnn.typecast(o32, ttnn.bfloat16)
            o32.deallocate(True)
        return out
