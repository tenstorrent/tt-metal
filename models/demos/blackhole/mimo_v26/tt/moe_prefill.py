# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2.6-Flash prefill routed experts on the 4x8 mesh through ttnn.experimental.deepseek_prefill ops (bfp8 weights, plain SiLU, no clamp).

Adapted from the DSV4.1 ``prefill_unified_moe.py`` (git show old/ssinghal/dsv4p1:models/demos/blackhole/deepseek_v41_flash/tt/prefill_unified_moe.py)
with MiMo constants: 256 routed experts, top-8, 8 experts per chip, hidden 4096, intermediate 2048.

Mapping: dispatch group axis = mesh rows (cluster_axis=0, 4 chips per group); the 8 mesh columns are 8 independent groups of 32 experts each
(expert e -> group e // 32, chip (row) (e % 32) // 8, local slot e % 8 = ExpertMapping column-major). x is replicated over the columns, every row carries
its own tokens; each column returns the PARTIAL weighted sum over its group's experts; the 8 column partials must be summed (reduce over columns).
"""

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import routed_expert_weight_memory_config

NUM_ROUTED, HIDDEN, INTER, TOPK = 256, 4096, 2048, 8
ROWS, COLS = 4, 8
EPC = NUM_ROUTED // (ROWS * COLS)  # 8
W_DTYPE = ttnn.bfloat8_b


def global_expert(row, col, local):
    return ExpertMapping.get_global_expert_idx(
        group=col,
        chip=row,
        local_expert=local,
        experts_per_chip=EPC,
        dispatch_group_size=ROWS,
        num_dispatch_groups=COLS,
    )


def make_ckc(fidelity="hifi2", fp32acc=False, packer_l1_acc=True):
    fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}[fidelity]
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32acc, packer_l1_acc=packer_l1_acc
    )


class Shared:
    """Per (mesh, tokens-per-row) constants: dispatch table, local->global id table, buffer sizes."""

    def __init__(self, md, n_tokens, cf=4, num_links=2, workers=2):
        self.md, self.N, self.cf = md, n_tokens, cf
        assert n_tokens % 64 == 0  # masked_bincount: tokens % 64 cores
        self.epc, self.meta_len, self.max_buf, self.max_per_expert = compute_constants(
            n_tokens, NUM_ROUTED, TOPK, ROWS * COLS, ROWS, cf
        )
        assert self.epc == EPC
        table = ExpertMapping.create_dispatch_table(NUM_ROUTED, ROWS, COLS)
        self.table_host = table
        self.dispatch_table = ttnn.from_torch(
            table,
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=md.shape, dims=(None, 0)),
        )
        gidx = ExpertMapping.create_global_expert_idx_table(EPC, ROWS, COLS)
        t = ttnn.from_torch(
            gidx,
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=md.shape, dims=(1, 0)),
        )
        self.gidx = ttnn.squeeze(ttnn.squeeze(t, 0), 0)
        self.num_links, self.workers = num_links, workers
        self.topology = ttnn.Topology.Linear


def build_expert_weights(md, w_of_expert, nd_sharded=True):
    """w_of_expert(e) -> (gate [K,N], up [K,N], down [N,K]) bf16 torch. Returns three lists of EPC per-device bfp8 DRAM ND-sharded tensors."""
    mapper = ttnn.ShardTensor2dMesh(md, mesh_shape=md.shape, dims=(0, 1))
    outs = {"gate": [], "up": [], "down": []}
    for l in range(EPC):
        ws = [[w_of_expert(global_expert(r, c, l)) for c in range(COLS)] for r in range(ROWS)]
        for i, proj in enumerate(("gate", "up", "down")):
            host = torch.stack([torch.stack([ws[r][c][i] for c in range(COLS)]) for r in range(ROWS)])  # [R,C,K,N]
            t = ttnn.from_torch(host, mesh_mapper=mapper, layout=ttnn.TILE_LAYOUT, dtype=W_DTYPE)
            t = ttnn.squeeze(ttnn.squeeze(t, dim=0), dim=0)
            outs[proj].append(
                ttnn.to_device(t, md, memory_config=routed_expert_weight_memory_config(md, t.shape[-1], nd_sharded))
            )
    return outs["gate"], outs["up"], outs["down"]


class MiMoPrefillMoE:
    """Routed experts of one layer. ``forward(x_rm, scores, indices)`` -> per-column PARTIAL weighted sums [1,1,N,D] (tile, DRAM)."""

    def __init__(self, md, weights, ckc=None, expert_op="unified", cf=4):
        self.md = md
        self.gate_projs, self.up_projs, self.down_projs = weights
        self.ckc = ckc or make_ckc()
        self.expert_op = expert_op
        self.cf = cf
        self._shared = {}
        self.last_counts = None
        self.mm_chunk = int(os.environ.get("MIMO_MM_CHUNK", "256"))

    def shared(self, N):
        if N not in self._shared:
            self._shared[N] = Shared(self.md, N, self.cf)
        return self._shared[N]

    def _ffn_mm(self, toks, l):
        """Plain ttnn matmuls with fp32 accumulation on the extracted tokens (routed_expert_ffn's fixed blocking does not fit fp32 DEST); M chunked."""

        def ffn(t_):
            g_ = ttnn.matmul(t_, self.gate_projs[l], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
            u_ = ttnn.matmul(t_, self.up_projs[l], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
            h_ = ttnn.multiply(
                g_, u_, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)], dtype=ttnn.bfloat16
            )
            ttnn.deallocate(g_)
            ttnn.deallocate(u_)
            o_ = ttnn.matmul(h_, self.down_projs[l], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
            ttnn.deallocate(h_)
            return o_

        M, K = toks.shape[-2], toks.shape[-1]
        ch = self.mm_chunk
        if M <= ch:
            return ffn(toks)
        parts = []
        for r0 in range(0, M, ch):
            sl = ttnn.slice(toks, [r0, 0], [min(M, r0 + ch), K])
            parts.append(ffn(sl))
            ttnn.deallocate(sl)
        o = ttnn.concat(parts, dim=-2)
        for p_ in parts:
            ttnn.deallocate(p_)
        return o

    def forward(self, x_rm, scores, indices, keep_counts=False, upto=99):
        """x_rm [1,N,D] bf16 ROW_MAJOR; scores [N,1,1,k] bf16 RM; indices [N,1,1,k] uint16 RM (router outputs; N = tokens of this mesh row)."""
        md = self.md
        N = x_rm.shape[1]
        sh = self.shared(N)
        dp = ttnn.experimental.deepseek_prefill
        idx_t = ttnn.to_layout(ttnn.reshape(indices, [N, TOPK]), ttnn.TILE_LAYOUT)
        hist = dp.masked_bincount(idx_t, sh.dispatch_table, NUM_ROUTED, TOPK)
        ttnn.deallocate(idx_t)
        offsets, counts, region_offsets, _ = dp.offset_cumsum(
            hist,
            cluster_axis=0,
            num_links=sh.num_links,
            experts_per_chip=EPC,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_l1_small_for_semaphores=True,
        )
        ttnn.deallocate(hist)
        if keep_counts:
            self.last_counts = counts
        if upto == 1:  # stage-timing hook: counts only
            return offsets, counts, region_offsets
        disp, meta = dp.dispatch(
            input_tensor=x_rm,
            indices_tensor=ttnn.reshape(indices, [1, N, TOPK]),
            expert_offsets_tensor=offsets,
            expert_dispatch_table_tensor=sh.dispatch_table,
            padding_config=None,
            scales_tensor=None,
            fp8_scaled_input=False,
            dispatch_group_size=ROWS,
            experts_per_chip=EPC,
            num_routed_experts=NUM_ROUTED,
            num_experts_per_tok=TOPK,
            metadata_len=sh.meta_len,
            max_dispatch_buffer_token_size=sh.max_buf,
            cluster_axis=0,
            num_links=sh.num_links,
            topology=sh.topology,
            fp8_output=False,
            subdevice_id=None,
            num_workers_per_sender=sh.workers,
            use_l1_small_for_semaphores=True,
        )
        ttnn.deallocate(offsets)
        if upto == 2:  # stage-timing hook: after dispatch
            return disp, meta, counts, region_offsets
        disp2 = ttnn.reshape(disp, [sh.max_buf, x_rm.shape[2]])
        if self.expert_op in ("loop", "mm"):
            # bf16-activation path (user decision: activations stay bf16): tilize the dispatched bf16 buffer (stays bf16), then per local expert
            # extract -> routed_expert_ffn (bf16 intermediates, bfp8 weights, DRAM-interleaved) -> insert in place. Costs max_per_expert rows per expert
            # (extract is fixed-size) instead of the real count.
            tiled = ttnn.to_layout(disp2, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            out = tiled
            for l in range(EPC):
                toks = dp.extract(
                    tiled,
                    region_offsets,
                    counts,
                    sh.gidx,
                    local_expert_id=l,
                    max_dispatched_tokens_per_expert=sh.max_per_expert,
                )
                if self.expert_op == "loop":
                    o = dp.routed_expert_ffn(
                        toks,
                        self.gate_projs[l],
                        self.up_projs[l],
                        self.down_projs[l],
                        compute_kernel_config=self.ckc,
                        output=None,
                    )
                else:
                    o = self._ffn_mm(toks, l)
                ttnn.deallocate(toks)
                out = dp.insert(out, o, region_offsets, counts, sh.gidx, local_expert_id=l)
                ttnn.deallocate(o)
        elif self.expert_op == "fused":
            out = ttnn.empty(
                disp2.shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            dp.moe_fused_swiglu(
                disp2,
                self.gate_projs,
                self.up_projs,
                self.down_projs,
                counts,
                sh.gidx,
                input_m_tiles=sh.max_per_expert // ttnn.TILE_SIZE,
                core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
                compute_kernel_config=self.ckc,
                activation=ttnn.RoutedExpertActivation.Silu,
                output=out,
                expert_region_offsets=region_offsets,
                read_x_at_offset=True,
            )
        else:
            out = dp.unified_routed_expert_moe(
                disp2,
                region_offsets,
                counts,
                sh.gidx,
                self.gate_projs,
                self.up_projs,
                self.down_projs,
                max_dispatched_tokens_per_expert=sh.max_per_expert,
                compute_kernel_config=self.ckc,
                activation=ttnn.RoutedExpertActivation.Silu,
            )
        ttnn.deallocate(disp)
        if upto == 3:  # stage-timing hook: after the expert op
            return out, meta, counts, region_offsets
        out5 = ttnn.reshape(out, [1, 1, sh.max_buf, out.shape[-1]])
        comb = dp.combine(
            out5,
            meta,
            counts,
            region_offsets,
            dispatch_group_size=ROWS,
            experts_per_chip=EPC,
            num_experts_per_tok=TOPK,
            seq_len_per_chip=N,
            cluster_axis=0,
            num_links=sh.num_links,
            topology=sh.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            init_zeros=False,
            use_fp8_combine=False,
            use_l1_small_for_semaphores=True,
        )
        ttnn.deallocate(out)
        ttnn.deallocate(meta)
        if upto == 4:  # stage-timing hook: after combine
            return comb
        if not keep_counts:
            ttnn.deallocate(counts)
        ttnn.deallocate(region_offsets)
        w5 = ttnn.reshape(scores, [1, 1, N, TOPK, 1])
        i5 = ttnn.reshape(indices, [1, 1, N, TOPK])
        summed = dp.post_combine_reduce(
            comb, w5, i5, sh.dispatch_table, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(comb)
        return summed


def spread_pad_routing(n_rows, n_pad, topk=TOPK, num_experts=NUM_ROUTED, offset=0):
    """Routing (idx [n_rows, n_pad, topk] int, w [n_rows, n_pad, topk] float) for DEAD padding slots of a decode batch (users padded up to the 64-token
    multiple masked_bincount needs). Dead slots still dispatch: if they all pick the same experts (e.g. 0..7) one chip overflows its dispatch
    buffer (seen: worst chip load 1280 rows vs max_buf 1248 at N=64, cf=4). Round-robin over all experts gives every expert at most
    ceil(n_rows*n_pad*topk/num_experts) pad tokens. Their outputs are garbage-free (zero hidden) and must be ignored by the caller.
    """
    import torch

    k = torch.arange(n_rows * n_pad * topk) + offset
    idx = (k * 1).reshape(n_rows, n_pad, topk) % num_experts
    # a token must not pick the same expert twice: consecutive k are distinct mod num_experts while topk <= num_experts
    return idx.long(), torch.full((n_rows, n_pad, topk), 1.0 / topk)
