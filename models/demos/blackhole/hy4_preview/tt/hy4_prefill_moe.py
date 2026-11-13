# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview prefill routed experts: the ``ttnn.experimental.deepseek_prefill`` pipeline of the DSV4 tree (tt/prefill_unified_moe.py, which stays
untouched), re-parameterised for 256 experts / top-8 / hidden 6144 / 8 experts per chip.

Mapping (unchanged): mesh 4 rows x 8 columns, x replicated over the 8 columns, every row carries its own tokens; dispatch group = mesh ROW axis
(cluster_axis 0, 4 chips), 8 independent groups = the columns, each serving 32 of the 256 experts (8 per chip): expert e on column e // 32, row
(e % 32) // 8, local slot e % 8 (the same map as the decode ``moe_compute`` placement, tt/hy4_moe_place.py), so the ring weights of the decode block are read IN PLACE
(``ring=(tt_w0_w1, tt_w2)``): no second weight copy (a bfp4 copy would be another 12 GiB per chip).
Per chunk: router on the 64 own rows of a column (N = 512 per row, n_own = N / 8), all-gather of hidden + routing over the columns, masked_bincount / offset_cumsum /
dispatch / unified_routed_expert_moe (ClampedSiluGlu, limit 10) / combine / post_combine_reduce, reduce-scatter back to the own rows.
"""
import os

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants

NUM_ROUTED, HIDDEN, INTER, TOPK = 256, 6144, 2048, 8
ROWS, COLS = 4, 8
EPC = NUM_ROUTED // (ROWS * COLS)  # 8
_SHARED = {}


def _ckc():
    fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}[
        os.environ.get("HY4_UNI_FID", "lofi")
    ]
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid,
        math_approx_mode=False,
        fp32_dest_acc_en=os.environ.get("HY4_UNI_FP32ACC", "0") == "1",
        packer_l1_acc=True,
    )


class Shared:
    """Per (mesh, tokens-per-row) constants shared by all layers."""

    def __init__(self, md, n_tokens, cf=None):
        self.md, self.N = md, n_tokens
        self.cf = int(os.environ.get("HY4_UNI_CF", "4")) if cf is None else cf
        assert n_tokens % 64 == 0
        self.epc, self.meta_len, self.max_buf, self.max_per_expert = compute_constants(
            n_tokens, NUM_ROUTED, TOPK, ROWS * COLS, ROWS, self.cf
        )
        assert self.epc == EPC
        table = ExpertMapping.create_dispatch_table(NUM_ROUTED, ROWS, COLS)
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
        self.ckc = _ckc()
        self.num_links = int(os.environ.get("HY4_UNI_LINKS", "2"))
        self.workers = int(os.environ.get("HY4_UNI_WORKERS", "2"))
        self.l1_small = os.environ.get("HY4_UNI_L1SMALL", "1") == "1"


def get_shared(md, n_tokens):
    key = (id(md), n_tokens)
    if key not in _SHARED:
        _SHARED[key] = Shared(md, n_tokens)
    return _SHARED[key]


def row_topology():
    return ttnn.Topology.Ring if os.environ.get("HY4_UNI_TOPO", "ring") == "ring" else ttnn.Topology.Linear


class Hy4UnifiedMoE:
    def __init__(self, md, layer_id, ring):
        """ring = (w0_w1, w2): the moe_compute decode weight tensors of the layer, read in place (the same tensor is passed EPC times)."""
        self.md, self.layer_id = md, layer_id
        self.gate_projs, self.up_projs, self.down_projs = [ring[0]] * EPC, [ring[0]] * EPC, [ring[1]] * EPC

    def forward(self, x_rm, scores, indices):
        """x_rm [1,N,D] bf16 ROW_MAJOR; scores [N,1,1,k] bf16 RM; indices [N,1,1,k] uint16 RM (router outputs of the whole chunk of this row).
        Returns the per-column PARTIAL weighted sums [1,1,N,D] (tile, DRAM)."""
        md = self.md
        N = x_rm.shape[1]
        sh = get_shared(md, N)
        idx2 = ttnn.reshape(indices, [N, TOPK])
        idx_t = ttnn.to_layout(idx2, ttnn.TILE_LAYOUT)
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(idx_t, sh.dispatch_table, NUM_ROUTED, TOPK)
        ttnn.deallocate(idx_t)
        offsets, counts, region_offsets, _ = ttnn.experimental.deepseek_prefill.offset_cumsum(
            hist,
            cluster_axis=0,
            num_links=sh.num_links,
            experts_per_chip=EPC,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_l1_small_for_semaphores=sh.l1_small,
        )
        ttnn.deallocate(hist)
        idx3 = ttnn.reshape(indices, [1, N, TOPK])
        disp, meta = ttnn.experimental.deepseek_prefill.dispatch(
            input_tensor=x_rm,
            indices_tensor=idx3,
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
            topology=row_topology(),
            fp8_output=False,
            subdevice_id=None,
            num_workers_per_sender=sh.workers,
            use_l1_small_for_semaphores=sh.l1_small,
        )
        ttnn.deallocate(offsets)
        disp2 = ttnn.reshape(disp, [sh.max_buf, x_rm.shape[2]])
        out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            disp2,
            region_offsets,
            counts,
            sh.gidx,
            self.gate_projs,
            self.up_projs,
            self.down_projs,
            max_dispatched_tokens_per_expert=sh.max_per_expert,
            compute_kernel_config=sh.ckc,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )
        ttnn.deallocate(disp)
        out5 = ttnn.reshape(out, [1, 1, sh.max_buf, out.shape[-1]])
        comb = ttnn.experimental.deepseek_prefill.combine(
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
            topology=row_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            init_zeros=False,
            use_fp8_combine=False,
            use_l1_small_for_semaphores=sh.l1_small,
        )
        for t in (out, meta, counts, region_offsets):
            ttnn.deallocate(t)
        w5 = ttnn.reshape(scores, [1, 1, N, TOPK, 1])
        i5 = ttnn.reshape(indices, [1, 1, N, TOPK])
        summed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb, w5, i5, sh.dispatch_table, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(comb)
        return summed


def reduce_scatter_tokens(part, cc):
    """[1,1,N,D] per-column partial sums (rows ordered [column][own chunk][32]) -> [1,1,N/8,D]: this column's own rows summed over the 8 columns
    (reduce-scatter over the hidden dim + all_to_all to the own tokens: the DSV4 ``DSV41_UNI_RS=hidden`` mode)."""
    r_ = ttnn.reduce_scatter(
        part, dim=3, cluster_axis=1, num_links=cc.num_links, topology=cc.topology, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(part)
    out = ttnn.experimental.all_to_all_async_generic(
        r_, in_dim=3, out_dim=2, num_links=cc.num_links, topology=ttnn.Topology.Ring, cluster_axis=1
    )
    ttnn.deallocate(r_)
    return out


def route_cols(gate, hh_own, mc, cc):
    """Route the own rows hh_own [1,1,n,D] (router once on n rows) and all-gather the routing over the 8 columns (packed: 32 tokens x 2*TOPK fp32 per page)
    -> (weights [8n,1,1,k] bf16 RM, indices [8n,1,1,k] uint16 RM, tensors to free)."""
    n = hh_own.shape[2]
    gy = int(os.environ.get("HY4_PF_RT_GY", "0")) or min(4, max(1, n // 128))
    sc, ix = gate.forward(hh_own, grid=(gy, 8))
    f32 = lambda a: ttnn.typecast(ttnn.to_layout(ttnn.reshape(a, [1, 1, n, TOPK]), ttnn.TILE_LAYOUT), ttnn.float32)
    ixf, scf = f32(ix), f32(sc)
    pk_t = ttnn.concat([ixf, scf], dim=3)
    pk = ttnn.reshape(ttnn.to_layout(pk_t, ttnn.ROW_MAJOR_LAYOUT), [1, 1, n // 32, 32 * 2 * TOPK])
    pg_w = mc.allgather(pk, cc, axis=1, dim=2)
    pg = ttnn.reshape(pg_w, [1, 1, COLS * n, 2 * TOPK])
    ix_all = ttnn.reshape(
        ttnn.typecast(ttnn.slice(pg, [0, 0, 0, 0], [1, 1, COLS * n, TOPK]), ttnn.uint16), [COLS * n, 1, 1, TOPK]
    )
    sc_all = ttnn.reshape(
        ttnn.typecast(ttnn.slice(pg, [0, 0, 0, TOPK], [1, 1, COLS * n, 2 * TOPK]), ttnn.bfloat16),
        [COLS * n, 1, 1, TOPK],
    )
    return sc_all, ix_all, (sc, ix, ixf, scf)


def moe_cols(um, gate, hh_own, mc, cc):
    """Column-split MoE of a whole chunk. hh_own [1,1,n,D] tile bf16: the n = N/8 normed hidden rows THIS column owns (own 32-token chunks in order) -> [1,1,n,D]."""
    D = hh_own.shape[3]
    h = mc.allgather(hh_own, cc, axis=1, dim=2)
    sc_all, ix_all, junk = route_cols(gate, hh_own, mc, cc)
    x_rm = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [1, h.shape[2], D])
    ttnn.deallocate(h)
    part = um.forward(x_rm, sc_all, ix_all)
    for t_ in (x_rm, sc_all, ix_all) + tuple(junk):
        ttnn.deallocate(t_)
    return reduce_scatter_tokens(part, cc)
