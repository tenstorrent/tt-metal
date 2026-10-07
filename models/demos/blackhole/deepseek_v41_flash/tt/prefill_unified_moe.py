# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill routed experts through the ``ttnn.experimental.deepseek_prefill`` pipeline (one count-driven expert program per chunk).

Mapping (documented choice, see PREFILL_UNIFIED_MOE_NOTES.md): the mesh is 4 rows x 8 columns, x is replicated over the 8 columns and every
row carries different tokens (its own users).  Dispatch group axis = mesh ROWS (cluster_axis=0, ``dispatch_group_size`` = 4, each row holds the
tokens of its own users, ``seq_len_per_chip`` = tokens per row of the chunk); the 8 columns are 8 independent dispatch groups, each serving
48 of the 384 routed experts (12 per chip): expert e lives in group g = e // 48, chip (row) r = (e % 48) // 12, local slot e % 12
(``ExpertMapping`` column-major order of models/demos/deepseek_v3_d_p).  Every column routes the same tokens (the router runs replicated),
dispatch drops each (token, expert) pair into the right chip's buffer of the column's group, ``unified_routed_expert_moe`` runs the 12 local
experts on their real token counts (one program, M = 100s of rows per expert), ``combine`` returns the rows to their origin chip and
``post_combine_reduce`` forms the weighted sum over the (<= 6) slots, keeping only the experts of this column's group; the partial sums of the
8 columns are added by a reduce-scatter over the columns along the TOKEN dim, which leaves every column with its own 32-token chunks.

Weights: one (5120, 2304) / (2304, 5120) bf8 tensor per local expert and projection, DRAM ND-sharded (``routed_expert_weight_memory_config``),
cached as host tensorbins in a directory of its own (``DSV41_UNIFIED_CACHE``) -- the moe_compute weights of ``DSV41MoEBlock`` are a different layout.
"""

import os
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import routed_expert_weight_memory_config

NUM_ROUTED = 384
HIDDEN = 5120
INTER = 2304
TOPK = 6
ROWS, COLS = 4, 8
EPC = NUM_ROUTED // (ROWS * COLS)  # 12 experts per chip
UNIFIED_CACHE = os.environ.get("DSV41_UNIFIED_CACHE", "/mnt/tt-data/ssinghal/dsv4-weight-cache-unified")
W_DTYPE = ttnn.bfloat8_b
_SHARED = {}  # (id(mesh), tokens per row, cf) -> UnifiedMoEShared


def global_expert(row, col, local):
    return ExpertMapping.get_global_expert_idx(
        group=col,
        chip=row,
        local_expert=local,
        experts_per_chip=EPC,
        dispatch_group_size=ROWS,
        num_dispatch_groups=COLS,
    )


def _ckc():
    fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}[
        os.environ.get("DSV41_UNI_FID", "lofi")
    ]
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid,
        math_approx_mode=False,
        fp32_dest_acc_en=os.environ.get("DSV41_UNI_FP32ACC", "0") == "1",
        packer_l1_acc=True,
    )


class UnifiedMoEShared:
    """Per (mesh, tokens-per-row) constants shared by all layers: dispatch table, expert-id table, buffer sizes."""

    def __init__(self, md, n_tokens, cf=None):
        self.md, self.N = md, n_tokens
        self.cf = int(os.environ.get("DSV41_UNI_CF", "4")) if cf is None else cf
        assert n_tokens % 64 == 0
        self.epc, self.meta_len, self.max_buf, self.max_per_expert = compute_constants(
            n_tokens, NUM_ROUTED, TOPK, ROWS * COLS, ROWS, self.cf
        )
        assert self.epc == EPC
        table = ExpertMapping.create_dispatch_table(NUM_ROUTED, ROWS, COLS)  # [8, 385]
        self.dispatch_table = ttnn.from_torch(
            table,
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=md.shape, dims=(None, 0)),
        )  # per device [1, 385]
        gidx = ExpertMapping.create_global_expert_idx_table(EPC, ROWS, COLS)  # [8 groups, 4 chips, 12]
        t = ttnn.from_torch(
            gidx,
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=md.shape, dims=(1, 0)),
        )  # per device [1, 1, 12]
        self.gidx = ttnn.squeeze(ttnn.squeeze(t, 0), 0)
        self.ckc = _ckc()
        self.num_links = int(
            os.environ.get("DSV41_UNI_LINKS", "2")
        )  # fabric links of dispatch / combine / offset_cumsum (2 available per row hop)
        self.workers = int(os.environ.get("DSV41_UNI_WORKERS", "2"))  # worker cores per sender of the dispatch op
        self.l1_small = (
            os.environ.get("DSV41_UNI_L1SMALL", "1") == "1"
        )  # global semaphores of the CCL ops in L1_SMALL (device opened with l1_small_size > 0)


def row_topology():
    """dispatch / combine topology over the 4 rows (DSV41_UNI_TOPO=ring: use the column's wrap link; read per call so A/B modes can switch it)"""
    return ttnn.Topology.Ring if os.environ.get("DSV41_UNI_TOPO", "ring") == "ring" else ttnn.Topology.Linear


def get_shared(md, n_tokens):
    key = (id(md), n_tokens)
    if key not in _SHARED:
        _SHARED[key] = UnifiedMoEShared(md, n_tokens)
    return _SHARED[key]


def _cache_name(layer_id, name):
    return str(Path(UNIFIED_CACHE) / f"layer_{layer_id}" / f"local_{name}")


def _cache_complete(layer_id):
    d = Path(UNIFIED_CACHE) / f"layer_{layer_id}"
    return all(
        any(d.glob(f"local_{l}_{p}_dtype_{W_DTYPE.name}_*.tensorbin"))
        for l in range(EPC)
        for p in ("gate", "up", "down")
    )


def _wmc(md, n_dim, k_dim):
    """Weight memory config. DSV41_UM_SHARD_H (K tile rows per ND shard, 0 = op default of 1) is an experiment knob: a shard as tall as K pins
    every N-column of the op to ONE DRAM bank for all K rows (the access pattern of the moe_compute decode ring layout).
    """
    cfg = routed_expert_weight_memory_config(md, n_dim, dram_nd_sharded=True)
    h = int(os.environ.get("DSV41_UM_SHARD_H", "0"))
    if h <= 0:
        return cfg
    sp = cfg.nd_shard_spec
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([min(h, k_dim // 32) * 32, sp.shard_shape[-1]]),
            grid=sp.grid,
            orientation=sp.orientation,
        ),
    )


def build_expert_weights(md, layer_id, log=print, cache_only=False):
    """-> (gate_projs, up_projs, down_projs), lists of EPC per-device tensors (K, N) bf8 DRAM ND-sharded.
    Host cache miss: dequantise the checkpoint (fp4 + e8m0 scales -> bf16), the same source and rounding as the moe_compute cache.
    """
    os.makedirs(Path(UNIFIED_CACHE) / f"layer_{layer_id}", exist_ok=True)
    warm = _cache_complete(layer_id)
    src = None
    if not warm:
        from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as mw

        src = mw._Shards()
    mapper = ttnn.ShardTensor2dMesh(md, mesh_shape=md.shape, dims=(0, 1))
    outs = {"gate": [], "up": [], "down": []}
    for l in range(EPC):
        for proj, ck in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            host = None
            if not warm:
                from concurrent.futures import ThreadPoolExecutor

                from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as mw

                def one(rc):
                    r, c = rc
                    e = global_expert(r, c, l)
                    return mw._fp4_expert(src, f"layers.{layer_id}.ffn.experts.{e}.{ck}", torch.bfloat16)  # [in, out]

                with ThreadPoolExecutor(max_workers=int(os.environ.get("DSV41_LOAD_THREADS", "24"))) as ex:
                    ws = list(ex.map(one, [(r, c) for r in range(ROWS) for c in range(COLS)]))
                host = torch.stack(ws).reshape(ROWS, COLS, *ws[0].shape)
            else:
                K, N = (INTER, HIDDEN) if proj == "down" else (HIDDEN, INTER)
                host = torch.empty(ROWS, COLS, K, N, dtype=torch.bfloat16)  # ignored on a cache hit
            t = ttnn.as_tensor(
                host,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                dtype=W_DTYPE,
                cache_file_name=_cache_name(layer_id, f"{l}_{proj}"),
            )
            if cache_only:
                del t
                continue
            t = ttnn.squeeze(ttnn.squeeze(t, dim=0), dim=0)
            outs[proj].append(ttnn.to_device(t, md, memory_config=_wmc(md, t.shape[-1], t.shape[-2])))
        if not warm:
            log(f"unified moe weights layer {layer_id}: expert slot {l + 1}/{EPC} built")
    return None if cache_only else (outs["gate"], outs["up"], outs["down"])


CHECK_N = int(os.environ.get("DSV41_UNI_CHECK", "0"))


class DSV41UnifiedMoE:
    _calls = 0
    """Routed experts of one layer: ``forward(h)`` takes the router weights/indices of the whole chunk and the hidden states of the whole
    chunk (all tokens of the mesh row, replicated over the columns) and returns the per-column PARTIAL weighted sums [1,1,N,D] (tile, DRAM).
    """

    def __init__(self, md, layer_id, weights=None, log=print, ring=None):
        """ring: (w0_w1, w2) = the moe_compute decode weight tensors of the layer; the op then reads them in place (RING_WEIGHTS mode, 8 columns,
        no second weight copy): the same tensor is passed EPC times in the gate/up/down lists."""
        self.md, self.layer_id = md, layer_id
        if ring is not None:
            self.gate_projs = [ring[0]] * EPC
            self.up_projs = [ring[0]] * EPC
            self.down_projs = [ring[1]] * EPC
        else:
            self.gate_projs, self.up_projs, self.down_projs = (
                weights if weights is not None else build_expert_weights(md, layer_id, log=log)
            )

    def forward(self, x_rm, scores, indices, upto=99, overlap=None):
        """x_rm [1,N,D] bf16 ROW_MAJOR; scores [N,1,1,k] bf16 RM; indices [N,1,1,k] uint16 RM (the router outputs, N = tokens of this row)."""
        md = self.md
        N = x_rm.shape[1]
        sh = get_shared(md, N)
        self.mask = sh.dispatch_table  # experts_in_dispatch_group (same table as the dispatch op)
        idx2 = ttnn.reshape(indices, [N, TOPK])
        idx_t = ttnn.to_layout(idx2, ttnn.TILE_LAYOUT)
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(idx_t, self.mask, NUM_ROUTED, TOPK)
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
        if (
            CHECK_N and DSV41UnifiedMoE._calls < CHECK_N
        ):  # DSV41_UNI_CHECK=<n>: eager-only capacity check of the first n calls (host read: never inside a trace capture)
            DSV41UnifiedMoE._calls += 1
            ttnn.synchronize_device(md)
            devs = ttnn.get_device_tensors(counts)
            load, cmax = 0, 0
            for r in range(ROWS):
                for c in range(COLS):
                    cnt = ttnn.to_torch(devs[r * COLS + c]).reshape(-1)[:NUM_ROUTED].long()
                    gids = [global_expert(r, c, l) for l in range(EPC)]
                    chip = int(sum(-(-int(cnt[g]) // 32) * 32 for g in gids))
                    load = max(load, chip)
                    cmax = max(cmax, int(cnt[gids].max()))
            print(
                f"UNI_CHECK layer {self.layer_id} N={N}: max chip load {load} rows of {sh.max_buf} ({'OVERFLOW' if load > sh.max_buf else 'ok'}), max expert count {cmax} of {sh.max_per_expert}",
                flush=True,
            )
        if upto == 1:
            return offsets, counts, region_offsets
        idx3 = ttnn.reshape(indices, [1, N, TOPK])
        ov = None
        if (
            overlap is not None
        ):  # (SDOverlap, hook): the hook (shared expert) runs on sub-device 1 while dispatch runs on sub-device 0
            ov = overlap[0]
            ov.load()
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
            subdevice_id=ov.d_id if ov is not None else None,
            num_workers_per_sender=sh.workers,
            use_l1_small_for_semaphores=sh.l1_small,
        )
        if ov is not None:
            overlap[1]()
            ov.clear()
        ttnn.deallocate(offsets)
        if upto == 2:
            return disp, meta, counts, region_offsets
        disp2 = ttnn.reshape(disp, [sh.max_buf, x_rm.shape[2]])
        if os.environ.get("DSV41_UNI_EXPERT", "unified") == "fused":
            # moe_fused_swiglu: plain SiLU (no clamp, as the moe_compute path), output dtype selectable (bf16 default); precision experiment, see the notes
            odt = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("DSV41_UNI_FUSED_OUT", "bf16")]
            out = ttnn.empty(
                disp2.shape, dtype=odt, layout=ttnn.TILE_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}[
                os.environ.get("DSV41_UNI_FID", "lofi")
            ]
            ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                disp2,
                self.gate_projs,
                self.up_projs,
                self.down_projs,
                counts,
                sh.gidx,
                input_m_tiles=sh.max_per_expert // ttnn.TILE_SIZE,
                core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
                ),
                activation=ttnn.RoutedExpertActivation.Silu,
                output=out,
                expert_region_offsets=region_offsets,
                read_x_at_offset=True,
            )
        else:
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
        if upto == 3:
            return out, meta, counts, region_offsets
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
        ttnn.deallocate(out)
        ttnn.deallocate(meta)
        ttnn.deallocate(counts)
        ttnn.deallocate(region_offsets)
        if upto == 4:
            return comb
        w5 = ttnn.reshape(scores, [1, 1, N, TOPK, 1])
        i5 = ttnn.reshape(indices, [1, 1, N, TOPK])
        summed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb, w5, i5, sh.dispatch_table, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(comb)
        return summed


def reduce_scatter_tokens(part, cc, piece=None, mode=None, free=True):
    """[1,1,N,D] per-column partial sums, rows ordered [column c][own chunk g][32] -> [1,1,N/8,D] = this column's own rows summed over the 8 columns
    (reduce-scatter over the token dim; ``piece`` > 0 splits it into reduce-scatters of 8 x piece/8 rows taken from every column block: larger
    reduce-scatters corrupted a tile-row block on this build in the attention all-reduce, see DSV41PrefillAttention._allreduce).
    """
    R = part.shape[2]
    mode = os.environ.get("DSV41_UNI_RS", "hidden") if mode is None else mode
    if mode == "hidden":
        # reduce-scatter over the HIDDEN dim (the generic ttnn.reduce_scatter the deepseek_prefill reduce module uses) + all_to_all to own tokens
        r_ = ttnn.reduce_scatter(
            part,
            dim=3,
            cluster_axis=1,
            num_links=cc.num_links,
            topology=cc.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if free:
            ttnn.deallocate(part)
        out = ttnn.experimental.all_to_all_async_generic(
            r_, in_dim=3, out_dim=2, num_links=cc.num_links, topology=ttnn.Topology.Ring, cluster_axis=1
        )
        ttnn.deallocate(r_)
        return out
    piece = int(os.environ.get("DSV41_UNI_RS_PIECE", "0")) if piece is None else piece
    kw = dict(
        dim=2,
        multi_device_global_semaphore=None,
        num_links=cc.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=cc.topology,
        cluster_axis=1,
        barrier_semaphore=None,
    )

    def rs(t):
        k = dict(
            kw,
            multi_device_global_semaphore=cc.get_rs_ping_pong_semaphore(),
            barrier_semaphore=cc.get_barrier_semaphore(),
        )
        return ttnn.experimental.reduce_scatter_minimal_async(t, **k)

    if not piece:
        out = rs(part)
        ttnn.deallocate(part)
        return out
    per = R // COLS  # rows per column block
    m = piece // COLS  # rows of every column block per piece
    outs = []
    for q in range(0, per, m):
        slabs = [ttnn.slice(part, [0, 0, c * per + q, 0], [1, 1, c * per + q + m, part.shape[3]]) for c in range(COLS)]
        pc = ttnn.concat(slabs, dim=2)
        for t in slabs:
            ttnn.deallocate(t)
        outs.append(rs(pc))
        ttnn.deallocate(pc)
    ttnn.deallocate(part)
    out = ttnn.concat(outs, dim=2)
    for o in outs:
        ttnn.deallocate(o)
    return out


def route_cols(gate, hh_own, h, mc, cc, mode=None, dbg=None):
    """Routing of the gathered rows h [1,1,8n,D] (row order [column][own chunk][32]) -> (weights [8n,1,1,k] bf16 RM, indices [8n,1,1,k] uint16 RM, tensors to free).
    mode "full": router on every gathered row (8x the router work); "own": router on the own rows hh_own only + all-gather of the (fp32 packed) routing.
    """
    n, D = hh_own.shape[2], hh_own.shape[3]
    mode = os.environ.get("DSV41_UNI_ROUTE", "own") if mode is None else mode
    if mode == "full":
        N = h.shape[2]
        parts = [gate.forward(ttnn.slice(h, [0, 0, 32 * i, 0], [1, 1, 32 * (i + 1), D])) for i in range(N // 32)]
        sc_all = ttnn.concat([p_[0] for p_ in parts], dim=0)
        ix_all = ttnn.concat([p_[1] for p_ in parts], dim=0)
        for p_ in parts:
            ttnn.deallocate(p_[0])
            ttnn.deallocate(p_[1])
        return sc_all, ix_all, ()
    if os.environ.get("DSV41_UNI_ROUTER", "batched") == "batched":
        # ONE router for all own rows (matmul / activations on n rows, router_select over n rows: bit-identical to the 32-row slices)
        sc, ix = gate._forward_fused(hh_own, grid=(min(4, max(1, n // 128)), 8))
        parts = []
    else:
        parts = [gate.forward(ttnn.slice(hh_own, [0, 0, 32 * i, 0], [1, 1, 32 * (i + 1), D])) for i in range(n // 32)]
        sc = ttnn.concat([p_[0] for p_ in parts], dim=0) if len(parts) > 1 else parts[0][0]
        ix = ttnn.concat([p_[1] for p_ in parts], dim=0) if len(parts) > 1 else parts[0][1]
    f32 = lambda a: ttnn.typecast(ttnn.to_layout(ttnn.reshape(a, [1, 1, n, TOPK]), ttnn.TILE_LAYOUT), ttnn.float32)
    ixf, scf = f32(ix), f32(sc)
    pk_t = ttnn.concat([ixf, scf], dim=3)  # fp32 tile [1,1,n,12]: expert ids (exact in fp32) | weights (bf16 values)
    # narrow tensors (<= 1 tile wide) are corrupted by the all-gather (blocks of 32 rows go missing): gather wide row-major pages instead,
    # 32 tokens x 12 values = 384 fp32 per page (n is a multiple of 32)
    pk = ttnn.reshape(ttnn.to_layout(pk_t, ttnn.ROW_MAJOR_LAYOUT), [1, 1, n // 32, 32 * 2 * TOPK])
    pg_w = mc.allgather(pk, cc, axis=1, dim=2)  # [1,1,8n/32,384]
    if dbg is not None:
        dbg.extend([pk, pg_w])
    pg = ttnn.reshape(pg_w, [1, 1, COLS * n, 2 * TOPK])
    ix_all = ttnn.reshape(
        ttnn.typecast(ttnn.slice(pg, [0, 0, 0, 0], [1, 1, COLS * n, TOPK]), ttnn.uint16), [COLS * n, 1, 1, TOPK]
    )
    sc_all = ttnn.reshape(
        ttnn.typecast(ttnn.slice(pg, [0, 0, 0, TOPK], [1, 1, COLS * n, 2 * TOPK]), ttnn.bfloat16),
        [COLS * n, 1, 1, TOPK],
    )
    pg = pk_t = pk = pg_w = None
    return sc_all, ix_all, (sc, ix, ixf, scf)


def moe_cols(um, gate, hh_own, mc, cc, overlap=None):
    """Column-split MoE of a whole chunk. hh_own [1,1,n,D] tile bf16: the (n = N/8) normed hidden rows THIS column owns (own 32-token chunks in order).
    Routes the own rows (router per 32 rows), all-gathers hidden + routing over the 8 columns (row order [column][own chunk][32]: dispatch does not
    care about the token order), runs the pipeline and reduce-scatters the column partial sums back -> [1,1,n,D] (own rows, same order).
    """
    D = hh_own.shape[3]
    h = mc.allgather(hh_own, cc, axis=1, dim=2)
    sc_all, ix_all, junk = route_cols(gate, hh_own, h, mc, cc)
    x_rm = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [1, h.shape[2], D])
    ttnn.deallocate(h)
    part = um.forward(x_rm, sc_all, ix_all, overlap=overlap)
    for t_ in (x_rm, sc_all, ix_all) + tuple(junk):
        ttnn.deallocate(t_)
    return reduce_scatter_tokens(part, cc)
