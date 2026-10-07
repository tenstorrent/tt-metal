# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4.1-Flash decoder layer for PREFILL, built around an existing decode ``DSV41Layer`` (weights shared).

The token-wise blocks (mHC, router, MoE, shared expert) run on chunks of ``T`` = 32 consecutive tokens of the mesh row (the
decode kernels are verified up to 32 tokens per device); attention runs once over all R = U*Sp tokens of the row
(``DSV41PrefillAttention``). The routed experts are the decode-format ``moe_compute`` weights: ``DSV41PrefillMoE`` is a second
``TTMoEDecode`` front-end with batch_per_device = T that SHARES the expert weights of the decode MoE (no second copy).
"""

import os
from pathlib import Path

import torch
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode, _TTMoEDecodeBuffers
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.demos.blackhole.deepseek_v41_flash.tt import moe_overlap, pf_tune

_PROF_EVERY = int(
    os.environ.get("DSV41_PROF_EVERY", "0")
)  # >0: drain the device profiler every k chunk groups (op-table profiling)
_MARKS = (
    []
)  # (phase, device-operation id at its end); written to $DSV41_PROF_MARKS at the end of every layer forward (profiling only)


_MARKS_FILE = os.environ.get("DSV41_PROF_MARKS")


def _mark(tag):
    if _PROF_EVERY or _MARKS_FILE:
        from ttnn import _ttnn

        _MARKS.append((tag, int(_ttnn.get_device_operation_id())))


def _dump_marks():
    if _MARKS_FILE:
        import json

        json.dump(_MARKS, open(_MARKS_FILE, "w"))


ROUTER_BATCH = (
    os.environ.get("DSV41_ROUTER_BATCH", "0") == "1"
)  # one router call for the whole grouped chunk (router_select takes T up to 256)
_MOE_G_ENV = os.environ.get(
    "DSV41_MOE_G", "auto"
)  # 32-token chunks per moe_compute call: "auto" (default), or an explicit 1 / 2 / 4 / 8 for every shape
# Grouped (T=256) moe_compute for every users-per-row. At 8 users/row it used to throw "Statically allocated circular buffers ... clash with L1 buffers" because the program's persistent global
# semaphore was created at the program's first compile while its ~650 KB L1 outputs were live, i.e. in the MIDDLE of L1, capping every later static CB region (decode avoids it with
# DSV41MoEBlock.warmup; prefill now has DSV41PrefillMoE.warmup, DSV41_PREFILL_MOE_WARM=0 disables). Verified with the warmup: U = 1, 4, 8, 16 (U = 8: 40 layers ISL 128 / 1k); other U use the same code path.


UNGROUPED_USERS = (
    8,
)  # users per mesh row that stay on the T=32 path: the grouped (T=256) program + column split intermittently HANGS in the traced chunk replay at 8 users/row
# (full-model demo/gate runs, MoEComputeDeviceOperation never finishes; U=1/2/4/16/32 and the same cell with DSV41_MOE_G=1 are fine). Explicit DSV41_MOE_G=8 overrides.


def moe_g_for(users_per_row):
    return int(_MOE_G_ENV) if _MOE_G_ENV != "auto" else (1 if users_per_row in UNGROUPED_USERS else 8)


COLSPLIT_MODE = os.environ.get(
    "DSV41_COLSPLIT", "auto"
)  # "auto" (default): on when the shape allows; "0" forces off; "1" forces on (raises if impossible)


def colsplit_active(U, C):
    """Column split of the token-wise work over the 8 mesh columns needs the grouped MoE (G=8) and a multiple of 8 chunks of 32 tokens per mesh row
    (U users x C tokens). Otherwise the layer falls back to the replicated path (grouped MoE when the chunk count allows, else T=32 slices).
    """
    from models.demos.blackhole.deepseek_v41_flash.tt import uni_policy

    # the unified prefill MoE does not use the grouped (G=8) moe_compute program, so it also column-splits the users-per-row values that keep G=1 there (U=8)
    ok = (moe_g_for(U) == 8 or uni_policy.in_use()) and (U * C) % 256 == 0
    if COLSPLIT_MODE == "0":
        return False
    if COLSPLIT_MODE == "1":
        assert (
            ok
        ), f"DSV41_COLSPLIT=1 needs the grouped MoE (G=8) and users*chunk % 256 == 0 (U={U}, C={C}, G={moe_g_for(U)})"
        return True
    return ok


_G1_BUFFERS = {}  # id(mesh) -> shared T=32 moe_compute buffers of the fallback front-ends

_FREE = set(os.environ.get("DSV41_PF_FREE", "a,a_c,h,hh,h_tok,m,sh,x2,hs").split(","))


def _drain(L, i, every=3):
    if _PROF_EVERY and i % every == every - 1:
        ttnn.synchronize_device(L.mesh_device)
        ttnn.ReadDeviceProfiler(L.mesh_device)


def _free(tag, t, keep=False):
    if tag in _FREE and t is not None:
        ttnn.deallocate(t)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "deepseek_v41_flash.yaml"


def with_height_shard(cfg, T):
    """moe_compute holds at most 32 * output_height_shard_dim * num_data_parallel_cores tokens per call (4 dispatch devices x T): raise
    output_height_shard_dim for T > 128 (env DSV41_OHSD overrides). No-op for T <= 128."""
    d = int(os.environ.get("DSV41_OHSD", "0")) or (4 if T <= 128 else 4 * (T // 128))
    if d == cfg.compute.output_height_shard_dim:
        return cfg
    return cfg.model_copy(update={"compute": cfg.compute.model_copy(update={"output_height_shard_dim": d})})


def big_batch_scores_config(T, k=6):
    """dispatch_input_expert_scores_memory_config for T > 96 tokens/device (stock: one token per core, T=128 needs 16x8 cores > 120)."""
    if os.environ.get("DSV41_SCORES_CFG", "l1") == "l1":
        return ttnn.L1_MEMORY_CONFIG
    sh = T // 64
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        [sh, k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


class DSV41PrefillMoE:
    """``DSV41MoEBlock.forward`` at batch_per_device = T over the expert weights of an existing decode MoE block."""

    def __init__(self, moe_block, T=32, buffers=None, g=None):
        md = moe_block.mesh_device
        G = (
            moe_g_for(moe_block.decode.config.batch_per_device) if g is None else g
        )  # G consecutive 32-token chunks per moe_compute call
        T = T * G
        text = CONFIG_PATH.read_text()
        text = text.replace("batch_per_device: 4 ", f"batch_per_device: {T} ", 1)
        text = text.replace("num_shared_experts: 1", "num_shared_experts: 0").replace(
            "  shared_expert_ids_to_devices: fully_replicated\n", ""
        )
        cfg = TTMoEDecodeConfig.from_yaml(text, topology=ttnn.Topology.Linear)
        if cfg.mesh_shape != tuple(md.shape):
            cfg = cfg.with_mesh_shape(tuple(md.shape))
        cfg = with_height_shard(cfg, T)
        if T > 96:
            cfg = cfg.model_copy(update={"dispatch_input_expert_scores_memory_config": big_batch_scores_config(T)})
        if cfg.num_fast_reduce_outputs == 1:
            cfg = cfg.model_copy(
                update={"reduce": cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})}
            )
        self.T, self.md, self.gate = T, md, moe_block.gate
        dec = object.__new__(TTMoEDecode)
        dec.config = cfg
        dec.expert_state = moe_block.decode.expert_state  # shared weights
        if buffers is None:
            bd = cfg.buffers.model_dump()
            bd["compute_tilize_drain_core"] = ttnn.experimental.get_moe_tilize_drain_core(
                md,
                cfg.compute.output_height_shard_dim,
                auto_output_width_shard_dim(cfg.hidden_size, matmul_ring_size=effective_matmul_ring_size(md)),
                cfg.hidden_size,
                mux_core_range_set=cfg.compute.mux_core_range_set,
            )
            buffers = _TTMoEDecodeBuffers(md, **bd)
        dec.buffers = buffers
        self.decode = dec
        self.g1 = None
        if (
            G > 1
        ):  # fallback front-end at T = 32 for chunk counts that are not a multiple of G (shares the expert weights, own shared buffers)
            self.g1 = DSV41PrefillMoE(moe_block, T=32, buffers=_G1_BUFFERS.get(id(md)), g=1)
            _G1_BUFFERS.setdefault(id(md), self.g1.decode.buffers)

    def warmup(self):
        """Compile this front-end's moe_compute program (and its T=32 fallback) once with a small free hole under the persistent L1 allocations
        (see ``DSV41MoEBlock.warmup``): the program's persistent 320 B global semaphore is created on its first compile, while the call's ~560-650 KB
        L1 outputs are live, and would otherwise land mid-L1 (~846 KB) and cap the static CB region of every later program (the prefill SDPA at
        head_dim 512 then throws 'Statically allocated circular buffers ... clash with L1 buffers'; seen at U=1). Called once, eagerly, before the
        first prefill compile / trace capture. DSV41_PREFILL_MOE_WARM=0 disables it."""
        if getattr(self, "_warm", False):
            return
        self._warm = True
        md, T = self.md, self.T
        H = self.decode.config.hidden_size
        nb = ttnn.get_memory_view(md, ttnn.BufferType.L1).num_banks
        row = lambda n: ttnn.empty(
            [1, 1, 32, 32 * nb * n],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hole, fence = row(1), row(2)
        ttnn.deallocate(hole)
        up = lambda shape, lay: ttnn.from_torch(
            torch.zeros(shape),
            device=md,
            dtype=ttnn.bfloat16,
            layout=lay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        out = self.forward(up([1, 1, T, H], ttnn.TILE_LAYOUT), up([T, 1, 1, H], ttnn.ROW_MAJOR_LAYOUT))
        ttnn.synchronize_device(md)
        ttnn.deallocate(out)
        ttnn.deallocate(fence)
        if self.g1 is not None:
            self.g1.warmup()

    def forward(self, tt_x_gate, tt_x_tokens):
        return self._forward(tt_x_gate, tt_x_tokens)

    def forward_routed(self, tt_x_tokens, scores, indices):
        """moe_compute with the routing already done (scores bf16 / indices uint16, row-major [T,1,1,k])."""
        return self.decode.forward(tt_x=tt_x_tokens, tt_scores=scores, tt_indices=indices, layer_id=0)

    def _forward(self, tt_x_gate, tt_x_tokens):
        n = tt_x_gate.shape[2] // 32
        if n > 1 and ROUTER_BATCH and os.environ.get("DSV41_ROUTER", "fused") == "fused":
            scores, indices = self.gate.forward(tt_x_gate)  # router_select handles T = 32 * n rows in one program
        elif (
            n > 1
        ):  # the router kernels take <= 32 tokens: run them per 32-token slice, concatenate the [T,1,1,k] results
            parts = [
                self.gate.forward(ttnn.slice(tt_x_gate, [0, 0, 32 * i, 0], [1, 1, 32 * (i + 1), tt_x_gate.shape[3]]))
                for i in range(n)
            ]
            scores = ttnn.concat([p[0] for p in parts], dim=0)
            indices = ttnn.concat([p[1] for p in parts], dim=0)
        else:
            scores, indices = self.gate.forward(tt_x_gate)
        if indices.dtype != ttnn.uint16:
            indices = ttnn.typecast(indices, ttnn.uint16)
        if indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        if scores.dtype != ttnn.bfloat16:
            scores = ttnn.typecast(scores, ttnn.bfloat16)
        if scores.layout != ttnn.ROW_MAJOR_LAYOUT:
            scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        return self.decode.forward(tt_x=tt_x_tokens, tt_scores=scores, tt_indices=indices, layer_id=0)


def shared_big(sh, h):
    """Shared expert on M = G*32 tokens in one go (the tuned DSV41SharedExpertV2 configs are per_core_M = 1, i.e. 32 tokens, and re-read the
    35 MB of weights per call): same maths / dtypes (bfp8 weights, HiFi4 fp32 accumulate, bf16 gate/up, fp32 out), automatic matmul configs.
    h [1,1,M,D] bf16 -> [1,1,M,D] fp32."""
    ckc = pf_tune.shared_ckc(sh, h.device())
    gu = ttnn.linear(
        h,
        sh.w01,
        dtype=sh.mid_dtype,
        compute_kernel_config=ckc,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g, u = gu[:, :, :, : sh.inter], gu[:, :, :, sh.inter :]
    act = ttnn.multiply(
        g,
        u,
        input_tensor_a_activations=sh.act_a,
        input_tensor_b_activations=sh.act_b,
        dtype=sh.mid_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(gu)
    out = ttnn.linear(
        act,
        sh.w2,
        dtype=ttnn.float32,
        compute_kernel_config=ckc,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(act)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Packed mHC carrier (DSV41_PF_MHC=packed, column-split path only): the residual streams of the column's N = 32*n8 own tokens live in ONE tensor
# [1,1,N,4C] fp32 (stream i at columns [i*C, (i+1)*C)) and the ffn pre coefficients in [1,1,N,4]; see tt/mhc_packed.py. The model-level code keeps
# calling layer.forward(xs, pres) with lists of 32-token chunks; a PkList (a one-element list holding the packed tensor) is what a packed layer
# returns and accepts, ``unpack_streams`` converts back to the [32,1,4,C] chunk lists (Engram layers, head, hooks).


def pk_mhc_enabled():
    return pf_tune._env("DSV41_PF_MHC", "0") == "packed"


class PkList(list):
    packed = True


def _rm_reshape_tile(t, shape):
    """tile [T,...] -> row-major -> reshape -> tile (fp32 stays fp32)."""
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)


def pack_streams(xs, pres):
    """lists of [32,1,4,C] / [32,1,1,4] fp32 chunks -> (PkList([x [1,1,N,4C]]), PkList([pre [1,1,N,4]]))."""
    C = xs[0].shape[3]
    px = [_rm_reshape_tile(x, [1, 1, 32, 4 * C]) for x in xs]
    pp = [_rm_reshape_tile(p, [1, 1, 32, 4]) for p in pres]
    x = ttnn.concat(px, dim=2) if len(px) > 1 else px[0]
    pre = ttnn.concat(pp, dim=2) if len(pp) > 1 else pp[0]
    if len(px) > 1:
        for t in px + pp:
            ttnn.deallocate(t)
    return PkList([x]), PkList([pre])


def unpack_streams(xs, pres):
    """PkList pair -> lists of [32,1,4,C] / [32,1,1,4] chunks (the packed tensors are freed); lists pass through."""
    if not getattr(xs, "packed", False):
        return xs, pres
    x, pre = xs[0], pres[0]
    N, C = int(x.shape[2]), int(x.shape[3]) // 4
    outs, pouts = [], []
    for c in range(N // 32):
        xc = ttnn.slice(x, [0, 0, 32 * c, 0], [1, 1, 32 * (c + 1), x.shape[3]])
        outs.append(_rm_reshape_tile(xc, [32, 1, 4, C]))
        ttnn.deallocate(xc)
        pc = ttnn.slice(pre, [0, 0, 32 * c, 0], [1, 1, 32 * (c + 1), pre.shape[3]])
        pouts.append(_rm_reshape_tile(pc, [32, 1, 1, 4]))
        ttnn.deallocate(pc)
    ttnn.deallocate(x)
    ttnn.deallocate(pre)
    return outs, pouts


class DSV41PrefillLayer:
    def __init__(self, layer, prefill_attn, pmoe, T=32):
        self.L, self.pa, self.pmoe, self.T = layer, prefill_attn, pmoe, T
        self.debug = None  # set to a dict: per-chunk lists of a_c / hh / m / sh are kept (not freed) for diagnostics

    colsplit = False  # set by the model per chunk size (colsplit_active)
    umoe = None  # tt/prefill_unified_moe.py DSV41UnifiedMoE (DSV41_PREFILL_MOE=unified), set by the model builder

    def _moe_unified(self, st, n8):
        """Routed experts through the deepseek_prefill pipeline for the whole chunk (tt/prefill_unified_moe.py ``moe_cols``): the own 32-token chunks of
        all n8 groups go through ONE route / all-gather / dispatch / experts / combine / reduce-scatter -> list of the own [1,1,32,D] chunks.
        """
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import moe_cols

        L, T = self.L, self.T
        hh_own = ttnn.concat([e[4] for e in st], dim=2) if n8 > 1 else st[0][4]
        ov = None
        self._sh_pre = None
        if (
            moe_overlap.enabled()
            and hasattr(L.shared, "w01")
            and (moe_overlap.explicit() or hasattr(L.shared, "wg"))
            and hh_own.shape[2] <= moe_overlap.MAX_M
        ):  # shared expert on a sub-device concurrently with the dispatch (tt/moe_overlap.py)
            ov = moe_overlap.SDOverlap.get(L.mesh_device)
            box = []
            overlap = (ov, lambda: box.append(ov.shared(L.shared, hh_own)))
        else:
            overlap = None
        own = moe_cols(self.umoe, L.moe.gate, hh_own, L.mesh_config, L.ccl, overlap=overlap)  # [1,1,n8*32,D]
        if overlap is not None:
            self._sh_pre = (
                box[0],
                hh_own if n8 > 1 else None,
            )  # hh_own is read by async shared-expert kernels: freed with the shared output
        elif n8 > 1:
            ttnn.deallocate(hh_own)
        if (
            n8 == 1
        ):  # U=2 x C=128: a full-extent slice aliases ``own`` (freeing it would free the result: "Tensor is not allocated")
            return [own]
        D = own.shape[3]
        outs = [ttnn.slice(own, [0, 0, g * T, 0], [1, 1, (g + 1) * T, D]) for g in range(n8)]
        ttnn.deallocate(own)
        return outs

    def forward(self, xs, pres, S, s0=0):
        if self.colsplit:
            n8 = int(xs[0].shape[2]) // self.T if getattr(xs, "packed", False) else len(xs)
            if (
                pk_mhc_enabled() and n8 > 1 and self.umoe is None
            ):  # n8 == 1 (U=2 x C=128) segfaults in the packed path's all_gather; the packed carrier has no unified-MoE branch
                return self.forward_cols_pk(xs, pres, S, s0)
            return self.forward_cols(xs, pres, S, s0)
        xs, pres = unpack_streams(xs, pres)
        return self.forward_full(xs, pres, S, s0)

    def _pk(self):
        L = self.L
        if getattr(self, "_pk_m", None) is None:
            from models.demos.blackhole.deepseek_v41_flash.tt.mhc_packed import PackedMHC

            self._pk_m = (PackedMHC(L.mhc_attn), PackedMHC(L.mhc_ffn))
        return self._pk_m

    def _route_own(self, hh, n8):
        """hh [1,1,N,D] bf16 (own chunks) -> (scores bf16 [n8*8*T,1,1,k], indices uint16 [n8*8*T,1,1,k]) row-major for the all-gathered tokens
        (group g, column j, token t at row (g*8 + j)*T + t): the router runs once per own chunk, then results are all-gathered over the columns.
        """
        L, T = self.L, self.T
        gate, mc, cc = self.pmoe.gate, L.mesh_config, L.ccl
        D = int(hh.shape[3])
        N = n8 * T
        ws, ixs = [], []
        for g in range(n8):
            hc = ttnn.slice(hh, [0, 0, g * T, 0], [1, 1, (g + 1) * T, D])
            w, i = gate.forward(hc)
            ttnn.deallocate(hc)
            ws.append(w)
            ixs.append(i)
        w_all = ttnn.concat(ws, dim=0) if n8 > 1 else ws[0]
        i_all = ttnn.concat(ixs, dim=0) if n8 > 1 else ixs[0]
        k = int(w_all.shape[3])
        tile32 = lambda t, dt: ttnn.typecast(ttnn.to_layout(ttnn.reshape(t, [1, 1, N, k]), ttnn.TILE_LAYOUT), dt)
        both = ttnn.concat(
            [tile32(w_all, ttnn.float32), tile32(ttnn.typecast(i_all, ttnn.uint32), ttnn.float32)], dim=3
        )
        g_ = mc.allgather(ttnn.reshape(both, [n8, 1, T, 2 * k]), cc, axis=1, dim=1)  # [n8,8,T,2k]
        g_ = ttnn.reshape(g_, [1, 1, n8 * 8 * T, 2 * k])
        sc = ttnn.slice(g_, [0, 0, 0, 0], [1, 1, n8 * 8 * T, k])
        ix = ttnn.slice(g_, [0, 0, 0, k], [1, 1, n8 * 8 * T, 2 * k])
        rm = lambda t, dt: ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(t, dt), ttnn.ROW_MAJOR_LAYOUT), [n8 * 8 * T, 1, 1, k]
        )
        sc_rm = rm(sc, ttnn.bfloat16)
        ix_rm = ttnn.typecast(ttnn.typecast(ix, ttnn.uint32), ttnn.uint16)
        ix_rm = ttnn.reshape(ttnn.to_layout(ix_rm, ttnn.ROW_MAJOR_LAYOUT), [n8 * 8 * T, 1, 1, k])
        for t in (w_all, i_all, both, g_, sc, ix):
            ttnn.deallocate(t)
        return sc_rm, ix_rm

    def forward_cols_pk(self, xs, pres, S, s0=0):
        """forward_cols with the mHC pieces on the packed stream layout (one call per sub-block over all of the column's own tokens instead
        of a loop over 32-token chunks of 4->32 padded streams). xs / pres: lists of chunks (first layer, after Engram) or PkLists.
        """
        L, T = self.L, self.T
        converted = not getattr(xs, "packed", False)
        if converted:
            xs, pres = pack_streams(xs, pres)
        x, pre_in = xs[0], pres[0]
        N, D = int(x.shape[2]), int(x.shape[3]) // 4
        n8 = N // T
        assert self.umoe is not None or self.pmoe.T == 8 * T, "column split needs DSV41_MOE_G=8 (or the unified MoE)"
        mc, cc = L.mesh_config, L.ccl
        eps = L.eps
        pka, pkf = self._pk()
        f32 = lambda t: t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)
        _mark("start")
        pre_a, post_a, comb_a = pka.mixes(x)
        h_own = pka.collapse_norm(x, pre_in, L.attn_norm_w, eps)  # [1,1,N,D] bf16, own chunks g = 0..n8-1
        _mark("mhc_attn")
        hg = mc.allgather(ttnn.reshape(h_own, [n8, 1, T, D]), cc, axis=1, dim=1)  # [n8,8,T,D]: (group, column, token)
        ttnn.deallocate(h_own)
        h = ttnn.reshape(hg, [1, 1, n8 * 8 * T, D])
        pa = self.pa
        pa.rs_tokens = True
        a = pa.forward_dyn(h) if pa.dyn is not None else pa.forward(h, S, s0=s0)  # [1,1,n8*32,D] own chunks
        pa.rs_tokens = False
        _free("h", h)
        _mark("attention")
        a32 = f32(a)
        x2 = pka.expand(a32, x, post_a, comb_a)
        for t in (a32, a, post_a, comb_a) if a32 is not a else (a, post_a, comb_a):
            ttnn.deallocate(t)
        pre_f, post_f, comb_f = pkf.mixes(x2)
        hh = pkf.collapse_norm(x2, pre_a, L.ffn_norm_w, eps)  # [1,1,N,D] bf16
        ttnn.deallocate(pre_a)
        _mark("mhc_expand_collapse")
        route_own = os.environ.get("DSV41_PF_ROUTE_OWN", "1") == "1"
        if (
            route_own
        ):  # router on the column's own chunks once, gathered with the hidden states (instead of 8x redundantly per group)
            sc_all, ix_all = self._route_own(hh, n8)
        hha = mc.allgather(ttnn.reshape(hh, [n8, 1, T, D]), cc, axis=1, dim=1)  # [n8,8,T,D]
        ms = []
        for g in range(n8):
            hh_g = ttnn.reshape(ttnn.slice(hha, [g, 0, 0, 0], [g + 1, 8, T, D]), [1, 1, 8 * T, D])
            tok_g = ttnn.reshape(ttnn.to_layout(hh_g, ttnn.ROW_MAJOR_LAYOUT), [8 * T, 1, 1, D])
            if route_own:
                k_ = sc_all.shape[3]
                sc_g = ttnn.slice(sc_all, [g * 8 * T, 0, 0, 0], [(g + 1) * 8 * T, 1, 1, k_])
                ix_g = ttnn.slice(ix_all, [g * 8 * T, 0, 0, 0], [(g + 1) * 8 * T, 1, 1, k_])
                mg = self.pmoe.forward_routed(tok_g, sc_g, ix_g)
                ttnn.deallocate(sc_g)
                ttnn.deallocate(ix_g)
            else:
                mg = self.pmoe.forward(hh_g, tok_g)  # [1,1,256,D/8]
            ttnn.deallocate(hh_g)
            ttnn.deallocate(tok_g)
            m_own = ttnn.experimental.all_to_all_async_generic(
                mg, in_dim=3, out_dim=2, num_links=cc.num_links, topology=ttnn.Topology.Ring, cluster_axis=1
            )
            ttnn.deallocate(mg)
            ms.append(m_own)
        ttnn.deallocate(hha)
        if route_own:
            ttnn.deallocate(sc_all)
            ttnn.deallocate(ix_all)
        _mark("moe+allgather")
        sh_all = shared_big(L.shared, hh)
        ttnn.deallocate(hh)
        _mark("shared")
        m_all = f32(ttnn.concat(ms, dim=2) if n8 > 1 else ms[0])
        for t in ms:
            ttnn.deallocate(t)
        x3 = pkf.expand(m_all, x2, post_f, comb_f, sh_all)
        for t in (m_all, sh_all, x2, post_f, comb_f):
            ttnn.deallocate(t)
        if converted:  # the caller owns (and frees) the chunk lists it passed in, we own the packed copies we made
            ttnn.deallocate(x)
            ttnn.deallocate(pre_in)
        _mark("expand_out")
        _dump_marks()
        return PkList([x3]), PkList([pre_f])

    def forward_cols(self, xs, pres, S, s0=0):
        """Column-split layer: xs / pres hold only the 32-token chunks THIS column owns (chunk i of the row belongs to column i % 8, so list
        element g = chunk 8g + column); the 8 columns no longer repeat the token-wise work (mHC, router input, shared expert, expand).
        Per group g of 8 chunks (256 consecutive tokens): all_gather of the attention input / MoE input over the columns, reduce_scatter
        of the attention output over tokens, all_to_all of the MoE output (hidden shard -> own tokens). Needs DSV41_MOE_G=8.
        """
        L, T = self.L, self.T
        n8 = len(xs)
        assert self.umoe is not None or self.pmoe.T == 8 * T, "column split needs DSV41_MOE_G=8 (or the unified MoE)"
        mc, cc = L.mesh_config, L.ccl
        eps = L.eps
        _mark("start")
        mix, hg = [], []
        for i_, (x, p) in enumerate(zip(xs, pres)):
            _drain(L, i_)
            mix.append(L.mhc_attn.mixes(x))
            h_own = L.mhc_attn.collapse_norm(x, p, L.attn_norm_w, eps)
            hg.append(mc.allgather(h_own, cc, axis=1, dim=2))  # [1,1,256,D] consecutive tokens of group g
            _free("hs", h_own)
        _mark("mhc_attn")
        h = ttnn.concat(hg, dim=2) if n8 > 1 else hg[0]
        if n8 > 1:  # (n8 == 1: h IS hg[0])
            for t in hg:
                ttnn.deallocate(t)
        pa = self.pa
        pa.rs_tokens = True
        a = pa.forward_dyn(h) if pa.dyn is not None else pa.forward(h, S, s0=s0)  # [1,1,n8*32,D] own chunks
        pa.rs_tokens = False
        _free("h", h)
        _mark("attention")
        st = []
        for g in range(n8):
            _drain(L, g)
            a_c = ttnn.slice(a, [0, 0, g * T, 0], [1, 1, (g + 1) * T, a.shape[3]])
            x2 = L.mhc_attn.expand(a_c, xs[g], mix[g][1], mix[g][2])
            _free("a_c", a_c)
            f_pre, f_post, f_comb = L.mhc_ffn.mixes(x2)
            hh = L.mhc_ffn.collapse_norm(x2, mix[g][0], L.ffn_norm_w, eps)
            st.append((x2, f_pre, f_post, f_comb, hh))
        _free("a", a)
        _mark("mhc_expand_collapse")
        ms = []
        if (
            self.umoe is not None
        ):  # DSV41_PREFILL_MOE=unified: the routed experts of the whole chunk in one count-driven pipeline
            ms = self._moe_unified(st, n8)
        for g in range(n8 if self.umoe is None else 0):
            _drain(L, g, 2)
            hh_g = mc.allgather(st[g][4], cc, axis=1, dim=2)  # [1,1,256,D]
            tok_g = ttnn.reshape(ttnn.to_layout(hh_g, ttnn.ROW_MAJOR_LAYOUT), [8 * T, 1, 1, hh_g.shape[3]])
            mg = self.pmoe.forward(hh_g, tok_g)  # [1,1,256,D/8]
            ttnn.deallocate(hh_g)
            ttnn.deallocate(tok_g)
            m_own = ttnn.experimental.all_to_all_async_generic(
                mg, in_dim=3, out_dim=2, num_links=cc.num_links, topology=ttnn.Topology.Ring, cluster_axis=1
            )
            ttnn.deallocate(mg)
            ms.append(m_own)
        _mark("moe+allgather")
        if getattr(self, "_sh_pre", None) is not None:  # computed on the shared-expert sub-device during the dispatch
            sh_all, hh_all = self._sh_pre[0], self._sh_pre[1]
            self._sh_pre = None
            if hh_all is None:
                hh_all = st[0][4]
        else:
            hh_all = ttnn.concat([e[4] for e in st], dim=2) if n8 > 1 else st[0][4]
            sh_all = shared_big(L.shared, hh_all)
        _mark("shared")
        outs, pres_out = [], []
        for g in range(n8):
            _drain(L, g)
            x2, f_pre, f_post, f_comb, hh = st[g]
            sh = ttnn.slice(sh_all, [0, 0, g * T, 0], [1, 1, (g + 1) * T, sh_all.shape[3]])
            x3 = L.mhc_ffn.expand(ms[g], x2, f_post, f_comb, sh)
            outs.append(ttnn.to_memory_config(x3, ttnn.DRAM_MEMORY_CONFIG))
            pres_out.append(ttnn.to_memory_config(f_pre, ttnn.DRAM_MEMORY_CONFIG))
            for tag, t in (("x2", x2), ("hh", hh), ("m", ms[g]), ("sh", sh)):
                _free(tag, t)
        ttnn.deallocate(sh_all)
        if n8 > 1:
            ttnn.deallocate(hh_all)
        _mark("expand_out")
        _dump_marks()
        return outs, pres_out

    def forward_full(self, xs, pres, S, s0=0):
        """xs: list of n chunks [T,1,4,D] fp32 (the mesh row's R = n*T tokens, user-major), pres: list of [T,1,1,4] fp32.
        -> (list of new streams, list of ffn_pre)."""
        L, T = self.L, self.T
        n = len(xs)
        pmoe = (
            self.pmoe if n % max(1, self.pmoe.T // T) == 0 else self.pmoe.g1
        )  # chunk count not a multiple of G: T=32 slices
        eps = L.eps
        _mark("start")
        mix, hs = [], []
        for i, (x, p) in enumerate(zip(xs, pres)):
            mix.append(L.mhc_attn.mixes(x))  # (pre, post, comb)
            hs.append(L.mhc_attn.collapse_norm(x, p, L.attn_norm_w, eps))
            if _PROF_EVERY and i % (4 * _PROF_EVERY) == 4 * _PROF_EVERY - 1:
                ttnn.synchronize_device(L.mesh_device)
                ttnn.ReadDeviceProfiler(L.mesh_device)
        if _PROF_EVERY:
            ttnn.synchronize_device(L.mesh_device)
            ttnn.ReadDeviceProfiler(L.mesh_device)
        h = ttnn.concat(hs, dim=2) if n > 1 else hs[0]
        for t in hs:
            _free("hs", t)
        if self.debug is not None:
            self.debug["hs"], self.debug["h"] = list(hs), h
        _mark("mhc_attn")
        a = self.pa.forward_dyn(h) if self.pa.dyn is not None else self.pa.forward(h, S, s0=s0)
        _mark("attention")
        _free("h", h)
        if self.debug is not None:
            ttnn.synchronize_device(L.mesh_device)
            cols = tuple(L.mesh_device.shape)[1]
            self.debug["a_full"] = a
            self.debug["a_early"] = [
                ttnn.to_torch(ttnn.get_device_tensors(a)[r * cols]).float().reshape(-1, a.shape[3])
                for r in range(tuple(L.mesh_device.shape)[0])
            ]
        if _PROF_EVERY:
            ttnn.synchronize_device(L.mesh_device)
            ttnn.ReadDeviceProfiler(L.mesh_device)
        outs, pres_out = [], []
        G = max(
            1, getattr(pmoe, "T", T) // T
        )  # MoE call covers G chunks of T tokens per device (moe_compute at T*G tokens/device)
        for g0 in range(0, n, G):
            cs = list(range(g0, min(n, g0 + G)))
            if len(cs) < G:  # tail group smaller than the MoE batch: run its chunks one by one through a G=1 front-end
                raise RuntimeError(f"chunk count {n} is not a multiple of the MoE group {G}")
            st = []
            for c in cs:
                a_c = ttnn.slice(a, [0, 0, c * T, 0], [1, 1, (c + 1) * T, a.shape[3]])
                if self.debug is not None:
                    self.debug.setdefault("a", []).append(a_c)
                x2 = L.mhc_attn.expand(a_c, xs[c], mix[c][1], mix[c][2])
                _free("a_c", a_c)
                f_pre, f_post, f_comb = L.mhc_ffn.mixes(x2)
                hh, h_tok = L.mhc_ffn.collapse_norm_rm(x2, mix[c][0], L.ffn_norm_w, eps)
                st.append((x2, f_pre, f_post, f_comb, hh, h_tok))
            _mark("mhc_expand_collapse")
            if G == 1:
                hh_g, tok_g = st[0][4], st[0][5]
            else:
                hh_g = ttnn.concat([e[4] for e in st], dim=2)
                tok_g = ttnn.concat([e[5] for e in st], dim=0)
            mg = pmoe.forward(hh_g, tok_g)
            mg = L.mesh_config.allgather(mg, L.ccl, axis=1, dim=3)
            _mark("moe+allgather")
            big = (
                G > 1
                and os.environ.get("DSV41_SHARED_BIG", "1") == "1"
                and hasattr(L.shared, "w01")
                and hasattr(L.shared, "act_a")
            )
            sh_g = shared_big(L.shared, hh_g) if big else None
            for j, (x2, f_pre, f_post, f_comb, hh, h_tok) in enumerate(st):
                m = mg if G == 1 else ttnn.slice(mg, [0, 0, j * T, 0], [1, 1, (j + 1) * T, mg.shape[3]])
                _mark("slice")
                sh = (
                    ttnn.slice(sh_g, [0, 0, j * T, 0], [1, 1, (j + 1) * T, sh_g.shape[3]])
                    if big
                    else L.shared.forward(hh)
                )
                _mark("shared")
                if self.debug is not None:
                    for k_, v_ in (("hh", hh), ("m", m), ("sh", sh), ("x2", x2)):
                        self.debug.setdefault(k_, []).append(v_)
                x3 = L.mhc_ffn.expand(m, x2, f_post, f_comb, sh)
                outs.append(ttnn.to_memory_config(x3, ttnn.DRAM_MEMORY_CONFIG))
                pres_out.append(ttnn.to_memory_config(f_pre, ttnn.DRAM_MEMORY_CONFIG))
                _mark("expand_out")
                for tag, t in (("x2", x2), ("hh", hh), ("h_tok", h_tok), ("m", m), ("sh", sh)):
                    _free(tag, t)
            if _PROF_EVERY and (g0 // G) % _PROF_EVERY == _PROF_EVERY - 1:
                ttnn.synchronize_device(L.mesh_device)
                ttnn.ReadDeviceProfiler(L.mesh_device)
            if G > 1:
                for t in (hh_g, tok_g, mg) + ((sh_g,) if big else ()):
                    ttnn.deallocate(t)
        _free("a", a)
        _dump_marks()
        return outs, pres_out
