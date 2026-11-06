# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's MoE layer on Marko Staletovic's reworked MoE block (models/demos/moe_block_main, the C++ ops
ttnn.bringup.moe_block_main: no DRISC), setting MOE_BLOCK_SRC=main (MOE_IMPL=block, MOE_HEAD=fused). The same layer
interface as tt/moe_block_adapter.py's TtMoEBlockLayer (the carried block, MOE_BLOCK_SRC=carried):

    x  ffn_norm [1, 1, S/SP, H/TP] bf16 TILE (OWN_ROWS: the chip's own rows [1, 1, S/(SP TP), H])
    -> fused head (x gather, shared expert, router, routing exchange + planner, dispatch prep) -> sparse 2D dispatch ->
       routed experts + combine (one program) -> the tail
    -> [1, 1, S/SP, H/TP] (OWN_ROWS: [1, 1, S/(SP TP), H]) bf16 (fp32 under GLUE_FOLD=off)

The block's ops take [S/SP, H/TP] in and give it back (its fused tail: tilize + the shared expert + a TP all-to-all).
OWN_ROWS (P.23): the ops have no own-rows head / tail (the carried block's MoEOptions residual "owners"); here x goes
through one TP all-to-all into the head's [S/SP, H/TP] and the tail is the local one in two stock ops: the combine's
owner rows tilized + the shared expert's owner rows (no all-to-all back) - the carried TailA2A local mode's math.

What is this model's (re-applied from the carried adapter, each still meaningful on the new ops):
- expert numbering: the block holds expert b on chip b // epc (row-major); our chips hold checkpoint expert
  gtab[c, r, le] in slot le, so block expert b = (r C + c) epc + le is checkpoint expert perm[b] and the router's rows
  are permuted to match (moe_block_adapter.expert_permutation);
- the routed experts' weights in the model's EXPERTS_IMPL format (TtExperts._host_weights: e2m1 x ue8m0 dequantized,
  bfp4_mse rounded in the checkpoint's K groups) laid out [N, K] (FlatLayoutOptions.weights_nk, DS41-35);
- D5 / FIX.9: the shared expert at LoFi with GEMM_INPUTS' activations; the combine snapped (COMBINE_SNAP: bit-identical
  runs); FIX.15: the block's programs inside tt/cores.max_grid (set_max_grid: 11 x 10 on a Galaxy).
Not available on the new ops (the carried block's Python programs): NORM_INTO_MOE's in-place x (input_buffers() is
(None, None): ffn_norm writes its own output), TAIL_A2A_PREPARED / PREPARED_ALL / HOST_ALLOC's block hooks (the C++
ops patch addresses on their own cache hits), SE_PACK_PRECISE, DECODER_SKIP=replay128 (at_chunk).

Weights: per layer, the head's (router + shared expert: prepare_fused_head_weights) from the checkpoint at the
layer's first call; the routed experts laid out at load (host pack, ~1 min a layer) and kept in the weight cache
(<cache_dir>/<rows>x<cols>/L<layer>.<impl>.<fingerprint>.moe_main.<plan>.{w_gu,w_d}.tensorbin) when one is given.
"""

from __future__ import annotations

import hashlib
import os
import time
import types
from pathlib import Path

import torch
from loguru import logger

import ttnn

from . import ccl
from .ccl import dealloc
from .common import DRAM, TP_AXIS, cast
from .cores import max_grid
from .settings import settings

LINKS = 2  # links per direction
COMBINE_SNAP = 2.0**-19  # (as the carried block, owner 2026-10-27: order-independent fp32 sums)
_BLOCKS: dict = {}  # (id(mesh), chunk) -> (mesh, MoEBlockResources, MoEBlock)
WDTYPES = {"bfp4": "bf4", "bfp4_mse": "bf4", "bfp8": "bf8"}  # EXPERTS_IMPL -> the flat op's one weight format


def _mb():
    return ttnn.bringup.moe_block_main


def block_dims(cfg):
    """(MoEModelDims, flat activation name) of DeepSeek-V4.1-Flash at this checkpoint: sqrt(softplus) scores, top-k over
    all experts x route_scale, clamped SwiGLU (limit swiglu_limit) for the routed and the shared expert."""
    mb = _mb()
    assert float(cfg.swiglu_limit) == 10.0, f"the flat op's clamped_silu bakes limit 10, got {cfg.swiglu_limit}"
    act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=float(cfg.swiglu_limit))
    dims = mb.MoEModelDims(
        num_experts=int(cfg.n_experts),
        top_k=int(cfg.topk_experts),
        emb_dim=int(cfg.dim),
        routed_expert_hidden_dim=int(cfg.inter),
        routed_expert_activation=act,
        router=mb.RouterSpec(score_func=mb.RouterScoreFunc.SqrtSoftplus, route_scale=float(cfg.route_scale)),
        shared_expert_hidden_dim=int(cfg.inter),
        shared_expert_activation=act,
    )
    return dims, "clamped_silu"


def block_dtypes():
    """MoEDtypes: routed weights in EXPERTS_IMPL's format (one for gate / up and down: bfp8_down4 has no equivalent),
    the shared expert at LoFi (D5) with GEMM_INPUTS' activations (FIX.9), router bf16."""
    from .common import gemm_dtype

    impl = settings.get("EXPERTS_IMPL")
    assert (
        impl in WDTYPES
    ), f"MOE_BLOCK_SRC=main: EXPERTS_IMPL={impl} (the flat op takes one weight format: bfp8, bfp4*)"
    mb = _mb()
    d = mb.MoEDtypes()
    d.routed_weights = ttnn.bfloat8_b if WDTYPES[impl] == "bf8" else ttnn.bfloat4_b
    d.shared_fidelity = ttnn.MathFidelity.LoFi
    d.shared_activations = gemm_dtype()
    return d


def _apply_core_cap(mesh) -> None:
    """FIX.15 (MAX_WORKER_CORES): the block's programs inside tt/cores.max_grid (once a process, before any plan)."""
    g = max_grid(mesh)
    cap = (int(g.x), int(g.y)) if g is not None else (0, 0)
    _mb().set_max_grid(*cap)


def _host_routed(ck, layer: int, mesh, cfg):
    """[device r C + c][le] = (W1 [I, H], W3 [I, H], W2 [H, I]) bf16 as stored ([N, K]), in EXPERTS_IMPL's format
    (TtExperts._host_weights: the same dequantization, checks and bfp4_mse rounding) - chip (r, c)'s slot le holds
    checkpoint expert gtab[c, r, le], the block's expert (r C + c) epc + le."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

    from .experts import IMPLS, TtExperts

    rows, cols = tuple(mesh.shape)
    n = rows * cols
    impl = settings.get("EXPERTS_IMPL")
    me = types.SimpleNamespace(
        dgs=rows,
        ngroups=cols,
        layer=int(layer),
        H=int(cfg.dim),
        I=int(cfg.inter),
        epc=int(cfg.n_experts) // n,
        impl=impl,
        formats=IMPLS[impl][1],
        bfp4_stats=None,
    )
    gtab = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=me.epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    flat_w = TtExperts._host_weights(me, ck, gtab)
    return flat_w, me.bfp4_stats


class TtMoEBlockMainLayer:
    """One layer's MoE on the shared reworked MoE block: __call__(ffn_norm x) -> the MoE output."""

    def __init__(self, mesh, ck, cfg, layer: int, chunk: int, cache_dir=None, own_rows: bool = False):
        from .moe_block_adapter import expert_permutation

        assert settings.get("MOE_HEAD") == "fused", "MOE_BLOCK_SRC=main: the block's head is fused (MOE_HEAD=fused)"
        assert settings.get("DECODER_SKIP") != "replay128", "MOE_BLOCK_SRC=main: no at_chunk (DECODER_SKIP=replay128)"
        self.mesh, self.ck, self.cfg = mesh, ck, cfg
        self.layer, self.chunk = int(layer), int(chunk)
        self.own_rows = bool(own_rows)
        self.fold = ccl.glue_fold()
        self.cache_dir = cache_dir
        self.dims, self.act = block_dims(cfg)
        self.dtypes = block_dtypes()
        rows, cols = tuple(mesh.shape)
        self.epc = int(cfg.n_experts) // (rows * cols)
        self.perm = expert_permutation(rows, cols, self.epc)
        _apply_core_cap(mesh)
        t0 = time.time()
        blk = self.block()
        self.gidx, self.w_gu, self.w_d = self._routed(blk)
        self.head_w = None  # (the head's weights: at the layer's first call, planned by the block's head)
        logger.info(
            f"MOE_BLOCK_MAIN layer={layer} experts={cfg.n_experts} ({self.epc} a chip) chunk={chunk} "
            f"experts_impl={settings.get('EXPERTS_IMPL')} own_rows={self.own_rows} cap={_mb().max_grid()} "
            f"routed weights {time.time() - t0:.1f} s"
        )

    # ------------------------------------------------------------------ the block (one a mesh and chunk)
    def block(self):
        from models.demos.moe_block_main.tt.moe_block import MoEBlock

        key = (id(self.mesh), self.chunk)
        hit = _BLOCKS.get(key)
        if hit is None or hit[0] is not self.mesh:
            mb = _mb()
            cfg = mb.MoEBlockConfig(
                chunk_tokens=self.chunk, num_links=LINKS, combine_snap=COMBINE_SNAP, persist_l1_small=True
            )
            t0 = time.time()
            res = mb.MoEBlockResources(self.mesh, self.dims, cfg, self.dtypes)
            fw = _flat_weights_module()
            blk = MoEBlock(
                self.mesh,
                res,
                dtypes=self.dtypes,
                activation=fw.ACTS[self.act],
                max_tokens_per_expert=max(self.chunk, 256),
                layout=mb.FlatLayoutOptions(weights_nk=True),
            )
            logger.info(
                f"MOE_BLOCK_MAIN chunk={self.chunk}: built in {time.time() - t0:.1f} s, accumulators "
                f"{blk.accumulators}, {blk.splits} splits x {blk.loops} token loops, arena "
                f"{blk.arena_spec.memory_config.shard_spec.shape[0] * 64 >> 10} KB a core, x in DRAM {blk.x_dram}"
            )
            hit = _BLOCKS[key] = (self.mesh, res, blk)
        return hit[2]

    # ------------------------------------------------------------------ weights
    def _cache_paths(self, blk, lay):
        if self.cache_dir is None:
            return None
        from .experts import expert_cache_prefix

        prefix = expert_cache_prefix(
            self.cache_dir, self.ck, self.layer, settings.get("EXPERTS_IMPL"), self.mesh, self.cfg.n_experts
        )
        if prefix is None:
            return None
        key = hashlib.sha1(repr((sorted(lay.items()), "nk", str(self.dtypes.routed_weights))).encode()).hexdigest()[:12]
        return {n: Path(f"{prefix}.moe_main.{key}.{n}.tensorbin") for n in ("w_gu", "w_d")}

    def _routed(self, blk):
        """(gidx, w_gu, w_d): the routed experts in the flat plan's bank layout ([N, K] tiles), from the weight cache or
        the checkpoint."""
        mb, fw = _mb(), _flat_weights_module()
        rows, cols = tuple(self.mesh.shape)
        n = rows * cols
        kw = blk.expert_kw
        lay = mb.flat_routed_expert_plan(
            self.mesh,
            int(self.cfg.dim),
            int(self.cfg.inter),
            self.epc,
            int(self.cfg.n_experts),
            kw["max_tokens_per_expert"],
            weights_bf8=self.dtypes.routed_weights == ttnn.bfloat8_b,
            pin=1,
            layout=kw["layout"],
        ).as_dict()
        gids = torch.arange(int(self.cfg.n_experts), dtype=torch.int32).reshape(n, 1, self.epc)
        gidx = ttnn.from_torch(
            gids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        paths = self._cache_paths(blk, lay)
        if paths is not None and all(p.exists() for p in paths.values()):
            t0 = time.time()
            w = {k: ttnn.load_tensor(str(p), device=self.mesh) for k, p in paths.items()}
            logger.info(f"MOE_BLOCK_MAIN layer={self.layer} routed weights cache hit {time.time() - t0:.1f} s")
            return gidx, w["w_gu"], w["w_d"]
        t0 = time.time()
        flat_w, stats = _host_routed(self.ck, self.layer, self.mesh, self.cfg)
        t1 = time.time()
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:  # (torch's copies release the GIL: a chip's experts a task)
            packed = list(pool.map(lambda W_l: fw.flat_weight_regions(W_l, lay, nk=True), flat_w))
        del flat_w
        t2 = time.time()
        dtype, banks = self.dtypes.routed_weights, lay["banks"]
        w_gu = fw.bank_sharded([p[0] for p in packed], banks, dtype, self.mesh)
        w_d = fw.bank_sharded([p[1] for p in packed], banks, dtype, self.mesh)
        del packed
        t3 = time.time()
        if paths is not None:
            for k, t in (("w_gu", w_gu), ("w_d", w_d)):
                tmp = paths[k].with_name(paths[k].name.replace(".tensorbin", f".tmp{os.getpid()}.tensorbin"))
                tmp.parent.mkdir(parents=True, exist_ok=True)
                ttnn.dump_tensor(str(tmp), t)
                os.replace(tmp, paths[k])
        if stats is not None:
            logger.info(
                f"EXPERTS_BFP4 layer={self.layer} values_changed={stats['changed']}/{stats['total']} "
                f"({100 * stats['changed_frac']:.2f} %) worst_matrix_rel={stats['worst_rel']:.4g}"
            )
        logger.info(
            f"MOE_BLOCK_MAIN layer={self.layer} routed weights: checkpoint {t1 - t0:.1f} s, pack {t2 - t1:.1f} s, "
            f"upload {t3 - t2:.1f} s, cache write {time.time() - t3:.1f} s ({'cached' if paths else 'no cache'})"
        )
        return gidx, w_gu, w_d

    def _head_weights(self, blk):
        if self.head_w is None:
            mb = _mb()
            p = f"layers.{self.layer}.ffn."
            w = self.ck.get(p + "gate.weight")
            assert w.dtype == torch.bfloat16, f"router weight stored as {w.dtype}, expected bf16"
            b = self.ck.weight(p + "gate.bias", torch.float32)
            s = {k: self.ck.weight(f"{p}shared_experts.{k}.weight", torch.float32) for k in ("w1", "w3", "w2")}
            host = lambda t: ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32)  # noqa: E731
            self.head_w = mb.prepare_fused_head_weights(
                host(s["w1"]),
                host(s["w3"]),
                host(s["w2"]),
                None,
                host(w[self.perm]),
                host(b[self.perm]),
                resources=blk.resources,
                dtypes=blk.dtypes,
                accumulators=blk.accumulators,
                l1_free=blk.l1_free,
            )
        from models.demos.moe_block_main.tt.moe_block import MoEBlockWeights

        return MoEBlockWeights(head=self.head_w, expert_ids=self.gidx, gate_up=self.w_gu, down=self.w_d)

    # ------------------------------------------------------------------ the model's interface (TtMoEBlockLayer's)
    def input_buffers(self):
        """NORM_INTO_MOE: the new ops read x from any buffer (no persistent input of the block's): none."""
        return None, None

    def at_chunk(self, chunk: int):
        raise NotImplementedError("MOE_BLOCK_SRC=main: DECODER_SKIP=replay128 (at_chunk) is the carried block's")

    def __call__(self, x: ttnn.Tensor, xq_ready: bool = False) -> ttnn.Tensor:
        assert not xq_ready, "MOE_BLOCK_SRC=main: no NORM_INTO_MOE input buffers"
        assert x.layout == ttnn.TILE_LAYOUT
        blk = self.block()
        weights = self._head_weights(blk)
        xb = cast(x, ttnn.bfloat16)
        if not self.own_rows:
            y = blk(xb, weights)
        else:
            y = self._own_rows(blk, xb, weights)
        if xb is not x:
            dealloc(xb)
        if not self.fold:
            y32 = ttnn.typecast(y, ttnn.float32, memory_config=DRAM)
            dealloc(y)
            y = y32
        return y

    def _own_rows(self, blk, x_own, weights):
        """OWN_ROWS: x [1, 1, S/(SP TP), H] -> one TP all-to-all into [1, 1, S/SP, H/TP] -> head, dispatch, experts +
        combine -> the combine's owner rows [S/(SP TP), H] (row-major) tilized + the shared expert's owner rows."""
        mb = _mb()
        if blk.loops * blk.splits != 1 or blk.pad_rows:  # (token loops / splits / padding: the block's own forward)
            xc = ttnn.experimental.all_to_all_async_generic(
                x_own, in_dim=2, out_dim=3, num_links=2, memory_config=DRAM, cluster_axis=TP_AXIS
            )
            y = blk(xc, weights)
            dealloc(xc)
            out = ttnn.experimental.all_to_all_async_generic(
                y, in_dim=3, out_dim=2, num_links=2, memory_config=DRAM, cluster_axis=TP_AXIS
            )
            dealloc(y)
            return out
        x = ttnn.experimental.all_to_all_async_generic(
            x_own, in_dim=2, out_dim=3, num_links=2, memory_config=DRAM, cluster_axis=TP_AXIS
        )
        if blk.exchange_spec is None:
            blk.one_arena = blk._one_arena_fits()
        x_full = (
            ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if blk.x_dram else None
        )
        arena = ttnn.allocate_tensor_on_device(blk.arena_spec, self.mesh)
        head = mb.fused_head(
            x,
            weights.head,
            resources=blk.resources,
            dtypes=blk.dtypes,
            accumulators=blk.accumulators,
            x_full=x_full,
            l1_free=blk.l1_free,
            arena=arena,
        )
        dealloc(x)
        if x_full is not None:
            dealloc(x_full)
        if blk.exchange_spec is None:
            blk.exchange_spec = (
                False
                if blk.one_arena
                else mb.routed_experts_combine_arena_spec(
                    blk.resources, rows_cap=int(head.table.shape[-2]) * 32, **blk.expert_kw
                )
            )
        if blk.exchange_spec:
            ttnn.deallocate(arena)
            arena = ttnn.allocate_tensor_on_device(blk.exchange_spec, self.mesh)
        dispatched = mb.sparse_dispatch_2d(head.xbuf, head.route, blk.resources, arena=arena)
        y_own = mb.routed_experts_combine(
            dispatched,
            head.counts,
            head.regions,
            head.token_index,
            head.table,
            head.plan,
            weights.expert_ids,
            weights.gate_up,
            weights.down,
            blk.resources,
            arena=arena,
            **blk.expert_kw,
        )
        ttnn.deallocate(arena)
        se = head.se
        assert tuple(y_own.shape) == tuple(x_own.shape) and tuple(se.shape)[-2:] == tuple(x_own.shape)[-2:], (
            f"MOE_BLOCK_SRC=main OWN_ROWS: owner rows y {tuple(y_own.shape)}, shared {tuple(se.shape)}, x "
            f"{tuple(x_own.shape)}"
        )
        yt = ttnn.to_layout(y_own, ttnn.TILE_LAYOUT, memory_config=DRAM)  # (y_own, se: the block's, kept)
        out = ttnn.add(yt, se, memory_config=DRAM)
        dealloc(yt)
        return out


_FW = None


def _flat_weights_module():
    """The flat op's Python weight prep (ttnn/ttnn/bringup/moe_block_main_ttnn/flat_routed_expert/
    flat_routed_expert_weights.py; models/demos/moe_block_main/tt/flat_gather.flat_weights_module)."""
    global _FW
    if _FW is None:
        from models.demos.moe_block_main.tt.flat_gather import flat_weights_module

        _FW = flat_weights_module()
    return _FW


def build_moe_block_main_layer(mesh, ck, cfg, layer: int, chunk: int, cache_dir=None, own_rows: bool = False):
    return TtMoEBlockMainLayer(mesh, ck, cfg, layer, chunk, cache_dir=cache_dir, own_rows=own_rows)
