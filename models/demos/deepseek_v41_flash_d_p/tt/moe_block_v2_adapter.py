# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's MoE layer on Marko Staletovic's MoE block as merged in his stack (mstaletovic/moe-block-65124-
split-rows: models/demos/moe_block/tt/moe_block.py MoEBlock on the C++ ops of ttnn/cpp/.../experimental/moe_block),
run on this branch's fork of those ops, ttnn.bringup.moe_block (ttnn/ttnn/bringup/moe_block_ttnn): setting
MOE_BLOCK_SRC=block (MOE_IMPL=block, MOE_HEAD=fused). The same layer interface as tt/moe_block_main_adapter.py's
(MOE_BLOCK_SRC=main, his older block, kept as the fallback):

    x  ffn_norm [1, 1, S/SP, H/TP] bf16 TILE (OWN_ROWS: the chip's own rows [1, 1, S/(SP TP), H])
    -> moe_head (x gather, shared expert, router, routing exchange + planner, dispatch prep) -> sparse 2D dispatch ->
       routed experts + combine (one program) -> the tail
    -> [1, 1, S/SP, H/TP] (OWN_ROWS: [1, 1, S/(SP TP), H]) bf16 (fp32 under GLUE_FOLD=off)

Split rows (MOE_BLOCK_SPLIT_ROWS, Marko's tip): the ops send a token row in two fabric packets when the mesh's fabric
payload holds less than a row + its 32 B trailer (the combine) / a row (the dispatch) - they read the payload the mesh
opened with. MOE_BLOCK_SPLIT_ROWS=on opens the mesh at FABRIC_PAYLOAD_SPLIT (6208 B: half a 10240 B DeepSeek-V4.1 row
+ its trailer fits, the whole row does not) instead of 10880 B (tt/weighted_combine.fabric_payload).

What is this model's, on the new ops (ttnn/ttnn/bringup/moe_block_ttnn/CHANGELOG.md for the fork's side):
- expert numbering, routed weights in EXPERTS_IMPL's format laid out [N, K] (FlatLayoutOptions.weights_nk) - as the
  main adapter; the main block's weight cache files are read when the new flat plan lays the weights out the same
  (_reuse_main_cache), else the new layout is cached as <prefix>.moe_blk.<plan>.{w_gu,w_d}.tensorbin;
- FIX.15: the block's programs inside tt/cores.max_grid (set_max_grid);
- EXPERT_X_PIECES (P.59): FlatLayoutOptions.x_min_piece_bytes;
- the combine: Marko's int32 snap (mstaletovic/moe-block-int32-snap a7c1a9d78c0) at COMBINE_SNAP - exact, so
  order-free, up to (2^31 - 1) q = 65536 (the routed sums reach ~2^14.4 here: FIX.23); it replaces COMBINE_SPLIT's
  hi / lo fp32 sums (not ported);
- MESH_PROGRAMS (P.63, cached_slots): not ported - the new ops keep their per-core values in a DRAM args table and a
  cache hit rewrites only common args (common/args_table.hpp), which is what cached_slots bought;
- the shared expert at LoFi; NOT its bfloat8_b x / h (MoEDtypes.shared_activations) nor SE_PACK_PRECISE: the new
  moe_head has neither (its x and h stay bf16); NORM_INTO_MOE then gives x only (no xq twin);
- OWN_ROWS / FFN_FEED, LEAN_PY's MoE output, DECODER_SKIP=replay128's at_chunk: as the main adapter.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn

from . import ccl
from .ccl import dealloc
from .common import DRAM, TP_AXIS, cast
from .cores import max_grid, sub_cores
from .moe_block_main_adapter import _host_routed, _lean_out, x_min_piece_bytes
from .settings import settings

LINKS = 2  # links per direction
# the combine's snap q (int32 counts of q: exact up to (2^31 - 1) q = 65536; DeepSeek-V4.1's routed sums reach 21376,
# FIX.23 / DS41-235). The main block's fp32 snap used 2^-19 with hi / lo sums (COMBINE_SPLIT).
COMBINE_SNAP = 2.0**-15
FABRIC_PAYLOAD_SPLIT = 6208  # B: MOE_BLOCK_SPLIT_ROWS=on (half a 10240 B row + 32 B trailer fits; owner 2026-11-18)
WDTYPES = {"bfp4": "bf4", "bfp4_mse": "bf4", "bfp8": "bf8"}  # EXPERTS_IMPL -> the flat op's one weight format
_BLOCKS: dict = {}  # (id(mesh), chunk, x pieces) -> (mesh, MoEBlockResources, MoEBlock)
_INPUTS: dict = {}  # (id(mesh), chunk) -> (mesh, x bf16): NORM_INTO_MOE's input buffer
_FEED: dict = {}  # (id(mesh), chunk) -> (mesh, x bf16): FFN_FEED's head input (P.36)
_REUSE: dict = {}  # (new plan facts, old plan facts) -> bool
_FW = None
# the new flat plan's facts the routed weights' layout depends on (flat_routed_expert_weights.flat_weight_regions /
# bank_sharded; Marko's TtK3MoEBlockFfn._ROUTED_PACKING), and the main block's (moe_block_main) name for each
_ROUTED_PACKING = {
    "dram_banks": "banks",
    "gate_up_weight_slots": "n_rd",
    "column_groups_per_weight_slot": "rg",
    "gate_up_pairs_per_core": "np",
    "units_per_core": "nu",
    "gate_up_k_blocks": "nk_gu",
    "down_core_count": "nd",
    "down_columns_per_core": "pcds",
    "down_first_columns": "col0s",
    "down_weight_region_tiles": "wd_rot_tiles",
}


def _mb():
    return ttnn.bringup.moe_block


def _flat_weights_module():
    """The fork's flat op Python weight prep (ttnn/ttnn/bringup/moe_block_ttnn/flat_routed_expert/
    flat_routed_expert_weights.py: weights_nk)."""
    global _FW
    if _FW is None:
        path = Path(ttnn.__file__).parent / "bringup/moe_block_ttnn/flat_routed_expert/flat_routed_expert_weights.py"
        spec = importlib.util.spec_from_file_location("moe_block_ttnn_flat_weights", path)
        _FW = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_FW)
    return _FW


@contextlib.contextmanager
def _fork_ops():
    """MoEBlock (models/demos/moe_block) takes its ops from ttnn.experimental.moe_block in its constructor (self.mb):
    point that name at the fork while one is built, so every plan and call of the block is the fork's."""
    exp = ttnn.experimental
    had = hasattr(exp, "moe_block")
    old = getattr(exp, "moe_block", None)
    exp.moe_block = _mb()
    try:
        yield
    finally:
        if had:
            exp.moe_block = old
        else:
            delattr(exp, "moe_block")


def block_dims(cfg):
    """(MoEModelDims, flat activation name) of DeepSeek-V4.1-Flash: sqrt(softplus) scores, top-k over all experts x
    route_scale, clamped SwiGLU (limit swiglu_limit) for the routed and the shared expert."""
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
    """MoEDtypes: routed weights in EXPERTS_IMPL's format, the shared expert at LoFi (D5; its weights bfloat8_b, the
    default). The new head has no bfloat8_b activations (shared_activations) or pack_precise."""
    impl = settings.get("EXPERTS_IMPL")
    assert (
        impl in WDTYPES
    ), f"MOE_BLOCK_SRC=block: EXPERTS_IMPL={impl} (the flat op takes one weight format: bfp8, bfp4*)"
    d = _mb().MoEDtypes()
    d.routed_weights = ttnn.bfloat8_b if WDTYPES[impl] == "bf8" else ttnn.bfloat4_b
    d.shared_fidelity = ttnn.MathFidelity.LoFi
    return d


def _apply_core_cap(mesh) -> None:
    """FIX.15 (MAX_WORKER_CORES): the block's programs inside tt/cores.max_grid (once a process, before any plan)."""
    g = max_grid(mesh)
    _mb().set_max_grid(*((int(g.x), int(g.y)) if g is not None else (0, 0)))


def _packing(lay: dict) -> tuple:
    return tuple((k, repr(lay.get(k))) for k in _ROUTED_PACKING)


class _ForkMoEBlock:
    """models.demos.moe_block.tt.moe_block.MoEBlock built on the fork's ops (_fork_ops)."""

    def __new__(cls, *args, **kwargs):
        from models.demos.moe_block.tt.moe_block import MoEBlock

        with _fork_ops():
            return MoEBlock(*args, **kwargs)


class TtMoEBlockV2Layer:
    """One layer's MoE on the shared MoE block: __call__(ffn_norm x) -> the MoE output."""

    def __init__(self, mesh, ck, cfg, layer: int, chunk: int, cache_dir=None, own_rows: bool = False):
        from .moe_block_adapter import expert_permutation

        assert settings.get("MOE_HEAD") == "fused", "MOE_BLOCK_SRC=block: the block's head is fused (MOE_HEAD=fused)"
        assert settings.get("EXPERTS_PRECISION") == "glm", (
            "MOE_IMPL=block runs EXPERTS_PRECISION=glm only (owner decision D8, 2026-10-27): set "
            "DS41_EXPERTS_PRECISION=glm or DS41_MOE_IMPL=flat"
        )
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
        self.head_w = None  # (the head's weights: at the layer's first call)
        logger.info(
            f"MOE_BLOCK_V2 layer={layer} experts={cfg.n_experts} ({self.epc} a chip) chunk={chunk} "
            f"experts_impl={settings.get('EXPERTS_IMPL')} own_rows={self.own_rows} cap={_mb().max_grid()} "
            f"routed weights {time.time() - t0:.1f} s"
        )

    # ------------------------------------------------------------------ the block (one a mesh and chunk)
    def block(self):
        key = (id(self.mesh), self.chunk, x_min_piece_bytes())
        hit = _BLOCKS.get(key)
        if hit is None or hit[0] is not self.mesh:
            mb = _mb()
            cfg = mb.MoEBlockConfig(chunk_tokens=self.chunk, num_links=LINKS, persist_l1_small=True)
            cfg.combine_snap = COMBINE_SNAP
            t0 = time.time()
            res = mb.MoEBlockResources(self.mesh, self.dims, cfg, self.dtypes)
            blk = _ForkMoEBlock(
                self.mesh,
                res,
                dtypes=self.dtypes,
                max_tokens_per_expert=max(self.chunk, 256),
                layout=mb.FlatLayoutOptions(weights_nk=True, x_min_piece_bytes=x_min_piece_bytes()),
            )
            logger.info(
                f"MOE_BLOCK_V2 chunk={self.chunk}: built in {time.time() - t0:.1f} s, accumulators "
                f"{blk.accumulators}, {blk.splits} splits, head row blocks {blk.row_blocks}, pad rows {blk.pad_rows}, "
                f"arena {blk.arena_spec.memory_config.shard_spec.shape[0] * 64 >> 10} KB a core, fabric payload "
                f"{res.fabric_max_payload_bytes() if hasattr(res, 'fabric_max_payload_bytes') else '?'} B, "
                f"flat layout {blk.expert_kw['layout']}"
            )
            hit = _BLOCKS[key] = (self.mesh, res, blk)
        return hit[2]

    # ------------------------------------------------------------------ weights
    def _plan(self, blk) -> dict:
        """The flat routed-expert plan's layout facts (as_dict) the block's routed experts are laid out for."""
        kw = blk.expert_kw
        return (
            _mb()
            .flat_routed_expert_plan(
                self.mesh,
                int(self.cfg.dim),
                int(self.cfg.inter),
                self.epc,
                int(self.cfg.n_experts),
                kw["max_tokens_per_expert"],
                weights_bf8=self.dtypes.routed_weights == ttnn.bfloat8_b,
                pin=1,
                layout=kw["layout"],
            )
            .as_dict()
        )

    def _prefix(self):
        if self.cache_dir is None:
            return None
        from .experts import expert_cache_prefix

        return expert_cache_prefix(
            self.cache_dir, self.ck, self.layer, settings.get("EXPERTS_IMPL"), self.mesh, self.cfg.n_experts
        )

    def _cache_paths(self, lay):
        prefix = self._prefix()
        if prefix is None:
            return None
        key = hashlib.sha1(repr((sorted(lay.items()), "nk", str(self.dtypes.routed_weights))).encode()).hexdigest()[:12]
        return {n: Path(f"{prefix}.moe_blk.{key}.{n}.tensorbin") for n in ("w_gu", "w_d")}

    def _main_cache_paths(self, blk, lay):
        """The main block's (MOE_BLOCK_SRC=main) cache files of this layer when its flat plan lays the routed weights
        out as this block's does (the same facts, renamed - flat_weight_regions / bank_sharded are the same code but
        for the names and the main plan's slot sub-groups), else None. The main plan is the one
        moe_block_main_adapter._plan makes: the same cap, capacity, weights_nk and x pieces."""
        prefix = self._prefix()
        if prefix is None:
            return None
        main = ttnn.bringup.moe_block_main
        g = max_grid(self.mesh)
        main.set_max_grid(*((int(g.x), int(g.y)) if g is not None else (0, 0)))
        old = main.flat_routed_expert_plan(
            self.mesh,
            int(self.cfg.dim),
            int(self.cfg.inter),
            self.epc,
            int(self.cfg.n_experts),
            blk.expert_kw["max_tokens_per_expert"],
            weights_bf8=self.dtypes.routed_weights == ttnn.bfloat8_b,
            pin=1,
            layout=main.FlatLayoutOptions(weights_nk=True, x_min_piece_bytes=x_min_piece_bytes()),
        ).as_dict()
        rk = (_packing(lay), tuple(sorted((k, repr(v)) for k, v in old.items())))
        same = _REUSE.get(rk)
        if same is None:
            same = all(repr(lay.get(n)) == repr(old.get(o)) for n, o in _ROUTED_PACKING.items()) and int(
                old.get("n_rd_sg", 0)
            ) >= int(old.get("n_rd", 1))
            _REUSE[rk] = same
            logger.info(
                f"MOE_BLOCK_V2: the main block's routed weight layout is {'the same' if same else 'another'} "
                f"({ {n: lay.get(n) for n in _ROUTED_PACKING} } vs { {o: old.get(o) for o in _ROUTED_PACKING.values()} }, "
                f"n_rd_sg {old.get('n_rd_sg')})"
            )
        if not same:
            return None
        key = hashlib.sha1(repr((sorted(old.items()), "nk", str(self.dtypes.routed_weights))).encode()).hexdigest()[:12]
        return {n: Path(f"{prefix}.moe_main.{key}.{n}.tensorbin") for n in ("w_gu", "w_d")}

    def _routed(self, blk):
        """(gidx, w_gu, w_d): the routed experts in the flat plan's bank layout ([N, K] tiles), from the weight cache
        (this block's, else the main block's when laid out the same) or the checkpoint."""
        fw = _flat_weights_module()
        rows, cols = tuple(self.mesh.shape)
        n = rows * cols
        lay = self.lay = self._plan(blk)
        gids = torch.arange(int(self.cfg.n_experts), dtype=torch.int32).reshape(n, 1, self.epc)
        gidx = ttnn.from_torch(
            gids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        paths = self._cache_paths(lay)
        if paths is not None:
            for label, cand in (("", paths), (" (the main block's files)", self._main_cache_paths(blk, lay))):
                if cand is not None and all(p.exists() for p in cand.values()):
                    t0 = time.time()
                    w = {k: ttnn.load_tensor(str(p), device=self.mesh) for k, p in cand.items()}
                    logger.info(
                        f"MOE_BLOCK_V2 layer={self.layer} routed weights cache hit{label} {time.time() - t0:.1f} s"
                    )
                    return gidx, w["w_gu"], w["w_d"]
        t0 = time.time()
        flat_w, stats = _host_routed(self.ck, self.layer, self.mesh, self.cfg)
        t1 = time.time()
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:  # (torch's copies release the GIL: a chip's experts a task)
            packed = list(pool.map(lambda W_l: fw.flat_weight_regions(W_l, lay, nk=True), flat_w))
        del flat_w
        t2 = time.time()
        dtype, banks = self.dtypes.routed_weights, lay["dram_banks"]
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
            f"MOE_BLOCK_V2 layer={self.layer} routed weights: checkpoint {t1 - t0:.1f} s, pack {t2 - t1:.1f} s, "
            f"upload {t3 - t2:.1f} s, cache write {time.time() - t3:.1f} s ({'cached' if paths else 'no cache'})"
        )
        return gidx, w_gu, w_d

    def _head_weights(self, blk):
        if self.head_w is None:
            p = f"layers.{self.layer}.ffn."
            w = self.ck.get(p + "gate.weight")
            assert w.dtype == torch.bfloat16, f"router weight stored as {w.dtype}, expected bf16"
            b = self.ck.weight(p + "gate.bias", torch.float32)
            s = {k: self.ck.weight(f"{p}shared_experts.{k}.weight", torch.float32) for k in ("w1", "w3", "w2")}
            host = lambda t: ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32)  # noqa: E731
            self.head_w = _mb().prepare_moe_head_weights(
                host(s["w1"]),
                host(s["w3"]),
                host(s["w2"]),
                None,
                host(w[self.perm]),
                host(b[self.perm]),
                resources=blk.resources,
                dtypes=blk.dtypes,
                accumulators=blk.accumulators,
            )
        from models.demos.moe_block.tt.moe_block import MoEBlockWeights

        return MoEBlockWeights(head=self.head_w, expert_ids=self.gidx, gate_up=self.w_gu, down=self.w_d)

    # ------------------------------------------------------------------ the model's interface (TtMoEBlockLayer's)
    def input_buffers(self):
        """NORM_INTO_MOE=on (P.21): (x [1, 1, S/SP, H/TP] bf16, None) - a persistent buffer of the (mesh, chunk), pinned:
        ffn_norm writes x there. No xq: the new head reads bf16 x. Not with OWN_ROWS."""
        if self.own_rows:
            return None, None
        key = (id(self.mesh), self.chunk)
        bufs = _INPUTS.get(key)
        if bufs is None or bufs[0] is not self.mesh:
            rows, cols = tuple(self.mesh.shape)
            shape = [1, 1, self.chunk // rows, int(self.cfg.dim) // cols]
            x = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            if id(x) not in ccl._KEEP:
                ccl.pin(self.mesh, x)
            bufs = _INPUTS[key] = (self.mesh, x)
        return bufs[1], None

    def _one_call(self, blk) -> bool:
        """The OWN_ROWS path's own forward: the chunk in one call (no splits, the head in one row block - its owner
        rows then in plain order - and no padding rows)."""
        return blk.splits == 1 and blk.row_blocks == 1 and not blk.pad_rows

    def can_feed(self) -> bool:
        """FFN_FEED (P.36): whether this layer's head input can come from the attn_residual pass (feed_buffer)."""
        hit = getattr(self, "_can_feed", None)
        if hit is None:
            hit = self._can_feed = bool(self.own_rows and self._one_call(self.block()))
        return hit

    def feed_buffer(self) -> ttnn.Tensor:
        """FFN_FEED (P.36): the head's persistent bf16 input [1, 1, S/SP, H/TP], one a (mesh, chunk), pinned."""
        key = (id(self.mesh), self.chunk)
        hit = _FEED.get(key)
        if hit is None or hit[0] is not self.mesh:
            rows, cols = tuple(self.mesh.shape)
            shape = [1, 1, self.chunk // rows, int(self.cfg.dim) // cols]
            x = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            ccl.pin(self.mesh, x)
            hit = _FEED[key] = (self.mesh, x)
        return hit[1]

    def at_chunk(self, chunk: int) -> "TtMoEBlockV2Layer":
        """This layer at another chunk (DECODER_SKIP=replay128, P.31): the (mesh, chunk)'s own block, the routed
        weights shared when its flat plan lays them out as this one's, else laid out for it."""
        out = copy.copy(self)
        out.chunk = int(chunk)
        out.head_w = None
        out._can_feed = None
        blk = out.block()
        if _packing(self.lay) == _packing(out._plan(blk)):
            logger.info(f"MOE_BLOCK_V2 layer={self.layer} chunk={chunk}: routed weights shared with chunk {self.chunk}")
        else:
            logger.info(f"MOE_BLOCK_V2 layer={self.layer} chunk={chunk}: another flat plan - its own routed weights")
            out.gidx, out.w_gu, out.w_d = out._routed(blk)
        return out

    def __call__(self, x: ttnn.Tensor, xq_ready: bool = False, fed: bool = False) -> ttnn.Tensor:
        """x: ffn_norm's output (the column split [1, 1, S/SP, H/TP]; OWN_ROWS: [1, 1, S/(SP TP), H]) -> the MoE
        output, same rows. xq_ready: never (input_buffers gives no xq). fed (FFN_FEED=on, P.36): the attn_residual pass
        wrote the head's input into feed_buffer()."""
        assert x.layout == ttnn.TILE_LAYOUT
        assert not xq_ready, "MOE_BLOCK_SRC=block: the head takes no xq"
        blk = self.block()
        weights = self._head_weights(blk)
        xb = cast(x, ttnn.bfloat16)
        if not self.own_rows:
            assert not fed, "FFN_FEED: the feed is the OWN_ROWS path's"
            y = blk(xb, weights)
        else:
            y = self._own_rows(blk, xb, weights, fed=fed)
        if xb is not x:
            dealloc(xb)
        if not self.fold:
            y32 = ttnn.typecast(y, ttnn.float32, memory_config=DRAM)
            dealloc(y)
            y = y32
        return y

    def _own_rows(self, blk, x_own, weights, fed: bool = False):
        """OWN_ROWS: x [1, 1, S/(SP TP), H] -> one TP all-to-all into [1, 1, S/SP, H/TP] -> head, dispatch, experts +
        combine -> the combine's owner rows [S/(SP TP), H] (row-major) tilized + the shared expert's owner rows (the
        local tail: no all-to-all back). fed (FFN_FEED=on): no all-to-all, the head reads feed_buffer()."""
        mb = _mb()
        if not self._one_call(blk):  # (splits / row blocks / padding: the block's own forward and its tail)
            assert not fed, "FFN_FEED: can_feed() refuses the block's own forward"
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
        if fed:
            x = self.feed_buffer()
        else:
            x = ttnn.experimental.all_to_all_async_generic(
                x_own, in_dim=2, out_dim=3, num_links=2, memory_config=DRAM, cluster_axis=TP_AXIS
            )
        arena = ttnn.allocate_tensor_on_device(blk.arena_spec, self.mesh)
        head = mb.moe_head(
            x, weights.head, resources=blk.resources, dtypes=blk.dtypes, accumulators=blk.accumulators, arena=arena
        )
        if not fed:  # (the feed buffer is persistent)
            dealloc(x)
        dispatched = mb.sparse_dispatch_2d(head.dispatch_buffer, head.route_words, blk.resources, arena=arena)
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
        se = head.shared_expert_output
        assert tuple(y_own.shape)[-2:] == tuple(x_own.shape)[-2:] and tuple(se.shape)[-2:] == tuple(x_own.shape)[-2:], (
            f"MOE_BLOCK_SRC=block OWN_ROWS: owner rows y {tuple(y_own.shape)}, shared {tuple(se.shape)}, x "
            f"{tuple(x_own.shape)}"
        )
        yt = ttnn.to_layout(y_own, ttnn.TILE_LAYOUT, memory_config=DRAM)  # (y_own, se: the block's, kept)
        if len(yt.shape) != len(x_own.shape):
            yt = ttnn.reshape(yt, x_own.shape)
        if len(se.shape) != len(x_own.shape):
            se = ttnn.reshape(se, x_own.shape)
        ob = _lean_out(self.mesh, yt, se)  # (LEAN_PY=on, P.66: the load-time MoE output buffer)
        if ob is not None:
            ttnn.add(yt, se, memory_config=DRAM, output_tensor=ob, sub_core_grids=sub_cores(yt))  # (FIX.36: rule 18)
            dealloc(yt)
            return ob
        out = ttnn.add(yt, se, memory_config=DRAM, sub_core_grids=sub_cores(yt))
        dealloc(yt)
        return out


def build_moe_block_v2_layer(mesh, ck, cfg, layer: int, chunk: int, cache_dir=None, own_rows: bool = False):
    return TtMoEBlockV2Layer(mesh, ck, cfg, layer, chunk, cache_dir=cache_dir, own_rows=own_rows)
