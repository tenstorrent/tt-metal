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
- D5 / FIX.9: the shared expert at LoFi with GEMM_INPUTS' activations (bfloat8_b x / h tiles inside the fused head:
  MoEDtypes.shared_activations, the head's xq; P.37); SE_PACK_PRECISE: its bfp8 h packed in one rounding
  (MoEDtypes.shared_pack_precise, P.37); the combine snapped (COMBINE_SNAP: bit-identical runs); FIX.15: the block's
  programs inside tt/cores.max_grid (set_max_grid: 11 x 10 on a Galaxy);
- NORM_INTO_MOE (P.21, P.37): input_buffers() gives a persistent x and the head's bfloat8_b xq; ffn_norm writes both
  (rms_norm bfp8_twin) and the head reads xq as written (no typecast);
- DECODER_SKIP=replay128 (P.31, P.37): at_chunk - the layer at the decoder tail's chunk on that chunk's own block, the
  routed weights shared when its flat plan lays them out the same.
Not ported (owner, P.37 out of scope): the carried block's TAIL_A2A_PREPARED / PREPARED_ALL / HOST_ALLOC hooks (the C++
ops patch addresses on their own cache hits), mixed expert formats (EXPERTS_IMPL=bfp8_down4).

MOE_REPLICAS=runtime (P.46 / P.47, owner 2026-11-12): every layer gets one more slot a chip (spc = epc + 1; the
block's num_experts = chips x spc, its router ranks the checkpoint's experts: RouterSpec.candidates). Slots 0 .. epc - 1
of a chip are the off path's placement (router candidate b = block slot (b // epc) spc + b % epc). Load: the off path's
weight cache (the same files, a cache hit) into a transient epc-slot tensor, re-laid into the spc-slot tensor on the
device (expert_slot_copy, local jobs; tt/expert_slots.py run tables); the spare slot (slot epc) starts empty and
unmapped. set_replicas(copies) fills spare slots at runtime: one expert_slot_copy (a chip-to-chip fabric copy from the
expert's home slot, at most one a destination) for the destinations that changed, then the layer's slot map written
into the router bias (prepare_fused_head_weights slot_map: one extra tile; router_rows.hpp slot_of), in program order
on CQ0 - a slot is mapped only after its copy completed, and an unfilled slot is never mapped (holder 0 = the own
slot), so the empty map is the off path bit for bit.

Weights: per layer, the head's (router + shared expert: prepare_fused_head_weights) from the checkpoint at the
layer's first call; the routed experts laid out at load (host pack, ~1 min a layer) and kept in the weight cache
(<cache_dir>/<rows>x<cols>/L<layer>.<impl>.<fingerprint>.moe_main.<plan>.{w_gu,w_d}.tensorbin) when one is given.
"""

from __future__ import annotations

import copy
import hashlib
import json
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
COMBINE_SNAP = 2.0**-19  # (as the carried block, owner 2026-10-27: order-independent fp32 sums; COMBINE_SPLIT: > 32)
_BLOCKS: dict = {}  # (id(mesh), chunk) -> (mesh, MoEBlockResources, MoEBlock)
WDTYPES = {"bfp4": "bf4", "bfp4_mse": "bf4", "bfp8": "bf8"}  # EXPERTS_IMPL -> the flat op's one weight format


def _mb():
    return ttnn.bringup.moe_block_main


def block_dims(cfg, slots: int | None = None):
    """(MoEModelDims, flat activation name) of DeepSeek-V4.1-Flash at this checkpoint: sqrt(softplus) scores, top-k over
    all experts x route_scale, clamped SwiGLU (limit swiglu_limit) for the routed and the shared expert. slots
    (MOE_REPLICAS=runtime): the block's expert slots (chips x slots a chip, > n_experts) - the router still ranks the
    checkpoint's n_experts (RouterSpec.candidates), the head maps them to slots."""
    mb = _mb()
    assert float(cfg.swiglu_limit) == 10.0, f"the flat op's clamped_silu bakes limit 10, got {cfg.swiglu_limit}"
    act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=float(cfg.swiglu_limit))
    replicas = slots is not None and int(slots) != int(cfg.n_experts)
    router = mb.RouterSpec(score_func=mb.RouterScoreFunc.SqrtSoftplus, route_scale=float(cfg.route_scale))
    if replicas:
        router.candidates = int(cfg.n_experts)
    dims = mb.MoEModelDims(
        num_experts=int(slots) if replicas else int(cfg.n_experts),
        top_k=int(cfg.topk_experts),
        emb_dim=int(cfg.dim),
        routed_expert_hidden_dim=int(cfg.inter),
        routed_expert_activation=act,
        router=router,
        shared_expert_hidden_dim=int(cfg.inter),
        shared_expert_activation=act,
    )
    return dims, "clamped_silu"


def block_dtypes():
    """MoEDtypes: routed weights in EXPERTS_IMPL's format (one for gate / up and down: bfp8_down4 has no equivalent),
    the shared expert at LoFi (D5) with GEMM_INPUTS' activations (FIX.9, DS41-132: bfloat8_b x / h tiles in the fused
    head - its engine, x gather and dispatch prep; the router keeps bf16 x), router bf16. SE_PACK_PRECISE (DS41-137):
    the head packs its bfp8 h in one rounding from fp32 DEST (bfp8_pack_precise)."""
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
    d.shared_pack_precise = settings.get("SE_PACK_PRECISE") == "on"
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


# ---------------------------------------------------------------------------------------------------------------------
# MOE_COUNTS (P.45, owner 2026-11-11: the expert replication study's per-chunk expert loads)
_COUNTS: dict = {}  # id(mesh) -> (mesh, {layer: layer object}): the layers that ran since the last flush_counts


def counts_on() -> bool:
    return settings.get("MOE_COUNTS") == "on"


def _record_counts(layer_obj, counts: ttnn.Tensor) -> None:
    """MOE_COUNTS=on: the head's counts copied into the layer's own buffer (allocated at load; one single-page copy on
    the device: no allocation, no sync, no readback). A layer without a buffer (the decoder tail's copies, at_chunk)
    records nothing."""
    buf = layer_obj.counts_buf
    if buf is None:
        return
    ttnn.copy(counts, buf)
    _COUNTS.setdefault(id(layer_obj.mesh), (layer_obj.mesh, {}))[1][layer_obj.layer] = layer_obj


def flush_counts(mesh, start: int, chunk: int) -> dict | None:
    """MOE_COUNTS=on, after a model sync: every layer that ran since the last flush, read back at once (the buffers
    concatenated on the device, one readback: [chips x layers, slots]) -> {"chunk", "start", "slots_per_chip",
    "layers": {"<L>": {"chips": [chips][slots], "perm": [n_experts]}}}; with MOE_COUNTS_DIR set also written to
    <dir>/<MOE_COUNTS_TAG>_chunk<NN>.json (NN = start // chunk). None when no layer ran."""
    hit = _COUNTS.pop(id(mesh), None)
    if hit is None or hit[0] is not mesh or not hit[1]:
        return None
    lays = [hit[1][k] for k in sorted(hit[1])]
    bufs = [lay.counts_buf for lay in lays]
    cat = ttnn.concat(bufs, dim=0, memory_config=DRAM) if len(bufs) > 1 else bufs[0]
    a = ttnn.to_torch(cat, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    if cat is not bufs[0]:
        ttnn.deallocate(cat)
    n = mesh.get_num_devices()
    a = a.reshape(n, len(lays), -1).to(torch.int64)  # (chip r C + c, layer, slot)
    rec = {
        "chunk": int(start) // int(chunk),
        "start": int(start),
        "slots_per_chip": int(lays[0].spc),
        "layers": {
            str(lay.layer): {"chips": a[:, j].tolist(), "perm": [int(v) for v in lay.perm.tolist()]}
            for j, lay in enumerate(lays)
        },
    }
    out_dir = settings.get("MOE_COUNTS_DIR")
    if out_dir:
        path = Path(out_dir) / f"{settings.get('MOE_COUNTS_TAG')}_chunk{rec['chunk']:02d}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, path)
        logger.info(f"MOE_COUNTS chunk {rec['chunk']} (start {start}): {len(lays)} layers -> {path}")
    return rec


# ---------------------------------------------------------------------------------------------------------------------
# MOE_REPLICAS (P.46 / P.47)
_SLOT_TABLES: dict = {}  # (id(mesh), plan layouts, epc, tile bytes) -> (mesh, expert_slots.SlotTables)


def replicas_on() -> bool:
    return settings.get("MOE_REPLICAS") == "runtime"


def slot_map_words(perm, epc: int, spc: int, copies) -> torch.Tensor:
    """The fused head's slot map (router_rows.hpp slot_of; prepare_fused_head_weights slot_map): 1024 integer-valued
    float32 words - [0, E) router candidate b's own slot (b // epc) spc + b % epc, [E, 2 E) its copies n | first << 6,
    then the copies' slots (chip k's spare slot k spc + epc), each candidate's by chip. perm[b]: the checkpoint expert of
    candidate b; copies: [(checkpoint expert, chip)]."""
    E = len(perm)
    where = torch.empty(E, dtype=torch.long)
    where[perm] = torch.arange(E)
    by_cand: dict = {}
    chips = set()
    for x, k in copies:
        b = int(where[x])
        assert k != b // epc, f"replica table: expert {x}'s copy on its own chip {k}"
        assert k not in chips, f"replica table: chip {k} holds two copies"
        assert 0 <= k < E // epc, f"replica table: chip {k} out of range"
        chips.add(k)
        by_cand.setdefault(b, []).append(k)
    w = torch.zeros(1024, dtype=torch.float32)
    b_ = torch.arange(E)
    w[:E] = ((b_ // epc) * spc + b_ % epc).float()
    first = 0
    for b in sorted(by_cand):
        ks = sorted(by_cand[b])
        w[E + b] = float(len(ks) | (first << 6))
        for i, k in enumerate(ks):
            w[2 * E + first + i] = float(k * spc + epc)
        first += len(ks)
    assert 2 * E + first <= 1024, f"slot map: {first} copies past the tile"
    assert all(len(v) < 64 for v in by_cand.values()), "slot map: an expert with 64 copies or more"
    return w


def holder_of(key: int, n: int) -> int:
    """router_rows.hpp slot_of's holder of a token (key: chip << 16 | its owned row) among n + 1 slots."""
    return ((((key * 2654435761) & 0xFFFFFFFF) >> 16) & 0xFFFF) * (n + 1) >> 16


class TtMoEBlockMainLayer:
    """One layer's MoE on the shared reworked MoE block: __call__(ffn_norm x) -> the MoE output."""

    def __init__(self, mesh, ck, cfg, layer: int, chunk: int, cache_dir=None, own_rows: bool = False):
        from .moe_block_adapter import expert_permutation

        assert settings.get("MOE_HEAD") == "fused", "MOE_BLOCK_SRC=main: the block's head is fused (MOE_HEAD=fused)"
        # (D8, as the carried block: the flat op at LoFi with bfp8 x / h tiles; full is not plumbed)
        assert settings.get("EXPERTS_PRECISION") == "glm", (
            "MOE_IMPL=block runs EXPERTS_PRECISION=glm only (owner decision D8, 2026-10-27): set "
            "DS41_EXPERTS_PRECISION=glm or DS41_MOE_IMPL=flat"
        )
        self.mesh, self.ck, self.cfg = mesh, ck, cfg
        self.layer, self.chunk = int(layer), int(chunk)
        self.own_rows = bool(own_rows)
        self.fold = ccl.glue_fold()
        self.cache_dir = cache_dir
        rows, cols = tuple(mesh.shape)
        self.epc = int(cfg.n_experts) // (rows * cols)
        # MOE_REPLICAS=runtime (P.47): spc = epc + 1 slots a chip, the last one empty until set_replicas fills it
        self.replicas = replicas_on()
        self.spc = self.epc + (1 if self.replicas else 0)
        self.copies = []  # (checkpoint expert, chip r C + c): the spare slots filled and mapped
        self._views = []  # (at_chunk's copies of this layer: they share the copies, set_replicas updates them too)
        self._bias = None  # (the router bias in candidate order, for the slot map's writes)
        self.dims, self.act = block_dims(cfg, slots=rows * cols * self.spc)
        self.dtypes = block_dtypes()
        self.perm = expert_permutation(rows, cols, self.epc)
        self._last_head = None  # (the last fused_head call's outputs, OWN_ROWS path: tests read its counts / rows)
        # MOE_COUNTS=on (P.45): the head's counts [1, 1, 1, slots] uint32 a chip, copied here every call (load-time)
        self.counts_buf = (
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, rows * cols * self.spc]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
            )
            if counts_on()
            else None
        )
        _apply_core_cap(mesh)
        t0 = time.time()
        blk = self.block()
        self.gidx, self.w_gu, self.w_d = self._routed(blk)
        self.head_w = None  # (the head's weights: at the layer's first call, planned by the block's head)
        logger.info(
            f"MOE_BLOCK_MAIN layer={layer} experts={cfg.n_experts} ({self.epc} a chip"
            + (f", {self.spc} slots: {len(self.copies)} copies" if self.replicas else "")
            + f") chunk={chunk} "
            f"experts_impl={settings.get('EXPERTS_IMPL')} own_rows={self.own_rows} cap={_mb().max_grid()} "
            f"routed weights {time.time() - t0:.1f} s"
        )

    # ------------------------------------------------------------------ the block (one a mesh and chunk)
    def block(self):
        from models.demos.moe_block_main.tt.moe_block import MoEBlock

        split = settings.get("COMBINE_SPLIT") == "on"
        key = (id(self.mesh), self.chunk, split, self.spc)
        hit = _BLOCKS.get(key)
        if hit is None or hit[0] is not self.mesh:
            mb = _mb()
            cfg = mb.MoEBlockConfig(
                chunk_tokens=self.chunk,
                num_links=LINKS,
                combine_snap=COMBINE_SNAP,
                # (FIX.23, DS41-235: order-free sums beyond 32 - the routed sums reach ~2^14 here)
                combine_snap_split=split,
                persist_l1_small=True,
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
    def _plan(self, blk, slots: int | None = None) -> dict:
        """The block's flat routed-expert plan's layout facts (as_dict); slots: a chip's (default the block's, spc; the
        index-order cache's: epc)."""
        kw = blk.expert_kw
        slots = self.spc if slots is None else int(slots)
        layout = kw["layout"]
        return (
            _mb()
            .flat_routed_expert_plan(
                self.mesh,
                int(self.cfg.dim),
                int(self.cfg.inter),
                slots,
                slots * self.mesh.get_num_devices(),
                kw["max_tokens_per_expert"],
                weights_bf8=self.dtypes.routed_weights == ttnn.bfloat8_b,
                pin=1,
                layout=layout,
            )
            .as_dict()
        )

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
        the checkpoint (MOE_REPLICAS=runtime: the spc-slot layout re-laid on the device from the epc-slot one,
        _spare_slot)."""
        rows, cols = tuple(self.mesh.shape)
        n = rows * cols
        lay = self.lay = self._plan(blk)
        gids = torch.arange(n * self.spc, dtype=torch.int32).reshape(n, 1, self.spc)
        gidx = ttnn.from_torch(
            gids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        if self.replicas:
            w_gu, w_d = self._spare_slot(blk, lay)
        else:
            w_gu, w_d = self._routed_index(blk, lay)
        return gidx, w_gu, w_d

    def _routed_index(self, blk, lay):
        """(w_gu, w_d) of the index placement (epc slots a chip) in lay's bank layout: the weight cache, else the
        checkpoint (and then cached)."""
        fw = _flat_weights_module()
        paths = self._cache_paths(blk, lay)
        if paths is not None and all(p.exists() for p in paths.values()):
            t0 = time.time()
            w = {k: ttnn.load_tensor(str(p), device=self.mesh) for k, p in paths.items()}
            logger.info(f"MOE_BLOCK_MAIN layer={self.layer} routed weights cache hit {time.time() - t0:.1f} s")
            return w["w_gu"], w["w_d"]
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
        return w_gu, w_d

    def slot_tables(self, blk=None):
        """The block's copy run tables (tt/expert_slots.SlotTables: fills of the spare slot from each home slot, the
        load-time regroup), one a (mesh, weight layouts); built and checked by a debug readback at first use."""
        from .expert_slots import SlotTables, tile_bytes

        blk = blk or self.block()
        lay_i = self.lay_index = getattr(self, "lay_index", None) or self._plan(blk, slots=self.epc)
        key = (
            id(self.mesh),
            _weight_layout(self.lay),
            _weight_layout(lay_i),
            self.epc,
            str(self.dtypes.routed_weights),
        )
        hit = _SLOT_TABLES.get(key)
        if hit is None or hit[0] is not self.mesh:
            t0 = time.time()
            tb = tile_bytes(self.dtypes.routed_weights)
            tabs = SlotTables(self.lay, lay_i, self.epc, int(self.cfg.dim), int(self.cfg.inter), tb, self.mesh)
            tabs.verify()
            hit = _SLOT_TABLES[key] = (self.mesh, tabs)
            logger.info(
                f"MOE_BLOCK_MAIN MOE_REPLICAS=runtime: slot copy tables in {time.time() - t0:.2f} s - a slot "
                f"{tabs.lay_spc.slot_bytes() / 2**20:.2f} MiB in {len(tabs.host[0])} runs, regroup {len(tabs.regroup)} runs"
            )
        return hit[1]

    def _spare_slot(self, blk, lay):
        """MOE_REPLICAS=runtime: (w_gu, w_d) in the spc-slot layout lay - every chip's epc home slots as the index
        placement (the off path's weight cache, a cache hit, loaded into a transient epc-slot tensor and re-laid on the
        device by expert_slot_copy's local jobs), the spare slot (slot epc) left empty: nothing maps to it until
        set_replicas fills it (the flat expert skips a slot with no rows, weights included)."""
        mb = _mb()
        fw = _flat_weights_module()
        n = self.mesh.get_num_devices()
        lay_i = self.lay_index = self._plan(blk, slots=self.epc)
        w_i = self._routed_index(blk, lay_i)
        t0 = time.time()
        tabs = self.slot_tables(blk)
        out = []
        g = tabs.lay_spc  # (the spc-slot geometry: rows of each bank shard)
        banks = int(lay["banks"])
        rows_of = {"w_gu": -(-g.n_rd // banks) * g.E * g.gu * 32, "w_d": g.nd * g.sub * 32}
        for name, w in zip(("w_gu", "w_d"), w_i):
            rows = rows_of[name]
            shape = list(w.shape)
            shape[-2] = rows
            out.append(
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape(shape),
                    self.dtypes.routed_weights,
                    ttnn.TILE_LAYOUT,
                    self.mesh,
                    fw._bank_memory_config(rows, lay["banks"]),
                )
            )
        mb.expert_slot_copy(
            w_i[0], w_i[1], out[0], out[1], tabs.runs, blk.resources, [(c, c, tabs.regroup_table) for c in range(n)]
        )
        for w in w_i:
            ttnn.deallocate(w)
        logger.info(
            f"MOE_BLOCK_MAIN layer={self.layer} MOE_REPLICAS=runtime: {self.spc} slots a chip, the spare one empty; "
            f"re-laid on the device in {time.time() - t0:.2f} s (enqueue)"
        )
        return out[0], out[1]

    # ------------------------------------------------------------------ MOE_REPLICAS=runtime: the public API
    def set_replicas(self, copies) -> int:
        """The layer's spare slots from now on: copies [(checkpoint expert, chip r C + c)], at most one a chip, never on
        the expert's own chip. Enqueued on CQ0 in program order, between this layer's MoE before and after the call:
        one expert_slot_copy for every destination whose expert changed (source: the expert's home slot on its home
        chip; the destination's program waits until it is done reading its slot), then the slot map written into the
        router bias (after the copy: a slot is mapped only once filled). A chip left out is unmapped (its slot keeps
        its bytes, unused). Returns the copies made (0: the table did not change, nothing enqueued)."""
        assert self.replicas, "set_replicas needs MOE_REPLICAS=runtime"
        new = {}
        E = len(self.perm)
        where = torch.empty(E, dtype=torch.long)
        where[self.perm] = torch.arange(E)
        n = self.mesh.get_num_devices()
        for x, k in copies:
            x, k = int(x), int(k)
            assert 0 <= x < E and 0 <= k < n, f"replica ({x}, {k}) out of range"
            assert k not in new, f"chip {k} gets two copies in one layer"
            assert int(where[x]) // self.epc != k, f"expert {x}'s copy on its own chip {k}"
            new[k] = x
        old = {k: x for x, k in self.copies}
        if new == old:
            return 0
        jobs = []
        tabs = self.slot_tables()
        for k in sorted(new):
            if old.get(k) != new[k]:
                home, slot = divmod(int(where[new[k]]), self.epc)
                jobs.append((home, k, tabs.fill_table(slot)))
        members = [self] + list(self._views)
        if jobs:
            done = set()
            for m in members:  # (a view with its own weights: the same copies into them)
                if id(m.w_gu) in done:
                    continue
                done.add(id(m.w_gu))
                mb = _mb()
                mt = m.slot_tables()
                mb.expert_slot_copy(m.w_gu, m.w_d, m.w_gu, m.w_d, mt.runs, m.block().resources, jobs)
        self.copies = sorted((x, k) for k, x in new.items())
        for m in members:
            m.copies = self.copies
            if m.head_w is not None:
                m._write_slot_map()
        return len(jobs)

    def _write_slot_map(self) -> None:
        """The router bias rewritten with the current slot map (a host write on CQ0: after every program before it)."""
        E = len(self.perm)
        host = torch.zeros(E + 32, 32, dtype=torch.float32)
        host[:E] = self._bias.reshape(E, 1)
        host[E:] = slot_map_words(self.perm, self.epc, self.spc, self.copies).reshape(32, 32)
        t = ttnn.from_torch(
            host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(t, self.head_w.router_bias)

    def _head_weights(self, blk):
        if self.head_w is None:
            mb = _mb()
            p = f"layers.{self.layer}.ffn."
            w = self.ck.get(p + "gate.weight")
            assert w.dtype == torch.bfloat16, f"router weight stored as {w.dtype}, expected bf16"
            b = self.ck.weight(p + "gate.bias", torch.float32)
            self._bias = b[self.perm].float().contiguous()
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
                **(
                    {"slot_map": host(slot_map_words(self.perm, self.epc, self.spc, self.copies))}
                    if self.replicas
                    else {}
                ),
            )
        from models.demos.moe_block_main.tt.moe_block import MoEBlockWeights

        return MoEBlockWeights(head=self.head_w, expert_ids=self.gidx, gate_up=self.w_gu, down=self.w_d)

    # ------------------------------------------------------------------ the model's interface (TtMoEBlockLayer's)
    def input_buffers(self):
        """NORM_INTO_MOE=on (P.21): (x [1, 1, S/SP, H/TP] bf16, the head's bfloat8_b xq or None) - persistent
        buffers of the (mesh, chunk)'s block, pinned (tt/ccl.dealloc leaves them alone): ffn_norm writes x there and,
        with GEMM_INPUTS=bfp8, xq from it in the same program (rms_norm bfp8_twin: typecast(x, bfp8) bit for bit);
        __call__(x, xq_ready=True) hands xq to the head (no typecast). The block's ops read x from any buffer, so x is
        only the norm's output kept at one address. Not with OWN_ROWS (the head's x comes from a TP all-to-all)."""
        if self.own_rows:
            return None, None
        blk = self.block()
        key = (id(self.mesh), self.chunk)
        bufs = _INPUTS.get(key)
        if bufs is None or bufs[0] is not self.mesh:
            rows, cols = tuple(self.mesh.shape)
            shape = [1, 1, self.chunk // rows, int(self.cfg.dim) // cols]
            x = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            xq = blk.xq_buffer(shape) if blk.bf8_x else None
            for t in (x, xq):
                if t is not None and id(t) not in ccl._KEEP:
                    ccl.pin(self.mesh, t)
            bufs = _INPUTS[key] = (self.mesh, x, xq)
        return bufs[1], bufs[2]

    def can_feed(self) -> bool:
        """FFN_FEED (P.36): whether this layer's head input can come from the attn_residual pass (feed_buffer): the
        OWN_ROWS path with its one TP all-to-all into the head (no token loops / splits / padded rows)."""
        hit = getattr(self, "_can_feed", None)
        if hit is None:
            blk = self.block()
            hit = self._can_feed = bool(self.own_rows and blk.loops * blk.splits == 1 and not blk.pad_rows)
        return hit

    def feed_buffer(self) -> ttnn.Tensor:
        """FFN_FEED (P.36): the head's persistent bf16 input [1, 1, S/SP, H/TP] (the column split the TP all-to-all
        wrote), one a (mesh, chunk), pinned: the same address on every chip, written by mhc_post(feed_output=)."""
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

    def at_chunk(self, chunk: int) -> "TtMoEBlockMainLayer":
        """This layer at another chunk (DECODER_SKIP=replay128, P.31: the decoder tail's 1024-row chunk on 8x4): the
        (mesh, chunk)'s own block (resources, plans, programs; built here), the routed experts' device weights shared
        when that block's flat plan lays them out as this one's (the plan's weight-layout facts equal), else laid out
        for it (the weight cache, else the checkpoint); the router / shared expert laid out by that block's head at its
        first call."""
        out = copy.copy(self)
        out.chunk = int(chunk)
        out.head_w = None
        out._can_feed = None  # (FFN_FEED: that block's own answer)
        out.counts_buf = None  # (MOE_COUNTS: the decoder tail's rows are not the chunk's; only the main layer records)
        out._views = []
        if self.replicas:  # (set_replicas on this layer updates the view's map, and its weights when it has its own)
            self._views.append(out)
        blk = out.block()
        if _weight_layout(self.lay) == _weight_layout(out._plan(blk)):
            logger.info(
                f"MOE_BLOCK_MAIN layer={self.layer} chunk={chunk}: routed weights shared with chunk {self.chunk}"
            )
        else:
            logger.info(f"MOE_BLOCK_MAIN layer={self.layer} chunk={chunk}: another flat plan - its own routed weights")
            out.lay_index = None
            out.gidx, out.w_gu, out.w_d = out._routed(blk)
            if self.replicas and self.copies:  # (its own spare slots: filled with the layer's current copies)
                E = len(self.perm)
                where = torch.empty(E, dtype=torch.long)
                where[self.perm] = torch.arange(E)
                jobs = [(int(where[x]) // self.epc, k, int(where[x]) % self.epc) for x, k in self.copies]
                _mb().expert_slot_copy(
                    out.w_gu, out.w_d, out.w_gu, out.w_d, out.slot_tables().runs, blk.resources, jobs
                )
        return out

    def __call__(self, x: ttnn.Tensor, xq_ready: bool = False, fed: bool = False) -> ttnn.Tensor:
        """x: ffn_norm's output (the column split [1, 1, S/SP, H/TP]; OWN_ROWS: [1, 1, S/(SP TP), H]) -> the MoE
        output, same rows. xq_ready (NORM_INTO_MOE=on): x is input_buffers()[0] and the norm wrote input_buffers()[1]
        (xq) from it. fed (FFN_FEED=on, P.36): the attn_residual pass wrote the head's input into feed_buffer()
        (x is then only its staging: its shape stands for the rows)."""
        assert x.layout == ttnn.TILE_LAYOUT
        blk = self.block()
        weights = self._head_weights(blk)
        xq = None
        if xq_ready:
            xb_, xq = self.input_buffers()
            assert xq is not None and int(x.buffer_address()) == int(
                xb_.buffer_address()
            ), "xq_ready needs x = input_buffers()[0] (the norm wrote xq from it)"
        xb = cast(x, ttnn.bfloat16)
        if not self.own_rows:
            assert not fed, "FFN_FEED: the feed is the OWN_ROWS path's"
            y = blk(xb, weights, xq=xq)
            if self.counts_buf is not None:
                assert blk.loops * blk.splits == 1, "MOE_COUNTS: a block with token loops / splits has a head a loop"
                _record_counts(self, blk.last_head.counts)
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
        combine -> the combine's owner rows [S/(SP TP), H] (row-major) tilized + the shared expert's owner rows.
        fed (FFN_FEED=on, P.36): no all-to-all, the head reads feed_buffer() (written by the attn_residual pass)."""
        mb = _mb()
        if blk.loops * blk.splits != 1 or blk.pad_rows:  # (token loops / splits / padding: the block's own forward)
            assert not fed, "FFN_FEED: can_feed() refuses the block's own forward"
            xc = ttnn.experimental.all_to_all_async_generic(
                x_own, in_dim=2, out_dim=3, num_links=2, memory_config=DRAM, cluster_axis=TP_AXIS
            )
            y = blk(xc, weights)
            if self.counts_buf is not None:
                assert blk.loops * blk.splits == 1, "MOE_COUNTS: a block with token loops / splits has a head a loop"
                _record_counts(self, blk.last_head.counts)
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
        if blk.exchange_spec is None:
            blk.one_arena = blk._one_arena_fits()
        xq = blk.to_xq(x) if blk.bf8_x else None  # (GEMM_INPUTS=bfp8: the shared expert's x, one typecast)
        x_full = (
            ttnn.all_gather(x if xq is None else xq, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if blk.x_dram
            else None
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
            **({"xq": xq} if xq is not None else {}),
        )
        self._last_head = head
        _record_counts(self, head.counts)
        if not fed:  # (the feed buffer is persistent)
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
_INPUTS: dict = {}  # (id(mesh), chunk) -> (mesh, x bf16, xq bfp8 or None): NORM_INTO_MOE's input buffers
_FEED: dict = {}  # (id(mesh), chunk) -> (mesh, x bf16): FFN_FEED's head input (P.36)
# the flat plan's facts the routed weights' layout depends on (flat_routed_expert_weights.flat_weight_regions /
# bank_sharded): two plans equal in these lay the same weights out the same
_WEIGHT_KEYS = ("banks", "col0s", "n_rd", "n_rd_sg", "nd", "nk_gu", "np", "nu", "pcds", "rg", "wd_rot_tiles")


def _weight_layout(lay: dict) -> tuple:
    return tuple((k, repr(lay.get(k))) for k in _WEIGHT_KEYS)


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
