# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's MoE layer on Marko Staletovic's MoE block (models/demos/moe_block), setting MOE_IMPL=block
(tt/MOE_BLOCK.md; owner decisions D1-D10, reports/moe_block_decisions_2026-10-27.md).

One call replaces the router, experts, shared_expert and moe_add steps:

    x  ffn_norm [1, 1, S/SP, H] bf16 TILE (rows over axis 0, replicated over axis 1: tt/norm.py gather=True)
    -> MoEBlock (MoEOptions x_gathered, ops head, ops tail, combine_deterministic) with this layer's MoELayer:
       router partial logits on the chip's H/TP columns, reduce-scatter to the token owners, sqrt(softplus) + bias,
       router_topk; shared expert (TP-split ttnn matmuls) + row reduce-scatter; owner dispatch, flat experts + combine
       in one program; + shared, TP all-to-all back
    -> routed + shared [1, 1, S/SP, H/TP] bf16 (fp32 under GLUE_FOLD=off): what moe_add returns on the flat path

MOE_HEAD=fused: x is ffn_norm's column split [1, 1, S/SP, H/TP] (the fused head gathers it on its ports), the head is
one program (x gather, shared expert engine + row reduce-scatter, router, routing exchange + planner, dispatch prep),
the combine snaps its terms to COMBINE_SNAP (the head's planner has no fixed order); the tail as above.

Expert numbering. The block holds expert b on chip b // epc in row-major chip order. Our routed weights (and their
weight cache) hold checkpoint expert gtab[c, r, le] in slot le of chip (r, c) (ExpertMapping column-major,
tt/experts.py). The block's experts are numbered by those slots, b = (r C + c) epc + le, and the router's rows (gate
weight and bias) are permuted to match: block expert b is checkpoint expert perm[b] = gtab[c, r, le]. Ids are only an
order; the math is unchanged and the weight cache keeps hitting.

One MoEBlock per (mesh, chunk) serves every layer (its L1 arenas and programs are shared; L1 allows only one); it is
built at the first call, after the model's load-time L1 is placed (the exchange bakes its arena address). Each layer
holds its own device weights (MoELayer) and passes them with its call.
"""

from __future__ import annotations

import copy
import dataclasses

import torch
from loguru import logger

import ttnn

from . import ccl, host_alloc
from .ccl import dealloc
from .common import DRAM, cast
from .cores import from_torch, max_grid, sub_cores_tiled
from .experts import TtExperts
from .settings import settings

LINKS = 2  # links per direction (the block's exchange plans its ports for 2)
COMBINE_SNAP = 2.0**-19  # (MOE_HEAD=fused: order-independent sums below 2^24 x 2^-19 = 32; routed sums peak at 5.4)
_BLOCKS: dict = {}  # (id(mesh), chunk) -> (mesh, MoEBlock)
_DONE: dict = {}  # id(mesh) -> (mesh, the done words every layer's flat op shares; each launch leaves them zero)


def block_model(cfg):
    """The block's DSV41_FLASH at this checkpoint's expert count (scaled 2x2: 96), checked against the config."""
    from models.demos.moe_block.moe_block import DSV41_FLASH

    m = dataclasses.replace(DSV41_FLASH, num_experts=int(cfg.n_experts))
    got = (cfg.topk_experts, cfg.dim, cfg.inter, cfg.inter, float(cfg.route_scale), float(cfg.swiglu_limit))
    want = (
        m.top_k,
        m.emb_dim,
        m.routed_expert_hidden_dim,
        m.shared_expert_hidden_dim,
        m.router.route_scale,
        m.routed_expert_activation.limit,
    )
    assert (
        got == want
    ), f"MOE_IMPL=block: config (top_k, H, I, I_shared, route_scale, limit) {got} != DSV41_FLASH {want}"
    assert m.router.score_func == "sqrtsoftplus" and m.shared_expert_activation.limit == cfg.swiglu_limit
    return m


def block_options(mesh=None):
    """max_grid (FIX.15, MAX_WORKER_CORES): the block's own programs inside tt/cores.max_grid(mesh) (11 x 10 on a
    Galaxy; None elsewhere: the whole grid, as before). MOE_HEAD=fused (the default, owner 2026-10-27): the fused head on the column split, the combine snapped (owner
    2026-10-27: keep the snap; bit-identical runs). ops: D2 x gathered, D4 ops head, D7 fixed combine order. MOE_TAIL:
    ops (a program a stage) or fused (TailA2A: tilize + shared + TP all-to-all in one program); TAIL_A2A_PREPARED: the
    fused tail's descriptors prepared once and dispatched with run_prepared."""
    from models.demos.moe_block.moe_block import MoEOptions

    tail = settings.get("MOE_TAIL")
    tp = settings.get("TAIL_A2A_PREPARED") == "on"  # (P.2: the fused tail's descriptors as prepared programs)
    g = max_grid(mesh) if mesh is not None else None
    cap = (g.x, g.y) if g is not None else None
    if settings.get("MOE_HEAD") == "fused":
        return MoEOptions(head="fused", tail=tail, tail_prepared=tp, combine_snap=COMBINE_SNAP, max_grid=cap)
    return MoEOptions(
        head="ops", tail=tail, tail_prepared=tp, x_gathered=True, combine_deterministic=True, max_grid=cap
    )


def block_dtypes():
    """D5: the shared expert at LoFi (bfp8 weights, the block's default); routed bfp4 at LoFi, router bf16. The shared
    expert's x / h: GEMM_INPUTS (FIX.9, DS41-132; owner 2026-10-28, F208) - bfloat8_b, the reference's fp8 act quant of
    both (reference/ds41_ref.py _swiglu: _q8(x), _q8(h)); the router keeps bf16 x (the fused head typecasts a copy for
    the engine). SE_PACK_PRECISE: the fused head packs bfp8 h in one rounding (bfp8_pack_precise)."""
    from models.demos.moe_block.moe_block import MoEDtypes

    from .common import gemm_dtype

    return MoEDtypes(
        shared_fidelity=ttnn.MathFidelity.LoFi,
        shared_activations=gemm_dtype(),
        shared_pack_precise=settings.get("SE_PACK_PRECISE") == "on",
    )


def expert_permutation(rows: int, cols: int, epc: int) -> torch.Tensor:
    """perm[b] = the checkpoint expert our weights hold where the block puts expert b (module docstring)."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

    gtab = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    return torch.tensor(
        [int(gtab[c, r, le]) for r in range(rows) for c in range(cols) for le in range(epc)], dtype=torch.long
    )


class _RoutedWeights(TtExperts):
    """TtExperts' weight load (EXPERTS_IMPL, the weight cache and its marker, weights_nk, EXPERTS_LAYOUT: the same
    FlatRoutedExpert arguments, so the same cache files) without its weighted combine."""

    def _load_flat(self, flat_w, gtab, cache_prefix=None):
        self.fe = self.flat_op(flat_w, cache_prefix)


def _block_op(mesh, fe, g, n_experts: int, layout: dict):
    """Our laid-out flat op as the block's: its capacity (RoutedExperts.cap_of; same packing, asserted by replanned),
    the block's expert ids (chip k: [k epc, (k + 1) epc)), the done words shared by every layer."""
    from models.demos.moe_block.moe_block import RoutedExperts

    cap = RoutedExperts.cap_of(g)
    op = copy.copy(fe) if fe.m == cap else fe.replanned(cap, layout)
    gids = RoutedExperts.gids_of(g)
    op.gidx = from_torch(
        torch.tensor(gids, dtype=torch.int32).reshape(len(gids), 1, g.experts_per_chip),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    dealloc(fe.gidx)  # (the flat path's column-local ids)
    op._n_global = n_experts
    shared = _DONE.get(id(mesh))
    if shared is None or shared[0] is not mesh:
        _DONE[id(mesh)] = (mesh, op.done)
    else:
        dealloc(op.done)
        op.done = shared[1]
    return op


class TtMoEBlockLayer:
    """One layer's MoE on the shared MoE block: __call__(ffn_norm x) -> the MoE output [1, 1, S/SP, H/TP]."""

    def __init__(self, mesh, ck, cfg, layer: int, chunk: int, cache_dir=None):
        from models.demos.moe_block.moe_block import Geometry, MoELayer, MoEWeights, RoutedExperts, fuse_gate_up

        # D8: the block's exchange runs the flat op at LoFi with bfp8 x / h tiles (glm); full is not plumbed
        assert settings.get("EXPERTS_PRECISION") == "glm", (
            "MOE_IMPL=block runs EXPERTS_PRECISION=glm only (owner decision D8, 2026-10-27): the block's exchange "
            "calls the flat op at LoFi with bfp8 x / h tiles; set DS41_EXPERTS_PRECISION=glm or DS41_MOE_IMPL=flat"
        )
        assert (
            settings.get("EXPERTS_IMPL") == "bfp4"
        ), "MOE_IMPL=block: bfp4 routed weights (MoEBlock asserts bfloat4_b)"
        rows, cols = tuple(mesh.shape)
        self.mesh, self.layer, self.chunk = mesh, int(layer), int(chunk)
        self.fold = ccl.glue_fold()
        host_alloc.reserve_tail_out(mesh, self.chunk, int(cfg.dim))  # HOST_ALLOC=on (P.15): at load time
        m = self.model = block_model(cfg)
        g = self.g = Geometry.of(mesh, m, self.chunk, LINKS, zigzag=False)
        # (lines or a torus: the block reads the wraps from the open fabric; torus untested on DS41, owner D14)
        self.dtypes = block_dtypes()
        epc = g.experts_per_chip
        # routed experts: our load and cache (weights_nk, bfp4) in DS41's EXPERTS_LAYOUT (D6: marko by default, the
        # block's own layout_of), as the block's op
        rw = _RoutedWeights(
            mesh,
            ck,
            layer,
            num_experts=cfg.n_experts,
            hidden=cfg.dim,
            inter=cfg.inter,
            top_k=cfg.topk_experts,
            swiglu_limit=cfg.swiglu_limit,
            max_chunk=self.chunk,
            cache_dir=cache_dir,
        )
        assert rw.epc == epc, (rw.epc, epc)
        self.layout = rw.flat_layout
        if settings.get("EXPERTS_LAYOUT") == "marko":
            assert self.layout == RoutedExperts.layout_of(g, m), (self.layout, RoutedExperts.layout_of(g, m))
        op = _block_op(mesh, rw.fe, g, m.num_experts, self.layout)
        self.bfp4_stats = rw.bfp4_stats
        perm = expert_permutation(rows, cols, epc)
        p = f"layers.{layer}.ffn."

        def host():  # (router and shared expert from the checkpoint; read when uploaded: at load, and at block build)
            w = ck.get(p + "gate.weight")
            assert w.dtype == torch.bfloat16, f"router weight stored as {w.dtype}, expected bf16"
            b = ck.weight(p + "gate.bias", torch.float32)
            b = b - b.mean()  # (as tt/router.py: top-k is shift invariant; the block subtracts it back for the weights)
            s = {n: ck.weight(f"{p}shared_experts.{n}.weight", torch.float32) for n in ("w1", "w3", "w2")}
            return MoEWeights(
                router={"weight": w[perm].contiguous(), "bias": b[perm].contiguous()},
                routed=None,
                shared={"gate_up": fuse_gate_up(s["w1"], s["w3"]), "down": s["w2"]},
                latent=None,
            )

        self.weights = MoELayer.build(mesh, g, m, host, self.dtypes, routed_op=op, layout=self.layout)
        self.head = settings.get("MOE_HEAD")
        if self.head == "ops":  # (router and shared expert on the device now: no host work in the forward)
            self.weights.ops()
        # (fused: the head lays out this layer's router and shared expert at the layer's first call - it needs the head)
        logger.info(
            f"MOE_BLOCK layer={layer} experts={m.num_experts} ({epc} a chip) chunk={self.chunk} cap={op.m} "
            f"layout={settings.get('EXPERTS_LAYOUT')} {self.layout} head={self.head} shared_fidelity=LoFi"
        )

    def block(self):
        """The (mesh, chunk)'s MoEBlock, built at the first call (with this layer's weights)."""
        from models.demos.moe_block.moe_block import MoEBlock, WeightCache

        key = (id(self.mesh), self.chunk)
        hit = _BLOCKS.get(key)
        if hit is None or hit[0] is not self.mesh:
            blk = MoEBlock(
                self.mesh,
                self.model,
                self.chunk,
                self.weights,
                WeightCache(path=None, layer=self.layer),
                num_links=LINKS,
                dtypes=self.dtypes,
                options=block_options(self.mesh),
            )
            if settings.get("PREPARED_ALL") == "on":  # (P.14: the fused tail keyed on shapes, addresses patched)
                from .prepared_tail import install

                install(blk)
            host_alloc.install_moe(blk)  # HOST_ALLOC=on (P.15): the exchange's arena probe is the call's arena
            hit = _BLOCKS[key] = (self.mesh, blk)
        return hit[1]

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: ffn_norm [1, 1, S/SP, H] TILE (MOE_HEAD=fused: [1, 1, S/SP, H/TP]; bf16, or fp32: cast here) ->
        [1, 1, S/SP, H/TP] bf16 (GLUE_FOLD=off: fp32, as moe_add's output)."""
        assert x.layout == ttnn.TILE_LAYOUT
        xb = cast(x, ttnn.bfloat16)
        y = self.block()(xb, layer=self.weights)
        if xb is not x:
            dealloc(xb)
        if not self.fold:
            y32 = ttnn.typecast(y, ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(y))
            dealloc(y)
            y = y32
        return y


def build_moe_block_layer(mesh, ck, cfg, layer: int, chunk: int, cache_dir=None) -> TtMoEBlockLayer:
    return TtMoEBlockLayer(mesh, ck, cfg, layer, chunk, cache_dir=cache_dir)
