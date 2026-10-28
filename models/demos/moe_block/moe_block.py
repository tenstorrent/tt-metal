# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MoE block for prefill: one entry for every MoE model, from the residual as attention leaves it back to the same layout.

Design (see the "Galaxy MoE Block" page): token owners, 2D dispatch, flat experts + combine in one program. The residual
stays main's layout, sequence over SP (mesh rows) and width over TP (mesh columns). No weight matrix is duplicated over TP.

Bring-up (MOE_BLOCK_WORKLOG.md): every stage its own program(s), run in sequence; the exchange is the K3 token-split
block's routed-only path. Not built yet: the weight cache, padded chunks, the experts during the dispatch, the fold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

# (MOE_EXPERTS_DRISC=0: the routed experts' gate / up weights from Tensix readers, not the DRAM cores' DRISC senders -
# every default below, MoEOptions.experts_drisc and the MoELayer / RoutedExperts builders)
_DRISC = os.environ.get("MOE_EXPERTS_DRISC", "1") != "0"

TILE = ttnn.TILE_SIZE

# ---------------------------------------------------------------------------------------------------------------------
# Configuration: what the checkpoint says, nothing about the machine
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertActivation:
    """act(gate) * up. One type for every model; the naming is still open."""

    kind: Literal["silu", "clamped_silu", "situ", "swiglu_oai"]
    limit: float | None = None  # clamp to +-limit (clamped_silu, swiglu_oai)
    alpha: float | None = None  # swiglu_oai
    beta: float | None = None  # situ
    linear_beta: float | None = None  # situ

    def __post_init__(self):
        if self.kind in ("clamped_silu", "swiglu_oai"):
            assert self.limit is not None and self.limit > 0, f"{self.kind} needs a positive limit"
        if self.kind == "swiglu_oai":
            assert self.alpha is not None, "swiglu_oai needs alpha"
        if self.kind == "situ":
            assert self.beta is not None and self.linear_beta is not None, "situ needs beta and linear_beta"


@dataclass(frozen=True)
class RouterConfig:
    score_func: Literal["sigmoid", "softmax", "sqrtsoftplus"] = "sigmoid"
    route_scale: float = 1.0  # routed_scaling_factor
    normalize_topk: bool = True  # norm_topk_prob
    expert_groups: int = 1  # group-limited routing; 1 = plain top-k
    topk_groups: int = 1
    # (a proxy: the router picks over num_experts x fold candidates, candidate c is expert c // fold - a Galaxy's
    # router width on fewer chips; the bias must leave one candidate of each group eligible so a token's picks fold to
    # distinct experts. The fused head only.)
    fold: int = 1


@dataclass(frozen=True)
class MoEModel:
    num_experts: int
    top_k: int
    emb_dim: int  # residual width H
    routed_expert_hidden_dim: int  # I
    routed_expert_activation: ExpertActivation
    router: RouterConfig
    shared_expert_hidden_dim: int | None  # None: no shared expert
    shared_expert_activation: ExpertActivation | None
    latent_dim: int | None = None  # latent MoE (K3): the routed experts run at this width
    latent_norm: bool = True
    norm_eps: float = 1e-5

    @property
    def routed_width(self) -> int:  # Hr: on the wire and in the experts
        return self.latent_dim if self.latent_dim is not None else self.emb_dim

    def __post_init__(self):
        E, K, r = self.num_experts, self.top_k, self.router
        assert 0 < K <= E, f"top_k {K} must be in (0, {E}]"
        assert E % r.expert_groups == 0, f"{E} experts don't split into {r.expert_groups} groups"
        assert 1 <= r.topk_groups <= r.expert_groups
        assert K <= r.topk_groups * (E // r.expert_groups), "top_k can't exceed the experts of the kept groups"
        for name, d in (("emb_dim", self.emb_dim), ("routed_expert_hidden_dim", self.routed_expert_hidden_dim)):
            assert d % TILE == 0, f"{name} {d} is not a whole number of tiles"
        assert (self.shared_expert_hidden_dim is None) == (self.shared_expert_activation is None)
        if self.shared_expert_hidden_dim is not None:
            assert self.shared_expert_hidden_dim % TILE == 0
        if self.latent_dim is not None:
            assert self.latent_dim % TILE == 0 and self.latent_dim < self.emb_dim


@dataclass(frozen=True)
class WireFormat:
    """Rows between chips. Receivers dequantize into the consuming dtype."""

    dtype: ttnn.DataType = ttnn.bfloat16  # bfloat16, or fp8_e4m3 with scales
    scale_block: int = 0  # values per fp32 scale; 0: none, 128: the dispatch's fp8_scaled_input

    def __post_init__(self):
        assert self.dtype in (ttnn.bfloat16, ttnn.fp8_e4m3), f"wire dtype {self.dtype} not supported"
        if self.dtype == ttnn.bfloat16:
            assert self.scale_block == 0, "bf16 rows carry no scales"
        else:
            assert self.scale_block > 0 and self.scale_block % TILE == 0, "fp8 rows need a per-block scale"


@dataclass(frozen=True)
class MoEDtypes:
    routed_weights: ttnn.DataType = ttnn.bfloat4_b
    routed_activations: ttnn.DataType = ttnn.bfloat8_b  # x and h inside the experts
    routed_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    shared_weights: ttnn.DataType = ttnn.bfloat8_b
    shared_activations: ttnn.DataType = ttnn.bfloat16
    shared_fidelity: ttnn.MathFidelity = (
        ttnn.MathFidelity.HiFi2
    )  # (HiFi3: rel L2 1.6 -> 1.3% with bf8 weights, +11-25% time)
    projection_weights: ttnn.DataType = ttnn.bfloat8_b  # latent down / up
    tail_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2  # the fused tail's up projection; never LoFi (user)
    router_weights: ttnn.DataType = ttnn.bfloat16  # top-k selection in fp32
    dispatch_rows: WireFormat = field(default_factory=WireFormat)
    combine_rows: WireFormat = field(default_factory=WireFormat)


@dataclass(frozen=True)
class MoEOptions:
    shared_expert: Literal["tp_split", "fold_into_combine"] = "tp_split"  # never replicated over TP
    # the dispatch inside the experts' program (the combine's port kernels' first phase): the weights stream from the
    # dispatch start, the experts' x reads wait until every port of the chip has its rows in (fused head only)
    experts_during_dispatch: bool = False
    # the head's router fin: the TP partial logits summed in a fixed order (the predicted landing order - bit-identical
    # runs); False: in landing order (A/B only - the logits' last bits, near-tie picks and routing weights vary)
    deterministic: bool = True
    # the combine's sums: every routed term rounded to a multiple of this q (a power of two) before its fp32 add, so a
    # token's sum is bit-identical in any landing order while every partial sum stays below 2^24 q (GLM-5.3-Flash:
    # 2^-14, its largest calibrated sum ~2^8.4); None: plain fp32 adds in landing order
    combine_snap: float | None = None
    # the head: "fused" one program a chip (tt/fused_head.py: x all-gather, shared expert + reduce-scatter, router,
    # routing exchange + planner, dispatch prep, latent; its rows in chunks where one does not fit L1); "ops" a program
    # a stage; "auto": fused where it fits (FusedHead.fits)
    head: Literal["auto", "fused", "ops"] = "auto"
    # a fused head that FusedHead.fits allows but that then fails to build (a host check, the L1 it needs): False
    # raises (the default: a shape the fused head should take never silently runs the ops head); True takes the ops
    # head. The reason is logged either way
    head_fallback: bool = False
    # the fused head's router logits: on the engine's cores at the start, on their own router cores, or "auto" (the
    # head's cost model: tt/fused_head.py router_placement - the router's FLOPs a core vs the SE path)
    router: Literal["auto", "engine", "router_cores"] = "auto"
    # the fused head's layout (tt/head_plan.py HeadLayout: engine GY, MT, chunks, K step, the logits on the engine):
    # None the planner's (its cost model's best that fits L1); given - A/B sweeps
    head_layout: object = None
    # the fused head's ports: the router's partial logits ahead of the x gather (True), after it (False), or None the
    # head's choice (HeadSE3.logits_priority) - A/B sweeps
    logits_first: bool | None = None
    # the fused head's router: a row tile's fin + top-K over two cores (router_stream's split - the logits on the engine,
    # the engine a column narrower): the routing chain ~-45 us at K3's E 896 (2x4 / ring: 106-109 -> 63-65 us), the
    # same picks. For meshes whose chain binds (the Galaxy's N 32); off - the LoudBox meshes' SE path binds
    router_split: bool = False
    # the routed experts' gate / up weight readers: the DRAM cores (DRISC, the default) or Tensix readers (A/B)
    experts_drisc: bool = _DRISC
    # the tail: "fused" one program a chip - latent models: tt/tail_engine.py (the latent all-gather into the engine's
    # L1, rms norm, up projection, + the shared expert; the combine's rows tilized there); else tt/tail_a2a.py (the
    # combine's rows tilized, + the shared expert's owner rows, the TP all-to-all into the output); "ops" its ops;
    # "auto": fused where it is built
    tail: Literal["auto", "fused", "ops"] = "auto"
    # the residual at the block's boundary: "sp_tp" main's layout [S/SP, H/TP] a chip; "owners" each chip's own tokens
    # at full width [S/N, H] (row-major chip order - the owners' rows: a model whose residual is split by tokens over
    # every chip, GLM-5.3-Flash's). The ops head gathers rows instead of width; the fused head (it reads the width
    # split) takes x through a TP all-to-all. The tail is the shared expert's add (its output is the owners' rows)
    residual: Literal["sp_tp", "owners"] = "sp_tp"


@dataclass(frozen=True)
class WeightCache:
    """Routed experts: one entry per expert, keyed on (chip grid, routed width, I, dtype, flat layout).
    TP-split weights (shared expert, router, latent): keyed on the TP degree. Never keyed on the mesh topology."""

    path: Path | None
    layer: int


# ---------------------------------------------------------------------------------------------------------------------
# Geometry: derived from the mesh, the fabric and the chunk; never passed in
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    sp: int  # mesh rows
    tp: int  # mesh columns
    sp_wraps: bool  # the SP axis is a ring
    tp_wraps: bool  # the TP axis is a ring
    links: int  # links per direction, the same on every axis
    zigzag: bool  # the residual's sequence order
    chunk_tokens: int  # the global chunk, all chips together
    num_experts: int

    @property
    def chips(self) -> int:
        return self.sp * self.tp

    @property
    def tokens_per_row(self) -> int:  # residual rows per chip
        return self.chunk_tokens // self.sp

    @property
    def tokens_per_chip(self) -> int:  # the tokens each chip owns
        return self.chunk_tokens // self.chips

    @property
    def experts_per_chip(self) -> int:
        return self.num_experts // self.chips

    def __post_init__(self):
        assert self.sp >= 1 and self.tp >= 1 and self.links >= 1
        assert self.num_experts % self.chips == 0, f"{self.num_experts} experts don't divide over {self.chips} chips"
        assert self.chunk_tokens % (self.chips * TILE) == 0, (
            f"chunk {self.chunk_tokens}: every chip must own whole tiles of tokens "
            f"(a multiple of {self.chips * TILE})"
        )

    @staticmethod
    def of(mesh_device, model: MoEModel, chunk_tokens: int, num_links: int, zigzag: bool) -> Geometry:
        sp, tp = tuple(mesh_device.shape)
        cfg = ttnn.get_fabric_config()
        return Geometry(
            sp=sp,
            tp=tp,
            sp_wraps=cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_Y, ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
            tp_wraps=cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_X, ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
            links=num_links,
            zigzag=zigzag,
            chunk_tokens=chunk_tokens,
            num_experts=model.num_experts,
        )

    def chip_of_expert(self, e: int) -> tuple[int, int]:  # contiguous: chip c holds [c E/N, (c + 1) E/N)
        assert 0 <= e < self.num_experts
        return divmod(e // self.experts_per_chip, self.tp)  # (mesh row, mesh column)

    def owner_of_row_token(self, j: int) -> int:  # the TP chip of the row that owns the row's token j
        assert 0 <= j < self.tokens_per_row
        return j // self.tokens_per_chip


@dataclass(frozen=True)
class ChunkSpan:
    """Which of the chunk's rows are real. Host ints, or one-element device tensors on the traced path."""

    start: int | ttnn.Tensor  # first token's absolute position
    valid_tokens: int | ttnn.Tensor  # real tokens from start; right padding only


# ---------------------------------------------------------------------------------------------------------------------
# Weights: host torch tensors in HF orientation [out, in]; gate and up are one tensor
# ---------------------------------------------------------------------------------------------------------------------

GATE_UP_UNIT = 16  # gate_up alternates 16 rows of gate and 16 rows of up


def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor, unit: int = GATE_UP_UNIT) -> torch.Tensor:
    """[I, H] gate and up -> [2 I, H]: rows [2 u k, 2 u k + u) are gate rows [u k, u k + u), the next u rows up's"""
    I, H = gate.shape
    assert up.shape == (I, H) and I % unit == 0
    return torch.stack([gate.view(I // unit, unit, H), up.view(I // unit, unit, H)], dim=1).reshape(2 * I, H)


def split_gate_up(gate_up: torch.Tensor, unit: int = GATE_UP_UNIT) -> tuple[torch.Tensor, torch.Tensor]:
    two_i, H = gate_up.shape
    assert two_i % (2 * unit) == 0
    pairs = gate_up.view(two_i // (2 * unit), 2, unit, H)
    return pairs[:, 0].reshape(two_i // 2, H), pairs[:, 1].reshape(two_i // 2, H)


@dataclass
class MoEWeights:
    router: dict  # {"weight": [E, H], "bias": [E]}
    routed: list[dict] | None  # E dicts by global expert id: {"gate_up": [2 I, Hr], "down": [Hr, I]}
    shared: dict | None  # {"gate_up": [2 Is, H], "down": [H, Is]}
    latent: dict | None  # {"down": [Hr, H], "up": [H, Hr], "norm": [Hr]}

    def check(self, model: MoEModel):
        """The contract: every key and shape. fp32 or bf16 on the host; the block quantizes."""
        E, H, I, Hr = model.num_experts, model.emb_dim, model.routed_expert_hidden_dim, model.routed_width
        F = model.router.fold
        assert self.router["weight"].shape == (E * F, H), f"router weight {tuple(self.router['weight'].shape)}"
        assert self.router["bias"].shape == (E * F,)
        # (None: the routed experts come laid out already - MoELayer.build's routed_op)
        assert self.routed is None or len(self.routed) == E, f"{len(self.routed)} routed experts, the model has {E}"
        for e, w in enumerate(self.routed or ()):
            assert w["gate_up"].shape == (2 * I, Hr), f"expert {e} gate_up {tuple(w['gate_up'].shape)}"
            assert w["down"].shape == (Hr, I), f"expert {e} down {tuple(w['down'].shape)}"
        assert (self.shared is None) == (model.shared_expert_hidden_dim is None)
        if self.shared is not None:
            Is = model.shared_expert_hidden_dim
            assert self.shared["gate_up"].shape == (2 * Is, H) and self.shared["down"].shape == (H, Is)
        assert (self.latent is None) == (model.latent_dim is None)
        if self.latent is not None:
            assert self.latent["down"].shape == (Hr, H) and self.latent["up"].shape == (H, Hr)
            assert not model.latent_norm or self.latent["norm"].shape == (Hr,)
        for t in self._tensors():
            assert t.dtype in (torch.float32, torch.bfloat16), f"host weights must be fp32 or bf16, got {t.dtype}"

    def _tensors(self):
        yield from self.router.values()
        for w in self.routed or ():
            yield from w.values()
        for d in (self.shared, self.latent):
            if d is not None:
                yield from d.values()


class MoELayer:
    """One layer of a model whose MoE layers share one MoEBlock (its programs, arena and exchange): each call passes its
    layer (MoEBlock.forward(x, layer=...)). Holds the routed experts (laid out at build) and its other weights' host
    source: the ops head's router / shared expert are uploaded at their first use, the fused head lays its own out
    (its engine's layout) at the layer's first call through it."""

    def __init__(
        self, mesh, g: Geometry, model: MoEModel, host, experts: "RoutedExperts", dtypes: MoEDtypes, parent=None
    ):
        self.mesh, self.g, self.model, self.dtypes = mesh, g, model, dtypes
        self.host = host  # () -> MoEWeights (router, shared, latent; routed may be None)
        self.experts = experts
        self.parent = parent  # (the layer at another chunk this one shares its weights with)
        self._ops = None
        self._latent = None
        self.heads = {}  # id(FusedHead) -> that head's layout of this layer's weights
        self.tails = {}  # id(FusedTail) -> that tail's upload of this layer's latent up / norm (a latent model)

    @staticmethod
    def build(mesh, g: Geometry, model: MoEModel, weights, dtypes: MoEDtypes = MoEDtypes(), routed_op=None, drisc=_DRISC):
        """weights: MoEWeights, or a callable returning them (read when a head lays them out). routed_op: the layer's
        FlatRoutedExpert, laid out by the caller (its weight cache) with RoutedExperts.layout_of(g, model, drisc) -
        weights.routed is then None."""
        host = weights if callable(weights) else (lambda: weights)
        routed = None
        if routed_op is None:
            w = host()
            w.check(model)
            assert w.routed is not None, "routed experts: host weights or a laid-out op"
            routed = w.routed
        experts = RoutedExperts(mesh, g, model, routed, dtypes, drisc=drisc, op=routed_op)
        return MoELayer(mesh, g, model, host, experts, dtypes)

    def ops(self):
        """(Router, SharedExpert | None): the ops head's, uploaded at the first use (at another chunk: the parent's)."""
        if self._ops is None:
            if self.parent is not None:
                r, sh = self.parent.ops()
                self._ops = (r.at(self.mesh, self.g, self.model), sh)
            else:
                w = self.host()
                w.check(self.model)
                sh = None
                if self.model.shared_expert_hidden_dim:
                    sh = SharedExpert(self.mesh, self.g, self.model, w.shared, self.dtypes)
                self._ops = (Router(self.mesh, self.g, self.model, w.router, self.dtypes), sh)
        return self._ops

    def latent_ops(self):
        """LatentProjections (the ops path's latent down / norm / up): this layer's, uploaded at the first use."""
        if self._latent is None:
            w = self.host()
            self._latent = LatentProjections(self.mesh, self.g, self.model, w.latent, self.dtypes)
        return self._latent

    def at(self, mesh, g: Geometry, model: MoEModel, drisc=_DRISC):
        """This layer at another chunk (geometry): its weights shared - the router's top-k and the flat experts' plan
        made for the chunk (FlatRoutedExpert.replanned: the same weight packing)."""
        op = self.experts.op.replanned(RoutedExperts.cap_of(g), RoutedExperts.layout_of(g, model, drisc))
        experts = RoutedExperts(mesh, g, model, None, self.dtypes, drisc=drisc, op=op)
        return MoELayer(mesh, g, model, self.host, experts, self.dtypes, parent=self)


# ---------------------------------------------------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------------------------------------------------


class MoEBlock(LightweightModule):
    """One MoE layer: head (router, x all-gather, shared expert + its reduce-scatter, latent down), exchange (routing
    gather, planner, 2D dispatch, flat experts + combine into the owners' sums), tail (back to the residual's split).
    Bring-up: every stage its own program(s), run in sequence; stage_hook(name), if set, is called after each stage."""

    def __init__(
        self,
        mesh_device,  # SP = mesh rows, TP = mesh columns
        model: MoEModel,
        chunk_tokens: int,  # the global chunk, all chips together
        weights: MoEWeights | MoELayer | None,  # None: load everything from the cache; MoELayer: a model's layer
        cache: WeightCache,
        num_links: int = 2,  # links per direction on every axis; ring or line per axis comes from the open fabric
        dtypes: MoEDtypes = MoEDtypes(),
        options: MoEOptions = MoEOptions(),
        sequence_zigzag: bool = False,  # must match attention's sequence order
    ):
        self.mesh_device = mesh_device
        self.geometry = Geometry.of(mesh_device, model, chunk_tokens, num_links, zigzag=sequence_zigzag)
        self.model, self.dtypes, self.options, self.cache = model, dtypes, options, cache
        assert (
            options.shared_expert != "fold_into_combine" or model.latent_dim is None
        ), "latent MoE: the owners hold latent-width sums, the shared expert is full width"
        assert (
            options.shared_expert != "fold_into_combine" or model.shared_expert_hidden_dim is not None
        ), "nothing to fold: the model has no shared expert"
        g = self.geometry
        assert model.emb_dim % (g.tp * TILE) == 0, f"H {model.emb_dim} doesn't split into whole tiles over TP {g.tp}"
        assert g.tokens_per_chip % TILE == 0, "every chip owns whole tiles of tokens"
        # (the combine hands a y row to its owner as one packet: the row in bf16 + a 32 B trailer; a smaller fabric
        # payload hangs the exchange instead of failing)
        need, have = 2 * model.routed_width + 32, int(ttnn.get_tt_fabric_max_payload_size_bytes())
        assert have >= need, (
            f"fabric packet payload {have} B: the MoE block's combine needs >= {need} B (a {model.routed_width}-wide "
            f"bf16 row + its trailer); open the fabric with a larger payload"
        )
        if weights is None:
            assert cache.path is not None, "no weights and no cache to load them from"
            raise NotImplementedError("MoEBlock: loading from the weight cache is not built yet; pass weights")
        layer = weights if isinstance(weights, MoELayer) else None
        self.layer = layer  # (the layer the block was built with: the last used)
        if layer is None:
            weights.check(model)
        else:
            weights = layer.host()  # (the heads lay out the router / shared expert from the host weights)
        owners = options.residual == "owners"
        assert (
            not owners or model.latent_dim is None
        ), "owners residual: no latent (its up projection lands the width split)"
        # bring-up limits (each lifted by the piece that needs it)
        assert options.shared_expert == "tp_split", "bring-up: the shared expert TP-split, reduce-scattered in the head"
        assert not sequence_zigzag, "bring-up: natural sequence order"
        fits = FusedHead.fits(g, model, dtypes)
        assert options.head != "fused" or fits is True, f"the fused head: {fits}"
        self.head_mode = "fused" if options.head == "fused" or (options.head == "auto" and fits is True) else "ops"
        # (the fused head plans the combine itself - routing_a2a's planner writes no order table: combine_snap instead)
        assert not (
            getattr(options, "combine_deterministic", False) and self.head_mode == "fused"  # (not an option on llk)
        ), "combine_deterministic: the ops head's planner (the fused head's has no fixed order; combine_snap)"
        assert (
            not options.experts_during_dispatch or self.head_mode == "fused"
        ), "experts_during_dispatch: the fused head (its dispatch prep hands the exchange the dispatch buffer)"
        self.fused = None
        self.router = self.shared = None
        if layer is not None:
            if self.head_mode == "ops":
                self.router, self.shared = layer.ops()
        elif self.head_mode == "ops":
            self.router = Router(mesh_device, g, model, weights.router, dtypes)
            if model.shared_expert_hidden_dim is not None:
                self.shared = SharedExpert(mesh_device, g, model, weights.shared, dtypes)
        self.latent = None
        if model.latent_dim is not None:  # (the fused head does the latent down: only the up is used then)
            # (a MoELayer: its own latent pieces - use() swaps them, and the fused tail's latent up / norm, per layer)
            if layer is not None:
                self.latent = layer.latent_ops()
            else:
                self.latent = LatentProjections(mesh_device, g, model, weights.latent, dtypes)
        if layer is not None:
            self.experts = layer.experts
        else:
            self.experts = RoutedExperts(mesh_device, g, model, weights.routed, dtypes, drisc=options.experts_drisc)
        # (the fused head's planner and the exchange's combine agree on the accumulator count: planned here from the
        # cells the experts' plan leaves free and the combine's ports - the ports-apart mode)
        # (the count and the mode: the ports-apart mode where it fits, else the accumulators on the ports' cells - the
        # ports' cells vary per Galaxy, plan_combine sizes the combine as the exchange will)
        acc = acc_mode = None
        if self.head_mode == "fused":
            from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import plan_combine

            acc, acc_mode = plan_combine(
                mesh_device,
                self.experts.op.plan,
                g.tokens_per_chip,
                model.routed_width,
                (g.tp_wraps, g.sp_wraps),
                options.experts_during_dispatch,
            )
            if acc_mode != "ports apart":
                from loguru import logger

                logger.info(f"MoEBlock: combine accumulators on the ports' cells ({acc}): ports apart does not fit here")
        self.exchange = TokenExchange(
            mesh_device,
            g,
            model,
            self.experts,
            dispatch_in_program=options.experts_during_dispatch,
            accumulators=(acc, acc_mode) if acc is not None else None,
            combine_snap=options.combine_snap,
        )
        self.stage_hook = None  # (profiling: called with the stage name after every stage)
        self.keep_taps = False  # (debug: keep the last call's routing rows for taps(); "device": as a device tensor -
        # no host read in the forward, so a traced forward keeps them too: taps() reads them)
        self._taps = {}
        # The exchange's arena address is baked into its kernels: run the head and the tail once first (their ops'
        # one-time L1 - collective semaphores, program state - lands now), then build the exchange on the L1 left.
        # The fused head: its arena and the exchange's alternate in the L1 left, each taken for its own call, at the
        # address its programs baked (L1 is placed top-down). So: the tail's one-time L1 first (its CBs clash with the
        # head's arena); a hole for what the exchange keeps (its words: given back before its build); two for what the
        # head keeps after placing its arena, and after placing it again (the head frees them then) - nothing kept
        # stranded below the arena, where the other programs' circular buffers go; the head built and run once, its
        # arena settled (free L1 right above it filled) and freed.
        tfits = FusedTail.fits(g, model, self.head_mode) if not owners else "the owners residual (the tail is an add)"
        assert options.tail != "fused" or tfits is True, f"the fused tail: {tfits}"
        self.tail = None
        if options.tail == "fused" or (options.tail == "auto" and tfits is True):
            # (before the tail's warm-up: its persistent words go with the tail's other one-time L1)
            self.tail = FusedTail(
                mesh_device, g, model, layer.host() if layer is not None else weights, self._tp_topology(), dtypes
            )
            if layer is not None and self.tail.a2a is None:  # (a latent model: the layer's latent up / norm)
                layer.tails[id(self.tail)] = self.tail.w
        if self.head_mode == "fused":
            self._warm_up_tail()
            # (PERSIST_L1=small: the kept words go to L1_SMALL - no holes, which settle_arena would otherwise fill
            # with a filler kept in L1 below everything)
            from ttnn.bringup.combine_rf import persist_l1

            small = persist_l1.small()
            hole = None if small else _l1_hole(mesh_device, FusedHead.HOLE)
            holes = [] if small else [_l1_hole(mesh_device, FusedHead.HOLE_HEAD) for _ in range(2)]
            try:
                self.fused = FusedHead(
                    mesh_device,
                    g,
                    model,
                    weights,
                    holes,
                    deterministic=options.deterministic,
                    accumulators=acc,
                    router=options.router,
                    layout=options.head_layout,
                    logits_first=options.logits_first,
                    router_split=options.router_split,
                    fidelity=dtypes.shared_fidelity,
                )
                self._warm_up_head()
                self.fused.head.settle_arena()
            except (AssertionError, ValueError, RuntimeError) as e:
                # (a fused head that FusedHead.fits allows but does not build raises, unless options.head_fallback:
                # then the ops head; the partial head's buffers go with it)
                from loguru import logger

                fits = "build: " + " ".join(str(e).split())[:400]
                if options.head == "fused" or not options.head_fallback:
                    logger.error(
                        f"MoEBlock: the fused head does not build (mesh {g.sp}x{g.tp}, chunk {g.chunk_tokens}): "
                        f"{fits}; raising (MoEOptions.head_fallback=True takes the ops head)"
                    )
                    raise
                logger.warning(
                    f"MoEBlock: the fused head does not build here ({fits}); the ops head (MoEOptions.head_fallback)"
                )
                _free_partial_l1([self.fused], e.__traceback__.tb_next)
                self.fused = None
                for t in holes:
                    if t.is_allocated():
                        ttnn.deallocate(t)
                del e
                import gc

                gc.collect()
            if hole is not None:
                ttnn.deallocate(hole)
            if self.fused is None:
                self._head_ops_fallback(weights, dtypes, options)
        else:
            self._warm_up()
        self.exchange.build()
        if self.fused is not None:
            self.fused.attach(self.exchange)
            if layer is not None:  # (the layer's weights as this head laid them out)
                layer.heads[id(self.fused)] = (self.fused.w, self.fused.head.rw)
                self.fused.head.use_layer(id(layer), self.fused.head.rw)
        from loguru import logger

        # (the resolved plan in one line: what was picked and why - the generality map reads it)
        self.summary = (
            f"head {self.head_mode}{'' if fits is True else f' (fused: {fits})'}{self.fused.describe() if self.fused else ''}; tail "
            f"{'fused' if self.tail else 'ops'}{'' if tfits is True else f' (fused: {tfits})'}"
        )
        logger.info(
            f"MoEBlock: mesh {g.sp}x{g.tp}, chunk {g.chunk_tokens} ({g.tokens_per_row} rows a mesh row, "
            f"{g.tokens_per_chip} owned a chip), {g.experts_per_chip} experts a chip, routed width {model.routed_width}; "
            f"{self.summary}; flat layout {self.experts.layout_fields}; dtypes {dtypes}; options {options}"
        )
        # (the fused head's outputs reused each forward, either tail: freed, they come back at other addresses between a
        # model's layers and the prepared program is rebuilt every call, ~90 ms of host)
        if self.fused is not None:
            self.fused.head.persist = True
            if self.tail is not None and getattr(self.tail, "a2a", None) is None:  # (its y / se: the persistent ones)
                self.tail.eng.cache_addrs = True

    def _mark(self, name):
        if self.stage_hook is not None:
            self.stage_hook(name)

    def _x_shape(self):
        g, H = self.geometry, self.model.emb_dim
        if self.options.residual == "owners":
            return (1, 1, g.tokens_per_chip, H)
        return (1, 1, g.tokens_per_row, H // g.tp)

    def use(self, layer: MoELayer):
        """Run the following forwards with this layer's weights (programs, arena and exchange shared; the fused head:
        the layer's weights in its layout - laid out at the layer's first call - and the layer's own programs)."""
        if layer is self.layer:
            return
        if self.fused is not None:
            lw = layer.heads.get(id(self.fused))
            if lw is None:
                lw = layer.heads[id(self.fused)] = self.fused.layer_weights(layer.host())
            self.fused.w = lw[0]
            self.fused.head.use_layer(id(layer), lw[1])
        else:
            self.router, self.shared = layer.ops()
        self.experts = self.exchange.experts = layer.experts
        if self.model.latent_dim is not None:  # (a latent model: the layer's latent down / norm / up, the tail's too)
            self.latent = layer.latent_ops()
            if self.tail is not None and self.tail.a2a is None:
                tw = layer.tails.get(id(self.tail))
                if tw is None:
                    tw = layer.tails[id(self.tail)] = self.tail.layer_weights(layer.host())
                self.tail.w = tw
        self.layer = layer

    def forward(self, x: ttnn.Tensor, chunk: ChunkSpan | None = None, layer: MoELayer | None = None) -> ttnn.Tensor:
        """x: [1, 1, S/SP, H/TP] TILE, the residual as attention leaves it (MoEOptions.residual "owners": [1, 1, S/N, H],
        the chip's own tokens). Returns the same layout. chunk: which rows are real; None = all of them. layer: this
        call's weights (MoELayer; None: the last used)."""
        g, H = self.geometry, self.model.emb_dim
        assert tuple(x.shape) == self._x_shape(), (
            f"x {tuple(x.shape)}, expected {self._x_shape()} on each chip "
            f"(chunk {g.chunk_tokens}, mesh {g.sp}x{g.tp}, H {H}, residual {self.options.residual})"
        )
        assert x.layout == ttnn.TILE_LAYOUT
        if chunk is not None:
            raise NotImplementedError("MoEBlock: padded chunks (ChunkSpan) are not built yet")
        if layer is not None:
            self.use(layer)
        if self.fused is not None:
            # (one L1 allocation a forward: the exchange's arena, the head's range inside it - the head runs there first)
            eng = self.fused.head.eng
            arena = self.exchange.take_arena(inside=(eng.base, eng.arena_bytes))
            if self.options.residual == "owners":  # (the fused head reads the width split [S/SP, H/TP])
                xs = self._tp_a2a(x, cat_dim=2, split_dim=3)
                self._mark("x_a2a")
                se, out = self.fused(xs, arena)
                ttnn.deallocate(xs)
            else:
                se, out = self.fused(x, arena)
            self._mark("head")
            if self.keep_taps == "device":
                self._taps = dict(rows_dev=out.pop("rows"))
            elif self.keep_taps:
                self._taps = dict(rows=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(out["rows"])])
            # owners' sums [S/N, Hr] (the fused tail: the combine's row-major rows, tilized in its program)
            y_own = self.exchange.prepped((out["xbuf"], out["route"]), hook=self.stage_hook, tilize=self.tail is None)
            persist = getattr(self.fused.head, "persist", False)  # (the head's outputs reused across forwards)
            for k in ("rows", "xbuf", "route", "latent"):
                if not persist and k in out and out[k].is_allocated():
                    ttnn.deallocate(out[k])
            if self.tail is not None:
                y = self.tail(y_own, se)
                if not persist:
                    ttnn.deallocate(se)
                self._mark("tail")
                return y
            return self._tail(y_own, se)
        rows, own, se = self._head(x)
        # ---- exchange ----
        if self.keep_taps == "device":  # (a device copy - the exchange consumes the routing rows: a traced forward)
            self._taps = dict(rows_dev=ttnn.clone(rows))
        elif self.keep_taps:  # (host copies: the exchange consumes the routing rows)
            self._taps = dict(rows=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(rows)])
        y_own = self.exchange(own, rows, hook=self.stage_hook, tilize=self.tail is None)  # owners' sums [S/N, Hr]
        if own is not x:  # (the owners residual: x is the owners' rows)
            ttnn.deallocate(own)
        if rows.is_allocated():
            ttnn.deallocate(rows)
        if self.tail is not None:
            y = self.tail(y_own, se)
            ttnn.deallocate(se)
            self._mark("tail")
            return y
        return self._tail(y_own, se)

    def _head_ops_fallback(self, weights, dtypes, options):
        """The ops head after a fused head that did not build (auto): its router / shared expert, the exchange planned
        without the fused head's accumulators, the ops warm-up."""
        g, m, mesh = self.geometry, self.model, self.mesh_device
        self.head_mode = "ops"
        if self.layer is not None:
            self.router, self.shared = self.layer.ops()
        else:
            self.router = Router(mesh, g, m, weights.router, dtypes)
            if m.shared_expert_hidden_dim is not None:
                self.shared = SharedExpert(mesh, g, m, weights.shared, dtypes)
        self.exchange = TokenExchange(
            mesh,
            g,
            m,
            self.experts,
            dispatch_in_program=options.experts_during_dispatch,
            combine_snap=options.combine_snap,
        )
        self._warm_up()

    OPS_KEEP = 16 * 1024  # (L1 a core the ops' one-time words take at the top: ~3 KB at M3 on Galaxy)

    def _warm_up(self):
        """The head and the tail once on zeros (the exchange skipped): every op they run allocates its one-time L1. The
        ops place theirs during their first call, under their large L1 intermediates (the logits reduce-scatter's: 8
        words 650 KB down on Galaxy) - stranded mid-L1 once those go, and the exchange's arena gets only the L1 below.
        So a free block at the top, held apart by a spacer under it: the words land there, intermediates (larger)
        below the spacer."""
        mesh = self.mesh_device
        top = _l1_hole(mesh, self.OPS_KEEP)
        spacer = _l1_hole(mesh, 64)
        ttnn.deallocate(top)
        self._warm_up_ops()
        ttnn.deallocate(spacer)

    def _warm_up_ops(self):
        g, m = self.geometry, self.model
        mesh = self.mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)
        z = lambda shape: ttnn.from_torch(  # noqa: E731
            torch.zeros(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        x0 = z(self._x_shape())
        rows, own, se = self._head(x0, mark=False)
        if own is not x0:
            ttnn.deallocate(own)
        ttnn.deallocate(rows)
        ttnn.deallocate(x0)
        if self.tail is not None:  # (the fused tail on the combine's row-major layout)
            y0 = ttnn.from_torch(
                torch.zeros(1, 1, g.tokens_per_chip, m.routed_width),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
            )
            out = self.tail(y0, se)
            ttnn.deallocate(y0)
            ttnn.deallocate(se)
        else:
            out = self._tail(z((g.tokens_per_chip, m.routed_width)), se, mark=False)
        ttnn.deallocate(out)
        ttnn.synchronize_device(mesh)

    def _zeros(self, shape):
        return ttnn.from_torch(
            torch.zeros(shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _warm_up_tail(self):
        """The tail once on zeros (the fused head's warm-up: before the head - see __init__)."""
        g, m = self.geometry, self.model
        se = self._zeros(
            (1, 1, g.tokens_per_row, m.emb_dim // g.tp) if self.latent else (1, 1, g.tokens_per_chip, m.emb_dim)
        )
        if self.tail is not None:
            y0 = ttnn.from_torch(
                torch.zeros(1, 1, g.tokens_per_chip, m.routed_width),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            out = self.tail(y0, se)
            for t in (out, y0, se):
                ttnn.deallocate(t)
            ttnn.synchronize_device(self.mesh_device)
            return
        out = self._tail(self._zeros((g.tokens_per_chip, m.routed_width)), se, mark=False)
        ttnn.deallocate(out)
        ttnn.synchronize_device(self.mesh_device)

    def _warm_up_head(self):
        """The fused head once on zeros (its programs built; its arena freed)."""
        g, m = self.geometry, self.model
        x0 = self._zeros((1, 1, g.tokens_per_row, m.emb_dim // g.tp))
        se, out = self.fused(x0)
        for t in [se, x0] + list(out.values()):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh_device)

    def _head(self, x, mark=True):
        """-> (the joint routing rows, the owners' rows [1, 1, S/N, Hr], the shared expert's [S/SP, H/TP] or None)."""
        mark = self._mark if mark else (lambda _: None)
        if self.options.residual == "owners":
            return self._head_owners(x, mark)
        # ---- head ----
        partial = self.router.logits(x)  # [S/SP, E] fp32 partial over H
        mark("router")
        logits = ttnn.reduce_scatter(  # owners' [S/N, E]
            partial, dim=2, cluster_axis=1, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(partial)
        mark("logits_rs")
        rows = self.router.top_k(logits)  # joint routing rows [S/N / 32, 32 x 2K] uint16
        ttnn.deallocate(logits)
        mark("topk")
        x_full = ttnn.all_gather(x, dim=3, cluster_axis=1)  # [S/SP, H]
        mark("x_ag")
        se = None
        if self.shared is not None:
            se_part = self.shared(x_full)  # [S/SP, H] partial over I_s
            mark("shared")
            if self.latent is None:  # (to the owners, by rows: added to the routed sums before the tail's all-to-all)
                se = self._tp_rs_rows(se_part)  # [S/N, H]
            else:  # (K3: the latent up lands the width split, so the shared output does too)
                se = ttnn.reduce_scatter(se_part, dim=3, cluster_axis=1)  # [S/SP, H/TP]
            ttnn.deallocate(se_part)
            mark("shared_rs")
        if self.latent is not None:
            lat_part = self.latent.down(x_full)  # [S/SP, Hr/TP]
            mark("latent_down")
            own = self._tp_a2a(lat_part, cat_dim=3, split_dim=2)  # the owners' latent rows [S/N, Hr]
            ttnn.deallocate(lat_part)
            mark("latent_a2a")
        else:
            own = ttnn.mesh_partition(x_full, dim=2, cluster_axis=1)  # [S/N, H]
            mark("own_rows")
        ttnn.deallocate(x_full)
        return rows, own, se

    def _head_owners(self, x, mark):
        """The owners residual: x [S/N, H] -> the mesh row's rows gathered over TP [S/SP, H]; the router's partial logits
        on this chip's H/TP columns of them (its weight split as in the width layout), reduce-scattered back to the
        owners; the shared expert on the gathered rows, reduce-scattered to the owners; the owners' rows are x."""
        x_full = self._tp_ag_rows(x)  # [S/SP, H]
        mark("x_ag")
        x_tp = ttnn.mesh_partition(x_full, dim=3, cluster_axis=1)  # [S/SP, H/TP]
        partial = self.router.logits(x_tp)
        ttnn.deallocate(x_tp)
        mark("router")
        logits = ttnn.reduce_scatter(partial, dim=2, cluster_axis=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        mark("logits_rs")
        rows = self.router.top_k(logits)
        ttnn.deallocate(logits)
        mark("topk")
        se = None
        if self.shared is not None:
            se_part = self.shared(x_full)
            mark("shared")
            se = self._tp_rs_rows(se_part)  # [S/N, H]
            ttnn.deallocate(se_part)
            mark("shared_rs")
        ttnn.deallocate(x_full)
        return rows, x, se

    def _tail(self, y_own, se, mark=True):
        """owners' sums [S/N, Hr] (+ the shared expert's output) -> [1, 1, S/SP, H/TP] (owners residual: [S/N, H])."""
        mark = self._mark if mark else (lambda _: None)
        g = self.geometry
        y_own = ttnn.reshape(y_own, [1, 1, g.tokens_per_chip, self.model.routed_width])
        if self.options.residual == "owners":
            if se is None:
                return y_own
            y = ttnn.add(y_own, se)
            ttnn.deallocate(y_own)
            self._free_se(se)
            mark("add")
            return y
        if self.latent is not None:
            y_row = self._tp_ag_rows(y_own)  # [S/SP, Hr]
            ttnn.deallocate(y_own)
            mark("tail_ag")
            out = self.latent.up(y_row)  # norm, up projection split by columns: [S/SP, H/TP]
            ttnn.deallocate(y_row)
            mark("latent_up")
            if se is not None:
                y = ttnn.add(out, se)
                ttnn.deallocate(out)
                self._free_se(se)
                out = y
                mark("add")
            return out
        if se is not None:  # (the shared expert's owner rows, added before the all-to-all carries both)
            y = ttnn.add(y_own, se)
            ttnn.deallocate(y_own)
            self._free_se(se)
            y_own = y
            mark("add")
        out = self._tp_a2a(y_own, cat_dim=2, split_dim=3)  # back to the width split [S/SP, H/TP]
        ttnn.deallocate(y_own)
        mark("tail_a2a")
        return out

    def _free_se(self, se):
        if not (self.fused is not None and getattr(self.fused.head, "persist", False)):  # (the head's, reused)
            ttnn.deallocate(se)

    # ---- TP collectives (cluster axis 1: a line, or a ring on a torus that wraps it) ----
    def _tp_topology(self):
        return ttnn.Topology.Ring if self.geometry.tp_wraps else ttnn.Topology.Linear

    def _tp_a2a(self, t, cat_dim, split_dim):
        """All-to-all over TP: t split into TP blocks along split_dim, block j to chip j; the received blocks
        concatenated along cat_dim in source order. (all_to_all_async_generic: its in_dim is the concatenation dim.)"""
        return ttnn.experimental.all_to_all_async_generic(
            t,
            in_dim=cat_dim,
            out_dim=split_dim,
            cluster_axis=1,
            num_links=self.geometry.links,
            topology=self._tp_topology(),
        )

    def _tp_ag_rows(self, t):
        """All-gather by rows over TP (combine_rf's fabric all-gather: ~6% under ttnn's at these shapes)."""
        from ttnn.bringup.combine_rf.fabric_all_gather_py import fabric_all_gather

        return fabric_all_gather(t, cluster_axis=1, topology=self._tp_topology(), num_links=self.geometry.links, dim=2)

    def _tp_rs_rows(self, t):
        """Reduce-scatter by rows over TP (fabric_reduce_scatter: ~17% under ttnn's on the shared expert's partials;
        bf16 - not for the fp32 logits, where it rounds)."""
        from ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter import fabric_reduce_scatter

        return fabric_reduce_scatter(t, cluster_axis=1, topology=self._tp_topology(), num_links=self.geometry.links)

    def taps(self) -> dict:
        """Debug: the last forward's routing rows (keep_taps = True, or "device": read now)."""
        if "rows_dev" in self._taps:
            return dict(rows=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(self._taps["rows_dev"])])
        return self._taps

    def input_buffer(self) -> ttnn.Tensor:
        """A persistent x [1, 1, S/SP, H/TP] TILE: forward(input_buffer()) reads it in place (the fused head's own x -
        no copy). A traced forward is captured on it; each replay's x is written into it first."""
        if self.fused is not None and self.options.residual != "owners":
            return self.fused.x
        if getattr(self, "_x_in", None) is None:
            g = self.geometry
            self._x_in = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(self._x_shape())),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._x_in

    def close(self):
        """Free the block's device memory (weights, arenas, persistent words): another block - another chunk or model -
        can then be built on this mesh. The block is unusable after."""
        ttnn.synchronize_device(self.mesh_device)
        _free_device_tensors([self], ("models.demos", "ttnn.bringup"), l1_only=False)
        import gc

        gc.collect()  # (the global semaphores go with their last reference)


def _act_dev(kind: ExpertActivation, g, u):
    """act(gate) * up on device tiles (ttnn elementwise ops)."""
    if kind.kind == "silu":
        return ttnn.multiply(ttnn.silu(g), u)
    if kind.kind == "situ":
        b, lb = kind.beta, kind.linear_beta
        a = ttnn.multiply(ttnn.multiply(ttnn.tanh(ttnn.multiply(g, 1.0 / b)), b), ttnn.sigmoid(g))
        return ttnn.multiply(a, ttnn.multiply(ttnn.tanh(ttnn.multiply(u, 1.0 / lb)), lb))
    if kind.kind == "clamped_silu":
        lim = kind.limit
        return ttnn.multiply(ttnn.silu(ttnn.clamp(g, max=lim)), ttnn.clamp(u, min=-lim, max=lim))
    if kind.kind == "swiglu_oai":
        lim, al = kind.limit, kind.alpha
        gc = ttnn.clamp(g, max=lim)
        return ttnn.multiply(
            ttnn.add(ttnn.clamp(u, min=-lim, max=lim), 1.0), ttnn.multiply(gc, ttnn.sigmoid(ttnn.multiply(gc, al)))
        )
    raise ValueError(kind)


def act_host(kind: ExpertActivation, g, u):
    """act(gate) * up on host tensors (the reference)."""
    F = torch.nn.functional
    if kind.kind == "silu":
        return F.silu(g) * u
    if kind.kind == "situ":
        b, lb = kind.beta, kind.linear_beta
        return b * torch.tanh(g / b) * torch.sigmoid(g) * lb * torch.tanh(u / lb)
    if kind.kind == "clamped_silu":
        return F.silu(g.clamp(max=kind.limit)) * u.clamp(-kind.limit, kind.limit)
    if kind.kind == "swiglu_oai":
        gc = g.clamp(max=kind.limit)
        return (u.clamp(-kind.limit, kind.limit) + 1) * gc * torch.sigmoid(kind.alpha * gc)
    raise ValueError(kind)


# the flat op's names for the routed activations
_FLAT_ACT = {"silu": "silu", "situ": "situ", "clamped_silu": "clamped_silu", "swiglu_oai": "swigluoai"}


def _mesh_mapper(mesh, dims):
    """dims = (tensor dim over mesh rows, tensor dim over mesh columns); None replicates."""
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _weight(mesh, t, dtype, dims):
    return ttnn.from_torch(
        t.contiguous().float(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mesh_mapper(mesh, dims),
    )


def _cfg(fidelity):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _free_partial_l1(roots, tb):
    """Force-free the L1 tensors of a build that failed part way: those of roots and of the partly built objects on its
    traceback (their constructors' self) - the moe_block.tt objects, walked through their attributes (programs it
    cached may still hold them: the ops the fallback runs need that L1)."""
    while tb is not None:
        roots.append(tb.tb_frame.f_locals.get("self"))
        tb = tb.tb_next
    _free_device_tensors(roots, ("models.demos.moe_block.tt",), l1_only=True)


def l1_tensors(roots, modules=("models.demos", "ttnn.bringup")):
    """The allocated L1 device tensors reachable from roots (as _free_device_tensors walks them) -> [(path, tensor)]."""
    seen, out, stack = set(), [], [(r, n) for r, n in roots if r is not None]
    ours = lambda o: (getattr(o, "__module__", None) or "").startswith(modules)  # noqa: E731
    while stack:
        o, path = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, ttnn.Tensor):
            if o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
                if o.memory_config().buffer_type in (ttnn.BufferType.L1, ttnn.BufferType.L1_SMALL):
                    out.append((path, o))
        elif isinstance(o, (list, tuple, set)):
            stack.extend((v, f"{path}[{i}]") for i, v in enumerate(o))
        elif isinstance(o, dict):
            stack.extend((v, f"{path}[{k!r}]") for k, v in o.items())
        elif callable(o) and hasattr(o, "__code__") and ours(o):
            for c in o.__closure__ or ():
                try:
                    stack.append((c.cell_contents, f"{path}<closure>"))
                except ValueError:
                    pass
            stack.extend((v, f"{path}.{k}") for k, v in vars(o).items())
        elif ours(type(o)) and hasattr(o, "__dict__"):
            stack.extend((v, f"{path}.{k}") for k, v in vars(o).items())
    return out


def _free_device_tensors(roots, modules, l1_only):
    """Force-free the device tensors reachable from roots through objects (and closures) of the given modules."""
    seen, stack = set(), [r for r in roots if r is not None]
    ours = lambda o: (getattr(o, "__module__", None) or "").startswith(modules)  # noqa: E731
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, ttnn.Tensor):
            dev = o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated()
            if dev and (not l1_only or o.memory_config().buffer_type in (ttnn.BufferType.L1, ttnn.BufferType.L1_SMALL)):
                ttnn.deallocate(o, force=True)
        elif isinstance(o, (list, tuple, set)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif callable(o) and hasattr(o, "__code__") and ours(o):  # (a closure: its cells, its attributes)
            for c in o.__closure__ or ():
                try:
                    stack.append(c.cell_contents)
                except ValueError:  # (an empty cell)
                    pass
            stack.extend(vars(o).values())
        elif ours(type(o)) and hasattr(o, "__dict__"):
            stack.extend(vars(o).values())


def _l1_hole(mesh, nbytes):
    """An L1 placeholder of nbytes on every core (freed later: the next allocations of up to that size land there)."""
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([grid.x * grid.y, nbytes // 4]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [1, nbytes // 4], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


class FusedHead:
    """The head as one program a chip (tt/fused_head.py HeadSE3): the x all-gather over TP straight into the shared
    expert's engine, its down + reduce-scatter, the router (logits, top-K, the joint routing rows), the routing exchange
    + the combine planner (into the exchange's plan tensors), the dispatch prep (each chip's own rows into the dispatch
    buffer, its route words) and the latent down + all-to-all. Its arena and the exchange's share the L1 left: the block
    frees it after the head runs."""

    HOLE = 24 * 1024  # (L1 a core kept above the head's arena for what the exchange keeps: its words, ~17 KB)
    HOLE_HEAD = 8 * 1024  # (two, for what the head keeps after placing its arena, and again: ~5 KB, ~1 KB)

    @staticmethod
    def fits(g: Geometry, model: MoEModel, dtypes: MoEDtypes):
        """True, or why not."""
        act = model.shared_expert_activation
        from models.demos.moe_block.tt.head_plan import HeadShape, head_layouts

        if model.shared_expert_hidden_dim is None:
            return "no shared expert"
        shape = HeadShape(
            rows=g.tokens_per_row,
            H=model.emb_dim,
            I=model.shared_expert_hidden_dim // g.tp,
            G=g.tp,
            E=model.num_experts * model.router.fold,
            K=model.top_k,
            latent=model.latent_dim,
            fold=model.router.fold,
            links=g.links,
            ring=g.tp_wraps,
        )
        if not head_layouts(shape):
            return f"{g.tokens_per_row} rows a mesh row: no engine layout (row chunks of whole owner blocks)"
        if act.kind == "situ" and (act.beta, act.linear_beta) != (4.0, 25.0):
            return "situ: only Kimi's (beta 4, linear beta 25)"
        if act.kind == "clamped_silu" and act.limit != 10.0:
            return "clamped silu: only limit 10 (the engine's compile-time limit)"
        if act.kind not in ("silu", "situ", "clamped_silu"):
            return f"shared activation {act.kind}"
        rc = model.router
        if rc.score_func not in ("sigmoid", "sqrtsoftplus") or not rc.normalize_topk or rc.expert_groups != 1:
            return "router: sigmoid / sqrtsoftplus scores, normalized top-K, no expert groups"
        # (lines or rings on either axis - the gather's / RS's / routing exchange's ring schedules; 2 links: the
        # exchange's ports and the tail's K halves are a link each)
        if g.links != 2:
            return f"{g.links} links (built for 2)"
        # (LoFi: the engine's own fidelity - DeepSeek-V4.1's choice; the router's logits stay HiFi3 either way)
        fids = (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4)
        if dtypes.shared_weights != ttnn.bfloat8_b or dtypes.shared_fidelity not in fids:
            return "shared expert: bf8 weights at LoFi / HiFi2 / 3 / 4"
        return True

    def __init__(
        self,
        mesh,
        g: Geometry,
        model: MoEModel,
        w: MoEWeights,
        l1_hole=None,
        deterministic=True,
        accumulators=12,
        router="auto",
        layout=None,
        logits_first=None,
        router_split=False,
        fidelity=ttnn.MathFidelity.HiFi2,  # (the shared expert's: MoEDtypes.shared_fidelity)
    ):
        from models.demos.moe_block.tt.fused_head import HeadSE3, RouterSpec, RoutingSpec

        self.g, self.model = g, model
        H, Is, latent = model.emb_dim, model.shared_expert_hidden_dim, model.latent_dim
        E, K, N = model.num_experts, model.top_k, g.chips
        grid = mesh.compute_with_storage_grid_size()
        if layout is not None:  # (a given layout: its logits placement too)
            router = "engine" if layout.on_engine else "router_cores"
        split = bool(layout.split) if layout is not None else router_split
        self.head = HeadSE3(
            mesh,
            # (the engine's columns: the last for the movers - a harvested grid has fewer; the router's split takes
            # the one left of them)
            GX=min(11, grid.x - 1) - (1 if split and layout is not None else 0),  # (planned: the planner's GX)
            mover_col=min(11, grid.x - 1),
            GY=layout.GY if layout is not None else None,
            MT=layout.MT if layout is not None else None,
            KB=layout.KB if layout is not None else None,
            compact=layout.compact if layout is not None else None,
            rows=g.tokens_per_row,
            H=H,
            I=Is // g.tp,
            glu=model.shared_expert_activation.kind,
            fidelity=fidelity,
            num_links=g.links,
            rs_dim=-1 if latent else -2,  # (K3: the latent up lands the width split, the shared output too)
            rs_mode="fused",
            # (deterministic: the router fin's sum in a fixed order, the predicted landing order)
            router=RouterSpec(
                E=E * model.router.fold,
                K=K,
                fold=model.router.fold,
                route_scale=model.router.route_scale,
                logits_first=logits_first,
                ordered=deterministic,
                split=split,
                on_engine={"auto": None, "engine": True, "router_cores": False}[router],
                score=model.router.score_func,
            ),
            latent=latent,
            # (the planner's row capacity: the head's (2 S K + 32 EPC: the router cores' L1), within the exchange's
            # worst case - its plan tensors take a prefix)
            routing=RoutingSpec(A=accumulators, torus=(g.tp_wraps, g.sp_wraps)),
            l1_hole=l1_hole,
        )
        self.w, rw = self.layer_weights(w)
        # (the programs bake x's address: one x buffer - a new x tensor each call rebuilt every chip's program, ~175 ms of
        # host; patching its address into the cached programs' args instead costs ~3 ms a call)
        self.x = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, g.tokens_per_row, H // g.tp]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self.head.rw = rw

    def layer_weights(self, w: MoEWeights):
        """A layer's shared expert (and latent down) and router laid out for this head: (engine weights, router
        weights). The shared expert TP-split: chip column c holds I_s slice c (gate / up columns, down rows)."""
        from ttnn.bringup.combine_rf.lin2d import glu_weight

        g, model = self.g, self.model
        Is, N = model.shared_expert_hidden_dim, g.chips
        gate, up = split_gate_up(w.shared["gate_up"].float())
        Ic = Is // g.tp
        gu, wd = [], []
        for d in range(N):
            c = d % g.tp
            sl = slice(c * Ic, (c + 1) * Ic)
            gu.append(glu_weight(gate[sl].T.contiguous(), up[sl].T.contiguous()))
            wd.append(w.shared["down"].float()[:, sl].T.contiguous())
        wl = [w.latent["down"].float()] * N if model.latent_dim else None
        w_dev = self.head.upload(gu, wd, w_latent=wl)
        return w_dev, self.head.router.upload(w.router["weight"].float(), w.router["bias"].float())

    def describe(self):
        """The resolved head: engine layout, chunks, the router's placement."""
        h = self.head
        GY, MT, NCH = h._lay
        on = h.router is not None and h.router.on_engine
        pred = f", KB {h.eng.KB}, predicted {h.planned.t:.0f} us" if h.planned is not None else f", KB {h.eng.KB}"
        return f" (engine {h.eng.GX}x{GY} cores, MT {MT}, {NCH} chunk{'s' if NCH > 1 else ''}, router {'on the engine' if on else f'on {len(h.router.cores)} router cores'}{pred})"

    def attach(self, exchange):
        """The planner writes the exchange's plan tensors (its combine baked their addresses)."""
        blk = exchange.block
        assert (
            blk.A == self.head.routing.A
        ), f"the combine's accumulators {blk.A} != the fused head planner's {self.head.routing.A}"
        assert self.head.routing.rows_cap <= blk.rows_cap
        for k, t in self.head.routing.outs.items():
            ttnn.deallocate(t)
        self.head.routing.outs = {k: blk.planner.outs[k] for k in ("counts", "regions", "token_index", "table", "plan")}
        self.head.state["key"] = None  # (the programs take the new tensors)

    def __call__(self, x, arena=None):
        """x [1, 1, S/SP, H/TP] -> (the shared expert's output: [S/N, H] the owners' rows, or (latent) [S/SP, H/TP];
        the outputs: rows, xbuf, route (and latent)). The arena's L1 given back after. arena: (an L1 tensor holding the
        head's arena range, its bytes a core) - the block's one arena a forward: run in it, the caller frees it."""
        if int(x.buffer_address()) != int(self.x.buffer_address()):  # (MoEBlock.input_buffer(): read in place)
            ttnn.copy(x, self.x)
        if arena is not None:
            self.head.use_arena(*arena)
        se = self.head.run(self.x, self.w)
        out = dict(self.head.outputs)
        if arena is None:
            self.head.free_arena()
        return se, out


class FusedTail:
    """The tail as one program a chip. K3 (a latent - tt/tail_engine.py TailEngine): the latent all-gather over TP into
    the engine's L1, the combine's own rows tilized there, rms norm (folded: g into W_up, 1 / rms on the fp32 sums), the
    up projection split by columns, + the shared expert's output -> [S/SP, H/TP]. No latent (tt/tail_a2a.py TailA2A):
    the combine's own rows tilized, + the shared expert's owner rows, the TP all-to-all straight into [S/SP, H/TP]."""

    @staticmethod
    def fits(g: Geometry, model: MoEModel, head_mode):
        """True, or why not. Either head: both land the shared expert's output width split ([S/SP, H/TP]) with a latent,
        its owner rows ([S/N, H]) without."""
        if model.latent_dim is None:
            if (model.emb_dim // g.tp) % 32 or g.tokens_per_chip % 32:
                return "whole tiles of H / TP and of a chip's owned rows"
            return True
        if not model.latent_norm:  # ("auto" takes the ops tail, which skips the norm; the summary line says why)
            return "latent_norm off (the fused tail always rms-normalizes the latent)"
        if g.links != 2:
            return f"{g.links} links (a link a K half)"
        if (model.emb_dim // g.tp) % 32 or model.latent_dim % 64:
            return "whole tiles of H / TP and of each latent K half"
        return True

    def __init__(self, mesh, g: Geometry, model: MoEModel, w: MoEWeights, topology, dtypes: MoEDtypes = MoEDtypes()):
        from models.demos.moe_block.tt.tail_engine import TailEngine

        self.a2a = None
        if model.latent_dim is None:
            from models.demos.moe_block.tt.tail_a2a import TailA2A

            self.a2a = TailA2A(mesh, rows=g.tokens_per_row, H=model.emb_dim, topology=topology, num_links=g.links)
            return
        H, Hr, tp = model.emb_dim, model.latent_dim, g.tp
        self.eng = TailEngine(
            mesh,
            rows=g.tokens_per_row,
            Hr=Hr,
            Ho=H // tp,
            chunks=tp,
            eps=model.norm_eps,
            topology=topology,
            num_links=g.links,
            fidelity=dtypes.tail_fidelity,
        )
        self.g, self.model = g, model
        self.w = self.layer_weights(w)

    def layer_weights(self, w: MoEWeights):
        """A layer's latent up (split by columns over TP) and norm, uploaded in the engine's layout."""
        g, model = self.g, self.model
        H, Hr, tp = model.emb_dim, model.latent_dim, g.tp
        up = w.latent["up"].float()  # [H, Hr]
        Ho = H // tp
        assert model.latent_norm, "the fused tail rms-normalizes the latent (FusedTail.fits)"
        gn = w.latent["norm"].float()
        return self.eng.upload([up.T[:, (d % tp) * Ho : (d % tp + 1) * Ho].contiguous() for d in range(g.chips)], gn)

    def __call__(self, y_rm, se):
        """y_rm: the combine's row-major output [1, 1, S/N, Hr]; se [1, 1, S/SP, H/TP] TILE (no latent: [1, 1, S/N, H]
        or None) -> [1, 1, S/SP, H/TP]."""
        if self.a2a is not None:
            return self.a2a.run(y_rm, se)
        return self.eng.run(y_rm, self.w, se)


class Router:
    """Gate weight split along H over TP ([H/TP, E] a chip, replicated over SP); fp32 partial logits, reduce-scattered
    by tokens to the owners (the block does the collective), then the selection once per token on its owner
    (combine_rf/router_topk.py: sigmoid + correction bias selects, the picked scores renormalized x route_scale,
    written straight as the joint routing rows the planner and the routing gather read)."""

    def __init__(self, mesh, g: Geometry, model: MoEModel, w: dict, dtypes: MoEDtypes):
        from ttnn.bringup.combine_rf.router_topk import router_topk_prepare

        r = model.router
        if not (r.score_func in ("sigmoid", "sqrtsoftplus") and r.normalize_topk and r.expert_groups == 1):
            raise NotImplementedError(
                f"MoEBlock router: sigmoid / sqrtsoftplus + bias, normalized, one group (got {r})"
            )
        self.score_func = r.score_func
        E = model.num_experts * r.fold  # (the candidates: a fold picks over E x fold, candidate c is expert c // fold)
        self.w = _weight(mesh, w["weight"].T, dtypes.router_weights, (None, 0))  # [H, E], H over the columns
        self.cfg = _cfg(ttnn.MathFidelity.HiFi4)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        bias = w["bias"].view(1, E).float()
        self.bias32 = ttnn.from_torch(
            bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )  # (the biased scores in one add)
        self.bias_row = ttnn.from_torch(
            bias, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep
        )  # (each pick's score = biased - bias)
        self.S = g.tokens_per_chip
        self.rtopk = router_topk_prepare(
            mesh, S=self.S, E=E, K=model.top_k, route_scale=r.route_scale, bias_mode=True, fold=r.fold
        )

    def at(self, mesh, g: Geometry, model: MoEModel):
        """This router at another chunk: its weights shared, its top-k prepared for the chunk's owned rows."""
        import copy

        from ttnn.bringup.combine_rf.router_topk import router_topk_prepare

        r = model.router
        out = copy.copy(self)
        out.S = g.tokens_per_chip
        out.rtopk = router_topk_prepare(
            mesh,
            S=out.S,
            E=model.num_experts * r.fold,
            K=model.top_k,
            route_scale=r.route_scale,
            bias_mode=True,
            fold=r.fold,
        )
        return out

    def logits(self, x):
        """[1, 1, S/SP, H/TP] -> partial logits [1, 1, S/SP, E] fp32."""
        return ttnn.linear(x, self.w, compute_kernel_config=self.cfg, dtype=ttnn.float32)

    def top_k(self, logits):
        """the owners' complete logits [1, 1, S/N, E] -> joint routing rows [S/N / 32, 32 x 2K] uint16."""
        lg = ttnn.reshape(logits, [self.S, int(logits.shape[-1])])
        # (the biased scores in L1: the top-k's 64 B reads at L1 latency; freed before the exchange's arena)
        if self.score_func == "sqrtsoftplus":  # (DeepSeek V4: sqrt(softplus(logits)), fp32)
            sp = ttnn.softplus(lg, memory_config=ttnn.L1_MEMORY_CONFIG)
            b = ttnn.add(
                sp,
                self.bias32,
                input_tensor_a_activations=[ttnn.UnaryOpType.SQRT],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(sp)
        else:
            b = ttnn.add(
                lg,
                self.bias32,
                input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        rows = self.rtopk(self.bias_row, b)
        ttnn.deallocate(b)
        return rows


class SharedExpert:
    """TP-split: gate / up split by columns over I_s (whole gate / up pairs a chip), down split by rows; the output is
    a partial [S/SP, H] the block reduce-scatters over TP. mm2d program configs (glm53_flash_d_p mm_configs); the
    activation as one op where ttnn has it: SiLU -> one gate / up matmul ([up | gate] columns a chip) + ttnn.swiglu
    (first half x silu(second half)); SiTU -> ttnn.situ_glu on the two matmuls' outputs; others elementwise."""

    def __init__(self, mesh, g: Geometry, model: MoEModel, w: dict, dtypes: MoEDtypes):
        gate, up = split_gate_up(w["gate_up"])  # [I_s, H] each
        dt = dtypes.shared_weights
        self.act = model.shared_expert_activation
        self.fused = self.act.kind == "silu"
        if self.fused:  # chip column c: [H, 2 I_s / TP] = [up slice c | gate slice c]
            Is, TP = gate.shape[0], g.tp
            sl = Is // TP
            cat = torch.cat(
                [torch.cat([up[c * sl : (c + 1) * sl], gate[c * sl : (c + 1) * sl]], dim=0) for c in range(TP)], dim=0
            )  # [TP 2 sl, H]
            self.wgu = _weight(mesh, cat.T, dt, (None, 1))  # [H, 2 I_s / TP]
        else:
            self.wg = _weight(mesh, gate.T, dt, (None, 1))  # [H, I_s / TP]
            self.wu = _weight(mesh, up.T, dt, (None, 1))
        self.wd = _weight(mesh, w["down"].T, dt, (None, 0))  # [I_s / TP, H]
        self.cfg = _cfg(dtypes.shared_fidelity)
        self.adt = dtypes.shared_activations

    def _lin(self, a, wt, dtype):
        import models.demos.glm53_flash_d_p.tt.mm_configs as mmc

        return ttnn.linear(
            a, wt, compute_kernel_config=self.cfg, dtype=dtype, program_config=mmc.linear_config(a, wt, dtype)
        )

    def __call__(self, x_full):
        if self.fused:
            gu = self._lin(x_full, self.wgu, self.adt)
            a = ttnn.swiglu(gu, dim=-1)
            ttnn.deallocate(gu)
        else:
            g_ = self._lin(x_full, self.wg, self.adt)
            u_ = self._lin(x_full, self.wu, self.adt)
            if self.act.kind == "situ":
                a = ttnn.situ_glu(g_, u_, self.act.beta, self.act.linear_beta)
            else:
                a = _act_dev(self.act, g_, u_)
            ttnn.deallocate(g_)
            ttnn.deallocate(u_)
        out = self._lin(a, self.wd, ttnn.bfloat16)
        ttnn.deallocate(a)
        return out


class LatentProjections:
    """Down split by columns (each chip a quarter of the latent width for every row token: the owners then get their
    rows by an all-to-all), up split by columns after the latent all-gather and norm (lands the width split, no
    reduce-scatter)."""

    def __init__(self, mesh, g: Geometry, model: MoEModel, w: dict, dtypes: MoEDtypes):
        dt = dtypes.projection_weights
        self.wd = _weight(mesh, w["down"].T, dt, (None, 1))  # [H, Hr / TP]
        self.wu = _weight(mesh, w["up"].T, dt, (None, 1))  # [Hr, H / TP]
        Hr = model.latent_dim
        self.norm = None
        if model.latent_norm:
            self.norm = ttnn.from_torch(
                w["norm"].float().view(1, 1, Hr // 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        self.eps = model.norm_eps
        self.cfg = _cfg(ttnn.MathFidelity.HiFi2)

    def down(self, x_full):
        return ttnn.linear(x_full, self.wd, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)

    def up(self, y_row):
        y = y_row
        if self.norm is not None:
            y = ttnn.rms_norm(y_row, epsilon=self.eps, weight=self.norm, compute_kernel_config=self.cfg)
        out = ttnn.linear(y, self.wu, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
        if y is not y_row:
            ttnn.deallocate(y)
        return out


class RoutedExperts:
    """The flat routed expert op with this layer's weights: experts_per_chip whole experts a chip (global expert e on
    chip e // experts_per_chip, row-major chip order), laid out by the close-out plan (pure DRISC, even splits,
    rows_hint = chunk x top_k / experts). Experts that are the same host tensors on every chip are packed once."""

    @staticmethod
    def gids_of(g: Geometry):
        EPC = g.experts_per_chip
        return [list(range(c * EPC, (c + 1) * EPC)) for c in range(g.chips)]

    @staticmethod
    def cap_of(
        g: Geometry,
    ):  # (an expert sees each token at most once; the flat op is built for >= 256 tokens an expert)
        return max(g.chunk_tokens, 256)

    @staticmethod
    def layout_of(g: Geometry, model: MoEModel, drisc=_DRISC):
        # (one x helper: a second one costs ~104 KB of every core's arena, which the exchange's combine needs at the
        # LoudBox's per-chip load - flat_layouts.GLM_FLASH)
        lay = dict(rows_hint=round(g.chunk_tokens * model.top_k / model.num_experts), x_helpers=1)
        if not drisc:  # (MoEOptions.experts_drisc off: Tensix weight readers)
            lay.update(drisc=False, drisc_in_overlap=False)
        return lay

    def __init__(self, mesh, g: Geometry, model: MoEModel, routed: list | None, dtypes: MoEDtypes, drisc=_DRISC, op=None):
        """op: a FlatRoutedExpert the caller laid out (gids_of, cap_of, layout_of; its weight cache); routed None."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        assert dtypes.routed_weights == ttnn.bfloat4_b, "bring-up: bfp4 routed weights"
        self.gids = self.gids_of(g)
        self.cap = self.cap_of(g)
        self.layout_fields = self.layout_of(g, model, drisc)
        self.act = _FLAT_ACT[model.routed_expert_activation.kind]
        if op is not None:
            self.op = op
            return
        conv = {}

        def flat_w(d):  # contract {"gate_up" [2I, Hr], "down" [Hr, I]} -> (Wg [Hr, I], Wu [Hr, I], Wd [I, Hr])
            k = id(d)
            if k not in conv:
                gate, up = split_gate_up(d["gate_up"])
                conv[k] = (gate.T.contiguous().float(), up.T.contiguous().float(), d["down"].T.contiguous().float())
            return conv[k]

        per_dev = [[flat_w(routed[e]) for e in gl] for gl in self.gids]
        self.op = FlatRoutedExpert(
            mesh,
            per_dev,
            m=self.cap,
            H=model.routed_width,
            I=model.routed_expert_hidden_dim,
            gids=self.gids,
            n_global=model.num_experts,
            wdtype="bf4",
            act=self.act,
            pin=1,
            layout=self.layout_fields,
        )
        del per_dev, conv


class TokenExchange:
    """Routing gather, planner, dispatch prep, 2D sparse dispatch and the fused flat experts + combine: the K3
    token-split block's routed-only path (models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py) on the
    owners' rows. Built by the block after its head and tail ran once (the arena's address is baked into its kernels),
    rebuilt (with a warning) only if L1 allocated since moved the arena."""

    def __init__(
        self,
        mesh,
        g: Geometry,
        model: MoEModel,
        experts: RoutedExperts,
        dispatch_in_program=False,
        accumulators=None,
        combine_snap=None,
    ):
        self.mesh, self.g, self.model, self.experts = mesh, g, model, experts
        self.combine_snap = combine_snap
        self.dispatch_in_program = dispatch_in_program
        self.accumulators = accumulators  # (the combine's, planned with the fused head; None: its own choice)
        self.block = None
        self.builds = 0

    def build(self):
        from ttnn.bringup.combine_rf.routing_a2a import worst_rows

        from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import K3Dims, TtK3TokenSplitMoE

        g, m = self.g, self.model
        Hr, S, EPC = m.routed_width, g.tokens_per_chip, g.experts_per_chip
        d = K3Dims(H=Hr, latent=Hr, I=m.routed_expert_hidden_dim, E=m.num_experts, K=m.top_k, act=self.experts.act)
        self.block = TtK3TokenSplitMoE(
            self.mesh,
            None,
            d,
            S,
            cap=self.experts.cap,
            rows_cap=worst_rows(g.chips, S, m.top_k, EPC),  # (every row a chip can get: packed routing too)
            routed_only=True,
            flat_layout=self.experts.layout_fields,
            torus=(g.tp_wraps, g.sp_wraps),
            joint_routing=True,
            combine_auto_cfg=True,
            combine_accumulators=self.accumulators,
            dispatch_in_program=self.dispatch_in_program,
            combine_snap=self.combine_snap,
        )
        self.builds += 1

    def __call__(self, own, rows, hook=None, tilize=True):
        """own: the owners' rows [1, 1, S/N, Hr] TILE; rows: the joint routing rows -> [S/N, Hr] TILE (tilize False: the
        combine's row-major output, persistent)."""
        if self.block is None or not self.block.arena_fits():
            if self.block is not None:
                from loguru import logger

                logger.warning("MoEBlock exchange: L1 moved the arena, rebuilding")
            self.build()
        x2 = ttnn.reshape(own, [self.g.tokens_per_chip, self.model.routed_width])
        return self.block.routed(x2, rows, None, self.experts.op, hook=hook, tilize=tilize)

    def take_arena(self, inside=None):
        """The forward's one L1 allocation (MOE_ONE_ARENA=0 turns it off): the exchange's arena taken ahead for the fused
        head to run in first -> (tensor, bytes a core), or None - the in-program dispatch (its own), or the head's arena
        range inside=(base, bytes) not within the exchange's (a head larger than the exchange's arena: K2.7, K3)."""
        if self.dispatch_in_program or os.environ.get("MOE_ONE_ARENA", "1") == "0":
            return None
        a, n = self.block.arena_addr, self.block.arena_bytes
        if inside is not None and not (a <= inside[0] and inside[0] + inside[1] <= a + n):
            if not getattr(self, "_two_logged", False):
                from loguru import logger

                logger.info(f"MoEBlock: two arenas a forward (the head's [{inside[0]}, +{inside[1]}) not in [{a}, +{n})")
                self._two_logged = True
            return None
        if inside is not None and not getattr(self, "_one_logged", False):
            from loguru import logger

            logger.info(f"MoEBlock: one arena a forward (the head's [{inside[0]}, +{inside[1]}) in [{a}, +{n}))")
            self._one_logged = True
        return self.block.take_arena(), n

    def prepped(self, prep, hook=None, tilize=True):
        """The fused head did the routing gather, the planner (into this block's plan tensors) and the dispatch prep:
        prep = (the dispatch buffer with each chip's own rows, the route words) -> [S/N, Hr] TILE (tilize False: the
        combine's row-major output, persistent). The arena's address is checked where it is taken (take_arena /
        alloc_arena), not by allocating it once more here."""
        return self.block.routed_prepped(prep, self.experts.op, hook=hook, tilize=tilize)


# ---------------------------------------------------------------------------------------------------------------------
# The models (reference configs: models/demos/deepseek_v3_d_p/reference/*_config.py)
# ---------------------------------------------------------------------------------------------------------------------

KIMI_K27 = MoEModel(
    num_experts=384,
    top_k=8,
    emb_dim=7168,
    routed_expert_hidden_dim=2048,
    routed_expert_activation=ExpertActivation("silu"),
    router=RouterConfig(route_scale=2.827),
    shared_expert_hidden_dim=2048,
    shared_expert_activation=ExpertActivation("silu"),
)
GLM_53 = MoEModel(
    num_experts=256,
    top_k=8,
    emb_dim=6144,
    routed_expert_hidden_dim=2048,
    routed_expert_activation=ExpertActivation("silu"),
    router=RouterConfig(route_scale=2.5),
    shared_expert_hidden_dim=2048,
    shared_expert_activation=ExpertActivation("silu"),
)
GLM_53_FLASH = MoEModel(  # (swiglu_limit 10: silu(min(gate, 10)) x clamp(up, +-10), routed and shared)
    num_experts=288,
    top_k=8,
    emb_dim=4096,
    routed_expert_hidden_dim=2048,
    routed_expert_activation=ExpertActivation("clamped_silu", limit=10.0),
    router=RouterConfig(route_scale=2.5),
    shared_expert_hidden_dim=2048,
    shared_expert_activation=ExpertActivation("clamped_silu", limit=10.0),
)
MINIMAX_M3 = MoEModel(
    num_experts=128,
    top_k=4,
    emb_dim=6144,
    routed_expert_hidden_dim=3072,
    routed_expert_activation=ExpertActivation("swiglu_oai", limit=7.0, alpha=1.702),
    router=RouterConfig(route_scale=2.0),
    shared_expert_hidden_dim=3072,
    shared_expert_activation=ExpertActivation("swiglu_oai", limit=7.0, alpha=1.702),
)
KIMI_K3 = MoEModel(
    num_experts=896,
    top_k=16,
    emb_dim=7168,
    routed_expert_hidden_dim=3072,
    routed_expert_activation=ExpertActivation("situ", beta=4.0, linear_beta=25.0),
    router=RouterConfig(route_scale=1.0),
    shared_expert_hidden_dim=6144,
    shared_expert_activation=ExpertActivation("situ", beta=4.0, linear_beta=25.0),
    latent_dim=3584,
    latent_norm=True,
)
# Router and activation as DSv4-Flash's; confirm against the DSv4.1 checkpoint. Hash-routed layers are out of scope.
DSV41_FLASH = MoEModel(
    num_experts=384,
    top_k=6,
    emb_dim=5120,
    routed_expert_hidden_dim=2304,
    routed_expert_activation=ExpertActivation("clamped_silu", limit=10.0),
    router=RouterConfig(score_func="sqrtsoftplus", route_scale=1.5),
    shared_expert_hidden_dim=2304,
    shared_expert_activation=ExpertActivation("clamped_silu", limit=10.0),
    norm_eps=1e-6,
)

MODELS = {
    "glm53": GLM_53,
    "glm53_flash": GLM_53_FLASH,
    "kimi_k3": KIMI_K3,
    "kimi_k27": KIMI_K27,
    "minimax_m3": MINIMAX_M3,
    "dsv41_flash": DSV41_FLASH,
}
