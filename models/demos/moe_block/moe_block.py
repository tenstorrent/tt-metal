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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

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
    shared_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi3
    projection_weights: ttnn.DataType = ttnn.bfloat8_b  # latent down / up
    router_weights: ttnn.DataType = ttnn.bfloat16  # top-k selection in fp32
    dispatch_rows: WireFormat = field(default_factory=WireFormat)
    combine_rows: WireFormat = field(default_factory=WireFormat)


@dataclass(frozen=True)
class MoEOptions:
    shared_expert: Literal["tp_split", "fold_into_combine"] = "tp_split"  # never replicated over TP
    experts_during_dispatch: bool = False  # weights stream from the dispatch start
    deterministic: bool = False  # fixed sum order, bit-identical runs (debug)


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
    routed: list[dict]  # E dicts by global expert id: {"gate_up": [2 I, Hr], "down": [Hr, I]}
    shared: dict | None  # {"gate_up": [2 Is, H], "down": [H, Is]}
    latent: dict | None  # {"down": [Hr, H], "up": [H, Hr], "norm": [Hr]}

    def check(self, model: MoEModel):
        """The contract: every key and shape. fp32 or bf16 on the host; the block quantizes."""
        E, H, I, Hr = model.num_experts, model.emb_dim, model.routed_expert_hidden_dim, model.routed_width
        assert self.router["weight"].shape == (E, H), f"router weight {tuple(self.router['weight'].shape)}"
        assert self.router["bias"].shape == (E,)
        assert len(self.routed) == E, f"{len(self.routed)} routed experts, the model has {E}"
        for e, w in enumerate(self.routed):
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
        for w in self.routed:
            yield from w.values()
        for d in (self.shared, self.latent):
            if d is not None:
                yield from d.values()


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
        weights: MoEWeights | None,  # None: load everything from the cache
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
        if weights is None:
            assert cache.path is not None, "no weights and no cache to load them from"
            raise NotImplementedError("MoEBlock: loading from the weight cache is not built yet; pass weights")
        weights.check(model)
        # bring-up limits (each lifted by the piece that needs it)
        assert options.shared_expert == "tp_split", "bring-up: the shared expert TP-split, reduce-scattered in the head"
        assert not options.experts_during_dispatch, "bring-up: the experts start after the dispatch"
        assert not sequence_zigzag, "bring-up: natural sequence order"
        self.router = Router(mesh_device, g, model, weights.router, dtypes)
        self.shared = None
        if model.shared_expert_hidden_dim is not None:
            self.shared = SharedExpert(mesh_device, g, model, weights.shared, dtypes)
        self.latent = None
        if model.latent_dim is not None:
            self.latent = LatentProjections(mesh_device, g, model, weights.latent, dtypes)
        self.experts = RoutedExperts(mesh_device, g, model, weights.routed, dtypes)
        self.exchange = TokenExchange(mesh_device, g, model, self.experts)
        self.stage_hook = None  # (profiling: called with the stage name after every stage)
        self.keep_taps = False  # (debug: keep the last call's routing rows for taps())
        self._taps = {}
        # The exchange's arena address is baked into its kernels: run the head and the tail once first (their ops'
        # one-time L1 - collective semaphores, program state - lands now), then build the exchange on the L1 left.
        self._warm_up()
        self.exchange.build()
        from loguru import logger

        logger.info(
            f"MoEBlock: mesh {g.sp}x{g.tp}, chunk {g.chunk_tokens} ({g.tokens_per_row} rows a mesh row, "
            f"{g.tokens_per_chip} owned a chip), {g.experts_per_chip} experts a chip, routed width {model.routed_width}; "
            f"flat layout {self.experts.layout_fields}; dtypes {dtypes}; options {options}"
        )

    def _mark(self, name):
        if self.stage_hook is not None:
            self.stage_hook(name)

    def forward(self, x: ttnn.Tensor, chunk: ChunkSpan | None = None) -> ttnn.Tensor:
        """x: [1, 1, S/SP, H/TP] TILE, the residual as attention leaves it. Returns the same layout.
        chunk: which rows are real; None = all of them."""
        g, H = self.geometry, self.model.emb_dim
        assert tuple(x.shape) == (1, 1, g.tokens_per_row, H // g.tp), (
            f"x {tuple(x.shape)}, expected (1, 1, {g.tokens_per_row}, {H // g.tp}) on each chip "
            f"(chunk {g.chunk_tokens} over SP {g.sp}, H {H} over TP {g.tp})"
        )
        assert x.layout == ttnn.TILE_LAYOUT
        if chunk is not None:
            raise NotImplementedError("MoEBlock: padded chunks (ChunkSpan) are not built yet")
        rows, own, se = self._head(x)
        # ---- exchange ----
        if self.keep_taps:  # (host copies: the exchange consumes the routing rows)
            self._taps = dict(rows=[ttnn.to_torch(t) for t in ttnn.get_device_tensors(rows)])
        y_own = self.exchange(own, rows, hook=self.stage_hook)  # owners' sums [S/N, Hr]
        ttnn.deallocate(own)
        if rows.is_allocated():
            ttnn.deallocate(rows)
        return self._tail(y_own, se)

    def _warm_up(self):
        """The head and the tail once on zeros (the exchange skipped): every op they run allocates its one-time L1."""
        g, m = self.geometry, self.model
        mesh = self.mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)
        z = lambda shape: ttnn.from_torch(  # noqa: E731
            torch.zeros(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        x0 = z((1, 1, g.tokens_per_row, m.emb_dim // g.tp))
        rows, own, se = self._head(x0, mark=False)
        ttnn.deallocate(x0)
        ttnn.deallocate(own)
        ttnn.deallocate(rows)
        out = self._tail(z((g.tokens_per_chip, m.routed_width)), se, mark=False)
        ttnn.deallocate(out)
        ttnn.synchronize_device(mesh)

    def _head(self, x, mark=True):
        """-> (the joint routing rows, the owners' rows [1, 1, S/N, Hr], the shared expert's [S/SP, H/TP] or None)."""
        mark = self._mark if mark else (lambda _: None)
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

    def _tail(self, y_own, se, mark=True):
        """owners' sums [S/N, Hr] (+ the shared expert's output) -> [1, 1, S/SP, H/TP]."""
        mark = self._mark if mark else (lambda _: None)
        g = self.geometry
        y_own = ttnn.reshape(y_own, [1, 1, g.tokens_per_chip, self.model.routed_width])
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
                ttnn.deallocate(se)
                out = y
                mark("add")
            return out
        if se is not None:  # (the shared expert's owner rows, added before the all-to-all carries both)
            y = ttnn.add(y_own, se)
            ttnn.deallocate(y_own)
            ttnn.deallocate(se)
            y_own = y
            mark("add")
        out = self._tp_a2a(y_own, cat_dim=2, split_dim=3)  # back to the width split [S/SP, H/TP]
        ttnn.deallocate(y_own)
        mark("tail_a2a")
        return out

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
        """Debug: the last forward's routing rows (keep_taps = True)."""
        return self._taps

    def close(self):
        pass


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


class Router:
    """Gate weight split along H over TP ([H/TP, E] a chip, replicated over SP); fp32 partial logits, reduce-scattered
    by tokens to the owners (the block does the collective), then the selection once per token on its owner
    (combine_rf/router_topk.py: sigmoid + correction bias selects, the picked scores renormalized x route_scale,
    written straight as the joint routing rows the planner and the routing gather read)."""

    def __init__(self, mesh, g: Geometry, model: MoEModel, w: dict, dtypes: MoEDtypes):
        from ttnn.bringup.combine_rf.router_topk import router_topk_prepare

        r = model.router
        if not (r.score_func == "sigmoid" and r.normalize_topk and r.expert_groups == 1):
            raise NotImplementedError(f"MoEBlock router: sigmoid + bias, normalized, one group only (got {r})")
        E = model.num_experts
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
        self.rtopk = router_topk_prepare(mesh, S=self.S, E=E, K=model.top_k, route_scale=r.route_scale, bias_mode=True)

    def logits(self, x):
        """[1, 1, S/SP, H/TP] -> partial logits [1, 1, S/SP, E] fp32."""
        return ttnn.linear(x, self.w, compute_kernel_config=self.cfg, dtype=ttnn.float32)

    def top_k(self, logits):
        """the owners' complete logits [1, 1, S/N, E] -> joint routing rows [S/N / 32, 32 x 2K] uint16."""
        lg = ttnn.reshape(logits, [self.S, int(logits.shape[-1])])
        # (the biased scores in L1: the top-k's 64 B reads at L1 latency; freed before the exchange's arena)
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
    a partial [S/SP, H] the block reduce-scatters over TP."""

    def __init__(self, mesh, g: Geometry, model: MoEModel, w: dict, dtypes: MoEDtypes):
        gate, up = split_gate_up(w["gate_up"])  # [I_s, H] each
        dt = dtypes.shared_weights
        self.wg = _weight(mesh, gate.T, dt, (None, 1))  # [H, I_s / TP]
        self.wu = _weight(mesh, up.T, dt, (None, 1))
        self.wd = _weight(mesh, w["down"].T, dt, (None, 0))  # [I_s / TP, H]
        self.act = model.shared_expert_activation
        self.cfg = _cfg(dtypes.shared_fidelity)
        self.adt = dtypes.shared_activations

    def __call__(self, x_full):
        g_ = ttnn.linear(x_full, self.wg, compute_kernel_config=self.cfg, dtype=self.adt)
        u_ = ttnn.linear(x_full, self.wu, compute_kernel_config=self.cfg, dtype=self.adt)
        a = _act_dev(self.act, g_, u_)
        ttnn.deallocate(g_)
        ttnn.deallocate(u_)
        out = ttnn.linear(a, self.wd, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
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

    def __init__(self, mesh, g: Geometry, model: MoEModel, routed: list, dtypes: MoEDtypes):
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        assert dtypes.routed_weights == ttnn.bfloat4_b, "bring-up: bfp4 routed weights"
        N, EPC = g.chips, g.experts_per_chip
        self.gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
        self.cap = g.chunk_tokens  # (an expert sees each token at most once)
        # (one x helper: a second one costs ~104 KB of every core's arena, which the exchange's combine needs at the
        # LoudBox's per-chip load - flat_layouts.GLM_FLASH)
        self.layout_fields = dict(rows_hint=round(g.chunk_tokens * model.top_k / model.num_experts), x_helpers=1)
        self.act = _FLAT_ACT[model.routed_expert_activation.kind]
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

    def __init__(self, mesh, g: Geometry, model: MoEModel, experts: RoutedExperts):
        self.mesh, self.g, self.model, self.experts = mesh, g, model, experts
        self.block = None
        self.builds = 0

    def build(self):
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
            rows_cap=-(-g.chips * S * min(m.top_k, EPC) // 32) * 32 + 32 * EPC,
            routed_only=True,
            flat_layout=self.experts.layout_fields,
            torus=(g.tp_wraps, g.sp_wraps),
            joint_routing=True,
            combine_auto_cfg=True,
        )
        self.builds += 1

    def __call__(self, own, rows, hook=None):
        """own: the owners' rows [1, 1, S/N, Hr] TILE; rows: the joint routing rows -> [S/N, Hr] TILE."""
        if self.block is None or not self.block.arena_fits():
            if self.block is not None:
                from loguru import logger

                logger.warning("MoEBlock exchange: L1 moved the arena, rebuilding")
            self.build()
        x2 = ttnn.reshape(own, [self.g.tokens_per_chip, self.model.routed_width])
        return self.block.routed(x2, rows, None, self.experts.op, hook=hook)


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
GLM_53_FLASH = MoEModel(
    num_experts=288,
    top_k=8,
    emb_dim=4096,
    routed_expert_hidden_dim=2048,
    routed_expert_activation=ExpertActivation("silu"),
    router=RouterConfig(route_scale=2.5),
    shared_expert_hidden_dim=2048,
    shared_expert_activation=ExpertActivation("silu"),
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
