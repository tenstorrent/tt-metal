# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MoE block for prefill: one entry for every MoE model, from the residual as attention leaves it back to the same layout.

Design (see the "Galaxy MoE Block" page): token owners, 2D dispatch, flat experts + combine in one program. The residual
stays main's layout, sequence over SP (mesh rows) and width over TP (mesh columns). No weight matrix is duplicated over TP.

Only the entry exists so far: configuration, geometry, the weight contract and the checks. The sub-modules are not built.
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
        if weights is None:
            assert cache.path is not None, "no weights and no cache to load them from"
        else:
            weights.check(model)
        # Not built yet: Router, SharedExpert, LatentProjections, RoutedExperts (the flat plan), TokenExchange
        # (routing gather, planner, 2D dispatch, experts + combine), L1Plan.

    def forward(self, x: ttnn.Tensor, chunk: ChunkSpan | None = None) -> ttnn.Tensor:
        """x: [1, 1, S/SP, H/TP] TILE, the residual as attention leaves it. Returns the same layout.
        chunk: which rows are real; None = all of them."""
        g, H = self.geometry, self.model.emb_dim
        assert tuple(x.shape) == (1, 1, g.tokens_per_row, H // g.tp), (
            f"x {tuple(x.shape)}, expected (1, 1, {g.tokens_per_row}, {H // g.tp}) on each chip "
            f"(chunk {g.chunk_tokens} over SP {g.sp}, H {H} over TP {g.tp})"
        )
        assert x.layout == ttnn.TILE_LAYOUT
        if chunk is not None and isinstance(chunk.valid_tokens, int) and isinstance(chunk.start, int):
            assert chunk.start >= 0 and 0 < chunk.valid_tokens <= g.chunk_tokens
        raise NotImplementedError("MoEBlock.forward: entry only")

    def close(self):
        pass


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
