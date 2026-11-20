# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which MoE a prefill block builds: TtMoe or the MoE block's C++ ops (ttnn.experimental.moe_block).

The model's adapter decides (PrefillModelAdapter.moe_impl, "ttmoe" by default); a caller's ``moe_impl`` ("block" /
"ttmoe"), else $MOE_IMPL, overrides it. Everything the block needs of a model comes through its adapter: its dims
from the adapter's model_config (moe_block_dims), the non-routed weights from moe_block_nonrouted_weights, the routed
experts from the TtMoe tensorbin cache (moe_block_routed_cache, else the caller's cache_path) and the persistent words'
slab from moe_block_words_slab. Nothing here names a model.

A mesh that runs the block opens with:
- L1_SMALL: the model's L1_SMALL_SIZE, its one figure, which holds the block's persistent words' slab
  (moe_block_words_slab) on top of the model's own ops' use;
- a fabric payload that holds one combine row (moe_block_fabric_payload_size / moe_block_device_params).
Paths that do not open their mesh for it (the synthetic-weight block tests, the TtMoe cache build) pass
moe_impl="ttmoe".
"""

from __future__ import annotations

import os
from typing import Optional

import ttnn

moe_block_utils = ttnn._ttnn.operations.experimental.moe_block

BLOCK = "block"
TTMOE = "ttmoe"


def model_adapter(model):
    """The adapter of `model`: an adapter itself, else the registered adapter whose model_config is this config class
    (common/prefill/adapter.adapter_of_model_config); None for a config no adapter names (the block tests' synthetic
    configs)."""
    if model is None or hasattr(model, "model_config"):
        return model
    from models.demos.common.prefill.adapter import adapter_of_model_config

    return adapter_of_model_config(model)


def model_config_of(model):
    """The config class of `model` (an adapter or a config class)."""
    return getattr(model, "model_config", model)


def moe_block_words_slab(model=None) -> int:
    """The MoE block's persistent words' slab a core for this model: its adapter's moe_block_words_slab, else
    MoEBlockConfig.words_slab_bytes."""
    adapter = model_adapter(model)
    own = getattr(adapter, "moe_block_words_slab", None)
    return int(own) if own is not None else int(moe_block_utils.MoEBlockConfig().words_slab_bytes)


def moe_impl_of(model=None, moe_impl: Optional[str] = None) -> str:
    """ "block" or "ttmoe" for a model (its adapter, or its config class)."""
    choice = moe_impl or os.environ.get("MOE_IMPL")
    if not choice:
        choice = getattr(model_adapter(model), "moe_impl", None) or TTMOE
    assert choice in (
        BLOCK,
        TTMOE,
    ), f"MOE_IMPL / moe_impl / adapter.moe_impl must be '{BLOCK}' or '{TTMOE}', got {choice!r}"
    return choice


_SCORE_FUNCS = {"sigmoid": "Sigmoid", "sqrtsoftplus": "SqrtSoftplus"}


def _expert_activation(cfg, name: str):
    """The block's ExpertActivation for a config's activation name, its parameters from the config."""
    kinds = moe_block_utils.ExpertActivationKind
    if name == "silu":
        return moe_block_utils.ExpertActivation(kind=kinds.Silu)
    if name == "situ":
        return moe_block_utils.ExpertActivation(
            kind=kinds.Situ, beta=cfg.ACTIVATION_SITU_BETA, linear_beta=cfg.ACTIVATION_SITU_LINEAR_BETA
        )
    if name == "clamped_silu":
        return moe_block_utils.ExpertActivation(kind=kinds.ClampedSilu, limit=cfg.SWIGLU_LIMIT)
    if name == "swigluoai":
        return moe_block_utils.ExpertActivation(kind=kinds.SwigluOai, limit=cfg.SWIGLU_LIMIT, alpha=cfg.SWIGLU_ALPHA)
    raise AssertionError(f"MoE block: no expert activation {name!r} (silu, situ, clamped_silu, swigluoai)")


def moe_block_dims(model):
    """(MoEModelDims, the routed experts' activation name) of a model (its adapter, or a config class), all from the
    config's fields: NUM_ROUTED_EXPERTS, NUM_EXPERTS_PER_TOKEN, EMB_SIZE, MOE_INTERMEDIATE_SIZE; the router's SCORE_FUNC
    (default sigmoid) and ROUTE_SCALE; the shared expert's SHARED_EXPERT_INTERMEDIATE_SIZE (default
    MOE_INTERMEDIATE_SIZE x NUM_SHARED_EXPERTS); ROUTED_EXPERT_ACTIVATION / SHARED_EXPERT_ACTIVATION (default silu, with
    their parameters); a latent MoE's ROUTED_EXPERT_HIDDEN_SIZE and LATENT_MOE_USE_NORM."""
    cfg = model_config_of(model)
    groups = getattr(cfg, "NUM_EXPERT_GROUPS", 1)
    assert groups == 1, f"MoE block: {cfg.__name__} routes over {groups} expert groups (the block: one, top-k of all)"
    score = getattr(cfg, "SCORE_FUNC", "sigmoid")
    assert score in _SCORE_FUNCS, f"MoE block: {cfg.__name__} SCORE_FUNC {score!r} (block: {sorted(_SCORE_FUNCS)})"
    routed = getattr(cfg, "ROUTED_EXPERT_ACTIVATION", "silu")
    shared_hidden = getattr(cfg, "SHARED_EXPERT_INTERMEDIATE_SIZE", None)
    if shared_hidden is None:
        shared_hidden = cfg.MOE_INTERMEDIATE_SIZE * getattr(cfg, "NUM_SHARED_EXPERTS", 1)
    kw = {}
    latent = getattr(cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None)
    if latent is not None and latent != cfg.EMB_SIZE:
        kw = dict(latent_dim=latent, latent_norm=bool(getattr(cfg, "LATENT_MOE_USE_NORM", False)))
    dims = moe_block_utils.MoEModelDims(
        num_experts=cfg.NUM_ROUTED_EXPERTS,
        top_k=cfg.NUM_EXPERTS_PER_TOKEN,
        emb_dim=cfg.EMB_SIZE,
        routed_expert_hidden_dim=cfg.MOE_INTERMEDIATE_SIZE,
        routed_expert_activation=_expert_activation(cfg, routed),
        router=moe_block_utils.RouterSpec(
            score_func=getattr(moe_block_utils.RouterScoreFunc, _SCORE_FUNCS[score]), route_scale=cfg.ROUTE_SCALE
        ),
        shared_expert_hidden_dim=shared_hidden,
        shared_expert_activation=_expert_activation(cfg, getattr(cfg, "SHARED_EXPERT_ACTIVATION", routed)),
        **kw,
    )
    return dims, routed


def moe_block_enabled(model=None, moe_impl: Optional[str] = None) -> bool:
    """True when a prefill block of this model (its adapter, or its config class) builds the MoE block instead of
    TtMoe."""
    return moe_impl_of(model, moe_impl) == BLOCK


def moe_block_fabric_payload(emb_dim: int) -> int:
    """Fabric payload bytes the block's combine needs: one row (emb_dim bf16) plus its trailer, 64-B aligned."""
    return (2 * emb_dim + 32 + 63) // 64 * 64


def moe_block_fabric_payload_size(model_cfg, fabric_payload: int, moe_impl: Optional[str] = None) -> int:
    """The fabric payload a mesh of this model opens with: the model's own, raised to one combine row of the routed
    width (Kimi-K3: the latent) when the MoE block is the model's MoE."""
    if not moe_block_enabled(model_cfg, moe_impl):
        return fabric_payload
    cfg = model_config_of(model_cfg)
    width = getattr(cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None) or cfg.EMB_SIZE
    return max(fabric_payload, moe_block_fabric_payload(width))


def moe_block_device_params(model_cfg, params: dict, moe_impl: Optional[str] = None) -> dict:
    """A device_params dict (a mesh fixture's) with its fabric router's payload raised to moe_block_fabric_payload_size.
    L1_SMALL and the fabric config stay the caller's."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

    router = params.get("fabric_router_config")
    own = router.max_packet_payload_size_bytes if router is not None else 0
    payload = moe_block_fabric_payload_size(model_cfg, own, moe_impl)
    if payload == own:
        return params
    return {**params, "fabric_router_config": create_fabric_router_config(payload)}
