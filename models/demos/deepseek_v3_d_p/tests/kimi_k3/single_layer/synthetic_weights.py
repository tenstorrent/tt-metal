# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic Kimi-K3 weights for one layer, at real dims, keyed exactly as the model consumes them.

Everything is a pure function of ``(layer, seed, variant)``, so LB and Galaxy build identical
weights and a failure reproduces from the test id alone. The large matrices (attention projections,
FFN, MoE) do not depend on the layer -- they are synthetic, and drawing them is most of a model
build's host time -- so every case reuses them. ``variant`` (the layer's position in its case)
keeps two layers of one case apart; the per-layer norms and AttnRes queries are always the layer's own. Norm weights are drawn around one rather than set to
one: a norm whose weight goes missing becomes ``torch.empty`` on device (#54841), and a unit weight
would hide that.

Key layout (plan §10): per layer ``attn_norm_weight``, ``ffn_norm_weight``, then ``kda_weights`` or
``mla_weights``, then ``ffn_weights`` (layer 0, dense) or the MoE quartet ``gate_weights``,
``shared_expert_weights``, ``latent_weights``, ``routed_expert_weights`` -- the last a
``PooledRoutedExperts`` provider rather than 896 host experts.
"""

from __future__ import annotations

import functools

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import CHECKPOINT_PREFIX, query_weight_names
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.utils import random_weights as kda_random_weights
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.synthetic_moe import PooledRoutedExperts
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    create_gate_weights,
    create_latent_weights,
    create_shared_expert_weights,
)

H = KimiK3Config.EMB_SIZE
WEIGHT_STD = 0.02
ROUTED_POOLS = 16

# Distinct streams per component, so adding one never reshuffles another's draws.
_COMPONENT = {"norms": 1, "kda": 2, "mla": 3, "dense": 4, "gate": 5, "shared": 6, "latent": 7, "routed": 8, "res": 9}


def component_seed(seed: int, layer: int, component: str) -> int:
    return (seed * 1_000 + layer) * 100 + _COMPONENT[component]


def shared_seed(seed: int, variant: int, component: str) -> int:
    """Seed of a layer-independent component; ``100 + variant`` never collides with a layer id (0..93)."""
    return component_seed(seed, 100 + variant, component)


def _gen(seed: int, layer: int, component: str) -> torch.Generator:
    return torch.Generator().manual_seed(component_seed(seed, layer, component))


def _normal(generator, *shape, std=WEIGHT_STD) -> torch.Tensor:
    return (torch.randn(*shape, generator=generator) * std).bfloat16()


def _around_one(generator, *shape, std=0.02) -> torch.Tensor:
    return (1.0 + torch.randn(*shape, generator=generator) * std).bfloat16()


# MLA at WEIGHT_STD everywhere writes ~0.06 RMS against the FFN's ~2.8: a nearly flat softmax over
# hundreds of keys averages V to almost nothing, and S1 could not see the attention branch at all
# (zeroing it passed). These bring its write to the FFN's order -- sharper scores (q_b), larger V
# (kv_b), and the output projection -- so each branch is visible above the bars; the S1 guard
# (`test_single_layer._assert_visible`) re-checks that per case.
MLA_Q_B_STD = 0.04
MLA_KV_B_STD = 0.06
MLA_O_STD = 0.05


@functools.lru_cache(maxsize=2)
def mla_weights(seed: int, variant: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(shared_seed(seed, variant, "mla"))
    heads = KimiK3Config.NUM_ATTENTION_HEADS
    nope, rope, v = KimiK3Config.QK_NOPE_HEAD_DIM, KimiK3Config.QK_ROPE_HEAD_DIM, KimiK3Config.V_HEAD_DIM
    q_rank, kv_rank = KimiK3Config.Q_LORA_RANK, KimiK3Config.KV_LORA_RANK
    return {
        "q_a_proj.weight": _normal(g, q_rank, H),
        "q_a_layernorm.weight": _around_one(g, q_rank),
        "q_b_proj.weight": _normal(g, heads * (nope + rope), q_rank, std=MLA_Q_B_STD),
        "kv_a_proj_with_mqa.weight": _normal(g, kv_rank + rope, H),
        "kv_a_layernorm.weight": _around_one(g, kv_rank),
        "kv_b_proj.weight": _normal(g, heads * (nope + v), kv_rank, std=MLA_KV_B_STD),
        "o_proj.weight": _normal(g, H, heads * v, std=MLA_O_STD),
        "g_proj.weight": _normal(g, heads * v, H),
    }


# The cached tensors are only read; the routed-expert provider re-derives its device experts on every
# build. maxsize 2: a case holds at most two variants.
@functools.lru_cache(maxsize=2)
def kda_weights(seed: int, variant: int = 0) -> dict[str, torch.Tensor]:
    return {
        name: tensor.bfloat16()
        for name, tensor in kda_random_weights(kimi_k3_kda_config(), shared_seed(seed, variant, "kda")).items()
    }


@functools.lru_cache(maxsize=2)
def dense_ffn_weights(seed: int, variant: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(shared_seed(seed, variant, "dense"))
    inter = KimiK3Config.INTERMEDIATE_SIZE
    return {"gate_proj": _normal(g, inter, H), "up_proj": _normal(g, inter, H), "down_proj": _normal(g, H, inter)}


@functools.lru_cache(maxsize=2)
def moe_weights(seed: int, variant: int = 0) -> dict:
    """The MoE quartet. bf16 gate, as TtMoEGatePrefill uploads it: an fp32 reference gate would flip
    top-k picks the device's rounded weights do not make (the 16th/17th score gap is ~3e-3 at 896)."""
    return {
        "gate_weights": create_gate_weights(
            KimiK3Config.NUM_ROUTED_EXPERTS, H, dtype=torch.bfloat16, seed=shared_seed(seed, variant, "gate")
        ),
        "shared_expert_weights": create_shared_expert_weights(
            H, KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE, seed=shared_seed(seed, variant, "shared")
        ),
        "latent_weights": create_latent_weights(
            H, KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE, seed=shared_seed(seed, variant, "latent")
        ),
        "routed_expert_weights": PooledRoutedExperts(
            KimiK3Config.NUM_ROUTED_EXPERTS,
            ROUTED_POOLS,
            KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
            KimiK3Config.MOE_INTERMEDIATE_SIZE,
            seed=shared_seed(seed, variant, "routed"),
        ),
    }


def make_layer_state(layer: int, *, seed: int, variant: int = 0, kv_only: bool = False) -> dict:
    """One layer's ``state_dict["layers"][i]`` entry for ``TtKimiK3Transformer``.

    ``variant``: the layer's position in its case (layers of one case must not share weights).
    ``kv_only`` leaves out the FFN (the block never builds it) and the FFN norm.
    """
    if not 0 <= layer < KimiK3Config.NUM_LAYERS:
        raise ValueError(f"layer {layer} out of range")
    norms = _gen(seed, layer, "norms")
    state = {"attn_norm_weight": _around_one(norms, H)}
    ffn_norm = _around_one(norms, H)

    if layer in KimiK3Config.mla_layer_ids():
        state["mla_weights"] = mla_weights(seed, variant)
    else:
        if kv_only:
            raise ValueError(f"layer {layer} is KDA; only an MLA layer can be kv_only")
        state["kda_weights"] = kda_weights(seed, variant)
    if kv_only:
        return state

    state["ffn_norm_weight"] = ffn_norm
    if layer < KimiK3Config.NUM_DENSE_LAYERS:
        state["ffn_weights"] = dense_ffn_weights(seed, variant)
    else:
        state.update(moe_weights(seed, variant))
    return state


_RES_SITES = ("self_attention", "mlp", "output")


def make_attn_res_weights(
    first_layer_idx: int, num_layers: int, *, seed: int, query_std: float = WEIGHT_STD, prefix: str = CHECKPOINT_PREFIX
) -> dict[str, torch.Tensor]:
    """The AttnRes checkpoint entries ``load_attn_res_weights`` reads for a window, plus the output read.

    A folded query is ``norm * proj``; with ``norm ~ 1`` its norm is about ``query_std * sqrt(d)``,
    which sets the softmax temperature over the candidates (0.02 -> |q| ~ 1.7).
    """
    weights = {}
    for name in query_weight_names(num_layers, prefix, first_layer_idx=first_layer_idx):
        layer = int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else KimiK3Config.NUM_LAYERS
        # A fixed index per factor and site, so the four entries of a layer can never share a draw
        # (a pre/post query collision would make a pre/post swap invisible).
        site = next(i for i, part in enumerate(_RES_SITES) if f"{part}_res_" in name or f"{part}_attn_res_" in name)
        factor = 0 if name.endswith("_norm.weight") else 1
        g = torch.Generator().manual_seed(component_seed(seed, layer, "res") * 10 + 2 * site + factor)
        if name.endswith("_proj.weight"):
            weights[name] = (torch.randn(1, H, generator=g) * query_std).bfloat16()
        else:
            weights[name] = _around_one(g, H, std=0.1)
    return weights


@functools.lru_cache(maxsize=1)
def make_embedding(seed: int) -> torch.Tensor:
    """``[vocab, d]`` bf16, drawn in bf16-sized chunks so the fp32 intermediate never exists whole."""
    g = torch.Generator().manual_seed(component_seed(seed, 0, "norms") + 17)
    out = torch.empty(KimiK3Config.VOCAB_SIZE, H, dtype=torch.bfloat16)
    step = 8192
    for start in range(0, KimiK3Config.VOCAB_SIZE, step):
        out[start : start + step] = torch.randn(min(step, KimiK3Config.VOCAB_SIZE - start), H, generator=g)
    return out


def make_final_norm(seed: int) -> torch.Tensor:
    return _around_one(_gen(seed, KimiK3Config.NUM_LAYERS, "norms"), H)
