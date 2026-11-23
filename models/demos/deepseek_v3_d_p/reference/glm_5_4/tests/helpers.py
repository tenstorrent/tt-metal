# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the GLM-5.4 reference tests: a tiny GLM-5.4 config and random weights for it."""

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import glm_5_4_hf_config

TINY_EXPERTS = 16  # > top-8 of GLMConfig, so routing selects


def tiny_glm54_config(*, seq_len: int, window: int = 4, index_topk: int = 8):
    """GLM-5.4 hf config at CPU-test size: L0 prelude (dense), L1-L2 loop (MoE), L3 coda (MoE).

    indexer_types full/full/shared/full: L2 reuses L1's top-k in pass 1 (so pass 2 of L2 must too), and the
    first coda layer owns an indexer, as at GLM-5.4 scale (L26 and L54 are full there).
    """
    c = glm_5_4_hf_config(seq_len)
    c.hidden_size = 128
    c.intermediate_size = 192
    c.num_attention_heads = c.num_key_value_heads = 4
    c.q_lora_rank = 64
    c.kv_lora_rank = 128
    c.qk_nope_head_dim = 32
    c.qk_rope_head_dim = 16
    c.v_head_dim = 32
    c.index_n_heads = 2
    c.index_head_dim = 32
    c.index_topk = index_topk
    c.num_hidden_layers = 4
    c.first_k_dense_replace = 1
    c.indexer_types = ["full", "full", "shared", "full"]
    c.loop_first_layer, c.loop_last_layer = 1, 2
    c.loop_window_size = window
    c.max_seq_len = seq_len
    return c


def _lin(g, out_f, in_f):
    return (torch.randn(out_f, in_f, generator=g) * in_f**-0.5).to(torch.bfloat16)


def _expert(g, hidden, inter):
    return {"gate_proj": _lin(g, inter, hidden), "up_proj": _lin(g, inter, hidden), "down_proj": _lin(g, hidden, inter)}


def tiny_layer_weights(config, layer: int, seed: int = 7) -> dict:
    """One TtPrefillBlock-form layer state dict for ``tiny_glm54_config``."""
    g = torch.Generator().manual_seed(seed + 100 * layer)
    h = config.hidden_size
    w = {
        "attn_norm_weight": (1.0 + 0.1 * torch.randn(h, generator=g)).to(torch.bfloat16),
        "mla_weights": random_mla_weights(config, seed=seed + 100 * layer + 1),
        "ffn_norm_weight": (1.0 + 0.1 * torch.randn(h, generator=g)).to(torch.bfloat16),
    }
    # random_mla_weights draws std 0.02 weights, sized for hidden 6144; rescale so the tiny layers mix.
    w["mla_weights"] = {
        k: (v.float() * 8.0).to(v.dtype) if v.dim() == 2 and "weights_proj" not in k else v
        for k, v in w["mla_weights"].items()
        # A shared layer owns no indexer, as in the device state dict.
        if not (config.indexer_types[layer] == "shared" and k.startswith("indexer."))
    }
    if layer < config.first_k_dense_replace:
        w["ffn_weights"] = _expert(g, h, config.intermediate_size)
    else:
        w["gate_weights"] = {
            "weight": _lin(g, TINY_EXPERTS, h),
            "e_score_correction_bias": torch.randn(TINY_EXPERTS, generator=g) * 0.01,
        }
        w["routed_expert_weights"] = [_expert(g, h, 32) for _ in range(TINY_EXPERTS)]
        w["shared_expert_weights"] = _expert(g, h, 32)
    return w


def tiny_gate_weights(config, loop_pass: int, *, per_channel: bool = False) -> dict:
    g = torch.Generator().manual_seed(1000 + loop_pass)
    h = config.hidden_size
    width = h if per_channel else 1
    return {
        "norm_weight": (1.0 + 0.1 * torch.randn(h, generator=g)).to(torch.bfloat16),
        "proj_weight": _lin(g, width, h),
        "proj_bias": (0.5 * torch.randn(width, generator=g)).to(torch.bfloat16),
    }


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])
