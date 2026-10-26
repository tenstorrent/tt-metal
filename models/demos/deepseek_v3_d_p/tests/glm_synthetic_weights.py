# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Random GLM-5.3 weights in TtPrefillBlock / TtPrefillTransformer state-dict form.

GLM has no runnable HF reference model wired (the adapter's reference_model_cls is None), so the
shared create_hf_model() + extract_tt_state_dict() random path cannot build it. These builders produce
the same dict shapes directly. They need no checkpoint, no ttnn cache and no golden trace, so a GLM
test built on them runs on any mesh with nothing staged.

Host RAM: one MoE layer is 256 experts x 3 x 6144 x 2048 bf16, about 19 GB, plus about 1.9 GB for the
embedding. Keep num_layers small (layers 0-2 are dense, so 4 layers is the smallest stack with a MoE).
"""

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused


def glm_norm_weight(hidden, seed):
    return (torch.randn(hidden, generator=torch.Generator().manual_seed(seed)) * 0.1 + 1.0).to(torch.bfloat16)


def glm_random_dense_ffn_weights(hidden, intermediate, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_proj": (torch.randn(intermediate, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "up_proj": (torch.randn(intermediate, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "down_proj": (torch.randn(hidden, intermediate, generator=g) * intermediate**-0.5).to(torch.bfloat16),
    }


def glm_random_moe_weights(hidden, moe_intermediate, n_routed, seed):
    g = torch.Generator().manual_seed(seed)
    hs, ds = hidden**-0.5, moe_intermediate**-0.5

    def _expert():
        return {
            "gate_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "up_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "down_proj": (torch.randn(hidden, moe_intermediate, generator=g) * ds).to(torch.bfloat16),
        }

    gate_weights = {
        "weight": (torch.randn(n_routed, hidden, generator=g) * hs).to(torch.bfloat16),
        "e_score_correction_bias": (torch.randn(n_routed, generator=g) * 0.01).to(torch.float32),
    }
    return gate_weights, [_expert() for _ in range(n_routed)], _expert()


def glm_random_layer_state_dict(config, model_cfg, layer_idx, seed):
    """One TtPrefillBlock state dict for global layer ``layer_idx``: dense FFN below
    first_k_dense_replace, MoE above. A GLM-5.3 ``shared`` layer owns no indexer, so its indexer
    weights are dropped (ttMLA would ignore them anyway)."""
    hidden = config.hidden_size
    base = seed + 10 * layer_idx
    mla_weights = random_mla_weights(config, seed=base)
    if indexer_layer_is_reused(config, layer_idx):
        mla_weights = {k: v for k, v in mla_weights.items() if not k.startswith("indexer.")}
    layer = {
        "attn_norm_weight": glm_norm_weight(hidden, base + 1),
        "mla_weights": mla_weights,
        "ffn_norm_weight": glm_norm_weight(hidden, base + 2),
    }
    if layer_idx < config.first_k_dense_replace:
        layer["ffn_weights"] = glm_random_dense_ffn_weights(hidden, config.intermediate_size, base + 3)
    else:
        gate_weights, routed, shared = glm_random_moe_weights(
            hidden, model_cfg.MOE_INTERMEDIATE_SIZE, model_cfg.NUM_ROUTED_EXPERTS, base + 3
        )
        layer.update(gate_weights=gate_weights, routed_expert_weights=routed, shared_expert_weights=shared)
    return layer


def glm_random_transformer_state_dict(config, model_cfg, num_layers, seed=42):
    """TtPrefillTransformer state dict (embed + layers 0..num_layers-1), the shape
    extract_tt_state_dict returns for the HF-backed variants. No norm / LM head: the chunked
    transformer has no tail unless MTP is on."""
    g = torch.Generator().manual_seed(seed)
    std = float(getattr(config, "initializer_range", 0.02))
    return {
        "embed_weight": (torch.randn(config.vocab_size, config.hidden_size, generator=g) * std).to(torch.bfloat16),
        "layers": [glm_random_layer_state_dict(config, model_cfg, i, seed) for i in range(num_layers)],
    }
