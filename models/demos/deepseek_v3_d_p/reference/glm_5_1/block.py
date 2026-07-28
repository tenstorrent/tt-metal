# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.1 decoder-*layer* (block) CPU reference.

GLM has no standalone HF reference model wired (see reference/glm_5_1_config.py); the DSA family
validates by *composing* the CPU references it already owns. This module assembles one full decoder
layer exactly as ``TtPrefillBlock.forward`` does:

    attn_norm_out = rms_norm(x, attn_norm_weight)
    mla_out       = SparseMLAReference(config, mla_weights).forward(attn_norm_out)   # DSA MLA (indexer)
    x             = x + mla_out
    ffn_norm_out  = rms_norm(x, ffn_norm_weight)
    ffn_out       = dense_ffn(ffn_norm_out)  OR  moe_reference(ffn_norm_out)
    out           = x + ffn_out

The DSA-MLA half reuses ``reference.cpu_deepseek_v32.SparseMLAReference`` (variant-parametrized, GLM
interleaved indexer) — the same truth that validates GLM sparse MLA at the op level; the FFN half is
plain torch (dense) or a caller-supplied MoE callable. Nothing here re-implements DSA math.
"""

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_1.moe import glm_moe_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with fp32 accumulation, matching TtDistributedRmsNorm."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight.float()).to(x.dtype)


def dense_ffn(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor) -> torch.Tensor:
    """GLM dense FFN: down( silu(x @ gate^T) * (x @ up^T) ). Projection weights are [out, in]."""
    xf = x.float()
    h = torch.nn.functional.silu(xf @ gate_proj.float().t()) * (xf @ up_proj.float().t())
    return (h @ down_proj.float().t()).to(x.dtype)


def glm_decoder_layer_reference(
    config,
    mla_weights,
    attn_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    seq_len: int,
    *,
    ffn_weights: dict | None = None,
    moe_weights: dict | None = None,
):
    """One GLM decoder layer on CPU (DSA-MLA + norm/residual + FFN), matching TtPrefillBlock.forward.

    Args:
        config: GLM HF-attribute config (glm_hf_config()); ``config.max_seq_len`` should be set.
        mla_weights: canonical MLA+indexer weights (cpu_deepseek_v32 ``Weights``), also fed to ttMLA.
        attn_norm_weight / ffn_norm_weight: the two RMSNorm gains [hidden].
        hidden_states: block input [1, seq, hidden] (pre-attn-norm).
        seq_len: sequence length (sizes the sparse-MLA KVPE buffer).
        ffn_weights: dense-layer FFN weights {"gate_proj","up_proj","down_proj"}, OR
        moe_weights: MoE-layer weights {"gate_weights","routed_expert_weights","shared_expert_weights"}.
            Exactly one must be given. The MoE uses GLM's own routing config (GLM51Config: 256 routed
            experts, single-group top-k n_group=topk_group=1, top-8, route_scale=2.5) — not DeepSeek's.

    Returns:
        (output [1, seq, hidden], kvpe_cache) — kvpe in the device layout for KVPE-PCC checks.
    """
    if (ffn_weights is None) == (moe_weights is None):
        raise ValueError("provide exactly one of ffn_weights (dense) or moe_weights (MoE)")

    x = hidden_states
    attn_norm_out = rms_norm(x, attn_norm_weight, config.rms_norm_eps)

    ref = SparseMLAReference(config, mla_weights, seq_len=seq_len)
    mla_out = ref.forward(attn_norm_out)  # [1, seq, hidden]
    x = x + mla_out

    ffn_norm_out = rms_norm(x, ffn_norm_weight, config.rms_norm_eps)
    if ffn_weights is not None:
        ffn_out = dense_ffn(ffn_norm_out, ffn_weights["gate_proj"], ffn_weights["up_proj"], ffn_weights["down_proj"])
    else:
        ffn_out = glm_moe_reference(
            ffn_norm_out,
            gate_weights=moe_weights["gate_weights"],
            routed_expert_weights=moe_weights["routed_expert_weights"],
            shared_expert_weights=moe_weights["shared_expert_weights"],
            emb_dim=config.hidden_size,
            num_experts_per_tok=GLM51Config.NUM_EXPERTS_PER_TOKEN,
            n_group=GLM51Config.NUM_EXPERT_GROUPS,
            topk_group=GLM51Config.NUM_LIMITED_GROUPS,
            routed_scaling_factor=GLM51Config.ROUTE_SCALE,
        )

    return x + ffn_out, ref.kvpe_cache
