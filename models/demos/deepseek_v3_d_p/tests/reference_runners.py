# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Reference runners for each model variant.

These run the variant's reference on CPU and return a torch tensor
suitable for PCC comparison against the TT output. Variants without a
bundled reference return `None` and the comparison is skipped at the call
site.
"""

import inspect
from copy import deepcopy
from typing import Optional

import torch

from models.demos.common.prefill.adapter import PrefillModelAdapter as TestVariant


def run_reference_moe(
    variant: TestVariant,
    *,
    config,
    gate_weights,
    routed_expert_weights,
    shared_expert_weights,
    x,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MoE reference on CPU."""
    if variant.reference_moe_cls is None:
        return None
    # Test params can override the variant's default MoE dims (expert count, hidden/intermediate size —
    # e.g. GLM-5.1's 256 experts / 6144 hidden vs the deepseek_v3 variant's 7168), so patch the reference
    # config from the actual generated weight shapes: gate.weight is [n_experts, hidden], each expert's
    # gate_proj is [moe_intermediate, hidden]. Without this the reference is built at the variant default
    # and load_state_dict fails with a size mismatch.
    cfg = deepcopy(config)
    cfg.n_routed_experts = gate_weights["weight"].shape[0]
    cfg.hidden_size = gate_weights["weight"].shape[1]
    if routed_expert_weights:
        cfg.moe_intermediate_size = routed_expert_weights[0]["gate_proj"].shape[0]
    moe = variant.reference_moe_cls(cfg)
    moe.load_state_dict(
        _pack_reference_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights),
        strict=True,
    )
    moe = moe.eval().to(torch.bfloat16)
    with torch.no_grad():
        return moe(x.to(torch.bfloat16))


def run_reference_mla(
    variant: TestVariant,
    *,
    config,
    weights,
    hidden_states,
    position_ids,
) -> Optional[torch.Tensor]:
    """Forward the variant's upstream MLA reference on CPU."""
    if variant.reference_attention_cls is None:
        return None
    _, q_len, _ = hidden_states.shape
    attn = variant.reference_attention_cls(config, layer_idx=0)
    # load_state_dict is non-strict (references carry submodules the TT weight dict doesn't), so a
    # missing weight silently stays at its random init and only shows up as a baffling PCC miss.
    # Gate-bearing variants (Kimi-K3) are checked explicitly for that reason.
    if getattr(config, "mla_use_output_gate", False):
        assert "g_proj.weight" in weights, (
            "config declares mla_use_output_gate but the MLA weight dict has no 'g_proj.weight'; "
            "the reference would run with a randomly-initialised gate"
        )
    attn.load_state_dict(weights, strict=False)
    attn = attn.eval().to(torch.bfloat16)
    causal = torch.triu(torch.full((q_len, q_len), float("-inf"), dtype=hidden_states.dtype), diagonal=1)
    # Bind by signature rather than assuming one generation's shape. The vendored DeepSeek/Kimi
    # attentions take `past_key_value` and derive rope internally; a transformers >= 5 attention
    # (e.g. Mistral4Attention) takes `past_key_values` and REQUIRES `position_embeddings`, because
    # rope moved up to the model level. Passing the wrong cache name silently lands in **kwargs;
    # omitting position_embeddings is a TypeError.
    fwd_params = inspect.signature(attn.forward).parameters
    kwargs = {
        "hidden_states": hidden_states,
        "attention_mask": causal[None, None],
        "position_ids": position_ids,
        "use_cache": False,
    }
    kwargs["past_key_values" if "past_key_values" in fwd_params else "past_key_value"] = None
    if "position_embeddings" in fwd_params:
        rotary_cls = getattr(variant, "reference_rotary_cls", None)
        assert rotary_cls is not None, (
            f"{type(attn).__name__} requires position_embeddings but {variant.name!r} exposes no "
            "reference_rotary_cls to build (cos, sin) from"
        )
        # Build and evaluate the rope tables in FLOAT32, then cast down. Computing them in bf16 is
        # catastrophic at long context: `position * inv_freq` loses its low bits, so the phase error
        # grows with position. Measured on Mistral at qk_rope_head_dim=64 -- a bf16 rotary holds up
        # to ~2k tokens and then falls off a cliff (PCC vs the absorbed reference 0.99998 at seq 512,
        # 0.958 at seq 5120), which reads exactly like a device bug and is not one. In fp32 the
        # absorbed (this family's) and unabsorbed (transformers') MLA formulations agree to 1.0000000.
        rotary = rotary_cls(config=config).float()
        with torch.no_grad():
            cos, sin = rotary(hidden_states.float(), position_ids)
            kwargs["position_embeddings"] = (cos.to(hidden_states.dtype), sin.to(hidden_states.dtype))
    with torch.no_grad():
        out = attn(**kwargs)
    # DeepseekV3Attention returns (out, attn_weights, past_kv); Kimi-K3's KimiMLAAttention returns a
    # bare tensor (upstream shape). Accept either.
    return out[0] if isinstance(out, tuple) else out


def _pack_reference_moe_state_dict(gate_weights, routed_expert_weights, shared_expert_weights) -> dict:
    sd = {
        "gate.weight": gate_weights["weight"],
        "gate.e_score_correction_bias": gate_weights["e_score_correction_bias"],
        "shared_experts.gate_proj.weight": shared_expert_weights["gate_proj"],
        "shared_experts.up_proj.weight": shared_expert_weights["up_proj"],
        "shared_experts.down_proj.weight": shared_expert_weights["down_proj"],
    }
    for i, w in enumerate(routed_expert_weights):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            sd[f"experts.{i}.{proj}.weight"] = w[proj]
    return {k: v.to(torch.bfloat16) for k, v in sd.items()}
