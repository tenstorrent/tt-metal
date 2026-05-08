# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HF 2D ``attention_mask`` (padding) + causal: backbone and causal LM vs full HF ``forward``."""

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.causal_lm_prefill import (
    causal_lm_prefill_logits_hf_reference_torch,
    causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16,
)
from models.tt_transformers.tt.mistral_small_4.model_backbone import (
    language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16,
    language_model_backbone_last_hidden_hf_reference_torch,
)


def _mini_mistral4_model_config():
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=1,
        n_routed_experts=64,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        n_shared_experts=1,
        max_position_embeddings=4096,
        rope_interleave=False,
        rope_parameters={
            "type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
            "max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale_all_dim": 1.0,
            "mscale": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_language_model_backbone_hybrid_padding_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

    torch.manual_seed(43)
    cfg = _mini_mistral4_model_config()
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"

    model = Mistral4Model(cfg).to(torch.bfloat16).eval()
    pad = int(cfg.pad_token_id or 0) % cfg.vocab_size

    s = 8
    input_ids = torch.tensor([[10, 11, 12, 13, 14, pad, pad, pad]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)

    expected = language_model_backbone_last_hidden_hf_reference_torch(
        input_ids, position_ids, model, attention_mask_2d=attention_mask
    )
    out = language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16(
        mesh_device,
        input_ids,
        position_ids,
        model,
        attention_mask_2d=attention_mask,
    )

    ok, msg = comp_pcc(expected, out, pcc=0.88)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.24, atol=0.30)
    assert close, amsg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_causal_lm_prefill_hybrid_padding_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    torch.manual_seed(43)
    cfg = _mini_mistral4_model_config()
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"

    model = Mistral4ForCausalLM(cfg).to(torch.bfloat16).eval()
    pad = int(cfg.pad_token_id or 0) % cfg.vocab_size

    s = 8
    input_ids = torch.tensor([[10, 11, 12, 13, 14, pad, pad, pad]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)

    expected = causal_lm_prefill_logits_hf_reference_torch(
        input_ids, position_ids, model, attention_mask_2d=attention_mask
    )
    out = causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16(
        mesh_device,
        input_ids,
        position_ids,
        model,
        attention_mask_2d=attention_mask,
    )

    ok, msg = comp_pcc(expected, out, pcc=0.88)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.26, atol=0.32)
    assert close, amsg
