# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``Mistral4HybridTextBackbone`` matches direct hybrid causal_lm / backbone calls."""

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.causal_lm_prefill import causal_lm_prefill_logits_hybrid_bf16
from models.tt_transformers.tt.mistral_small_4.model_backbone import language_model_backbone_hybrid_forward_bf16
from models.tt_transformers.tt.mistral_small_4.text_backbone import Mistral4HybridTextBackbone


def _mini_mistral4_causal_config():
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
def test_mistral4_hybrid_text_backbone_matches_functional_api(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    torch.manual_seed(43)
    cfg = _mini_mistral4_causal_config()
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"

    model = Mistral4ForCausalLM(cfg).to(torch.bfloat16).eval()
    backbone = Mistral4HybridTextBackbone(mesh_device, model, use_device_sdpa_attention=False)

    b, s = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.long)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)

    hidden_bb = backbone.last_hidden_prefill(input_ids, position_ids)
    hidden_fn = language_model_backbone_hybrid_forward_bf16(
        mesh_device, input_ids, position_ids, model.model, attention_mask_4d=None, use_device_sdpa_attention=False
    )
    ok_h, msg_h = comp_pcc(hidden_fn, hidden_bb, pcc=0.99)
    assert ok_h, msg_h
    close_h, amsg_h = comp_allclose(hidden_fn, hidden_bb, rtol=0.05, atol=0.05)
    assert close_h, amsg_h

    logits_bb = backbone.logits_prefill(input_ids, position_ids)
    logits_fn = causal_lm_prefill_logits_hybrid_bf16(
        mesh_device, input_ids, position_ids, model, attention_mask_4d=None, use_device_sdpa_attention=False
    )
    ok_l, msg_l = comp_pcc(logits_fn, logits_bb, pcc=0.90)
    assert ok_l, msg_l
    close_l, amsg_l = comp_allclose(logits_fn, logits_bb, rtol=0.20, atol=0.24)
    assert close_l, amsg_l
