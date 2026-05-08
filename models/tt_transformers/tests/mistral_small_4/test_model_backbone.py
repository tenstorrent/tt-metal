# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mini ``Mistral4Model`` backbone: ``embed_tokens`` → ``layers`` → ``norm`` (text LM stack only)."""

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.model_backbone import (
    language_model_backbone_forward_reference_torch,
    language_model_backbone_hybrid_forward_bf16,
)


def _mini_mistral4_model_config():
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    # Two layers: first dense (``layer_idx < first_k_dense_replace``), then MoE.
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
def test_language_model_backbone_hybrid_matches_reference(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

    torch.manual_seed(41)
    cfg = _mini_mistral4_model_config()
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"

    model = Mistral4Model(cfg).to(torch.bfloat16).eval()

    b, s = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.long)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)

    expected = language_model_backbone_forward_reference_torch(input_ids, position_ids, model)
    out = language_model_backbone_hybrid_forward_bf16(mesh_device, input_ids, position_ids, model)

    ok, msg = comp_pcc(expected, out, pcc=0.90)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.20, atol=0.24)
    assert close, amsg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_language_model_backbone_hybrid_device_sdpa_matches_reference(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

    torch.manual_seed(41)
    cfg = _mini_mistral4_model_config()
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"

    model = Mistral4Model(cfg).to(torch.bfloat16).eval()

    b, s = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.long)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)

    expected = language_model_backbone_forward_reference_torch(input_ids, position_ids, model)
    out = language_model_backbone_hybrid_forward_bf16(
        mesh_device, input_ids, position_ids, model, use_device_sdpa_attention=True
    )

    ok, msg = comp_pcc(expected, out, pcc=0.90)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.20, atol=0.24)
    assert close, amsg
