# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""KV-cache parity: hybrid / device SDPA attention paths vs Hugging Face ``DynamicCache``."""

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.decoder_layer import (
    decoder_layer_dense_forward_reference_torch,
    decoder_layer_dense_hybrid_forward_bf16,
)
from models.tt_transformers.tt.mistral_small_4.model_backbone import (
    language_model_backbone_hybrid_forward_incremental_bf16,
)


def _decoder_layer_config():
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=1,
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


def _mini_mistral4_all_dense():
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        first_k_dense_replace=2,
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
def test_decoder_layer_dense_hybrid_kv_two_steps_matches_hf(mesh_device, reset_seeds):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    torch.manual_seed(41)
    cfg = _decoder_layer_config()
    cfg._attn_implementation = "eager"
    layer = Mistral4DecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).eval()
    rot = Mistral4RotaryEmbedding(cfg).to(torch.bfloat16).eval()

    b, h = 1, cfg.hidden_size
    s_prefill = 4
    x1 = torch.randn(b, s_prefill, h, dtype=torch.bfloat16)
    position_ids1 = torch.arange(s_prefill, dtype=torch.long).unsqueeze(0)
    cos1, sin1 = rot(x1, position_ids=position_ids1)

    # Separate caches: HF and hybrid each call ``past_key_values.update()`` once per step.
    # Reusing one ``DynamicCache`` for both would double-append K/V (4→8) and break step-2 masks / matmuls.
    past_hf = DynamicCache(config=cfg)
    past_tt = DynamicCache(config=cfg)
    mask1_hf = create_causal_mask(
        cfg,
        x1,
        attention_mask=None,
        past_key_values=past_hf,
        position_ids=position_ids1,
    )
    mask1_tt = create_causal_mask(
        cfg,
        x1,
        attention_mask=None,
        past_key_values=past_tt,
        position_ids=position_ids1,
    )
    expected1 = decoder_layer_dense_forward_reference_torch(
        x1,
        (cos1, sin1),
        position_ids1,
        layer,
        attention_mask=mask1_hf,
        past_key_values=past_hf,
        use_cache=True,
    )
    out1 = decoder_layer_dense_hybrid_forward_bf16(
        mesh_device,
        x1,
        (cos1, sin1),
        position_ids1,
        layer,
        attention_mask=mask1_tt,
        past_key_values=past_tt,
    )
    ok1, msg1 = comp_pcc(expected1, out1, pcc=0.92)
    assert ok1, msg1
    close1, amsg1 = comp_allclose(expected1, out1, rtol=0.18, atol=0.22)
    assert close1, amsg1

    x2 = torch.randn(b, 1, h, dtype=torch.bfloat16)
    position_ids2 = torch.tensor([[s_prefill]], dtype=torch.long)
    cos2, sin2 = rot(x2, position_ids=position_ids2)
    mask2_hf = create_causal_mask(
        cfg,
        x2,
        attention_mask=None,
        past_key_values=past_hf,
        position_ids=position_ids2,
    )
    mask2_tt = create_causal_mask(
        cfg,
        x2,
        attention_mask=None,
        past_key_values=past_tt,
        position_ids=position_ids2,
    )
    expected2 = decoder_layer_dense_forward_reference_torch(
        x2,
        (cos2, sin2),
        position_ids2,
        layer,
        attention_mask=mask2_hf,
        past_key_values=past_hf,
        use_cache=True,
    )
    out2 = decoder_layer_dense_hybrid_forward_bf16(
        mesh_device,
        x2,
        (cos2, sin2),
        position_ids2,
        layer,
        attention_mask=mask2_tt,
        past_key_values=past_tt,
    )
    ok2, msg2 = comp_pcc(expected2, out2, pcc=0.92)
    assert ok2, msg2
    close2, amsg2 = comp_allclose(expected2, out2, rtol=0.18, atol=0.22)
    assert close2, amsg2


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_decoder_layer_dense_device_sdpa_kv_two_steps_matches_hf(mesh_device, reset_seeds):
    """
    Both steps use HF ``create_causal_mask`` + ttnn SDPA vs eager; use the same loose bounds as
    ``test_decoder_layer_dense_hybrid_device_sdpa_matches_hf_with_causal_mask`` (prefill and decode).
    """
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    torch.manual_seed(42)
    cfg = _decoder_layer_config()
    cfg._attn_implementation = "eager"
    layer = Mistral4DecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).eval()
    rot = Mistral4RotaryEmbedding(cfg).to(torch.bfloat16).eval()

    b, h = 1, cfg.hidden_size
    s_prefill = 4
    x1 = torch.randn(b, s_prefill, h, dtype=torch.bfloat16)
    position_ids1 = torch.arange(s_prefill, dtype=torch.long).unsqueeze(0)
    cos1, sin1 = rot(x1, position_ids=position_ids1)

    past_hf = DynamicCache(config=cfg)
    past_tt = DynamicCache(config=cfg)
    mask1_hf = create_causal_mask(
        cfg,
        x1,
        attention_mask=None,
        past_key_values=past_hf,
        position_ids=position_ids1,
    )
    mask1_tt = create_causal_mask(
        cfg,
        x1,
        attention_mask=None,
        past_key_values=past_tt,
        position_ids=position_ids1,
    )
    expected1 = decoder_layer_dense_forward_reference_torch(
        x1,
        (cos1, sin1),
        position_ids1,
        layer,
        attention_mask=mask1_hf,
        past_key_values=past_hf,
        use_cache=True,
    )
    out1 = decoder_layer_dense_hybrid_forward_bf16(
        mesh_device,
        x1,
        (cos1, sin1),
        position_ids1,
        layer,
        attention_mask=mask1_tt,
        past_key_values=past_tt,
        use_device_sdpa_attention=True,
    )
    ok1, msg1 = comp_pcc(expected1, out1, pcc=0.88)
    assert ok1, msg1
    close1, amsg1 = comp_allclose(expected1, out1, rtol=2.0, atol=1.2)
    assert close1, amsg1

    x2 = torch.randn(b, 1, h, dtype=torch.bfloat16)
    position_ids2 = torch.tensor([[s_prefill]], dtype=torch.long)
    cos2, sin2 = rot(x2, position_ids=position_ids2)
    mask2_hf = create_causal_mask(
        cfg,
        x2,
        attention_mask=None,
        past_key_values=past_hf,
        position_ids=position_ids2,
    )
    mask2_tt = create_causal_mask(
        cfg,
        x2,
        attention_mask=None,
        past_key_values=past_tt,
        position_ids=position_ids2,
    )
    expected2 = decoder_layer_dense_forward_reference_torch(
        x2,
        (cos2, sin2),
        position_ids2,
        layer,
        attention_mask=mask2_hf,
        past_key_values=past_hf,
        use_cache=True,
    )
    out2 = decoder_layer_dense_hybrid_forward_bf16(
        mesh_device,
        x2,
        (cos2, sin2),
        position_ids2,
        layer,
        attention_mask=mask2_tt,
        past_key_values=past_tt,
        use_device_sdpa_attention=True,
    )
    ok2, msg2 = comp_pcc(expected2, out2, pcc=0.88)
    assert ok2, msg2
    close2, amsg2 = comp_allclose(expected2, out2, rtol=2.0, atol=1.2)
    assert close2, amsg2


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_language_model_backbone_incremental_matches_hf_one_shot(mesh_device, reset_seeds):
    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

    torch.manual_seed(43)
    cfg = _mini_mistral4_all_dense()
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"
    model = Mistral4Model(cfg).to(torch.bfloat16).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 7), dtype=torch.long)
    with torch.no_grad():
        full = model(input_ids, use_cache=False).last_hidden_state[:, -1:, :]

    past = DynamicCache(config=cfg)
    h1, past = language_model_backbone_hybrid_forward_incremental_bf16(
        mesh_device, input_ids[:, :6], None, model, past_key_values=past, use_cache=True
    )
    h2, _past = language_model_backbone_hybrid_forward_incremental_bf16(
        mesh_device, input_ids[:, 6:7], None, model, past_key_values=past, use_cache=True
    )

    past_hf = DynamicCache(config=cfg)
    with torch.no_grad():
        out_p = model(input_ids[:, :6], use_cache=True, past_key_values=past_hf)
        expected_last = model(
            input_ids[:, 6:7], use_cache=True, past_key_values=out_p.past_key_values
        ).last_hidden_state

    ok, msg = comp_pcc(expected_last, h2, pcc=0.90)
    assert ok, msg
    close, amsg = comp_allclose(expected_last, h2, rtol=0.20, atol=0.24)
    assert close, amsg

    okf, msgf = comp_pcc(full, h2, pcc=0.90)
    assert okf, msgf
    closef, amsgf = comp_allclose(full, h2, rtol=0.20, atol=0.24)
    assert closef, amsgf

    past_mid = DynamicCache(config=cfg)
    with torch.no_grad():
        ref_h1 = model(input_ids[:, :6], use_cache=True, past_key_values=past_mid).last_hidden_state
    okm, msgm = comp_pcc(ref_h1, h1, pcc=0.90)
    assert okm, msgm
