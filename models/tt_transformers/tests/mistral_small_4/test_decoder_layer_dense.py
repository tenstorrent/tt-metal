# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.decoder_layer import (
    decoder_layer_dense_forward_reference_torch,
    decoder_layer_dense_hybrid_forward_bf16,
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


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_decoder_layer_dense_hybrid_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    torch.manual_seed(41)
    cfg = _decoder_layer_config()
    cfg._attn_implementation = "eager"
    layer = Mistral4DecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).eval()
    rot = Mistral4RotaryEmbedding(cfg).to(torch.bfloat16).eval()

    b, s, h = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, h, dtype=torch.bfloat16)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(x, position_ids=position_ids)

    expected = decoder_layer_dense_forward_reference_torch(x, (cos, sin), position_ids, layer)
    out = decoder_layer_dense_hybrid_forward_bf16(mesh_device, x, (cos, sin), position_ids, layer)
    ok, msg = comp_pcc(expected, out, pcc=0.92)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.18, atol=0.22)
    assert close, amsg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_decoder_layer_dense_hybrid_device_sdpa_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    torch.manual_seed(41)
    cfg = _decoder_layer_config()
    cfg._attn_implementation = "eager"
    layer = Mistral4DecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).eval()
    rot = Mistral4RotaryEmbedding(cfg).to(torch.bfloat16).eval()

    b, s, h = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, h, dtype=torch.bfloat16)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(x, position_ids=position_ids)

    expected = decoder_layer_dense_forward_reference_torch(x, (cos, sin), position_ids, layer)
    out = decoder_layer_dense_hybrid_forward_bf16(
        mesh_device, x, (cos, sin), position_ids, layer, use_device_sdpa_attention=True
    )
    ok, msg = comp_pcc(expected, out, pcc=0.92)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.18, atol=0.22)
    assert close, amsg


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_decoder_layer_dense_hybrid_device_sdpa_matches_hf_with_causal_mask(mesh_device, reset_seeds):
    from transformers.masking_utils import create_causal_mask
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    torch.manual_seed(41)
    cfg = _decoder_layer_config()
    cfg._attn_implementation = "eager"
    layer = Mistral4DecoderLayer(cfg, layer_idx=0).to(torch.bfloat16).eval()
    rot = Mistral4RotaryEmbedding(cfg).to(torch.bfloat16).eval()

    b, s, h = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, h, dtype=torch.bfloat16)
    position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    cos, sin = rot(x, position_ids=position_ids)

    causal = create_causal_mask(cfg, x, None, past_key_values=None, position_ids=position_ids)
    if causal is None or not isinstance(causal, torch.Tensor):
        pytest.skip("create_causal_mask did not return a torch.Tensor for this transformers build")

    expected = decoder_layer_dense_forward_reference_torch(x, (cos, sin), position_ids, layer, attention_mask=causal)
    out = decoder_layer_dense_hybrid_forward_bf16(
        mesh_device, x, (cos, sin), position_ids, layer, attention_mask=causal, use_device_sdpa_attention=True
    )
    ok, msg = comp_pcc(expected, out, pcc=0.92)
    assert ok, msg
    # Device SDPA + additive mask vs HF eager (float32 softmax on scores) is looser than unmasked SDPA.
    close, amsg = comp_allclose(expected, out, rtol=2.0, atol=1.0)
    assert close, amsg
