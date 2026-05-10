# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase B: Hugging Face Mistral4 text decoder layer 0 on CPU (reference contract).

The public checkpoint stores FP8 expert/linear weights plus scale tensors; a plain
``load_state_dict`` into stock ``nn.Linear`` modules is therefore not expected to work
without HF's quantized loading path. These tests instead validate:

* ``Mistral4DecoderLayer`` + ``Mistral4RotaryEmbedding`` build from ``text_config`` and run
  a short forward pass with explicitly initialized weights (eager attention).
* Safetensors **index** keys for ``language_model.model.layers.0.`` match the layout we
  rely on for bring-up.
* Optionally (slow, when shards are available): filtered safetensors load returns tensors.
"""

import json
import os

import pytest
import torch
import torch.nn as nn

from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID, text_decoder_layer_state_dict_prefix


pytest.importorskip("transformers")
pytest.importorskip(
    "transformers.models.mistral4.modeling_mistral4", reason="Mistral4 classes require a recent transformers build"
)


def _text_config_eager_attn():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")
    text = cfg.text_config
    if text is None:
        pytest.skip("Config has no text_config")
    if hasattr(text, "attn_implementation"):
        text.attn_implementation = "eager"
    if hasattr(text, "_attn_implementation"):
        text._attn_implementation = "eager"
    return text


def _init_mistral4_decoder_layer_weights(layer: nn.Module) -> None:
    """Stock ``Mistral4DecoderLayer`` does not initialize MoE ``torch.empty`` buffers."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4NaiveMoe, Mistral4TopkRouter

    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, Mistral4TopkRouter):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, Mistral4NaiveMoe):
                nn.init.normal_(m.gate_up_proj, std=0.02)
                nn.init.normal_(m.down_proj, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def test_mistral4_decoder_layer0_cpu_forward_shape():
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    layer = Mistral4DecoderLayer(text, layer_idx=0).eval()
    _init_mistral4_decoder_layer_weights(layer)
    rotary = Mistral4RotaryEmbedding(text).eval()

    batch, seq = 1, 8
    hidden = text.hidden_size
    x = torch.randn(batch, seq, hidden, dtype=torch.bfloat16)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary(x, position_ids)

    with torch.no_grad():
        y = layer(
            x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )

    assert y.shape == x.shape
    assert torch.isfinite(y).all(), "forward produced non-finite values"


def test_mistral4_layer0_safetensors_index_key_layout():
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import hf_hub_download

    local_only = os.getenv("CI") == "true"
    try:
        index_path = hf_hub_download(
            HF_MODEL_ID,
            filename="model.safetensors.index.json",
            local_files_only=local_only,
        )
    except Exception as exc:
        pytest.skip(f"Could not obtain safetensors index: {exc}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    prefix = text_decoder_layer_state_dict_prefix(0)
    keys = [k for k in index["weight_map"] if k.startswith(prefix)]
    assert len(keys) > 0
    joined = " ".join(keys)
    assert "self_attn." in joined
    assert "mlp.gate.weight" in joined
    assert "mlp.experts.gate_up_proj" in joined
    assert "input_layernorm.weight" in joined


@pytest.mark.slow
def test_mistral4_layer0_filtered_safetensors_load_nonempty():
    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

    prefix = text_decoder_layer_state_dict_prefix(0)
    try:
        sd = load_hf_state_dict_filtered(HF_MODEL_ID, (prefix,))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"No checkpoint shards available: {exc}")

    assert len(sd) > 0
    assert any("input_layernorm.weight" in k for k in sd)
    assert all(k.startswith(prefix) for k in sd)
    # Representative dtypes from hub (metadata / tensor dtype)
    assert any("self_attn" in k for k in sd)
