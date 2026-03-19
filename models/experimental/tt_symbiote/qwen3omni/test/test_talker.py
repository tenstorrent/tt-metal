import copy

import pytest
import torch
from safetensors.torch import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerModel,
)

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.qwen3omni.tt.talker_attention import TTNNQwen3Attention


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

SAFETENSOR_PATH = "/home/ubuntu/tt-metal/models/experimental/qwen3omni/checkpoints/" "model-00013-of-00015.safetensors"

TALKER_PREFIX = "talker.model.layers.0.self_attn."


def load_talker_attention_weights():
    """Load only talker layer0 attention weights from safetensor."""

    loaded = {}

    with safe_open(SAFETENSOR_PATH, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(TALKER_PREFIX):
                loaded[key] = f.get_tensor(key)

    if not loaded:
        raise RuntimeError("No talker attention weights found")

    return loaded


def build_talker_attention_layer():
    """Create minimal talker model with 1 layer."""

    full_config = AutoConfig.from_pretrained(MODEL_NAME)

    text_config = copy.deepcopy(full_config.talker_config.text_config)

    text_config.num_hidden_layers = 1

    state_dict = load_talker_attention_weights()

    # Map full-model keys to talker-model keys: talker.model.layers.0.self_attn.* -> layers.0.self_attn.*
    mapped_sd = {k.replace("talker.model.", "", 1): v for k, v in state_dict.items()}

    text_model = Qwen3OmniMoeTalkerModel(text_config).to(torch.bfloat16)

    text_model.load_state_dict(mapped_sd, strict=False)

    text_model.eval()

    torch_attn = text_model.layers[0].self_attn

    torch_attn.rotary_emb = text_model.rotary_emb

    torch_attn = torch_attn.to("cpu")

    class _Config:
        hidden_size = text_config.hidden_size

    return torch_attn, _Config()


def compute_pcc(a, b):
    a = a.flatten()
    b = b.flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_talker_attention(device):
    torch_attn, config = build_talker_attention_layer()

    print("Testing layer:", type(torch_attn))

    # ------------------------
    # Convert to TTNN
    # ------------------------

    ttnn_attn = TTNNQwen3Attention.from_torch(torch_attn)

    set_device(ttnn_attn, device)

    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    # ------------------------
    # Input
    # ------------------------

    batch = 1
    seq_len = 8192
    hidden_size = config.hidden_size

    hidden_states = torch.randn(
        batch,
        seq_len,
        hidden_size,
        dtype=torch.bfloat16,
    )

    # ------------------------
    # RoPE
    # ------------------------

    position_ids = torch.arange(seq_len).unsqueeze(0)

    cos, sin = torch_attn.rotary_emb(hidden_states, position_ids)

    # ------------------------
    # PyTorch attention
    # ------------------------

    with torch.no_grad():
        torch_out, _ = torch_attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )

    # ------------------------
    # Convert to TTNN
    # ------------------------

    hidden_states_tt = TorchTTNNTensor(hidden_states)

    cos_tt = TorchTTNNTensor(cos)
    sin_tt = TorchTTNNTensor(sin)

    # ------------------------
    # TTNN attention
    # ------------------------

    ttnn_out, _ = ttnn_attn(
        hidden_states_tt,
        position_embeddings=(cos_tt, sin_tt),
        attention_mask=None,
        past_key_values=None,
    )

    ttnn_out_torch = ttnn_out.to_torch

    # ------------------------
    # PCC
    # ------------------------

    pcc = compute_pcc(torch_out, ttnn_out_torch)

    print("PCC:", pcc)
    assert torch_out.shape == ttnn_out_torch.shape

    assert pcc > 0.99
