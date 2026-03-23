import copy
import json
import os
import warnings

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerModel,
)

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.qwen3omni.tt.talker_attention import TTNNQwen3Attention


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Local directory containing model.safetensors.index.json and shard files.
CHECKPOINT_DIR = os.environ.get(
    "QWEN3OMNI_CHECKPOINT_DIR",
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695",
)

TALKER_PREFIX = "talker.model.layers.0.self_attn."


def load_state_dict_filtered_local(ckpt_dir, key_prefix):
    """Load only weights with the given prefix from a local checkpoint (index + shards)."""
    from safetensors.torch import safe_open as safetensors_safe_open

    ckpt_dir = os.path.abspath(ckpt_dir)
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    file_to_keys = {}
    for key, file_name in weight_map.items():
        if key.startswith(key_prefix):
            file_to_keys.setdefault(file_name, []).append(key)

    loaded = {}
    for file_name, keys in file_to_keys.items():
        path = os.path.join(ckpt_dir, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Shard not found: {path}")
        with safetensors_safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                loaded[key] = f.get_tensor(key)
    return loaded


def build_talker_attention_layer():
    """Create minimal talker model with 1 layer."""

    full_config = AutoConfig.from_pretrained(MODEL_NAME)

    text_config = copy.deepcopy(full_config.talker_config.text_config)

    text_config.num_hidden_layers = 1

    text_model = Qwen3OmniMoeTalkerModel(text_config).to(torch.bfloat16)
    try:
        state_dict = load_state_dict_filtered_local(CHECKPOINT_DIR, TALKER_PREFIX)
        if not state_dict:
            raise RuntimeError(f"No weights found with prefix {TALKER_PREFIX!r} in {CHECKPOINT_DIR}")
        # Map full-model keys to talker-model keys: talker.model.layers.0.self_attn.* -> layers.0.self_attn.*
        mapped_sd = {k.replace("talker.model.", "", 1): v for k, v in state_dict.items()}
        text_model.load_state_dict(mapped_sd, strict=False)
    except (FileNotFoundError, RuntimeError) as e:
        warnings.warn(
            f"{e}. Falling back to random-initialized talker layer for parity test.",
            RuntimeWarning,
        )

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
