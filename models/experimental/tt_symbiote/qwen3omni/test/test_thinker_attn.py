import copy
import json
import os
import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.qwen3omni.tt.thinker_attention import TTNNQwen3OmniAttention


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Local directory containing model.safetensors.index.json and shard files
# for thinker.model.layers.0.self_attn (e.g. HF cache snapshot or a copy).
CHECKPOINT_DIR = os.environ.get(
    "QWEN3OMNI_CHECKPOINT_DIR",
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695",
)
THINKER_ATTN_PREFIX = "thinker.model.layers.0.self_attn."


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
    for key, file in weight_map.items():
        if key.startswith(key_prefix):
            file_to_keys.setdefault(file, []).append(key)

    loaded = {}
    for file, keys in file_to_keys.items():
        path = os.path.join(ckpt_dir, file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Shard not found: {path}")
        with safetensors_safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                loaded[key] = f.get_tensor(key)
    return loaded


def load_thinker_layer0_attention_from_checkpoint(checkpoint_dir, model_name=MODEL_NAME):
    """
    Load only the first Thinker self-attention from a local checkpoint.
    Builds only the 1-layer Thinker text model (minimal memory), not the full 40GB model.
    Returns (torch_attn, config) where config has .hidden_size for the test.
    """
    full_config = AutoConfig.from_pretrained(model_name)
    # Use Thinker text config with a single layer so we only allocate one decoder layer
    text_config = copy.deepcopy(full_config.thinker_config.text_config)
    text_config.num_hidden_layers = 1

    filtered_sd = load_state_dict_filtered_local(checkpoint_dir, THINKER_ATTN_PREFIX)
    if not filtered_sd:
        raise RuntimeError(f"No weights found with prefix {THINKER_ATTN_PREFIX!r} in {checkpoint_dir}")
    # Map full-model keys to text-model keys: thinker.model.layers.0.self_attn.* -> model.layers.0.self_attn.*
    mapped_sd = {k.replace("thinker.", "", 1): v for k, v in filtered_sd.items()}

    # Build only the 1-layer Thinker text model (small)
    text_model = Qwen3OmniMoeThinkerTextModel(text_config).to(torch.bfloat16)
    text_model.load_state_dict(mapped_sd, strict=False)
    text_model.eval()

    torch_attn = text_model.layers[0].self_attn
    # Attention module doesn't have rotary_emb; the text model does. Attach for the test.
    torch_attn.rotary_emb = text_model.rotary_emb
    torch_attn = torch_attn.to("cpu")

    # Return a config-like object with .hidden_size for the test
    class _Config:
        hidden_size = text_config.hidden_size

    return torch_attn, _Config()


def compute_pcc(a, b):
    a = a.flatten()
    b = b.flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_qwen_attention(device):
    # ---------------------------
    # Load thinker.model.layers.0.self_attn from local safetensors
    # ---------------------------
    torch_attn, config = load_thinker_layer0_attention_from_checkpoint(CHECKPOINT_DIR)

    print("Testing layer:", type(torch_attn))

    # ---------------------------
    # Convert to TTNN
    # ---------------------------
    ttnn_attn = TTNNQwen3OmniAttention.from_torch(torch_attn)

    set_device(ttnn_attn, device)

    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    # ---------------------------
    # Create random input
    # ---------------------------
    batch = 1
    seq_len = 32
    hidden_size = config.hidden_size

    hidden_states = torch.randn(
        batch,
        seq_len,
        hidden_size,
        dtype=torch.bfloat16,
    )

    # ---------------------------
    # Generate RoPE embeddings
    # ---------------------------
    position_ids = torch.arange(seq_len).unsqueeze(0)

    cos, sin = torch_attn.rotary_emb(hidden_states, position_ids)

    # ---------------------------
    # Run PyTorch attention
    # ---------------------------
    with torch.no_grad():
        torch_out, _ = torch_attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )

    # ---------------------------
    # Convert inputs to TTNN
    # ---------------------------
    hidden_states_tt = TorchTTNNTensor(hidden_states)

    cos_tt = TorchTTNNTensor(cos)
    sin_tt = TorchTTNNTensor(sin)

    # ---------------------------
    # Run TTNN attention
    # ---------------------------
    ttnn_out, _ = ttnn_attn(
        hidden_states_tt,
        position_embeddings=(cos_tt, sin_tt),
        attention_mask=None,
        past_key_values=None,
    )

    ttnn_out_torch = ttnn_out.to_torch

    # ---------------------------
    # Compare outputs
    # ---------------------------
    pcc = compute_pcc(torch_out, ttnn_out_torch)

    print("PCC:", pcc)

    assert pcc > 0.99
