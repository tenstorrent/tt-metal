# Test RoPE operator

import os

import pytest
import torch
from transformers import AutoConfig

import ttnn

# from models.demos.gpt_oss.tt.rope import GptOssRotaryEmbedding as GptOssRotaryEmbeddingTT
from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssRotaryEmbedding, apply_rotary_pos_emb


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (4, 8)}.get(
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
        )
    ],
    indirect=True,
)
@pytest.fixture
def hf_config():
    """Load GPT-OSS config for testing"""
    path = os.getenv("HF_MODEL", "models/demos/gpt_oss/reference")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 32),
    ],
)
def test_rope_op(
    mesh_device,
    mode,
    seq_len,
    hf_config,
):
    position_ids = torch.arange(seq_len).unsqueeze(0)

    RopeEmbeddings = GptOssRotaryEmbedding(hf_config)
    torch_inputs = torch.randn(1, seq_len, hf_config.hidden_size)
    cos, sin = RopeEmbeddings(torch_inputs, position_ids)
    TP = 1
    q_torch = torch.randn(hf_config.num_attention_heads // TP, seq_len, hf_config.head_dim)
    k_torch = torch.randn(hf_config.num_attention_heads // TP, seq_len, hf_config.head_dim)

    q_rope_torch, k_rope_torch = apply_rotary_pos_emb(q_torch, k_torch, cos, sin)
