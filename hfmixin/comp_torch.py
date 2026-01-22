"""Comparte two types of rotary embeddings implemented in pure torch: HF style vs meta lib style.
The model used as reference will be Llama in both cases"""

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
import mllama_model

hidden_size = 512
num_heads = 4
num_kv_heads = 2

USE_AUTOCONFIG = False

if USE_AUTOCONFIG:
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    hf_config = AutoConfig.from_pretrained(model_id)
else:
    from transformers.models.llama.configuration_llama import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rope_theta=500000.0,  # Use the same theta as Meta
        rope_scaling=None,  # This forces "default" RoPE type (vanilla)
    )

hf_rotary_emb = LlamaRotaryEmbedding(hf_config)


# type(config)
# # returns <class 'transformers.models.llama.configuration_llama.LlamaConfig'>

assert isinstance(hf_config, LlamaConfig)

head_dim = hidden_size // num_heads
wq = q_proj = nn.Linear(hidden_size, num_heads * head_dim)
wk = k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
hf_rotary_emb = LlamaRotaryEmbedding(hf_config)

rope_theta = hf_config.rope_theta
max_seq_len = 32

m_freq_cis = mllama_model.precompute_freqs_cis(dim=hidden_size // num_heads, end=max_seq_len * 2, theta=rope_theta)

# m_freq_cis.shape (end, dim / 2)


def hf_rotate(hidden_states, positions_ids):
    # hidden_states (batch_size, seq_len, hidden_size)
    # position_ids (batch_size, seq_len)

    position_embeddings = hf_rotary_emb(hidden_states, positions_ids)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = q_proj(hidden_states).view(hidden_shape)  # (batch_shape, seq_len, num_heads, head_dim)
    query_states = query_states.transpose(1, 2)  # (batch_shape, num_heads, seq_len, head_dim)
    key_states = k_proj(hidden_states).view(hidden_shape)  # (batch_shape, seq_len, num_heads, head_dim)
    key_states = key_states.transpose(1, 2)  # (batch_shape, num_heads, seq_len, , head_dim)

    cos, sin = position_embeddings
    # cos.shape: (batch_size, seq_len, head_dim)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    #  query_states.shape: (batch_size, num_heads, seq_len, head_dim)
    #  key_states.shape: (batch_size, num_kv_heads, seq_len, head_dim)

    return query_states, key_states, position_embeddings


def pre_meta_to_hf(x: torch.Tensor) -> torch.Tensor:
    """This is meant to be applied before applying a meta rotary emb, so the rotation is applied
    to the same elements as in the HF version

    x = [x1, x2, x3, x4]

    To rotate
    hf = [x1, x3, x2, x4]
    meta = [x1, x2, x3, x4]

    """
    dim = x.shape[-1]
    half_dim = dim // 2
    x_shape = x.shape
    return torch.cat([x[..., :half_dim][..., None], x[..., half_dim:][..., None]], dim=-1).view(x_shape)


def post_hf_to_meta(x: torch.Tensor) -> torch.Tensor:
    x_shape = x.shape  # (batch_size, seq_len, num_heads, head_dim)
    return x.view((*x_shape[:-1], -1, 2)).transpose(3, 4).reshape(x_shape)


def mllama_rotate(hidden_states, positions_ids):
    start_pos = int(positions_ids[0, 0])
    seq_len = len(positions_ids[0])
    freq_cis = m_freq_cis[start_pos : start_pos + seq_len]

    xq, xk = wq(hidden_states), wk(hidden_states)

    xq = xq.view(batch_size, seq_len, num_heads, head_dim)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim)

    xq, xk = pre_meta_to_hf(xq), pre_meta_to_hf(xk)
    xq, xk = mllama_model.apply_rotary_emb(xq, xk, freq_cis)
    xq, xk = post_hf_to_meta(xq), post_hf_to_meta(xk)

    return xq, xk, freq_cis


batch_size = 4
seq_len = 16
assert seq_len <= max_seq_len
positions_ids = torch.LongTensor(batch_size * [list(range(seq_len))])
hidden_states = torch.rand(batch_size, seq_len, hidden_size)


rot_query_states, rot_key_states, position_embeddings = hf_rotate(hidden_states, positions_ids)
rot_xq, rot_xk, freq_cis = mllama_rotate(hidden_states, positions_ids)

rot_m_query_states = rot_xq.transpose(1, 2)
rot_m_key_states = rot_xk.transpose(1, 2)

assert (
    torch.isclose(rot_query_states, rot_m_query_states).to(torch.float).mean() > 0.999
), "rotated queries are not equal"
assert torch.isclose(rot_key_states, rot_m_key_states).to(torch.float).mean() > 0.999, "rotated keys are not equal"

print("PASSES!!")
