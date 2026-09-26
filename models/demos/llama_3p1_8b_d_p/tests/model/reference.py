# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent float32 HF-frame oracle. No production numerical helpers are imported."""


import math

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.demos.llama_3p1_8b_d_p.tests.utils import read_raw_weights


def rms(x, weight, epsilon):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon) * weight.float()


@torch.no_grad()
def reference_layer(hidden, state_dict, config):
    """A full causal prefix, in raw HF half-split coordinates throughout attention."""
    weights = {name: value.float() for name, value in state_dict.items()}
    normalized = rms(hidden, weights["input_layernorm.weight"], config.rms_norm_eps)
    n_heads, n_kv = config.num_attention_heads, config.num_key_value_heads
    dim = config.hidden_size // n_heads
    q = F.linear(normalized, weights["self_attn.q_proj.weight"]).reshape(-1, n_heads, dim).transpose(0, 1)
    k = F.linear(normalized, weights["self_attn.k_proj.weight"]).reshape(-1, n_kv, dim).transpose(0, 1)
    v = F.linear(normalized, weights["self_attn.v_proj.weight"]).reshape(-1, n_kv, dim).transpose(0, 1)
    rotary = LlamaRotaryEmbedding(config=config)
    cos, sin = rotary(hidden.unsqueeze(0), torch.arange(len(hidden)).unsqueeze(0))
    cos, sin = cos[0].unsqueeze(0), sin[0].unsqueeze(0)
    half = dim // 2
    q = q * cos + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin
    k = k * cos + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
    # Each head uses the same independent PyTorch full-causal reference. Query blocks bound host
    # score storage without changing any probabilities or truncating the prefix.
    attended = torch.empty((n_heads, len(hidden), dim), dtype=torch.float32)
    keys = torch.arange(len(hidden))
    for head in range(n_heads):
        for start in range(0, len(hidden), 128):
            end = min(start + 128, len(hidden))
            scores = q[head, start:end] @ k[head // (n_heads // n_kv)].T / math.sqrt(dim)
            positions = torch.arange(start, end)
            scores.masked_fill_(keys[None] > positions[:, None], -torch.inf)
            attended[head, start:end] = scores.softmax(dim=-1) @ v[head // (n_heads // n_kv)]
    delta = F.linear(
        attended.transpose(0, 1).reshape(len(hidden), config.hidden_size), weights["self_attn.o_proj.weight"]
    )
    residual = hidden + delta
    normalized = rms(residual, weights["post_attention_layernorm.weight"], config.rms_norm_eps)
    output = residual + F.linear(
        F.silu(F.linear(normalized, weights["mlp.gate_proj.weight"]))
        * F.linear(normalized, weights["mlp.up_proj.weight"]),
        weights["mlp.down_proj.weight"],
    )
    for tensor in (output, k, v):
        assert torch.isfinite(tensor).all()
    # Conversion occurs only at the comparison boundary. This is independent of QKVProjection.
    adjacent = torch.tensor([coordinate for pair in zip(range(half), range(half, dim)) for coordinate in pair])
    return output, k[..., adjacent].contiguous(), v


def chat_tokens(checkpoint_path, *, slot, length=None, user_text=None):
    """Render one Instruct template; explicit user text is never repeated to meet a length."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    text = (
        "Explain why the sky looks blue in daylight. Use a short and clear answer. "
        if slot == 0
        else "List the steps to make a cup of tea. Explain each step in simple words. "
    )
    # A single chat template preserves one BOS and real role delimiters. Fixed-boundary cases
    # explicitly truncate user content; the short main quality case keeps the complete template,
    # including its assistant generation header.
    if user_text is not None and (not isinstance(user_text, str) or not user_text.strip()):
        raise ValueError("user_text must be a nonempty string")
    content = user_text if user_text is not None else text * (max(1, length) if length is not None else 1)
    messages = [{"role": "user", "content": content}]
    all_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if length is not None and len(all_ids) < length:
        raise ValueError("chat fixture did not reach requested logical length")
    ids = torch.tensor(all_ids if length is None else all_ids[:length], dtype=torch.int64)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "slot": slot,
        "logical_length": len(ids),
        "chat_template_applied": True,
        "user_text_override": user_text is not None,
        "prefix_of_rendered_chat": length is not None,
        "add_generation_prompt": True,
        "prompt_text": tokenizer.decode(ids.tolist(), skip_special_tokens=False),
        "token_ids": ids.tolist(),
    }
    return ids, metadata


@torch.no_grad()
def reference_prefill(checkpoint_path, token_ids, *, num_layers=32, selected_logit_positions=None):
    """Incremental raw-checkpoint reference, with all layer hidden/K/V values retained for gates."""
    config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
    embed = read_raw_weights(checkpoint_path, ["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    hidden = F.embedding(token_ids, embed).float()
    del embed
    names = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    layers = []
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}."
        raw = read_raw_weights(checkpoint_path, [prefix + name for name in names])
        weights = {name.removeprefix(prefix): tensor for name, tensor in raw.items()}
        hidden, key, value = reference_layer(hidden, weights, config)
        layers.append({"hidden": hidden, "k": key, "v": value})
        del raw, weights
    final = read_raw_weights(checkpoint_path, ["model.norm.weight", "lm_head.weight"])
    norm = rms(hidden, final["model.norm.weight"], config.rms_norm_eps)
    if selected_logit_positions is None:
        selected_logit_positions = torch.arange(len(token_ids))
    selected_logit_positions = torch.as_tensor(selected_logit_positions, dtype=torch.int64)
    logits = F.linear(norm[selected_logit_positions], final["lm_head.weight"].float())
    assert torch.isfinite(logits).all()
    return {"layers": layers, "normalized": norm, "logit_positions": selected_logit_positions, "logits": logits}
