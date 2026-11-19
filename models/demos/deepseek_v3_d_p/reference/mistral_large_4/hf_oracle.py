# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The HF oracle for Mistral Large 4: transformers' own ``mistral_large4`` text model, vendored.

The checkpoint needs transformers 5.21 (model type ``mistral_large4``, added upstream by huggingface/transformers
b5e1cb8b015b "Add Mistral Large 4 (#50121)", 2026-10-29); this workspace has 5.12.1. The text-model classes below are
copied from that commit's ``modeling_mistral_large4.py`` with only these changes:
  * absolute imports, no hub-kernel / output-capture / config-default decorators (no effect on CPU eager);
  * ``create_causal_mask`` without ``allow_is_causal_skip`` (new in 5.17; 5.12 never skips the mask for eager);
  * the routed experts read their FP8 weights from the checkpoint when they run (``StreamedExperts``, the same loop as
    ``MistralLarge4TextExperts``: 14.5 GB of FP8 per layer, 1 TB in all, does not fit in host memory);
  * a causal-LM wrapper with the ConditionalGeneration head (LM head + final logit softcapping), text only;
  * the weights are loaded straight from the consolidated checkpoint with the upstream conversion mapping's names
    (conversion_mapping.py ``mistral_large4`` / ``mistral_large4_text``), FP8 blocks dequantized exactly.
The config is the checkpoint's ``text_config`` with MistralLarge4TextConfig.__post_init__'s derived fields.
"""

from __future__ import annotations

import json
import math
from collections import UserDict
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PreTrainedConfig
from transformers.masking_utils import create_causal_mask
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, dequant, expert_store, load_dense


# ----------------------------------------------------------------------------------------------- config
def text_config(path) -> PreTrainedConfig:
    raw = json.loads((Path(path) / "config.json").read_text())["text_config"]
    cfg = PreTrainedConfig()
    for k, v in raw.items():
        setattr(cfg, k, v)
    # MistralLarge4TextConfig.__post_init__
    cfg.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    cfg.head_dim = cfg.qk_rope_head_dim
    cfg.num_local_experts = cfg.n_routed_experts
    cfg.rope_parameters = dict(raw["rope_parameters"])
    cfg._attn_implementation = "eager"
    return cfg


# ----------------------------------------------------------------------------------------------- upstream classes
class MistralLarge4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MistralLarge4TextRotaryEmbedding(nn.Module):
    """LlamaRotaryEmbedding with a learned tensor for scaling the inverse frequencies."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("alpha", torch.zeros_like(self.inv_freq))

    @torch.no_grad()
    def forward(self, x, position_ids):
        scaled_inv_freq = (
            self.inv_freq.to(device=x.device, dtype=torch.float)
            * self.alpha.to(device=x.device, dtype=torch.float).exp()
        )
        freqs = position_ids[..., None].float() * scaled_inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MistralLarge4TextSoftmaxScaling(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(num_heads))
        self.bias = nn.Parameter(torch.empty(num_heads))

    def forward(self, query_states: torch.Tensor, position_ids: torch.Tensor, head_dim_first: bool) -> torch.Tensor:
        log_pos = (position_ids + 1).float().log()  # convert to float to avoid overflow
        log_pos, scale, bias = (x.to(query_states.dtype) for x in (log_pos, self.scale, self.bias))
        if head_dim_first:
            scale = scale[None, :, None] * log_pos[:, None, :] + bias[None, :, None]
        else:
            scale = scale[None, None, :] * log_pos[:, :, None] + bias[None, None, :]
        return query_states * scale.unsqueeze(-1)


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    q1, q2 = q[..., 0::2], q[..., 1::2]
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    if k is not None:
        k1, k2 = k[..., 0::2], k[..., 1::2]
        k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    else:
        k_embed = None
    return q_embed, k_embed


class MistralLarge4TextIndexer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.wq_b = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.q_input_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.softmax_scaling = MistralLarge4TextSoftmaxScaling(self.n_heads)
        self.use_rope = config.no_rope_layers[layer_idx]

    @torch.no_grad()
    def forward(self, hidden_states, position_embeddings, attention_mask, position_ids, past_key_values=None):
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.q_input_norm(hidden_states)
        q = self.wq_b(q)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = self.softmax_scaling(q, position_ids, head_dim_first=False)
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        if self.use_rope:
            q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        k = k.squeeze(2)
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        if attention_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            index_scores = index_scores + attention_mask
        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)


def eager_attention_forward(module, query, key, value, attention_mask, scaling: float, dropout: float = 0.0, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_apply_mscale(rope_parameters, scaling):
    if rope_parameters.get("rope_type", "default") != "default":
        mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
        if mscale_all_dim:
            mscale = yarn_get_mscale(rope_parameters["factor"], mscale_all_dim)
            scaling = scaling * mscale * mscale
    return scaling


class MistralLarge4TextAttention(nn.Module):
    """DSA attention from GlmMoeDsa, with no Q-LoRA and KV sharing."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.use_rope = config.no_rope_layers[layer_idx]
        self.is_kv_shared_layer = config.kv_source_layers[layer_idx] != layer_idx
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim if self.use_rope else 0
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim + (0 if self.use_rope else config.qk_rope_head_dim)
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.kv_a_proj_with_mqa = (
            nn.Linear(self.hidden_size, config.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias)
            if not self.is_kv_shared_layer
            else None
        )
        self.kv_a_layernorm = (
            MistralLarge4RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps) if not self.is_kv_shared_layer else None
        )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
        self.scaling = yarn_apply_mscale(config.rope_parameters, self.qk_head_dim ** (-0.5))
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else MistralLarge4TextIndexer(config, layer_idx)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        self.softmax_scaling = MistralLarge4TextSoftmaxScaling(self.num_heads)
        self.store_full_length_kv = any(src == layer_idx for src in config.kv_source_layers[layer_idx + 1 :])

    def expand_kv(self, kv_nope, k_rot):
        batch_size, _, seq_length, _ = kv_nope.shape
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)
        key_states = kv_nope.new_empty(*kv_nope.shape[:-1], self.qk_nope_head_dim + self.qk_rope_head_dim)
        key_states[..., : self.qk_nope_head_dim].copy_(k_nope)
        key_states[..., self.qk_nope_head_dim :].copy_(k_rot)
        return key_states, value_states

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        position_ids,
        past_key_values=None,
        prev_topk_indices=None,
        **kwargs,
    ):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        q_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if self.is_kv_shared_layer:
            k_pass, k_rot = shared_kv_states["rope" if self.use_rope else "nope"]
        else:
            compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
            kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
            k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        if self.use_rope:
            cos, sin = position_embeddings
            if self.is_kv_shared_layer:
                q_rot, _ = apply_rotary_pos_emb_interleave(q_rot, None, cos, sin)
            else:
                q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        if self.store_full_length_kv:
            shared_kv_states["rope" if self.use_rope else "nope"] = (k_pass, k_rot)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        query_states = self.softmax_scaling(query_states, position_ids, head_dim_first=True)
        key_states, value_states = self.expand_kv(k_pass, k_rot)
        if self.indexer is not None:
            topk_indices = self.indexer(hidden_states, position_embeddings, attention_mask[:, 0, :, :], position_ids)
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices
        index_mask = (
            topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
            .scatter(-1, topk_indices.long(), False)
            .unsqueeze(1)
        )
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask & ~index_mask
        else:
            attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            position_ids=position_ids,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices


class MistralLarge4TextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralLarge4TextTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.softmax(-1)
        group_scores = (
            scores.view(-1, self.num_group, self.num_experts // self.num_group).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class StreamedExperts(nn.Module):
    """MistralLarge4TextExperts' eager loop over the hit experts, each expert's gate_up_proj (w1; w3) and down_proj
    (w2) dequantized from the checkpoint's FP8 when it runs."""

    def __init__(self, config, store, layer_idx: int, next_layer: int | None):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.act_fn = ACT2FN[config.hidden_act]
        self.store, self.layer_idx, self.next_layer = store, layer_idx, next_layer

    def forward(self, hidden_states, top_k_index, top_k_weights):
        le = self.store.get(self.layer_idx, next_layer=self.next_layer)
        dt = hidden_states.dtype
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = torch.cat([le.weight(expert_idx, "w1", dt), le.weight(expert_idx, "w3", dt)])
            gate, up = nn.functional.linear(current_state, gate_up).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, le.weight(expert_idx, "w2", dt))
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class MistralLarge4TextMoE(nn.Module):
    def __init__(self, config, store, layer_idx: int, next_layer: int | None):
        super().__init__()
        self.experts = StreamedExperts(config, store, layer_idx, next_layer)
        self.gate = MistralLarge4TextTopkRouter(config)
        self.shared_experts = MistralLarge4TextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class MistralLarge4TextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, store, next_moe: int | None):
        super().__init__()
        self.self_attn = MistralLarge4TextAttention(config, layer_idx)
        self.mlp = (
            MistralLarge4TextMoE(config, store, layer_idx, next_moe)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else MistralLarge4TextMLP(config)
        )
        self.input_layernorm = MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        shared_kv_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        prev_topk_indices=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            prev_topk_indices=prev_topk_indices,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


class MistralLarge4TextModel(nn.Module):
    def __init__(self, config, store):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        n = config.num_hidden_layers
        moe = [i for i in range(n) if config.mlp_layer_types[i] == "sparse"]
        nxt = {a: b for a, b in zip(moe, moe[1:])}
        self.layers = nn.ModuleList([MistralLarge4TextDecoderLayer(config, i, store, nxt.get(i)) for i in range(n)])
        self.norm = MistralLarge4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = None  # built on the CPU after the meta-device construction (load_oracle)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        assert causal_mask is not None, "the indexer needs an explicit causal mask"
        shared_kv_states = UserDict()
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        topk_indices = None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                shared_kv_states=shared_kv_states,
                prev_topk_indices=topk_indices,
            )
        return self.norm(hidden_states)


class MistralLarge4ForCausalLM(nn.Module):
    """MistralLarge4ForConditionalGeneration's text path: language model, LM head, final logit softcapping."""

    def __init__(self, config, store):
        super().__init__()
        self.config = config
        self.model = MistralLarge4TextModel(config, store)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = config.final_logit_softcapping

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping
        return SimpleNamespace(logits=logits)


# ----------------------------------------------------------------------------------------------- weights
# module path (in MistralLarge4TextModel) -> checkpoint name, per the upstream conversion mapping
def _layer_names(cfg, i: int) -> dict[str, str]:
    p, q = f"model.layers.{i}.", f"layers.{i}."
    m = {
        p + "input_layernorm.weight": q + "attention_norm.weight",
        p + "post_attention_layernorm.weight": q + "ffn_norm.weight",
        p + "self_attn.q_proj.weight": q + "attention.wq.weight",
        p + "self_attn.kv_b_proj.weight": q + "attention.wkv_b.weight",
        p + "self_attn.o_proj.weight": q + "attention.wo.weight",
        p + "self_attn.softmax_scaling.scale": q + "attention.ssmax_scaling.s",
        p + "self_attn.softmax_scaling.bias": q + "attention.ssmax_scaling.b",
    }
    if cfg.kv_source_layers[i] == i:
        m[p + "self_attn.kv_a_proj_with_mqa.weight"] = q + "attention.wkv_a_with_mqa.weight"
        m[p + "self_attn.kv_a_layernorm.weight"] = q + "attention.kv_a_norm.weight"
    if cfg.indexer_types[i] == "full":
        a, b = p + "self_attn.indexer.", q + "attention.indexer."
        m.update(
            {
                a + "wq_b.weight": b + "wq_b.weight",
                a + "wk.weight": b + "wk.weight",
                a + "k_norm.weight": b + "k_norm.weight",
                a + "k_norm.bias": b + "k_norm.bias",
                a + "weights_proj.weight": b + "weights_proj.weight",
                a + "q_input_norm.weight": b + "q_input_norm.weight",
                a + "q_input_norm.bias": b + "q_input_norm.bias",
                a + "softmax_scaling.scale": b + "ssmax_scaling.s",
                a + "softmax_scaling.bias": b + "ssmax_scaling.b",
            }
        )
    if cfg.mlp_layer_types[i] == "sparse":
        m[p + "mlp.gate.weight"] = q + "gate.weight"
        for hf, ms in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
            m[p + f"mlp.shared_experts.{hf}.weight"] = q + f"shared_experts.{ms}.weight"
    else:
        for hf, ms in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
            m[p + f"mlp.{hf}.weight"] = q + f"feed_forward.{ms}.weight"
    return m


def _set(module: nn.Module, path: str, t: torch.Tensor) -> None:
    *mods, leaf = path.split(".")
    for m in mods:
        module = getattr(module, m)
    if leaf in module._parameters:
        module._parameters[leaf] = nn.Parameter(t, requires_grad=False)
    else:
        module._buffers[leaf] = t


def build_hf_model(path, num_layers: int | None, dtype: torch.dtype) -> MistralLarge4ForCausalLM:
    """The text model with the first ``num_layers`` layers (None: all), weights in ``dtype`` (the router weight in
    fp32, as upstream's _keep_in_fp32_modules_strict keeps it), routed experts streamed."""
    cfg = text_config(path)
    if num_layers:
        cfg.num_hidden_layers = num_layers
    ckpt = checkpoint(path)
    store = expert_store(ckpt, cfg.n_routed_experts)
    with torch.device("meta"):
        model = MistralLarge4ForCausalLM(cfg, store)
    names = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    for i in range(cfg.num_hidden_layers):
        names.update(_layer_names(cfg, i))
    fp32 = {k for k in names if k.endswith("mlp.gate.weight")}
    for k, t in load_dense(ckpt, {k: v for k, v in names.items() if k not in fp32}, dtype).items():
        _set(model, k, t)
    for k, t in load_dense(ckpt, {k: names[k] for k in fp32}, torch.float32).items():
        _set(model, k, t)
    rot = MistralLarge4TextRotaryEmbedding(cfg)  # fp32, on the CPU (upstream keeps rotary_emb in fp32)
    rot.alpha = ckpt.get("lerope.alpha").float()
    model.model.rotary_emb = rot
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    assert not meta, f"oracle tensors not loaded: {meta[:8]}"
    return model.eval()


__all__ = ["build_hf_model", "text_config", "dequant"]
