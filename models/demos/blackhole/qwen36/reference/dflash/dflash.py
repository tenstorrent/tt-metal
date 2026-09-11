# SPDX-FileCopyrightText: Copyright (c) 2026 Z Lab
#
# SPDX-License-Identifier: MIT

"""Vendored HF reference modeling for the DFlash drafter (``z-lab/Qwen3.6-27B-DFlash``).

Adapted from the upstream z-lab DFlash ``dflash/model.py`` (https://github.com/z-lab/dflash,
arXiv:2602.06036). The drafter checkpoint declares ``auto_map: {"AutoModel": "dflash.DFlashDraftModel"}``
but ships **no** modeling file, so the code has to be vendored to load it at all.

Deviations from upstream, all deliberate:

* The ``DFlash2`` classes (``GroupedDynamicCausalConv``, ``CandidateSelector``,
  ``DFlash2DraftModel``) are dropped. ``z-lab/Qwen3.6-27B-DFlash`` is a plain ``DFlashDraftModel``:
  its ``config.json`` declares no ``conv_kernel_size`` / ``selector_rank``, and its checkpoint has
  exactly 58 tensors (5 x 11 layer tensors + ``fc`` + ``hidden_norm`` + ``norm``) with no conv or
  codebook weights. Restore them from upstream if a DFlash2 checkpoint is ever targeted.
* The generation loop lives in :mod:`.generate` instead of here, because the Qwen3.6 target needs
  speculative rollback that upstream's loop does not implement (see that module).

Everything that shapes a tensor is upstream's, so this stays a faithful golden reference.

What the drafter is, in one paragraph: it is a 5-layer block-diffusion drafter for the
``Qwen/Qwen3.6-27B`` target. Per decode step it consumes (a) ``target_hidden`` — the target's
residual stream tapped at layers ``[1, 16, 31, 46, 61]`` and concatenated to ``5 * 5120 = 25600``
features — and (b) ``noise_embedding`` — the target's *input* embedding of a 16-token block whose
first slot is the confirmed anchor token and whose other 15 slots are all ``mask_token_id``. It
emits hidden states for the block; the *target's* ``lm_head`` turns those into the 15 draft tokens,
which the target then verifies in one forward. Context enters attention only as extra K/V
(``k_ctx``/``v_ctx``); queries are formed from the noise block alone, so all 15 positions are
drafted in parallel rather than autoregressively.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Default residual-stream taps when the checkpoint does not name them explicitly.

    Unused for Qwen3.6-27B-DFlash (its ``dflash_config`` declares ``target_layer_ids``), kept so a
    checkpoint that omits the key still builds the same taps upstream would.
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [round(start + (i * span) / (num_draft_layers - 1)) for i in range(num_draft_layers)]


def extract_context_feature(hidden_states: list[torch.Tensor], layer_ids: list[int]) -> torch.Tensor:
    """Concatenate the target's tapped residual streams into the drafter's context feature.

    ``hidden_states`` is HF's ``output_hidden_states`` tuple, whose entry ``i + 1`` is the *output*
    of decoder layer ``i`` (entry 0 is the embedding), hence the ``offset``.
    """
    offset = 1
    return torch.cat([hidden_states[layer_id + offset] for layer_id in layer_ids], dim=-1)


def _draft_config(config) -> dict:
    return getattr(config, "dflash_config", {}) or {}


def _draft_value(config, name, default=None):
    """Read a DFlash knob from ``dflash_config`` first, then the config top level, then ``default``."""
    return _draft_config(config).get(name, getattr(config, name, default))


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """RoPE for the drafter's split query/key lengths.

    ``k`` spans context + noise and gets the full ``cos``/``sin``; ``q`` spans the noise block only
    and gets the *trailing* ``q_len`` entries, which is what pins the drafted block to its absolute
    positions.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _attention_mask(query, key, *, is_causal, sliding_window):
    """Boolean ``[1, 1, q_len, kv_len]`` visibility mask in absolute token positions.

    ``key`` is the post-cache-update tensor, so ``kv_len - q_len`` is the absolute index of the
    block's first query — this is what makes the mask correct across decode steps.
    """
    query_position = key.shape[-2] - query.shape[-2] + torch.arange(query.shape[-2], device=query.device)[:, None]
    key_position = torch.arange(key.shape[-2], device=query.device)[None, :]
    visible = torch.ones((query.shape[-2], key.shape[-2]), dtype=torch.bool, device=query.device)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= query_position - key_position < sliding_window
        if not is_causal:
            visible &= key_position - query_position < sliding_window
    return visible[None, None]


class Qwen3DFlashAttention(nn.Module):
    """Qwen3 GQA where K/V span ``[context, noise]`` but Q spans the noise block only.

    ``layer_types`` decides the regime per layer: Qwen3.6-27B-DFlash is 4 causal
    ``sliding_attention`` layers (window 2048) then 1 **bidirectional** ``full_attention`` layer.
    The bidirectional layer is what lets the 15 masked slots see each other — the "diffusion" in
    block diffusion — so ``is_causal`` must stay ``False`` there.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None)
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        is_causal = getattr(config, "is_causal", None)
        self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # Context and noise share the K/V projections; only the context half is worth caching, which
        # is why the caller drops the trailing q_len entries after every step.
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        if attention_mask is None and (self.is_causal or self.sliding_window is not None):
            attention_mask = _attention_mask(q, k, is_causal=self.is_causal, sliding_window=self.sliding_window)

        attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    """The drafter. Returns hidden states — the *target's* ``lm_head`` produces the draft logits."""

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = _draft_value(
            config,
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = int(_draft_value(config, "block_size", 16))
        self.mask_token_id = _draft_value(config, "mask_token_id")
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """``noise_embedding`` is ``[b, q_len, H]``; ``target_hidden`` is ``[b, ctx_len, n_taps * H]``.

        ``position_ids`` must span ``ctx_len + q_len`` — the K/V axis, not the query axis.
        """
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden: torch.Tensor, output_head: nn.Module) -> torch.Tensor:
        """Draft logits, via the target's output head plus the optional DFlash logit shaping."""
        logits = output_head(hidden)
        logits = logits * float(_draft_value(self.config, "output_multiplier", 1.0))
        softcap = _draft_value(self.config, "final_logit_softcapping")
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits
