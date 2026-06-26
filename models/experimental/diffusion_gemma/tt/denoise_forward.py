# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN denoise-forward helpers for DiffusionGemma canvas attention.

Canvas queries attend to the frozen prompt prefix plus the current canvas as an
all-attend rectangle. The canonical path is maskless non-causal SDPA; explicit
masks are kept only for op A/B tests and non-canonical experiments. Sampling,
self-conditioning, and the multi-layer generation loop live in later W3/W4
helpers; this module owns the real denoise attention wiring.
"""

from __future__ import annotations

import torch
import ttnn

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask

NEG = -1.0e9


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def build_device_canvas_denoise_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    dtype=ttnn.bfloat16,
):
    """Build the canonical all-attend `[1, 1, C, P+C]` denoise mask on device."""
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, prompt_len + canvas_len)
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def denoise_attention_forward(
    tt_model,
    *,
    layer_idx: int,
    prompt_hidden=None,
    prompt_kv=None,
    canvas_hidden,
    attn_mask=None,
    q_rope_offset: int | None = None,
):
    """Run one DiffusionGemma denoise attention layer on canvas hidden states.

    Args:
        tt_model: `Gemma4Model` carrying the reused DiffusionGemma decoder weights.
        layer_idx: decoder layer to run.
        prompt_hidden: frozen prompt hidden states `[1, 1, P, H]` on device.
        prompt_kv: optional frozen projected prompt `(K, V)` heads. This is the
            cache-shaped input used by the eventual paged encoder KV read path.
        canvas_hidden: current canvas hidden states `[1, 1, C, H]` on device.
        attn_mask: optional prebuilt `[1, 1, C, P+C]` additive mask on device.
            Leave unset for the canonical all-attend denoise path.

    Returns:
        The attention output for the canvas positions `[1, 1, C, H]`.
    """
    if (prompt_hidden is None) == (prompt_kv is None):
        raise ValueError("pass exactly one of prompt_hidden or prompt_kv")
    prompt_len = prompt_kv[0].shape[-2] if prompt_kv is not None else prompt_hidden.shape[-2]
    canvas_len = canvas_hidden.shape[-2]
    q_rope_offset = prompt_len if q_rope_offset is None else q_rope_offset
    kv_hidden = None if prompt_kv is not None else ttnn.concat([prompt_hidden, canvas_hidden], dim=2)
    out = tt_model.layers[layer_idx].self_attn(
        canvas_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + canvas_len),
        is_decode=False,
        is_causal=False,
        kv_phase=KVCachePhase.DENOISE_READONLY,
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        prefix_kv=prompt_kv,
        q_rope_offset=q_rope_offset,
    )
    if kv_hidden is not None:
        kv_hidden.deallocate(True)
    return out


def _prompt_source_len(prompt_source):
    return prompt_source[0].shape[-2] if isinstance(prompt_source, (tuple, list)) else prompt_source.shape[-2]


def _denoise_layer_forward(tt_model, layer_idx, hidden_states, prompt_source, attn_mask, prompt_len, q_rope_offset):
    layer = tt_model.layers[layer_idx]
    residual = hidden_states
    normed = layer.input_layernorm.forward(hidden_states)
    prefix_kv = prompt_source if isinstance(prompt_source, (tuple, list)) else None
    kv_hidden = None if prefix_kv is not None else ttnn.concat([prompt_source, normed], dim=2)
    attn_output = layer.self_attn(
        normed,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + hidden_states.shape[-2]),
        is_decode=False,
        is_causal=False,
        kv_phase=KVCachePhase.DENOISE_READONLY,
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        prefix_kv=prefix_kv,
        q_rope_offset=q_rope_offset,
    )
    normed.deallocate(True)
    if kv_hidden is not None:
        kv_hidden.deallocate(True)

    attn_output = layer.post_attention_layernorm.forward(attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = layer.pre_feedforward_layernorm.forward(hidden_states)
    mlp_output = layer.shared_mlp(normed)
    normed.deallocate(True)

    if layer.enable_moe_block:
        mlp_normed = layer.post_feedforward_layernorm_1.forward(mlp_output)
        mlp_output.deallocate(True)
        expert_input = layer.pre_feedforward_layernorm_2.forward(residual)
        expert_output = layer.moe(residual, expert_input)
        expert_input.deallocate(True)
        expert_normed = layer.post_feedforward_layernorm_2.forward(expert_output)
        expert_output.deallocate(True)
        hidden_states = ttnn.add(mlp_normed, expert_normed)
        mlp_normed.deallocate(True)
        expert_normed.deallocate(True)
    else:
        hidden_states = mlp_output

    hidden_states = layer.post_feedforward_layernorm.forward(hidden_states)
    combined = ttnn.add(residual, hidden_states)
    residual.deallocate(True)
    hidden_states.deallocate(True)
    if layer.layer_scalar != 1.0:
        scaled = ttnn.mul(combined, layer.layer_scalar)
        combined.deallocate(True)
        combined = scaled
    return combined


def denoise_hidden_forward(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_hidden,
    q_rope_offset: int | None = None,
):
    """Run the short-prompt DiffusionGemma denoise backbone to final hidden states.

    ``prompt_hidden_by_layer`` provides the frozen encoder-side attention source
    for each decoder layer. Entries can be either `[1, 1, P, H]` hidden tensors
    (legacy shim) or projected `(K, V)` prompt heads produced by the encoder KV path.
    """
    if len(prompt_hidden_by_layer) != len(tt_model.layers):
        raise ValueError(
            f"prompt_hidden_by_layer has {len(prompt_hidden_by_layer)} entries but model has {len(tt_model.layers)} layers"
        )

    hidden_states = canvas_hidden
    prompt_len = _prompt_source_len(prompt_hidden_by_layer[0])
    q_rope_offset = prompt_len if q_rope_offset is None else q_rope_offset
    attn_mask = None
    for layer_idx in range(len(tt_model.layers)):
        hidden_states = _denoise_layer_forward(
            tt_model,
            layer_idx,
            hidden_states,
            prompt_hidden_by_layer[layer_idx],
            attn_mask,
            prompt_len,
            q_rope_offset,
        )
    return tt_model.norm.forward(hidden_states)


def denoise_logits_forward(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_hidden,
    q_rope_offset: int | None = None,
):
    """Run a short-prompt DiffusionGemma denoise logits forward.

    The returned logits cover all canvas positions, which the diffusion sampler
    consumes each denoise step.
    """
    hidden_states = denoise_hidden_forward(
        tt_model,
        prompt_hidden_by_layer=prompt_hidden_by_layer,
        canvas_hidden=canvas_hidden,
        q_rope_offset=q_rope_offset,
    )
    return tt_model._apply_lm_head(hidden_states, is_decode=False)


def collect_prompt_hidden_by_layer(tt_model, prompt_hidden):
    """Collect per-layer frozen prompt attention inputs for denoise K/V source.

    The DiffusionGemma decoder reads encoder K/V. Until the real paged encoder
    cache is threaded into this wrapper, this helper captures the tensor that
    feeds each layer's K/V projections: the prompt hidden states after that
    layer's input RMSNorm, while advancing the prompt through the normal causal
    layer stack. Returned tensors are owned by the caller.
    """
    hidden_states = prompt_hidden
    prompt_hidden_by_layer = []
    for layer_idx, layer in enumerate(tt_model.layers):
        prompt_hidden_by_layer.append(layer.input_layernorm.forward(hidden_states))
        hidden_states = layer(
            hidden_states,
            rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=hidden_states.shape[-2]),
            position_idx=None,
            page_table=None,
            kv_cache=None,
            is_decode=False,
            kv_phase=KVCachePhase.DENOISE_READONLY,
        )
    hidden_states.deallocate(True)
    return prompt_hidden_by_layer


def collect_prompt_kv_by_layer(tt_model, prompt_hidden):
    """Collect per-layer frozen prompt K/V heads for denoise prefix attention.

    This uses the existing Gemma4 prefill ``keep_kv`` path to capture K/V after
    per-head norm and RoPE. Those tensors match the shape carried by KV caches,
    so this is the narrow interface the future paged encoder-cache read should
    populate.
    """
    hidden_states = prompt_hidden
    prompt_kv_by_layer = []
    for layer_idx, layer in enumerate(tt_model.layers):
        hidden_states = layer(
            hidden_states,
            rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=hidden_states.shape[-2]),
            position_idx=None,
            page_table=None,
            kv_cache=None,
            is_decode=False,
            keep_kv=True,
            kv_phase=KVCachePhase.DENOISE_READONLY,
        )
        prompt_kv_by_layer.append(layer.self_attn._last_kv)
    hidden_states.deallocate(True)
    return prompt_kv_by_layer


def read_prompt_kv_cache_slice(kv_cache, *, prompt_len: int, seq_len_start: int = 0):
    """Read a frozen prompt K/V prefix from a contiguous Gemma4 KV cache.

    This is the non-paged cache adapter for W2: it reads the encoder-written K/V
    heads from `[B, heads, max_seq, head_dim]` cache tensors and returns the
    cache-shaped `(K, V)` prefix accepted by denoise attention. The underlying
    TTNN slice op requires full tiles along sequence, so bounds must be 32-aligned.
    """
    seq_len_end = seq_len_start + prompt_len
    if seq_len_start % ttnn.TILE_SIZE != 0 or seq_len_end % ttnn.TILE_SIZE != 0:
        raise ValueError("KV cache slice bounds must be multiples of 32")
    k_cache, v_cache = kv_cache
    return (
        ttnn.experimental.nlp_kv_cache_load_slice(
            k_cache,
            seq_len_start=seq_len_start,
            seq_len_end=seq_len_end,
        ),
        ttnn.experimental.nlp_kv_cache_load_slice(
            v_cache,
            seq_len_start=seq_len_start,
            seq_len_end=seq_len_end,
        ),
    )


def read_prompt_kv_cache_by_layer(
    tt_model, *, prompt_len: int, seq_len_start: int = 0, read_fn=read_prompt_kv_cache_slice
):
    """Read frozen prompt K/V prefixes from every layer's Gemma4 KV cache.

    This is the production-shaped prompt source for the denoise adapter: one
    `(K, V)` tuple per decoder layer, rather than the early single-layer test
    shim. The returned tensors are owned by the caller.
    """
    if len(tt_model.tt_kv_cache) != len(tt_model.layers):
        raise ValueError(
            f"tt_kv_cache has {len(tt_model.tt_kv_cache)} layers but model has {len(tt_model.layers)} layers"
        )
    return [read_fn(kv_cache, prompt_len=prompt_len, seq_len_start=seq_len_start) for kv_cache in tt_model.tt_kv_cache]


def embed_canvas_tokens(tt_model, canvas_tokens):
    """Embed device canvas token ids into `[1, 1, C, H]` TILE hidden states."""
    if len(canvas_tokens.shape) == 4 and canvas_tokens.shape[-1] == 1:
        canvas_len = canvas_tokens.shape[-2]
        token_ids = ttnn.reshape(canvas_tokens, (canvas_tokens.shape[0], canvas_len))
        token_ids = ttnn.to_layout(token_ids, ttnn.ROW_MAJOR_LAYOUT)
    else:
        canvas_len = canvas_tokens.shape[-1]
        token_ids = canvas_tokens
    canvas_hidden = tt_model.embed_tokens(token_ids)
    if token_ids is not canvas_tokens:
        token_ids.deallocate(True)
    if len(canvas_hidden.shape) == 3:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    elif canvas_hidden.shape[-2] != canvas_len:
        canvas_hidden = ttnn.reshape(canvas_hidden, (1, 1, canvas_len, tt_model.hidden_size))
    return ttnn.to_layout(canvas_hidden, ttnn.TILE_LAYOUT)


def denoise_logits_from_tokens(
    tt_model,
    *,
    prompt_hidden_by_layer,
    canvas_tokens,
    self_conditioning=None,
    prev_logits=None,
    q_rope_offset: int | None = None,
    self_conditioning_embedding_weight=None,
    self_conditioning_compute_kernel_config=None,
):
    """Embed canvas token ids, optionally self-condition, then run denoise logits."""
    canvas_hidden = embed_canvas_tokens(tt_model, canvas_tokens)
    if self_conditioning is not None:
        conditioned = self_conditioning.condition(
            canvas_hidden,
            prev_logits,
            self_conditioning_embedding_weight,
            compute_kernel_config=self_conditioning_compute_kernel_config,
        )
        canvas_hidden.deallocate(True)
        canvas_hidden = conditioned
    return denoise_logits_forward(
        tt_model,
        prompt_hidden_by_layer=prompt_hidden_by_layer,
        canvas_hidden=canvas_hidden,
        q_rope_offset=q_rope_offset,
    )


class DenoiseLogitsAdapter:
    """Stateful W2 logits callback for the W3 denoise controller.

    The controller calls ``logits_fn(canvas_tokens, step)``. This adapter turns
    that narrow callback into the real W2 path: token embedding, optional
    self-conditioning from the previous step's logits, and denoise logits forward.
    """

    def __init__(
        self,
        tt_model,
        *,
        prompt_hidden_by_layer,
        self_conditioning=None,
        self_conditioning_embedding_weight=None,
        self_conditioning_compute_kernel_config=None,
        q_rope_offset: int | None = None,
        logits_from_tokens=denoise_logits_from_tokens,
    ):
        self.tt_model = tt_model
        self.prompt_hidden_by_layer = prompt_hidden_by_layer
        self.self_conditioning = self_conditioning
        self.self_conditioning_embedding_weight = self_conditioning_embedding_weight
        self.self_conditioning_compute_kernel_config = self_conditioning_compute_kernel_config
        self.q_rope_offset = q_rope_offset
        self.logits_from_tokens = logits_from_tokens
        self.prev_logits = None

    def __call__(self, canvas_tokens, step: int):
        old_prev_logits = self.prev_logits
        logits = self.logits_from_tokens(
            self.tt_model,
            prompt_hidden_by_layer=self.prompt_hidden_by_layer,
            canvas_tokens=canvas_tokens,
            self_conditioning=self.self_conditioning,
            prev_logits=old_prev_logits,
            q_rope_offset=self.q_rope_offset,
            self_conditioning_embedding_weight=self.self_conditioning_embedding_weight,
            self_conditioning_compute_kernel_config=self.self_conditioning_compute_kernel_config,
        )
        self.prev_logits = logits
        if old_prev_logits is not None:
            old_prev_logits.deallocate(True)
        return logits

    def reset(self):
        if self.prev_logits is not None:
            self.prev_logits.deallocate(True)
            self.prev_logits = None


def make_denoise_logits_adapter_from_kv_cache(
    tt_model,
    *,
    prompt_len: int,
    seq_len_start: int = 0,
    self_conditioning=None,
    self_conditioning_embedding_weight=None,
    self_conditioning_compute_kernel_config=None,
    q_rope_offset: int | None = None,
    read_prompt_kv_fn=read_prompt_kv_cache_by_layer,
    adapter_cls=DenoiseLogitsAdapter,
):
    """Build a denoise logits adapter from the model's per-layer prompt KV cache."""
    prompt_kv_by_layer = read_prompt_kv_fn(
        tt_model,
        prompt_len=prompt_len,
        seq_len_start=seq_len_start,
    )
    return adapter_cls(
        tt_model,
        prompt_hidden_by_layer=prompt_kv_by_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=self_conditioning_embedding_weight,
        self_conditioning_compute_kernel_config=self_conditioning_compute_kernel_config,
        q_rope_offset=prompt_len if q_rope_offset is None else q_rope_offset,
    )
