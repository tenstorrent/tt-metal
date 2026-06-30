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

from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask
from models.experimental.diffusion_gemma.tt.diffusion_attention import denoise_attention
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    _rms_norm_dram,
    build_self_conditioning,
    build_self_conditioning_embedding_weight,
)
from models.experimental.diffusion_gemma.weight_mapping import GEMMA4_LM_PREFIX, remap_state_dict

NEG = -1.0e9


def default_self_conditioning_compute_kernel_config():
    """Use fp32 accumulation for production-vocab self-conditioning softmax."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _replicate_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def build_device_canvas_denoise_mask(
    mesh_device,
    *,
    prompt_len: int,
    canvas_len: int,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    dtype=ttnn.bfloat16,
):
    """Build a `[1, 1, C, P+C]` denoise mask on device."""
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        layer_type=layer_type,
        sliding_window=sliding_window,
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


def _layer_type_for_denoise(tt_model, layer_idx: int) -> str | None:
    layer_types = getattr(getattr(tt_model, "hf_config", None), "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_idx]
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    return getattr(attn_config, "layer_type", None)


def _sliding_window_for_denoise(tt_model, layer_idx: int) -> int | None:
    attn_config = getattr(getattr(tt_model.layers[layer_idx], "self_attn", None), "config", None)
    window = getattr(attn_config, "sliding_window", None)
    if window is not None:
        return window
    return getattr(getattr(tt_model, "hf_config", None), "sliding_window", None)


def _sliding_layer_needs_denoise_mask(prompt_len: int, canvas_len: int, sliding_window: int) -> bool:
    # HF's bidirectional sliding overlay allows abs(q_idx - kv_idx) <= sliding_window.
    return prompt_len + canvas_len - 1 > sliding_window


def _build_denoise_attn_mask_for_layer(
    tt_model,
    layer_idx: int,
    *,
    prompt_len: int,
    canvas_len: int,
    mask_builder=build_device_canvas_denoise_mask,
):
    layer_type = _layer_type_for_denoise(tt_model, layer_idx)
    if layer_type != "sliding_attention":
        return None

    sliding_window = _sliding_window_for_denoise(tt_model, layer_idx)
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(f"sliding_attention layer {layer_idx} requires a positive sliding_window")
    if not _sliding_layer_needs_denoise_mask(prompt_len, canvas_len, sliding_window):
        return None

    return mask_builder(
        tt_model.mesh_device,
        prompt_len=prompt_len,
        canvas_len=canvas_len,
        layer_type="sliding_attention",
        sliding_window=sliding_window,
    )


def _deallocate_optional_tensor(tensor) -> None:
    if tensor is not None and hasattr(tensor, "deallocate"):
        tensor.deallocate(True)


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
    out = denoise_attention(
        tt_model.layers[layer_idx].self_attn,
        canvas_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + canvas_len),
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


def _chunked_norm_forward(norm, hidden_states, *, chunk_size: int = 32):
    if getattr(norm, "with_scale", True) is False and getattr(norm, "tt_weight", None) is None:
        return _rms_norm_dram(hidden_states, epsilon=norm.eps, chunk_size=chunk_size)
    seq_len = hidden_states.shape[-2]
    if seq_len <= chunk_size:
        return norm.forward(hidden_states)

    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = ttnn.slice(
            hidden_states,
            [0, 0, start, 0],
            [hidden_states.shape[0], hidden_states.shape[1], end, hidden_states.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunks.append(norm.forward(chunk))
        chunk.deallocate(True)
    out = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for chunk in chunks:
        chunk.deallocate(True)
    return out


def _denoise_router_forward(router, hidden_states):
    normed = _chunked_norm_forward(router.norm, hidden_states)
    scaled = ttnn.mul(normed, router.scale)
    normed.deallocate(True)
    scaled = ttnn.mul(scaled, router.scalar_root_size)

    expert_scores = ttnn.linear(scaled, router.proj_weight)
    scaled.deallocate(True)

    router_probs = ttnn.softmax(expert_scores, dim=-1)
    expert_scores.deallocate(True)

    top_k_values, top_k_indices = ttnn.topk(router_probs, k=router.top_k, dim=-1)
    top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
    top_k_values = ttnn.div(top_k_values, top_k_sum)
    top_k_sum.deallocate(True)

    dense_routing = ttnn.scatter(
        ttnn.zeros_like(router_probs),
        dim=-1,
        index=top_k_indices,
        src=top_k_values,
    )
    router_probs.deallocate(True)
    top_k_values.deallocate(True)
    top_k_indices.deallocate(True)

    if router.per_expert_scale is not None:
        dense_routing = ttnn.mul(dense_routing, router.per_expert_scale)

    return dense_routing


def _denoise_moe_forward(moe, router_input, expert_input):
    dense_routing = _denoise_router_forward(moe.router, router_input)
    return moe.experts(expert_input, dense_routing)


def _denoise_layer_forward(tt_model, layer_idx, hidden_states, prompt_source, attn_mask, q_rope_offset):
    layer = tt_model.layers[layer_idx]
    residual = hidden_states
    normed = _chunked_norm_forward(layer.input_layernorm, hidden_states)
    prefix_kv = prompt_source if isinstance(prompt_source, (tuple, list)) else None
    kv_hidden = None if prefix_kv is not None else ttnn.concat([prompt_source, normed], dim=2)
    attn_output = denoise_attention(
        layer.self_attn,
        normed,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=q_rope_offset + hidden_states.shape[-2]),
        attn_mask=attn_mask,
        kv_hidden_states=kv_hidden,
        prefix_kv=prefix_kv,
        q_rope_offset=q_rope_offset,
    )
    if kv_hidden is not None:
        kv_hidden.deallocate(True)

    attn_output = _chunked_norm_forward(layer.post_attention_layernorm, attn_output)
    hidden_states = ttnn.add(residual, attn_output)
    residual.deallocate(True)
    attn_output.deallocate(True)

    residual = hidden_states
    normed = _chunked_norm_forward(layer.pre_feedforward_layernorm, hidden_states)
    mlp_output = layer.shared_mlp(normed)
    normed.deallocate(True)

    if layer.enable_moe_block:
        mlp_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_1, mlp_output)
        mlp_output.deallocate(True)
        expert_input = _chunked_norm_forward(layer.pre_feedforward_layernorm_2, residual)
        expert_output = _denoise_moe_forward(layer.moe, residual, expert_input)
        expert_input.deallocate(True)
        expert_normed = _chunked_norm_forward(layer.post_feedforward_layernorm_2, expert_output)
        expert_output.deallocate(True)
        hidden_states = ttnn.add(mlp_normed, expert_normed)
        mlp_normed.deallocate(True)
        expert_normed.deallocate(True)
    else:
        hidden_states = mlp_output

    hidden_states = _chunked_norm_forward(layer.post_feedforward_layernorm, hidden_states)
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
    mask_builder=build_device_canvas_denoise_mask,
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
    for layer_idx in range(len(tt_model.layers)):
        attn_mask = _build_denoise_attn_mask_for_layer(
            tt_model,
            layer_idx,
            prompt_len=prompt_len,
            canvas_len=hidden_states.shape[-2],
            mask_builder=mask_builder,
        )
        hidden_states = _denoise_layer_forward(
            tt_model,
            layer_idx,
            hidden_states,
            prompt_hidden_by_layer[layer_idx],
            attn_mask,
            q_rope_offset,
        )
        _deallocate_optional_tensor(attn_mask)
    final_hidden = _chunked_norm_forward(tt_model.norm, hidden_states)
    hidden_states.deallocate(True)
    return final_hidden


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
    if canvas_tokens.shape[0] != 1:
        raise ValueError("embed_canvas_tokens currently supports batch=1")
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

    def owns_logits(self, logits) -> bool:
        """Return True when ``logits`` is retained for next-step self-conditioning."""
        return self.prev_logits is logits

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


def make_denoise_logits_adapter_from_checkpoint_state(
    tt_model,
    *,
    prompt_len: int,
    self_conditioning_state,
    embedding_weight,
    config=None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    eps: float | None = None,
    seq_len_start: int = 0,
    q_rope_offset: int | None = None,
    self_conditioning_dtype=ttnn.bfloat16,
    self_conditioning_compute_kernel_config=None,
    default_compute_kernel_config_fn=default_self_conditioning_compute_kernel_config,
    self_conditioning_builder=build_self_conditioning,
    embedding_weight_builder=build_self_conditioning_embedding_weight,
    adapter_builder=make_denoise_logits_adapter_from_kv_cache,
):
    """Build the full denoise logits adapter from remapped real-checkpoint pieces."""
    self_conditioning = self_conditioning_builder(
        tt_model.mesh_device,
        self_conditioning_state,
        config=config,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        eps=eps,
        dtype=self_conditioning_dtype,
    )
    if config is not None and hidden_size is None:
        hidden_size = config["hidden_size"] if isinstance(config, dict) else config.hidden_size
    embedding_weight_tt = embedding_weight_builder(
        tt_model.mesh_device,
        embedding_weight,
        hidden_size=hidden_size,
        dtype=self_conditioning_dtype,
    )
    if self_conditioning_compute_kernel_config is None:
        self_conditioning_compute_kernel_config = default_compute_kernel_config_fn()
    return adapter_builder(
        tt_model,
        prompt_len=prompt_len,
        seq_len_start=seq_len_start,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=embedding_weight_tt,
        self_conditioning_compute_kernel_config=self_conditioning_compute_kernel_config,
        q_rope_offset=q_rope_offset,
    )


def make_denoise_logits_adapter_from_remapped_state(
    tt_model,
    *,
    prompt_len: int,
    backbone_state,
    self_conditioning_state,
    embedding_key: str = GEMMA4_LM_PREFIX + "embed_tokens.weight",
    checkpoint_adapter_builder=make_denoise_logits_adapter_from_checkpoint_state,
    **kwargs,
):
    """Build a denoise adapter from ``weight_mapping.remap_state_dict`` outputs."""
    if embedding_key not in backbone_state:
        raise ValueError(f"missing tied embedding weight in backbone_state: {embedding_key}")
    return checkpoint_adapter_builder(
        tt_model,
        prompt_len=prompt_len,
        self_conditioning_state=self_conditioning_state,
        embedding_weight=backbone_state[embedding_key],
        **kwargs,
    )


def make_generation_logits_fn_builder_from_remapped_state(
    *,
    backbone_state,
    self_conditioning_state,
    adapter_builder=make_denoise_logits_adapter_from_remapped_state,
    **adapter_kwargs,
):
    """Return a ``tt.generate`` post-prefill builder for remapped checkpoint state."""

    def logits_fn_builder(
        tt_model,
        *,
        prompt_tokens=None,
        prompt_len: int,
        page_table=None,
        page_tables_per_layer=None,
    ):
        del prompt_tokens, page_table, page_tables_per_layer
        return adapter_builder(
            tt_model,
            prompt_len=prompt_len,
            backbone_state=backbone_state,
            self_conditioning_state=self_conditioning_state,
            **adapter_kwargs,
        )

    return logits_fn_builder


def make_generation_logits_fn_builder_from_checkpoint_state(
    dg_state_dict,
    *,
    remap_fn=remap_state_dict,
    remapped_builder=make_generation_logits_fn_builder_from_remapped_state,
    **adapter_kwargs,
):
    """Return a generation logits builder directly from raw DiffusionGemma state."""
    backbone_state, self_conditioning_state, _ignored_keys = remap_fn(dg_state_dict)
    return remapped_builder(
        backbone_state=backbone_state,
        self_conditioning_state=self_conditioning_state,
        **adapter_kwargs,
    )
