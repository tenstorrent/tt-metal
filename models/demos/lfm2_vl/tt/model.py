# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ttnn implementation of LiquidAI LFM2.5-VL-1.6B.

Architecture mirrors HuggingFace ``Lfm2VlForConditionalGeneration``:
  - SigLIP2 vision tower
  - multimodal projector (pixel-unshuffle + MLP)
  - LFM2 hybrid language backbone (short-conv + GQA layers)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

import ttnn


def _to_tile(x):
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def _linear(x, weight, bias=None):
    """Linear with optional bias. Inputs/weights are expected in TILE layout."""
    x = _to_tile(x)
    weight = _to_tile(weight)
    out = ttnn.linear(x, weight)
    if bias is not None:
        out = ttnn.add(out, bias)
    return out


class TtLfm2RMSNorm:
    def __init__(self, weight, eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)


class TtLayerNorm:
    def __init__(self, weight, bias, eps: float):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x):
        return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)


class TtSigLip2VisionAttention:
    """Separate Q/K/V projections with multi-head attention (SigLIP2)."""

    def __init__(self, config: Dict[str, Any], parameters):
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.parameters = parameters

    def __call__(self, x):
        q = _linear(x, self.parameters.q_proj.weight, getattr(self.parameters.q_proj, "bias", None))
        k = _linear(x, self.parameters.k_proj.weight, getattr(self.parameters.k_proj, "bias", None))
        v = _linear(x, self.parameters.v_proj.weight, getattr(self.parameters.v_proj, "bias", None))

        batch, seq_len, _ = x.shape
        q = ttnn.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch, seq_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (batch, seq_len, self.num_heads, self.head_dim))

        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        k_t = ttnn.permute(k, (0, 1, 3, 2))
        attn = ttnn.matmul(q, k_t)
        attn = ttnn.mul(attn, self.scale)
        attn = ttnn.softmax(attn, dim=-1)
        out = ttnn.matmul(attn, v)

        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (batch, seq_len, self.hidden_size))
        return _linear(out, self.parameters.out_proj.weight, getattr(self.parameters.out_proj, "bias", None))


class TtSigLip2VisionMLP:
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, x):
        x = _linear(x, self.parameters.fc1.weight, getattr(self.parameters.fc1, "bias", None))
        x = ttnn.gelu(x)
        x = _linear(x, self.parameters.fc2.weight, getattr(self.parameters.fc2, "bias", None))
        return x


class TtSigLip2VisionBlock:
    def __init__(self, config: Dict[str, Any], parameters):
        eps = config["layer_norm_eps"]
        self.norm1 = TtLayerNorm(parameters.layer_norm1.weight, parameters.layer_norm1.bias, eps)
        self.norm2 = TtLayerNorm(parameters.layer_norm2.weight, parameters.layer_norm2.bias, eps)
        self.attn = TtSigLip2VisionAttention(config, parameters.self_attn)
        self.mlp = TtSigLip2VisionMLP(parameters.mlp)

    def __call__(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = ttnn.add(residual, x)

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        return ttnn.add(residual, x)


class TtSigLip2VisionEncoder:
    """SigLIP2 tower: patch embed + position embed + transformer stack + post LN."""

    def __init__(self, config: Dict[str, Any], parameters):
        self.config = config
        self.parameters = parameters
        self.blocks = [TtSigLip2VisionBlock(config, parameters.layers[i]) for i in range(config["num_hidden_layers"])]
        self.post_layernorm = TtLayerNorm(
            parameters.post_layernorm.weight,
            parameters.post_layernorm.bias,
            config["layer_norm_eps"],
        )

    def __call__(self, pixel_values):
        # pixel_values: [B, num_patches, patch_dim] already flattened patches
        x = _to_tile(pixel_values)
        x = _linear(
            x,
            self.parameters.patch_embedding.weight,
            getattr(self.parameters.patch_embedding, "bias", None),
        )
        if hasattr(self.parameters, "position_embedding"):
            x = ttnn.add(x, self.parameters.position_embedding.weight)

        for block in self.blocks:
            x = block(x)

        return self.post_layernorm(x)


class TtLfm2VlProjector:
    """Pixel-unshuffle + 2-layer MLP projector (HF Lfm2VlMultiModalProjector)."""

    def __init__(self, config: Dict[str, Any], parameters):
        self.factor = config["downsample_factor"]
        self.parameters = parameters
        self.use_layer_norm = config.get("projector_use_layernorm", False)
        self.layer_norm = None
        if self.use_layer_norm and hasattr(parameters, "layer_norm"):
            self.layer_norm = TtLayerNorm(parameters.layer_norm.weight, parameters.layer_norm.bias, 1e-5)

    def _pixel_unshuffle_torch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, W, H, C] as in HF
        batch_size, width, height, channels = hidden_states.shape
        factor = self.factor
        hidden_states = hidden_states.reshape(batch_size, width, height // factor, channels * factor)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size, height // factor, width // factor, channels * factor * factor
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        return hidden_states

    def __call__(self, vision_features, spatial_shapes=None, device=None):
        """
        Args:
            vision_features: ttnn tensor [B, N, C] or torch [B, H, W, C]
            spatial_shapes: optional (H, W) grid for unshuffle; defaults to square N
        """
        # Run unshuffle + flatten on host for variable-resolution correctness, then linear on device.
        if not isinstance(vision_features, torch.Tensor):
            feats = ttnn.to_torch(vision_features)
        else:
            feats = vision_features

        if feats.ndim == 3:
            b, n, c = feats.shape
            if spatial_shapes is None:
                side = int(n**0.5)
                h = w = side
            else:
                h, w = int(spatial_shapes[0]), int(spatial_shapes[1])
            feats = feats[:, : h * w, :].reshape(b, h, w, c)
        elif feats.ndim != 4:
            raise ValueError(f"Expected vision features rank 3 or 4, got shape {tuple(feats.shape)}")

        unshuffled = self._pixel_unshuffle_torch(feats)
        flat = unshuffled.reshape(unshuffled.shape[0], -1, unshuffled.shape[-1])

        x = ttnn.from_torch(
            flat,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = _linear(x, self.parameters.linear_1.weight, getattr(self.parameters.linear_1, "bias", None))
        x = ttnn.gelu(x)
        x = _linear(x, self.parameters.linear_2.weight, getattr(self.parameters.linear_2, "bias", None))
        return x


class TtLfm2ShortConv:
    """LIV short convolution block: in_proj -> B*x, depthwise conv1d, C*y, out_proj."""

    def __init__(self, config: Dict[str, Any], parameters):
        self.config = config
        self.parameters = parameters
        self.hidden_size = config["hidden_size"]

    def __call__(self, x):
        projected = _linear(x, self.parameters.in_proj.weight)
        # split into B, C, x_proj along last dim
        b_gate, c_gate, x_proj = ttnn.split(projected, self.hidden_size, dim=-1)
        x_gated = ttnn.mul(b_gate, x_proj)

        x_rm = ttnn.to_layout(x_gated, ttnn.ROW_MAJOR_LAYOUT)
        x_ncl = ttnn.permute(x_rm, (0, 2, 1))  # [B, C, L]
        x_conv = ttnn.conv1d(
            x_ncl,
            self.parameters.conv.weight,
            groups=self.hidden_size,
        )
        x_conv = ttnn.permute(x_conv, (0, 2, 1))
        x_conv = _to_tile(x_conv)

        y = ttnn.mul(c_gate, x_conv)
        return _linear(y, self.parameters.out_proj.weight)


class TtLfm2Attention:
    """GQA attention with per-head Q/K RMSNorm and RoPE."""

    def __init__(self, config: Dict[str, Any], parameters):
        self.config = config
        self.parameters = parameters
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.hidden_size = config["hidden_size"]
        self.scale = self.head_dim**-0.5
        self.q_layernorm = TtLfm2RMSNorm(parameters.q_layernorm.weight, config["norm_eps"])
        self.k_layernorm = TtLfm2RMSNorm(parameters.k_layernorm.weight, config["norm_eps"])

    def _apply_rope(self, q, k, cos, sin):
        try:
            q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, None, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, None, is_decode_mode=False)
            return q, k
        except Exception:
            # Half-rotation fallback matching HF rotate_half style
            def rotate_half(t):
                half = t.shape[-1] // 2
                t1 = t[..., :half]
                t2 = t[..., half:]
                return ttnn.concat([ttnn.mul(t2, -1.0), t1], dim=-1)

            q = ttnn.add(ttnn.mul(q, cos), ttnn.mul(rotate_half(q), sin))
            k = ttnn.add(ttnn.mul(k, cos), ttnn.mul(rotate_half(k), sin))
            return q, k

    def __call__(self, x, cos_sin_cache=None, layer_past=None, use_cache: bool = False):
        batch, seq_len, _ = x.shape

        q = _linear(x, self.parameters.q_proj.weight)
        k = _linear(x, self.parameters.k_proj.weight)
        v = _linear(x, self.parameters.v_proj.weight)

        q = ttnn.reshape(q, (batch, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch, seq_len, self.num_kv_heads, self.head_dim))
        v = ttnn.reshape(v, (batch, seq_len, self.num_kv_heads, self.head_dim))

        # RMSNorm over head_dim: apply on [B, S, H, D] via reshape to last dim
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        if cos_sin_cache is not None:
            cos, sin = cos_sin_cache
            q, k = self._apply_rope(q, k, cos, sin)

        if use_cache and layer_past is not None:
            k_cache, v_cache = layer_past
            k = ttnn.concat([k_cache, k], dim=2)
            v = ttnn.concat([v_cache, v], dim=2)

        n_repeat = self.num_heads // self.num_kv_heads
        if n_repeat > 1:
            k = ttnn.repeat_interleave(k, n_repeat, dim=1) if hasattr(ttnn, "repeat_interleave") else ttnn.repeat(
                k, (1, n_repeat, 1, 1)
            )
            v = ttnn.repeat_interleave(v, n_repeat, dim=1) if hasattr(ttnn, "repeat_interleave") else ttnn.repeat(
                v, (1, n_repeat, 1, 1)
            )

        k_t = ttnn.permute(k, (0, 1, 3, 2))
        attn = ttnn.matmul(q, k_t)
        attn = ttnn.mul(attn, self.scale)
        attn = ttnn.softmax(attn, dim=-1)
        out = ttnn.matmul(attn, v)

        present = (k, v) if use_cache else None

        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (batch, -1, self.num_heads * self.head_dim))
        out = _linear(out, self.parameters.out_proj.weight)
        return out, present


class TtLfm2MLP:
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, x):
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        w1 = _linear(x, self.parameters.w1.weight)
        w3 = _linear(x, self.parameters.w3.weight)
        return _linear(ttnn.mul(ttnn.silu(w1), w3), self.parameters.w2.weight)


class TtLfm2DecoderLayer:
    """Pre-norm hybrid layer: short-conv or GQA, then feed-forward."""

    def __init__(self, config: Dict[str, Any], parameters, layer_type: str):
        self.layer_type = layer_type
        self.operator_norm = TtLfm2RMSNorm(parameters.operator_norm.weight, config["norm_eps"])
        self.ffn_norm = TtLfm2RMSNorm(parameters.ffn_norm.weight, config["norm_eps"])
        self.feed_forward = TtLfm2MLP(parameters.feed_forward)
        if layer_type == "full_attention":
            self.operator = TtLfm2Attention(config, parameters.self_attn)
        else:
            self.operator = TtLfm2ShortConv(config, parameters.conv)

    def __call__(self, x, cos_sin_cache=None, layer_past=None, use_cache: bool = False):
        residual = x
        h = self.operator_norm(x)
        present = None
        if self.layer_type == "full_attention":
            h, present = self.operator(h, cos_sin_cache=cos_sin_cache, layer_past=layer_past, use_cache=use_cache)
        else:
            h = self.operator(h)
        x = ttnn.add(residual, h)

        residual = x
        x = self.feed_forward(self.ffn_norm(x))
        x = ttnn.add(residual, x)
        return x, present


class TtLfm2VlModel:
    """LFM2.5-VL multimodal model (vision + projector + hybrid LM)."""

    def __init__(self, device, config: Dict[str, Any], parameters):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.image_token_id = config["image_token_id"]

        self.vision_tower = TtSigLip2VisionEncoder(config["vision_config"], parameters.vision_tower)
        self.multi_modal_projector = TtLfm2VlProjector(config, parameters.multi_modal_projector)

        self.layers: List[TtLfm2DecoderLayer] = []
        for i, layer_type in enumerate(config["layer_types"]):
            self.layers.append(TtLfm2DecoderLayer(config, parameters.layers[i], layer_type))

        self.embedding_norm = TtLfm2RMSNorm(parameters.embedding_norm.weight, config["norm_eps"])
        self.cos_sin_cache = None

    def set_rope_cache(self, cos_sin_cache: Tuple):
        self.cos_sin_cache = cos_sin_cache

    def get_image_features(self, pixel_values, spatial_shapes=None):
        vision_out = self.vision_tower(pixel_values)
        return self.multi_modal_projector(vision_out, spatial_shapes=spatial_shapes, device=self.device)

    def _merge_image_features(self, input_ids, inputs_embeds, image_features):
        """Scatter projected image tokens into placeholder positions (HF semantics)."""
        ids = ttnn.to_torch(input_ids).to(torch.long)
        embeds = ttnn.to_torch(inputs_embeds)
        feats = ttnn.to_torch(image_features)

        if feats.ndim == 3:
            feats = feats.reshape(-1, feats.shape[-1])

        mask = ids == self.image_token_id
        n_tokens = int(mask.sum().item())
        n_features = int(feats.shape[0])
        if n_tokens != n_features:
            raise ValueError(
                f"Image features and image tokens do not match, tokens: {n_tokens}, features: {n_features}"
            )

        embeds = embeds.clone()
        embeds[mask] = feats.to(dtype=embeds.dtype)
        return ttnn.from_torch(
            embeds,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def __call__(
        self,
        pixel_values=None,
        input_ids=None,
        inputs_embeds=None,
        spatial_shapes=None,
        layer_past=None,
        use_cache: bool = False,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            # Allow either path; if both provided prefer embeds after merge
            if input_ids is None and inputs_embeds is None:
                raise ValueError("Specify input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = ttnn.embedding(input_ids, self.parameters.embed_tokens.weight)
            inputs_embeds = _to_tile(inputs_embeds)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, spatial_shapes=spatial_shapes)
            if input_ids is None:
                raise ValueError("input_ids required to locate image placeholders when pixel_values are provided")
            x = self._merge_image_features(input_ids, inputs_embeds, image_features)
        else:
            x = _to_tile(inputs_embeds)

        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = layer_past[i] if layer_past is not None else None
            x, present = layer(
                x,
                cos_sin_cache=self.cos_sin_cache,
                layer_past=past,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        x = self.embedding_norm(x)
        if use_cache:
            return x, presents
        return x


class TtLfm2VlForConditionalGeneration:
    """Causal LM head on top of TtLfm2VlModel."""

    def __init__(self, device, config: Dict[str, Any], parameters):
        self.model = TtLfm2VlModel(device, config, parameters)
        self.lm_head_weight = parameters.lm_head.weight if hasattr(parameters, "lm_head") else parameters.embed_tokens.weight
        self.config = config
        self.device = device

    def set_rope_cache(self, cos_sin_cache):
        self.model.set_rope_cache(cos_sin_cache)

    def __call__(self, **kwargs):
        hidden = self.model(**kwargs)
        if isinstance(hidden, tuple):
            hidden, past = hidden
            logits = _linear(hidden, self.lm_head_weight)
            return logits, past
        return _linear(hidden, self.lm_head_weight)
