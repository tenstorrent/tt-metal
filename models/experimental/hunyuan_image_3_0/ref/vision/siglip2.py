# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 SigLIP2 vision encoder + aligner.
# Used as the golden reference for TT-Metal numeric (PCC) validation.
#
# Extracted / adapted from:
#   HunyuanImage-3.0/hunyuan_image_3/siglip2.py
#     Siglip2VisionEmbeddings          lines 54-152
#     Siglip2Attention / SdpaAttention lines 155-298
#     Siglip2MLP                       lines 301-313
#     Siglip2EncoderLayer              lines 316-363
#     Siglip2Encoder                   lines 366-444
#     Siglip2VisionTransformer         lines 478-540
#     LightProjector                   lines 543-564
#
# This port is SELF-CONTAINED (no `transformers` import): the only upstream
# helpers it depends on (`_prepare_4d_attention_mask`, `gelu_pytorch_tanh`) are
# reimplemented verbatim below so the golden has no heavy dependencies.
#
# Scope for HunyuanImage I2I (TI2I):
#   * Only `last_hidden_state` is consumed (modeling_hunyuan_image_3.py
#     _forward_vision_encoder -> vision_aligner). The multihead attention
#     pooling head produces `pooler_output`, which is UNUSED, so it is omitted
#     here (use_head defaults False).
#
# Config: siglip2-so400m-patch16-naflex
#   (ref/tokenizer/assets/config.json -> "vit" / "vit_aligner")
#
# Inputs (from Siglip2ImageProcessor, see image_processor.py):
#   pixel_values:          [B, max_num_patches, num_channels * patch_size**2]
#   attention_mask:        [B, max_num_patches]  (1 = real patch, 0 = pad)
#   spatial_shapes:        [B, 2]  (token_h, token_w) in PATCH units, per image
#
# References
# ----------
#   ref/weights.py                       — safetensors loaders
#   tt/vision/siglip2.py                 — TTNN port mirrored against this file

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.hunyuan_image_3_0.ref.model_config import ALIGNER_CONFIG, VIT_CONFIG
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_prefixed_state_dict


@dataclass
class SiglipConfig:
    hidden_size: int = VIT_CONFIG["hidden_size"]
    intermediate_size: int = VIT_CONFIG["intermediate_size"]
    num_attention_heads: int = VIT_CONFIG["num_attention_heads"]
    num_hidden_layers: int = VIT_CONFIG["num_hidden_layers"]
    num_channels: int = VIT_CONFIG["num_channels"]
    patch_size: int = VIT_CONFIG["patch_size"]
    num_patches: int = VIT_CONFIG["num_patches"]
    layer_norm_eps: float = VIT_CONFIG["layer_norm_eps"]
    attention_dropout: float = VIT_CONFIG["attention_dropout"]
    hidden_act: str = VIT_CONFIG["hidden_act"]

    @classmethod
    def from_dict(cls, cfg: dict) -> "SiglipConfig":
        fields = {k: cfg[k] for k in cfg if k in cls.__dataclass_fields__}
        return cls(**fields)


def prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None) -> torch.Tensor:
    """Verbatim port of transformers `_prepare_4d_attention_mask`.

    Expands a [B, src_len] padding mask (1 = keep) into an additive
    [B, 1, tgt_len, src_len] mask where padded KEY positions are -inf. Note this
    masks keys (columns) only, matching upstream SigLIP2.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def build_resize_matrix_host(
    num_patches: int,
    height: int,
    width: int,
    max_length: int,
) -> torch.Tensor:
    """Linear operator R [max_length, num_patches] for the naflex pos resize.

    Bilinear(+antialias) interpolation is linear in the input grid, so the whole
    resize is a single matrix: resized = R @ position_embedding_weight. R is
    extracted by running the same F.interpolate on the num_patches-identity
    (depends only on the target size, not on the weights). Padded rows (>= height*width)
    are set to R[0] so that R @ W reproduces the reference's ``out[h*w:] = resized[0]``.
    """
    side = int(num_patches**0.5)
    eye = torch.eye(num_patches, dtype=torch.float32).reshape(num_patches, 1, side, side)
    resized = F.interpolate(eye, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    r = resized.reshape(num_patches, height * width).transpose(0, 1).contiguous()

    out = torch.empty((max_length, num_patches), dtype=torch.float32)
    hw = height * width
    out[:hw] = r
    out[hw:] = r[0]
    return out


class Siglip2VisionEmbeddings(nn.Module):
    """Patch projection (Linear) + naflex resized positional embeddings."""

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """Resize positional embeddings to per-image size and pad to max_length.

        Args:
            positional_embeddings: [height, width, embed_dim]
            spatial_shapes:        [batch_size, 2]  (height, width) in patches
            max_length:            padded sequence length (= pixel_values.shape[1])
        Returns:
            [batch_size, max_length, embed_dim]
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # antialias is not supported for bf16/fp16 on CPU -> upcast there.
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # (1, dim, h, w) -> (h*w, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """pixel_values: [B, max_num_patches, num_channels * patch**2]."""
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )
        return patch_embeds + resized_positional_embeddings


class Siglip2Attention(nn.Module):
    """Multi-head self-attention (SDPA), bidirectional (is_causal=False)."""

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)
        return self.out_proj(attn_output)


class Siglip2MLP(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        if config.hidden_act != "gelu_pytorch_tanh":
            raise NotImplementedError(f"hidden_act={config.hidden_act!r} not supported; expected gelu_pytorch_tanh")
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    """Pre-LN transformer block: x + attn(LN1(x)); x + mlp(LN2(x))."""

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    def __init__(self, config: SiglipConfig, num_layers: int | None = None):
        super().__init__()
        self.config = config
        n = config.num_hidden_layers if num_layers is None else num_layers
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(n)])

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    """SigLIP2 vision tower. Returns last_hidden_state (pooling head omitted)."""

    def __init__(self, config: SiglipConfig, num_layers: int | None = None):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config, num_layers=num_layers)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        if attention_mask is not None:
            encoder_attention_mask = prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = None

        hidden_states = self.encoder(hidden_states, encoder_attention_mask)
        return self.post_layernorm(hidden_states)


class LightProjector(nn.Module):
    """vision_aligner: mlp_gelu, depth=2 -> Linear(1152, 4096), GELU, Linear(4096, 4096)."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = ALIGNER_CONFIG if config is None else config
        if config["projector_type"] == "linear":
            modules = nn.Linear(config["input_dim"], config["n_embed"])
        elif config["projector_type"] == "mlp_gelu":
            mlp = [nn.Linear(config["input_dim"], config["n_embed"])]
            for _ in range(1, config["depth"]):
                mlp.append(nn.GELU())
                mlp.append(nn.Linear(config["n_embed"], config["n_embed"]))
            modules = nn.Sequential(*mlp)
        else:
            raise ValueError(f"Unknown projector type: {config['projector_type']}")
        self.layers = modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def forward_vision_with_aligner(
    vision_model: Siglip2VisionTransformer,
    aligner: LightProjector,
    pixel_values: torch.Tensor,
    *,
    spatial_shapes: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """SigLIP2 vision tower + aligner (upstream ``_forward_vision_encoder``)."""
    image_embeds = vision_model(pixel_values, attention_mask, spatial_shapes)
    return aligner(image_embeds)


# ---------------------------------------------------------------------------
# Checkpoint loaders (real HunyuanImage-3.0-Instruct weights)
# ---------------------------------------------------------------------------
def load_siglip2_vision(
    model_dir=MODEL_DIR,
    *,
    num_layers: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> Siglip2VisionTransformer:
    """Build Siglip2VisionTransformer and load `vision_model.*` weights.

    Pass num_layers to load a partial stack (for fast layer-0 PCC tests).
    The unused pooling head (`vision_model.head.*`) is ignored.
    """
    cfg = SiglipConfig.from_dict(VIT_CONFIG)
    model = Siglip2VisionTransformer(cfg, num_layers=num_layers).to(dtype).eval()

    sd = load_prefixed_state_dict(model_dir, "vision_model.", dtype=dtype)
    # Drop unused pooling-head weights so load_state_dict(strict) stays clean.
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    if num_layers is not None:
        sd = {k: v for k, v in sd.items() if not _is_dropped_layer(k, num_layers)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _assert_clean_load(missing, unexpected)
    return model


def load_aligner(
    model_dir=MODEL_DIR,
    *,
    dtype: torch.dtype = torch.float32,
) -> LightProjector:
    """Build LightProjector and load `vision_aligner.*` weights."""
    model = LightProjector(ALIGNER_CONFIG).to(dtype).eval()
    sd = load_prefixed_state_dict(model_dir, "vision_aligner.", dtype=dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _assert_clean_load(missing, unexpected)
    return model


def _is_dropped_layer(key: str, num_layers: int) -> bool:
    prefix = "encoder.layers."
    if not key.startswith(prefix):
        return False
    idx = int(key[len(prefix) :].split(".", 1)[0])
    return idx >= num_layers


def _assert_clean_load(missing: list[str], unexpected: list[str]) -> None:
    # Allow zero missing; unexpected should be empty after head filtering.
    real_missing = [k for k in missing if not k.startswith("head.")]
    if real_missing:
        raise RuntimeError(f"Missing weights when loading SigLIP2 reference: {real_missing[:8]} ...")
    if unexpected:
        raise RuntimeError(f"Unexpected weights when loading SigLIP2 reference: {unexpected[:8]} ...")


# ---------------------------------------------------------------------------
# Quick smoke-test (random weights; checks shapes + mask plumbing only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = SiglipConfig.from_dict(VIT_CONFIG)
    vit = Siglip2VisionTransformer(cfg, num_layers=2).eval()
    aligner = LightProjector(ALIGNER_CONFIG).eval()

    B, S = 2, 64
    patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size
    pixel_values = torch.randn(B, S, patch_dim)
    spatial_shapes = torch.tensor([[8, 8], [4, 4]], dtype=torch.long)
    attn = torch.ones(B, S, dtype=torch.long)
    attn[1, 16:] = 0  # second image only has 4*4 = 16 real patches

    with torch.no_grad():
        last_hidden = vit(pixel_values, attn, spatial_shapes)
        projected = aligner(last_hidden)

    print(f"last_hidden_state: {tuple(last_hidden.shape)}  (expect ({B}, {S}, {cfg.hidden_size}))")
    print(f"aligned:           {tuple(projected.shape)}  (expect ({B}, {S}, {ALIGNER_CONFIG['n_embed']}))")
