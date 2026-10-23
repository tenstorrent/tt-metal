# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference (CPU golden) for the MiniMax-M3 vision encoder: the vision tower and the projector.

Plain PyTorch, no ttnn and no transformers. It follows Hugging Face transformers 5.12.1
``minimax_m3_vl`` (``MiniMaxM3VLVisionModel``, then ``MiniMaxM3VLMultiModalProjector``) op for op;
``tests/test_vision_reference_vs_hf.py`` checks that on CPU. The vision unit tests
(``tests/unit/test_vision_*_vs_ref.py``) compare the TT modules against this reference.

One forward takes all patches of a request as one sequence of L patches:

    pixel_values (L, 1176)   one row per patch: 3 channels x 2 frames x 14 x 14 pixels
      -> patch_embed         Conv3d with kernel = stride = (2, 14, 14), no bias       (L, 1280)
      -> pre_layrnorm
      -> layers              x = x + self_attn(layer_norm1(x))    3D RoPE, full attention, no mask
                             x = x + mlp(layer_norm2(x))          fc1, exact (erf) GELU, fc2
      -> tower output        (L, 1280); there is no final LayerNorm
      -> projector           linear_2(gelu(linear_1(x)))          per patch, 1280 -> 6144 -> 6144
                             reshape to (L / 4, 4 * 6144)         each 2x2 block of patches is 4 consecutive rows
                             merge_linear_2(gelu(merge_linear_1(x)))   24576 -> 6144 -> 6144
      -> vision tokens       (L / 4, 6144)

There is no attention mask: with several images in one call, every patch attends to the patches of
all of them, as in Hugging Face.

The module names are the ones the TT ``from_torch`` builders read (``patch_embed``, ``pre_layrnorm``,
``layers``, ``projector``). ``load_hf_state_dict`` takes the Hugging Face key layout
(``vision_tower.*``, ``multi_modal_projector.*``), which is also what ``tt/vision/loader.py`` returns
for the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MiniMaxM3VisionConfig:
    """Vision tower and projector dimensions. The defaults are the MiniMax-M3 checkpoint's."""

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    projector_hidden_size: int = 6144
    text_hidden_size: int = 6144

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads  # 80

    @property
    def patch_dim(self) -> int:
        """Values per flattened patch: channels x frames x patch_size x patch_size (1176)."""
        return self.num_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def merged_hidden_size(self) -> int:
        """Projector width after folding each block of spatial_merge_size**2 patches into one row (24576)."""
        return self.text_hidden_size * self.spatial_merge_size**2

    @classmethod
    def from_hf_config(cls, config: dict, **overrides) -> "MiniMaxM3VisionConfig":
        """Build from a config dict: the checkpoint's config.json (transformers 4.52.4 layout, with the merge and
        temporal sizes under ``img_token_compression_config``) or ``MiniMaxM3VLConfig.to_dict()`` (transformers 5.12).
        """
        vc = config["vision_config"]
        compression = vc.get("img_token_compression_config") or config.get("img_token_compression_config") or {}
        rope = vc.get("rope_parameters") or {}
        values = dict(
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_hidden_layers=vc["num_hidden_layers"],
            num_attention_heads=vc["num_attention_heads"],
            num_channels=vc.get("num_channels", 3),
            patch_size=vc["patch_size"],
            temporal_patch_size=vc.get("temporal_patch_size", compression.get("temporal_patch_size", 2)),
            spatial_merge_size=vc.get("spatial_merge_size", compression.get("spatial_merge_size", 2)),
            layer_norm_eps=vc.get("layer_norm_eps", 1e-5),
            rope_theta=rope.get("rope_theta", vc.get("rope_theta", 10000.0)),
            projector_hidden_size=config["projector_hidden_size"],
            text_hidden_size=config["text_config"]["hidden_size"],
        )
        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------------------------------------------
# 3D RoPE (MiniMaxM3VL3DRotaryEmbedding and apply_rotary_pos_emb_vision)
# ---------------------------------------------------------------------------------------------


def rope_axis_dim(head_dim: int) -> int:
    """Rotary dims per axis (T, H and W): 2 * (head_dim // 2) split in three, rounded down to even (26 for 80)."""
    return 2 * ((2 * (head_dim // 2) // 3) // 2)


def rope_positions(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """(num_images, 3) grid of (t, h, w) patches -> (L, 3) fp32 coordinates of every patch.

    The patches are in the image processor's order, where each spatial_merge_size x spatial_merge_size
    block of patches is consecutive.
    """
    m = spatial_merge_size
    coords = []
    for t, h, w in grid_thw.tolist():
        hi = torch.arange(h).unsqueeze(1).expand(-1, w)
        hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = torch.arange(w).unsqueeze(0).expand(h, -1)
        wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = torch.arange(t).repeat_interleave(h * w)
        coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    return torch.cat(coords).to(torch.float32)


def rope_cos_sin(
    grid_thw: torch.Tensor, head_dim: int, theta: float, spatial_merge_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the 3D RoPE, each (L, 3 * rope_axis_dim(head_dim)) in fp32 (78 columns for head_dim 80).

    The columns are the T, H and W frequency bands, then the same three again, to pair with rotate_half.
    """
    axis_dim = rope_axis_dim(head_dim)
    positions = rope_positions(grid_thw, spatial_merge_size)
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
    freqs = torch.cat([positions[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate the first cos.shape[-1] head dims of q and k, (1, L, heads, head_dim); the other dims pass through."""
    rot_dim = cos.shape[-1]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


# ---------------------------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------------------------


class VisionAttention(nn.Module):
    """Separate q, k, v and out projections with bias, 3D RoPE on q and k, softmax attention over all patches."""

    def __init__(self, config: MiniMaxM3VisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        d = config.hidden_size
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x (1, L, hidden) -> (1, L, hidden)."""
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(x).view(hidden_shape)
        k = self.k_proj(x).view(hidden_shape)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous()
        return self.out_proj(out.reshape(*input_shape, -1).contiguous())


class VisionMLP(nn.Module):
    def __init__(self, config: MiniMaxM3VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    """Pre-norm block: attention then MLP, each with a residual add."""

    def __init__(self, config: MiniMaxM3VisionConfig):
        super().__init__()
        self.self_attn = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), cos, sin)
        return x + self.mlp(self.layer_norm2(x))


class MultiModalProjector(nn.Module):
    """Per-patch GELU MLP, then each block of spatial_merge_size**2 patches folded into one row and a second GELU MLP."""

    def __init__(self, config: MiniMaxM3VisionConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.linear_1 = nn.Linear(config.hidden_size, config.projector_hidden_size)
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.text_hidden_size)
        self.merge_linear_1 = nn.Linear(config.merged_hidden_size, config.projector_hidden_size)
        self.merge_linear_2 = nn.Linear(config.projector_hidden_size, config.text_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tower output (L, hidden) -> vision tokens (L / spatial_merge_size**2, text_hidden_size)."""
        x = self.linear_2(F.gelu(self.linear_1(x)))
        x = x.reshape(x.shape[0] // self.spatial_merge_size**2, -1)
        return self.merge_linear_2(F.gelu(self.merge_linear_1(x)))


# Reference name prefix <-> Hugging Face MiniMaxM3VLModel name prefix. The first match wins.
_HF_PREFIXES = (
    ("patch_embed.", "vision_tower.embeddings.proj."),
    ("pre_layrnorm.", "vision_tower.pre_layrnorm."),
    ("layers.", "vision_tower.layers."),
    ("projector.", "multi_modal_projector."),
)


class MiniMaxM3VisionEncoder(nn.Module):
    """Vision tower plus projector: pixel_values (L, 1176) -> vision tokens (L / 4, 6144)."""

    def __init__(self, config: Optional[MiniMaxM3VisionConfig] = None):
        super().__init__()
        self.config = config = config or MiniMaxM3VisionConfig()
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.patch_embed = nn.Conv3d(config.num_channels, config.hidden_size, kernel, stride=kernel, bias=False)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(VisionEncoderLayer(config) for _ in range(config.num_hidden_layers))
        self.projector = MultiModalProjector(config)

    def rope(self, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        return rope_cos_sin(grid_thw, c.head_dim, c.rope_theta, c.spatial_merge_size)

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(L, patch_dim) -> (L, hidden): the Conv3d applied to each patch."""
        c = self.config
        x = pixel_values.view(-1, c.num_channels, c.temporal_patch_size, c.patch_size, c.patch_size)
        return self.patch_embed(x.to(self.patch_embed.weight.dtype)).view(-1, c.hidden_size)

    def forward_tower(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """(L, patch_dim) -> tower output (L, hidden), the last_hidden_state of MiniMaxM3VLVisionModel."""
        x = self.embed(pixel_values).to(self.pre_layrnorm.weight.dtype)
        cos, sin = self.rope(grid_thw)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        x = self.pre_layrnorm(x).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x.squeeze(0)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """(L, patch_dim) -> vision tokens (L / spatial_merge_size**2, text_hidden_size), the pooler_output of
        MiniMaxM3VLModel.get_image_features."""
        return self.projector(self.forward_tower(pixel_values, grid_thw))

    # ---- weights ----
    @torch.no_grad()
    def init_random_weights(self, seed: int = 0, std: float = 0.25) -> "MiniMaxM3VisionEncoder":
        """Seeded random weights for tests.

        Linear and Conv3d weights are N(0, 1 / fan_in), so activations stay near unit scale. Biases and
        LayerNorm shifts are N(0, std**2) and LayerNorm scales 1 + N(0, std**2), so a dropped bias or
        affine term shows up in the output.
        """
        g = torch.Generator().manual_seed(seed)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                module.weight.normal_(0.0, module.weight[0].numel() ** -0.5, generator=g)
                if module.bias is not None:
                    module.bias.normal_(0.0, std, generator=g)
            elif isinstance(module, nn.LayerNorm):
                module.weight.normal_(1.0, std, generator=g)
                module.bias.normal_(0.0, std, generator=g)
        return self

    def load_hf_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> "MiniMaxM3VisionEncoder":
        """Load weights in the Hugging Face MiniMaxM3VLModel key layout (vision_tower.*, multi_modal_projector.*).

        Other keys (the language model) are skipped, and so are layers past num_hidden_layers, so a shorter
        reference can take the first layers of the checkpoint. A patch embed weight flattened to
        (hidden, patch_dim) is reshaped back to the Conv3d layout.
        """
        own = {}
        for key, value in state_dict.items():
            name = next((ours + key[len(hf) :] for ours, hf in _HF_PREFIXES if key.startswith(hf)), None)
            if name is None:
                continue
            if name.startswith("layers.") and int(name.split(".")[1]) >= self.config.num_hidden_layers:
                continue
            if name == "patch_embed.weight":
                value = value.reshape(self.patch_embed.weight.shape)
            own[name] = value
        self.load_state_dict(own, strict=strict)
        return self

    def hf_state_dict(self) -> Dict[str, torch.Tensor]:
        """The weights under Hugging Face MiniMaxM3VLModel names (vision_tower.*, multi_modal_projector.*)."""
        out = {}
        for name, value in self.state_dict().items():
            ours, hf = next(p for p in _HF_PREFIXES if name.startswith(p[0]))
            out[hf + name[len(ours) :]] = value
        return out
