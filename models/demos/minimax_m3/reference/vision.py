# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference (CPU golden) for the MiniMax-M3 vision encoder: the vision tower and the projector.

Plain PyTorch, no transformers; its config and RoPE angles are tt/vision's (``config.py``, ``rope.py``). It
follows Hugging Face transformers 5.12.1 ``minimax_m3_vl`` (``MiniMaxM3VLVisionModel``, then
``MiniMaxM3VLMultiModalProjector``) op for op;
``tests/unit/test_vision_reference_vs_hf.py`` checks that on CPU. The vision unit tests
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
all of them. That is Hugging Face 5.12.1's behaviour, kept here for parity with it only: vLLM and
SGLang keep attention inside each image, and the TT model (M3VLVisionModel) takes one image per
forward.

The module names are the ones the TT ``from_torch`` builders read (``patch_embed``, ``pre_layrnorm``,
``layers``, ``projector``). ``load_hf_state_dict`` takes the Hugging Face key layout
(``vision_tower.*``, ``multi_modal_projector.*``), which is also what ``tt/vision/loader.py`` returns
for the checkpoint.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.rope import rope_cos_sin

# ---------------------------------------------------------------------------------------------
# 3D RoPE rotation (apply_rotary_pos_emb_vision)
# ---------------------------------------------------------------------------------------------


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

    def load_hf_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> "MiniMaxM3VisionEncoder":
        """Load weights in the Hugging Face MiniMaxM3VLModel key layout (vision_tower.*, multi_modal_projector.*).

        Other keys (the language model) are skipped, and so are layers past num_hidden_layers, so a shorter
        reference can take the first layers of the checkpoint. A vision_tower.* key outside the patch embed,
        pre_layrnorm and layers, such as a final LayerNorm or a learned position embedding, raises ValueError;
        load_state_dict, which is strict, raises for any other weight the reference does not have, such as an unknown
        multi_modal_projector.* one, and for any weight of the reference that is missing. A patch embed weight
        flattened to (hidden, patch_dim) is reshaped back to the Conv3d layout.
        """
        own, unmapped = {}, []
        for key, value in state_dict.items():
            name = next((ours + key[len(hf) :] for ours, hf in _HF_PREFIXES if key.startswith(hf)), None)
            if name is None:
                if key.startswith("vision_tower."):
                    unmapped.append(key)
                continue
            if name.startswith("layers.") and int(name.split(".")[1]) >= self.config.num_hidden_layers:
                continue
            if name == "patch_embed.weight":
                value = value.reshape(self.patch_embed.weight.shape)
            own[name] = value
        if unmapped:
            raise ValueError(f"vision tower weights with no place in the reference: {sorted(unmapped)}")
        self.load_state_dict(own, strict=True)
        return self

    def hf_state_dict(self) -> Dict[str, torch.Tensor]:
        """The weights under Hugging Face MiniMaxM3VLModel names (vision_tower.*, multi_modal_projector.*)."""
        out = {}
        for name, value in self.state_dict().items():
            ours, hf = next(p for p in _HF_PREFIXES if name.startswith(p[0]))
            out[hf + name[len(ours) :]] = value
        return out
