# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pixtral vision tower for Mistral-Small-4-119B — fully on device.

Architecture::

    image (B, 3, H, W)
      │
      ▼  patch_conv (3 → 1024, 14×14 stride 14)              [boundary: torch.unfold]
    patches (B, num_patches, 1024)
      │
      ▼  ln_pre (RMSNorm)
      ▼  24× PixtralBlock:
            x = x + Attention(RMSNorm(x))     (2D RoPE on q,k, non-causal SDPA)
            x = x + MLP(RMSNorm(x))           (SiLU-gated)
      │
      ▼
    features (B, num_patches, 1024)            ← consumed by multi-modal projector

Boundary note:
    ``patch_conv`` uses ``torch.nn.Unfold`` on the host to extract patches from
    the raw image tensor — this is image preprocessing, analogous to tokenization
    for text. The remaining model layers (ln_pre, all 24 attention/MLP blocks,
    RoPE rotation, SDPA) execute entirely in TTNN with no host fallback.
"""

from __future__ import annotations

from typing import Tuple

import torch

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    VISION_HEAD_DIM,
    VISION_HIDDEN_SIZE,
    VISION_NORM_EPS,
    VISION_NUM_CHANNELS,
    VISION_NUM_LAYERS,
    VISION_PATCH_SIZE,
    vision_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _load_norm_weight
from models.experimental.mistral_small_4_119b.tt.mistral4_vision_attention import TtPixtralAttention
from models.experimental.mistral_small_4_119b.tt.mistral4_vision_mlp import TtPixtralMLP
from models.experimental.mistral_small_4_119b.tt.mistral4_vision_rope import (
    TtPixtralRoPE2D,
    position_ids_from_grid,
)


# ── Pixtral norm uses a slightly larger epsilon than the text norm ─────────


def _vision_rms_norm(x: ttnn.Tensor, weight: ttnn.Tensor, compute_kernel_config) -> ttnn.Tensor:
    return ttnn.rms_norm(
        x,
        weight=weight,
        epsilon=VISION_NORM_EPS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )


# ── Patch embedding (Conv2d via unfold + linear) ───────────────────────────


class TtPixtralPatchConv:
    """
    Pixtral patch embedding: Conv2d(3, 1024, kernel=14, stride=14, bias=False).

    Implemented as host-side unfold + on-device linear. The unfold is image
    preprocessing — the matmul (the actual compute) runs on device.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        compute_kernel_config,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.compute_kernel_config = compute_kernel_config
        self.patch_size = VISION_PATCH_SIZE

        # HF Conv2d weight: [out_channels, in_channels, kH, kW] = [1024, 3, 14, 14].
        w_4d = state_dict["vision_tower.patch_conv.weight"].to(torch.bfloat16)
        # Reshape to [out, in*kH*kW] then transpose for matmul: [in*kH*kW, out]
        w = w_4d.reshape(VISION_HIDDEN_SIZE, -1).T.contiguous()
        self.weight = ttnn.as_tensor(
            w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        self._unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, image: torch.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        """
        Args:
            image: torch [1, 3, H, W] bf16 (H,W must be multiples of patch_size)
        Returns:
            patches: ttnn [1, 1, num_patches, 1024]
            h_patches, w_patches: int
        """
        assert image.ndim == 4 and image.shape[1] == VISION_NUM_CHANNELS
        H, W = image.shape[2], image.shape[3]
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # [1, 3, H, W] → [1, 3*kH*kW, num_patches] → [1, num_patches, 3*kH*kW]
        unfolded = self._unfold(image.to(torch.bfloat16)).permute(0, 2, 1).contiguous()
        # Upload [1, num_patches, 588]
        x = ttnn.as_tensor(
            unfolded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        out = ttnn.linear(
            x,
            self.weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, num_patches, 1024]
        ttnn.deallocate(x)
        num_patches = h_patches * w_patches
        out = ttnn.reshape(out, [1, 1, num_patches, VISION_HIDDEN_SIZE])
        return out, h_patches, w_patches


# ── Single Pixtral block ───────────────────────────────────────────────────


class TtPixtralBlock:
    """attention_norm → attention → residual → ffn_norm → mlp → residual."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_idx: int,
        compute_kernel_config,
        dtype=ttnn.bfloat16,
    ):
        self.compute_kernel_config = compute_kernel_config
        prefix = vision_layer_state_dict_prefix(layer_idx)

        self.attn_norm_w = _load_norm_weight(
            state_dict, prefix + "attention_norm.weight", VISION_HIDDEN_SIZE, mesh_device
        )
        self.ffn_norm_w = _load_norm_weight(state_dict, prefix + "ffn_norm.weight", VISION_HIDDEN_SIZE, mesh_device)
        self.attn = TtPixtralAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            layer_prefix=prefix,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
        )
        self.mlp = TtPixtralMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            layer_prefix=prefix,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        residual = x
        normed = _vision_rms_norm(x, self.attn_norm_w, self.compute_kernel_config)
        attn_out = self.attn.forward(normed, cos, sin)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        residual = x
        normed = _vision_rms_norm(x, self.ffn_norm_w, self.compute_kernel_config)
        mlp_out = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        x = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)
        return x


# ── Full Pixtral vision tower ──────────────────────────────────────────────


class TtPixtralVisionTower:
    """
    Mistral-Small-4 Pixtral vision tower.

    Args:
        mesh_device:        TTNN MeshDevice (vision weights are replicated)
        state_dict:         HF checkpoint dict filtered to ``vision_tower.*``
        num_layers:         layers to instantiate (1..24)
        dtype:              storage dtype for projection weights (bf16 default)
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        num_layers: int = VISION_NUM_LAYERS,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.num_layers = num_layers

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Patch embedding.
        self.patch_conv = TtPixtralPatchConv(
            mesh_device=mesh_device,
            state_dict=state_dict,
            compute_kernel_config=self.compute_kernel_config,
            dtype=dtype,
        )

        # ln_pre.
        self.ln_pre_w = _load_norm_weight(state_dict, "vision_tower.ln_pre.weight", VISION_HIDDEN_SIZE, mesh_device)

        # 2D RoPE table.
        self.rope = TtPixtralRoPE2D(mesh_device=mesh_device, head_dim=VISION_HEAD_DIM)

        # Transformer blocks.
        self.blocks: list[TtPixtralBlock] = [
            TtPixtralBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                layer_idx=i,
                compute_kernel_config=self.compute_kernel_config,
                dtype=dtype,
            )
            for i in range(num_layers)
        ]

    def forward(self, image: torch.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        """
        Args:
            image: torch [1, 3, H, W] bf16 (H,W multiples of patch_size).
        Returns:
            features: ttnn [1, 1, num_patches, 1024] replicated on the mesh
            h_patches, w_patches: int — patch grid dimensions
        """
        # Patch embed.
        x, h_patches, w_patches = self.patch_conv.forward(image)

        # Pre-norm.
        x = _vision_rms_norm(x, self.ln_pre_w, self.compute_kernel_config)

        # Build per-patch 2D RoPE lookup; this stays on device after lookup.
        position_ids = position_ids_from_grid(h_patches, w_patches)
        cos, sin = self.rope.lookup(position_ids)

        # 24 attention/MLP blocks.
        for blk in self.blocks:
            x = blk.forward(x, cos, sin)

        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        return x, h_patches, w_patches
