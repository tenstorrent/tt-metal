# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pixtral vision tower for Mistral-Small-4-119B — fully on device.

Architecture::

    image (B, 3, H, W)
      │
      ▼  patch_conv (3 → 1024, 14×14 stride 14)              [ttnn.conv2d on device]
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
    ``patch_conv`` is a real ``ttnn.conv2d`` (stride = kernel = 14, no padding,
    no bias). The image is permuted NCHW → flat-NHWC ``[1, 1, H*W, 3]`` on host
    and uploaded; the convolution itself runs on device. The rest of the tower
    (ln_pre, 24 attention/MLP blocks, RoPE, SDPA) is also fully on-device.
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )


# ── Patch embedding (folded as matmul) ─────────────────────────────────────


class TtPixtralPatchConv:
    """
    Pixtral patch embedding: Conv2d(3, 1024, kernel=14, stride=14, bias=False).

    Because kernel == stride and there is no padding, the conv is equivalent
    to a matmul with the per-patch 14×14×3 = 588 input values flattened into
    the channel dim. We pre-fold the weight on host once and the image on host
    per upload (``torch.nn.functional.unfold``), then run a single ``ttnn.linear``
    on device — bypassing ``ttnn.conv2d``'s internal ``fold`` + tilize +
    sharding setup that otherwise stalls ~3.6 ms per frame.
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
        self.dtype = dtype
        self.folded_in_channels = VISION_NUM_CHANNELS * self.patch_size * self.patch_size  # 588

        # HF Conv2d weight [out, in, kH, kW] = [1024, 3, 14, 14] → fold to
        # [in*kH*kW, out] = [588, 1024] so ttnn.linear can consume it.
        w_4d = state_dict["vision_tower.patch_conv.weight"].to(torch.bfloat16)
        assert w_4d.shape == (VISION_HIDDEN_SIZE, VISION_NUM_CHANNELS, self.patch_size, self.patch_size)
        w_folded = w_4d.reshape(VISION_HIDDEN_SIZE, self.folded_in_channels).T.contiguous()  # [588, 1024]
        self.weight = ttnn.as_tensor(
            w_folded,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        self._image_input_tt: ttnn.Tensor | None = None
        self._image_input_shape: tuple[int, int] | None = None

    def _fold_image_host(self, image: torch.Tensor) -> torch.Tensor:
        """[1, 3, H, W] → [1, 1, num_patches, 588] (im2col with kernel==stride)."""
        x = image.to(torch.bfloat16)
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # [1, 588, num_patches]
        num_patches = x.shape[-1]
        return x.transpose(1, 2).contiguous().reshape(1, 1, num_patches, self.folded_in_channels)

    def upload_image(self, image: torch.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        """Upload pre-folded image to a persistent device buffer for trace replay."""
        assert image.ndim == 4 and image.shape[1] == VISION_NUM_CHANNELS
        H, W = image.shape[2], image.shape[3]
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        x_host = self._fold_image_host(image)
        if self._image_input_tt is None or self._image_input_shape != (H, W):
            self._image_input_tt = ttnn.from_torch(
                x_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self._image_input_shape = (H, W)
        else:
            host_tt = ttnn.from_torch(
                x_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(host_tt, self._image_input_tt)
        return self._image_input_tt, h_patches, w_patches

    def forward_device(self, image_tt: ttnn.Tensor, h_patches: int, w_patches: int) -> Tuple[ttnn.Tensor, int, int]:
        """Pre-folded patch embedding: one matmul, fully trace-compatible."""
        out = ttnn.linear(
            image_tt,
            self.weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out, h_patches, w_patches

    def forward(self, image: torch.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        """
        Args:
            image: torch [1, 3, H, W] bf16 (H,W must be multiples of patch_size)
        Returns:
            patches: ttnn [1, 1, num_patches, 1024] in TILE_LAYOUT, DRAM
            h_patches, w_patches: int
        """
        assert image.ndim == 4 and image.shape[1] == VISION_NUM_CHANNELS
        H, W = image.shape[2], image.shape[3]
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        image_tt, h_patches, w_patches = self.upload_image(image)
        return self.forward_device(image_tt, h_patches, w_patches)


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

        self._rope_grid: tuple[int, int] | None = None
        self._rope_ids_tt: ttnn.Tensor | None = None
        self._cached_cos: ttnn.Tensor | None = None
        self._cached_sin: ttnn.Tensor | None = None
        self._forward_trace_id: int | None = None
        self._features_out: ttnn.Tensor | None = None

    def cache_rope_for_grid(self, h_patches: int, w_patches: int) -> None:
        """Pre-upload position ids and cache cos/sin for a fixed patch grid (trace-safe)."""
        grid = (h_patches, w_patches)
        if self._rope_grid == grid:
            return
        position_ids = position_ids_from_grid(h_patches, w_patches)
        seq_len = position_ids.numel()
        self._rope_ids_tt = ttnn.as_tensor(
            position_ids.to(torch.int32).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._cached_cos, self._cached_sin = self.rope.lookup_device(self._rope_ids_tt, seq_len)
        self._rope_grid = grid

    def forward_device(self, image_tt: ttnn.Tensor, h_patches: int, w_patches: int) -> Tuple[ttnn.Tensor, int, int]:
        """
        Device-only forward for trace capture/replay.

        Requires ``cache_rope_for_grid`` for the same ``(h_patches, w_patches)``.
        Does not deallocate cached RoPE tensors (safe for trace replay).
        """
        self.cache_rope_for_grid(h_patches, w_patches)
        assert self._cached_cos is not None and self._cached_sin is not None

        x, h_patches, w_patches = self.patch_conv.forward_device(image_tt, h_patches, w_patches)
        x = _vision_rms_norm(x, self.ln_pre_w, self.compute_kernel_config)

        for blk in self.blocks:
            x = blk.forward(x, self._cached_cos, self._cached_sin)

        return x, h_patches, w_patches

    def forward(self, image: torch.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        """
        Args:
            image: torch [1, 3, H, W] bf16 (H,W multiples of patch_size).
        Returns:
            features: ttnn [1, 1, num_patches, 1024] replicated on the mesh
            h_patches, w_patches: int — patch grid dimensions
        """
        image_tt, h_patches, w_patches = self.patch_conv.upload_image(image)
        return self.forward_device(image_tt, h_patches, w_patches)

    def capture_forward_trace(self, image_tt: ttnn.Tensor, h_patches: int, w_patches: int) -> ttnn.Tensor:
        """Capture a replayable trace of ``forward_device`` (run once after JIT warmup)."""
        self.cache_rope_for_grid(h_patches, w_patches)
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        features_tt, _, _ = self.forward_device(image_tt, h_patches, w_patches)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self._forward_trace_id = trace_id
        self._features_out = features_tt
        return features_tt

    def execute_forward_trace(self, blocking: bool = False) -> ttnn.Tensor:
        """Replay the captured forward trace; returns the same output buffer as capture."""
        assert self._forward_trace_id is not None and self._features_out is not None
        ttnn.execute_trace(self.mesh_device, self._forward_trace_id, cq_id=0, blocking=blocking)
        return self._features_out
