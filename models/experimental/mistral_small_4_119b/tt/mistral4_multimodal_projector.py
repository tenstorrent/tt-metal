# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-3 multi-modal projector — fully on device, no host fallback during inference.

Flow:
    image_features (1, 1, num_patches, 1024)
      ▼  RMSNorm(1024, eps = text rms_norm_eps)
      ▼  Mistral3PatchMerger:
            2×2 spatial merge → linear (4*1024 → 1024)
      ▼  linear_1: 1024 → 4096
      ▼  GELU
      ▼  linear_2: 4096 → 4096
      ▼
    language_embeddings (1, 1, (h/2)*(w/2), 4096)

The 2×2 merge uses only 4D reshape + one 4D permute (no host round-trip,
no torch.nn.functional.unfold). HF's unfold gives feature ordering (c, kh, kw);
our on-device merge gives (kw, kh, c). We permute the merging_layer weight
columns once at load time to compensate — see `_load_merger_weight`.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    HIDDEN_SIZE,
    MMP_SPATIAL_MERGE_SIZE,
    NORM_EPS,
    VISION_HIDDEN_SIZE,
)


# ── Weight loader for the merging layer ────────────────────────────────────


def _load_merger_weight(
    state_dict: dict,
    mesh_device: ttnn.MeshDevice,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """
    Load ``multi_modal_projector.patch_merger.merging_layer.weight`` and permute
    its input dim from HF ordering (c, kh, kw) to our merge ordering (kw, kh, c).

    HF weight shape:  [vision_hidden_size, vision_hidden_size * spatial_merge_size**2]
                    = [1024, 4096]
    Returned tensor:  [4096, 1024] in TILE layout, replicated on the mesh, with
                      columns reordered so that ``merged_features @ weight`` yields
                      the same result as HF's unfold + linear path.
    """
    s = MMP_SPATIAL_MERGE_SIZE  # 2
    d = VISION_HIDDEN_SIZE  # 1024

    w_hf = state_dict["multi_modal_projector.patch_merger.merging_layer.weight"].to(torch.bfloat16)
    assert w_hf.shape == (d, d * s * s), f"unexpected merging_layer.weight shape {w_hf.shape}"

    # HF input feature index = c*s*s + kh*s + kw. Expose that structure:
    w_hf_4d = w_hf.reshape(d, d, s, s)  # [out, c, kh, kw]
    # Our merge produces feature index = kw*s*d + kh*d + c. Re-arrange weight to match:
    w_ours_4d = w_hf_4d.permute(0, 3, 2, 1).contiguous()  # [out, kw, kh, c]
    w_ours = w_ours_4d.reshape(d, d * s * s)  # [out, our_in]

    # Transpose to [in, out] for ttnn.linear and upload.
    w_for_matmul = w_ours.T.contiguous()  # [4096, 1024]
    return ttnn.as_tensor(
        w_for_matmul,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# ── Patch merger ───────────────────────────────────────────────────────────


class TtMistral3PatchMerger:
    """
    On-device 2×2 spatial patch merge + linear projection (no host fallback).

    Forward expects vision features of shape ``[1, 1, h*w, vision_hidden_size]``
    plus the patch grid dimensions (``h_patches``, ``w_patches``), and returns
    merged features of shape ``[1, 1, (h/2)*(w/2), vision_hidden_size]``.
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
        self.s = MMP_SPATIAL_MERGE_SIZE
        self.merge_weight = _load_merger_weight(state_dict, mesh_device, dtype=dtype)

    def forward(
        self,
        image_features: ttnn.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> ttnn.Tensor:
        s = self.s
        d = VISION_HIDDEN_SIZE
        assert (
            h_patches % s == 0 and w_patches % s == 0
        ), f"patch grid {h_patches}×{w_patches} must be divisible by spatial_merge_size={s}"
        h2 = h_patches // s
        w2 = w_patches // s

        # 4D-only merge: split rows into pairs, permute, then absorb columns via reshape.
        x = ttnn.reshape(image_features, [h_patches, w_patches, d])  # [h, w, d]
        x = ttnn.reshape(x, [h2, s, w_patches, d])  # [h/2, 2, w, d]
        x = ttnn.permute(x, [0, 2, 1, 3])  # [h/2, w, 2, d]
        x = ttnn.reshape(x, [h2, w2, s * s * d])  # [h/2, w/2, 4*d]
        x = ttnn.reshape(x, [1, 1, h2 * w2, s * s * d])  # [1, 1, num_out, 4096]

        out = ttnn.linear(
            x,
            self.merge_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, num_out, 1024]
        ttnn.deallocate(x)
        return out


# ── Multi-modal projector ──────────────────────────────────────────────────


class TtMistral3MultiModalProjector:
    """
    HF parity: norm → patch_merger → linear_1 → GELU → linear_2.

    No biases (config.multimodal_projector_bias = False).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Pre-merge RMSNorm uses the text model's rms_norm_eps (== NORM_EPS).
        from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _load_norm_weight
        from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _load_weight

        self.norm_w = _load_norm_weight(
            state_dict, "multi_modal_projector.norm.weight", VISION_HIDDEN_SIZE, mesh_device
        )

        self.patch_merger = TtMistral3PatchMerger(
            mesh_device=mesh_device,
            state_dict=state_dict,
            compute_kernel_config=self.compute_kernel_config,
            dtype=dtype,
        )

        # linear_1: vision_hidden (1024) → text_hidden (4096)
        # linear_2: text_hidden (4096) → text_hidden (4096)
        self.linear_1 = _load_weight(
            state_dict,
            "multi_modal_projector.linear_1.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [1024, 4096]

        self.linear_2 = _load_weight(
            state_dict,
            "multi_modal_projector.linear_2.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [4096, 4096]

    def forward(
        self,
        image_features: ttnn.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> ttnn.Tensor:
        """
        Args:
            image_features: ttnn [1, 1, h*w, 1024] from the vision tower
            h_patches, w_patches: patch grid dimensions
        Returns:
            ttnn [1, 1, (h/2)*(w/2), 4096] — language-embedding-shaped tokens
        """
        # Pre-merge norm.
        x = ttnn.rms_norm(
            image_features,
            weight=self.norm_w,
            epsilon=NORM_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # 2x2 merge + linear (4096 → 1024).
        x_merged = self.patch_merger.forward(x, h_patches, w_patches)
        ttnn.deallocate(x)

        # linear_1 + GELU (fused via activation arg).
        hidden = ttnn.linear(
            x_merged,
            self.linear_1,
            activation="gelu",
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, num_out, 4096]
        ttnn.deallocate(x_merged)

        # linear_2.
        out = ttnn.linear(
            hidden,
            self.linear_2,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, num_out, 4096]
        ttnn.deallocate(hidden)

        assert out.shape[-1] == HIDDEN_SIZE, f"projector output last dim {out.shape[-1]} != text hidden {HIDDEN_SIZE}"
        return out
