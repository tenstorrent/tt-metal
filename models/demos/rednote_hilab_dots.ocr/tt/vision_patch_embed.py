# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN vision patch embed for dots.ocr.

DotsPatchEmbed = Conv2d(3 -> 1536, kernel 14x14, stride 14) + RMSNorm(eps=1e-5).

The HF preprocessor hands the vision tower PRE-FLATTENED patches of shape
[num_patches, C * T * P * P] (T = temporal_patch_size = 1). Because the conv
stride equals the kernel size, each patch is convolved exactly once and the
convolution is mathematically a single linear projection with weight
``proj.weight.view(embed_dim, C * P * P)`` — the same trick used by ViT-style
TTNN ports (cf. reference_impl models/demos/qwen25_vl/tt/model.py, whose
patch-embed conv likewise collapses onto a matmul over flattened patches).

Parallelism plan (ARCHITECTURE.md): placement=replicate — the vision tower is
run-once per input, all weights are ``ReplicateTensorToMesh`` and the output
stays replicated, so the handoff into the column-parallel decoder needs no CCL.
On a single device the mesh_mapper degenerates gracefully (1x1 mesh).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class TtVisionPatchEmbed(LightweightModule):
    """dots.ocr DotsPatchEmbed: flattened patches -> linear(conv) -> RMSNorm.

    Args:
        mesh_device: ttnn mesh device handle (all weights replicated).
        state_dict: {"proj.weight": [E, C, P, P], "proj.bias": [E],
                     "norm.weight": [E]} torch tensors.
        dtype: on-device weight/activation dtype.
        eps: RMSNorm epsilon (dots.ocr vision uses 1e-5).
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, eps=1e-5):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps

        proj_w = state_dict["proj.weight"]  # [E, C, P, P]
        embed_dim = proj_w.shape[0]
        in_features = proj_w.shape[1] * proj_w.shape[2] * proj_w.shape[3]
        # temporal_patch_size == 1: input [N, C*P*P] flattening matches the
        # conv kernel's (C, kh, kw) flattening, so conv == x @ W_flat.T + b.
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        self.proj_weight = ttnn.from_torch(
            proj_w.reshape(embed_dim, in_features).T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.proj_bias = ttnn.from_torch(
            state_dict["proj.bias"].reshape(1, embed_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        # ttnn.rms_norm gamma format: [1, 1, E//32, 32] in ROW_MAJOR for
        # bf16. The ROW_MAJOR gamma path is bf16-only (an fp32 ROW_MAJOR
        # gamma is misread on device, PCC ~0) — fp32 gammas use TILE
        # [1, 1, 1, E] instead.
        if dtype == ttnn.float32:
            gamma, gamma_layout = state_dict["norm.weight"].reshape(1, 1, 1, embed_dim), ttnn.TILE_LAYOUT
        else:
            gamma, gamma_layout = (
                state_dict["norm.weight"].reshape(1, 1, embed_dim // TILE, TILE),
                ttnn.ROW_MAJOR_LAYOUT,
            )
        self.norm_weight = ttnn.from_torch(
            gamma,
            dtype=dtype,
            layout=gamma_layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [num_patches, C*P*P] TILE_LAYOUT, replicated across the mesh.

        Returns: [num_patches, embed_dim], replicated.
        """
        h = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.rms_norm(h, epsilon=self.eps, weight=self.norm_weight)
        ttnn.deallocate(h)
        return out
