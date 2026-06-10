# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN vision RMSNorm for dots.ocr.

dots.ocr vision RMSNorm (modeling_dots_vision.RMSNorm, eps=1e-5): normalize in
fp32 — x * rsqrt(mean(x^2, -1) + eps) — then scale by the learned weight. The
whole decomposed chain (pow -> mean -> rsqrt -> mul -> mul) maps onto the single
fused ``ttnn.rms_norm`` op (cf. KB entry ttnn_pow: the pow/mean/rsqrt/multiply
chain "is a fusion candidate into ttnn.rms_norm"); reference_impl
models/demos/qwen25_vl/tt/vision_rmsnorm.py uses the identical fused op with a
[1, 1, dim//32, 32] ROW_MAJOR gamma.

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — weight
is ``ReplicateTensorToMesh`` and the replicated activation stays replicated, so
no CCL is needed. On a single device the mesh_mapper degenerates gracefully.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class TtVisionRMSNorm(LightweightModule):
    """dots.ocr vision RMSNorm: fused ttnn.rms_norm with replicated weight.

    Args:
        mesh_device: ttnn mesh device handle (weight replicated).
        state_dict: {"weight": [dim]} torch tensor (HF key e.g.
            vision_tower.blocks.N.norm1.weight).
        dtype: on-device weight dtype.
        eps: RMSNorm epsilon (dots.ocr vision uses 1e-5).
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, eps=1e-5):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps

        weight = state_dict["weight"]
        dim = weight.shape[-1]
        # ttnn.rms_norm gamma format: [1, 1, dim//32, 32] in ROW_MAJOR for
        # bf16. The ROW_MAJOR gamma path is bf16-only (an fp32 ROW_MAJOR
        # gamma is misread on device, PCC ~0) — fp32 gammas use TILE
        # [1, 1, 1, dim] instead.
        if dtype == ttnn.float32:
            gamma, gamma_layout = weight.reshape(1, 1, 1, dim), ttnn.TILE_LAYOUT
        else:
            gamma, gamma_layout = weight.reshape(1, 1, dim // TILE, TILE), ttnn.ROW_MAJOR_LAYOUT
        self.weight = ttnn.from_torch(
            gamma,
            dtype=dtype,
            layout=gamma_layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [..., dim] TILE_LAYOUT, replicated across the mesh.

        Returns: same shape, replicated, DRAM interleaved.

        Tracy-driven placement (vision_block optimization phase): the norm
        output MUST stay DRAM interleaved. The earlier per-block isolation
        win (L1 pin, 51.2 -> 38.4 us/iter on the lone norm kernel) reverses
        catastrophically in composition: a 5.5 MB fp32 [1, 1, 896, 1536] L1
        interleaved activation stalls every downstream ttnn.linear (QKV 97 ->
        2963 us, fc1/fc3 178 -> ~2650 us at the production operating point) —
        the optimization skill's "never pin a LARGE activation to L1 in front
        of a matmul" layout-interaction stall. DRAM output costs +14 us on
        the norm and saves ~7.7 ms on the consuming matmuls per block. The
        width-sharded LayerNormShardedMultiCoreProgramConfig variant remains
        WORSE (69.4 us): the i2s/s2i bounce costs more than it saves.
        """
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
