# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr Qwen2 language-model RMSNorm.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`rmsnorm_forward`

    output = (x.float() * rsqrt(mean(x.float()**2, -1) + eps)).type_as(x) * weight

eps = 1e-6 (LM, distinct from the vision tower's 1e-5), hidden = 1536. The norm
is computed in fp32 (fp32_dest_acc + HiFi4) to match the reference's
float-then-cast path, then scaled by the (replicated) weight vector.

This is the Qwen2RMSNorm used by the decoder layers / final norm. It mirrors
``tt/vision_rmsnorm.py`` (TtVisionRMSNorm) exactly except for the epsilon.
"""
import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
SHARD_HEIGHT = TILE  # ttnn.rms_norm wants the weight laid out one tile high


class TtRMSNorm(LightweightModule):
    """dots Qwen2 LM RMSNorm replicated over a (mesh) device.

    Args:
        device: ttnn Device or MeshDevice.
        dim: feature dimension (1536).
        weight: torch.Tensor of shape [dim] (the ``norm.weight`` parameter).
        eps: epsilon (1e-6).
        weight_dtype: dtype to store the gamma weight in.
    """

    def __init__(
        self,
        device,
        dim,
        weight,
        eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.eps = eps

        # Reshape gamma to [1, 1, dim // TILE, TILE] row-major as ttnn.rms_norm expects.
        torch_weight = weight.reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # fp32 compute to match the reference float-then-cast normalization.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
        )
