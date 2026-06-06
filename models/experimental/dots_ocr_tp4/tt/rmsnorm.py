# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Local (replicated) RMSNorm for the TP4 replicated-hidden design.

Because the hidden stream is replicated full-width on every chip, RMSNorm is a
plain per-chip op over the full hidden dim — no cross-device variance combine,
so it is numerically exact (matches torch up to bf16 rounding).
"""

import ttnn

from models.experimental.dots_ocr_tp4.tt.common import to_replicated


class DotsOCRRMSNormTP4:
    def __init__(self, mesh_device, eps=1e-6):
        self.mesh_device = mesh_device
        self.eps = eps
        self.tt_weight = None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, mesh_device, torch_norm, eps=None):
        e = eps
        if e is None:
            e = getattr(torch_norm, "variance_epsilon", getattr(torch_norm, "eps", 1e-6))
        m = cls(mesh_device, eps=e)
        # Replicate gamma on every chip; [32, H] tile layout (broadcast over rows).
        weight = torch_norm.weight.data
        m.tt_weight = to_replicated(weight.unsqueeze(0).expand(32, -1), mesh_device, dtype=ttnn.bfloat16)
        return m

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        original_shape = x.shape
        if len(original_shape) == 3:
            x = ttnn.unsqueeze(x, 1)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.tt_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        if len(original_shape) == 3 and len(out.shape) == 4:
            out = ttnn.reshape(out, [out.shape[0], out.shape[2], out.shape[3]])
        return out
