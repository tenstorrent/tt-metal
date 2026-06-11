# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Local (replicated) RMSNorm for the TP4 replicated-hidden design.

Because the hidden stream is replicated full-width on every chip, RMSNorm is a
plain per-chip op over the full hidden dim — no cross-device variance combine,
so it is numerically exact (matches torch up to bf16 rounding).
"""

import ttnn

from models.experimental.dots_ocr_tp4.tt.common import to_replicated
from models.experimental.tt_symbiote.core.module import TTNNModule


class DotsOCRRMSNormTP4(TTNNModule):
    def __init__(self, mesh_device, eps=1e-6):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps
        self.tt_weight = None
        # LoFi (was HiFi4): RMSNorm accumulates in fp32 dest (fp32_dest_acc_en),
        # so the reduction stays full-precision; LoFi only drops mantissa bits in
        # the elementwise multiply, which is cheap accuracy for a faster kernel.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
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
        m.to_device(mesh_device)
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    def forward(self, x: ttnn.Tensor, out_memory_config=None) -> ttnn.Tensor:
        original_shape = x.shape
        # Decode (seq==1): keep the norm L1-resident so the surrounding decode
        # ops avoid a DRAM round-trip. Prefill stays DRAM-interleaved -- unless the
        # caller requests an explicit ``out_memory_config`` (e.g. the input_layernorm
        # output is put in L1 to feed the activation-read-bound prefill QKV matmul).
        is_decode = int(original_shape[-2]) == 1
        mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        out_mc = out_memory_config if out_memory_config is not None else mc
        if len(original_shape) == 3:
            x = ttnn.unsqueeze(x, 1)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=mc)
        out = ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.tt_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=out_mc,
        )
        if len(original_shape) == 3 and len(out.shape) == 4:
            out = ttnn.reshape(out, [out.shape[0], out.shape[2], out.shape[3]])
        return out
