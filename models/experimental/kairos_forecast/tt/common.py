# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Precision policy and upload helpers for the Kairos port.

Accuracy (the gated target): bf16 weights and activations, HiFi2 matmuls with fp32 accumulation, HiFi4 for the
norms, the RoPE rotation and the patch embedding, attention scores and softmax in fp32 (T5 does not scale them,
so they reach tens), HiFi4 + fp32 output for the head. Performance: bfp8 weights at LoFi, except the matmuls
that fold in a bf16 residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

TILE = 32
S_PAD = 96  # encoder rows per series: 64 tokens + [REG] + 31 pad (N is 64 on every natural input)
D_PAD = 32  # decoder rows per series: 2 tokens + 30 pad
NEG = -1e9  # additive mask; exp underflows to 0 exactly, as finfo.min does in the reference


def pad_to_tile(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass(frozen=True)
class Precision:
    bf8_weights: bool = False
    lofi: bool = False
    fp32_dest_acc_en: bool = True

    @classmethod
    def accuracy(cls):
        return cls()

    @classmethod
    def performance(cls):
        return cls(bf8_weights=True, lofi=True)

    @property
    def name(self):
        return "performance" if self.bf8_weights else "accuracy"

    def weight_dtype(self, *, keep_bf16=False):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_weights and not keep_bf16 else ttnn.bfloat16

    def mm(self, fidelity=None):
        import ttnn

        if fidelity is None:
            fidelity = ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2
        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=self.fp32_dest_acc_en, packer_l1_acc=True
        )

    def hifi4(self):
        import ttnn

        return self.mm(ttnn.MathFidelity.HiFi4)


def upload(device, t: torch.Tensor, *, dtype=None, layout=None):
    import ttnn

    dtype = ttnn.bfloat16 if dtype is None else dtype
    host = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}.get(dtype, torch.float32)
    return ttnn.from_torch(
        t.detach().to(host).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if layout is None else layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def upload_linear(device, w: torch.Tensor, dtype):
    """nn.Linear weight (out, in) -> device (in, out)."""
    return upload(device, w.t(), dtype=dtype)


def upload_norm(device, w: torch.Tensor):
    import ttnn

    return upload(device, w.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
