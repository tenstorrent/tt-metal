# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for HunyuanRMSNorm.
# Extracted verbatim from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py  lines 1025-1043
#
# Used as the golden reference for TT-Metal numeric validation.

import torch
import torch.nn as nn


class HunyuanRMSNorm(nn.Module):
    """
    HunyuanRMSNorm is equivalent to T5LayerNorm.

    Forward signature:
        x: [batch, seq_len, hidden_size]  (any dtype)
        returns same shape, cast back to input dtype

    Notes for TT-Metal port:
    - Input is upcast to float32 before variance computation.
    - cast_weight_fp32=True casts the learned weight to fp32 before the
      final multiply; in TT-Metal this maps to HiFi2 math fidelity +
      fp32_dest_acc_en=True in WormholeComputeKernelConfig.
    - ttnn.rms_norm requires weight shape [1, 1, hidden_size // 32, 32]
      due to the SHARD_HEIGHT=32 tile requirement.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, cast_weight_fp32: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.cast_weight_fp32 = cast_weight_fp32

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.cast_weight_fp32:
            return (self.weight.float() * hidden_states).to(input_dtype)
        else:
            return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, H = 1, 256, 4096
    x = torch.randn(B, S, H, dtype=torch.bfloat16)

    norm = HunyuanRMSNorm(H, eps=1e-6, cast_weight_fp32=False)
    out = norm(x)
    print(f"input  shape: {x.shape}  dtype: {x.dtype}")
    print(f"output shape: {out.shape}  dtype: {out.dtype}")
    print(f"output mean={out.float().mean():.6f}  std={out.float().std():.6f}")
