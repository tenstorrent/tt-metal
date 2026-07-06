# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `conv1_d` of coqui/XTTS-v2.

Reference submodule: `gpt.gpt.h.0.attn.c_attn`, a GPT-2 style
`transformers.pytorch_utils.Conv1D`. Despite the name it is NOT a convolution —
it is a linear layer that projects the last dim:

    Conv1D.forward(x): y = addmm(bias, x.view(-1, nx), weight).view(..., nf)
                     == x @ weight + bias

with `weight` shaped `[nx, nf]` (already in `x @ weight` orientation, so NO
transpose) and `bias` shaped `[nf]`. Captured shapes: in `[1, 33, 1024]`
(nx=1024), out `[1, 33, 3072]` (nf=3072).
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    """Precompute the ttnn weight/bias from the trained Conv1D."""
    import torch

    m = torch_module

    weight = ttnn.as_tensor(
        m.weight.detach().contiguous().to(torch.bfloat16),  # [nx, nf], no transpose
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    nf = m.weight.shape[-1]
    bias = ttnn.as_tensor(
        m.bias.detach().reshape(1, 1, nf).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # HiFi4 + fp32 accumulation: this projection feeds the 30-layer GPT2 stack whose
    # accumulated bf16 error otherwise flips near-tie greedy argmaxes in the AR
    # decoder. Full-fidelity matmul keeps the logits close enough to the fp32
    # reference that the free-running token sequence tracks HF for the whole horizon.
    _kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(x, *args, **kwargs):
        y = ttnn.matmul(x, weight, compute_kernel_config=_kernel_cfg)
        y = ttnn.add(y, bias)
        return y

    return forward


def conv1_d(x, *args, **kwargs):
    raise RuntimeError(
        "conv1_d requires build(device, torch_module) to bind trained weights; "
        "the bare callable has no parameters."
    )
