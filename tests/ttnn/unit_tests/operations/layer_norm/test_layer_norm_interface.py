# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The model's traced calls (spec PHASE0 comment, rule R4): tt-metal's device compute kernel config, passed as
it is, and the output placement of rule R12."""

import pytest
import torch

import ttnn
from ttnn.operations.layer_norm import layer_norm


@pytest.mark.parametrize("shape", [(1, 1, 128, 1024), (1, 1, 64, 4096)], ids=["model_1024", "model_4096"])
def test_model_call_with_device_compute_kernel_config(device, shape):
    torch.manual_seed(0)
    W = shape[-1]
    x, w, b = torch.randn(shape).bfloat16(), torch.randn(1, 32, W).bfloat16(), torch.randn(1, 32, W).bfloat16()
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    out = layer_norm(put(x), compute_kernel_config=config, weight=put(w), bias=put(b), epsilon=1e-06)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(out).double().flatten()
    xd = x.double()
    c = xd - xd.mean(-1, keepdim=True)
    ref = (c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + 1e-6) * w.double()[0, 0] + b.double()[0, 0]).flatten()
    pcc = torch.corrcoef(torch.stack([got, ref]))[0, 1].item()
    assert pcc >= 0.999, pcc
