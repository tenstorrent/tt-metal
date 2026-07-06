# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `s_e_layer` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.speaker_encoder.layer1.0.se`, a
`TTS.encoder.models.resnet.SELayer` (squeeze-excitation channel gate):

    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)          # global average pool over H,W
    y = self.fc(y).view(b, c, 1, 1)          # Linear(c,c//r)->ReLU->Linear(c//r,c)->Sigmoid
    return x * y                             # per-channel rescale

Native ttnn: the global average pool is `ttnn.mean` over the spatial dims, the
two `fc` linears are `ttnn.matmul` (+ bias) with `ttnn.relu` / `ttnn.sigmoid`,
and the gate is a broadcast `ttnn.multiply`. All in float32 for a clean PCC.
Input/output are NCHW to match the torch module.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.adaptive_avg_pool2d import build as _b_pool


HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind the SE fc weights and return a native ttnn forward closure."""
    import torch

    se = torch_module
    lin0 = se.fc[0]   # Linear(c, c//r)
    lin2 = se.fc[2]   # Linear(c//r, c)

    # global average pool: graduated leaf stub (adaptive_avg_pool2d)
    avg_pool = _b_pool(device, se.avg_pool)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.as_tensor(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    w0 = f32(lin0.weight.detach().t())                         # [c, c//r]
    b0 = f32(lin0.bias.detach().reshape(1, 1, 1, -1))
    w2 = f32(lin2.weight.detach().t())                         # [c//r, c]
    b2 = f32(lin2.bias.detach().reshape(1, 1, 1, -1))

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.as_tensor(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        c = int(x.shape[1])
        # global average pool over spatial (H, W) -> [1, c, 1, 1] (graduated leaf: adaptive_avg_pool2d)
        g = avg_pool(x)
        g = ttnn.reshape(g, (1, 1, 1, c))                      # channel to last dim for matmul
        y = ttnn.relu(ttnn.add(ttnn.matmul(g, w0, compute_kernel_config=compute_config), b0))
        y = ttnn.sigmoid(ttnn.add(ttnn.matmul(y, w2, compute_kernel_config=compute_config), b2))
        y = ttnn.reshape(y, (1, c, 1, 1))                      # back to NCHW per-channel scale
        return ttnn.multiply(x, y)

    return forward


def s_e_layer(*args, **kwargs):
    raise RuntimeError(
        "s_e_layer requires build(device, torch_module) to bind the SE fc weights; "
        "the bare callable has no parameters."
    )
