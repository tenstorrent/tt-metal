# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `linear_norm` (hexgrad/Kokoro-82M `predictor.duration_proj`,
a StyleTTS2 `LinearNorm` — a thin wrapper over `nn.Linear`).

`forward(x) = linear_layer(x)`. Native ttnn: `ttnn.linear` with the stored
weight (transposed to `x @ Wᵀ` orientation) and bias. fp32 for a clean PCC.
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    lin = torch_module.linear_layer
    w_t = ttnn.from_torch(
        lin.weight.detach().t().contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )
    bias = None
    if lin.bias is not None:
        bias = ttnn.from_torch(
            lin.bias.detach().reshape(1, -1).contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(
                x.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
            )
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        return ttnn.linear(x, w_t, bias=bias, compute_kernel_config=compute_config, memory_config=_DRAM)

    return forward


def linear_norm(*args, **kwargs):
    raise RuntimeError(
        "linear_norm requires build(device, torch_module) to bind the linear "
        "weight/bias; the bare callable has no parameters."
    )
