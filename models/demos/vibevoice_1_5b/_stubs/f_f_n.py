# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `f_f_n` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.encoder.stages.0.0.ffn`, a
`vibevoice.modular.modular_vibevoice_tokenizer.FFN` — plain
Linear(dim, ffn_dim) -> GELU -> Linear(ffn_dim, dim), channel-last
(B, T, C) input (same op `_stubs/block1_d.py` implements inline as `_ffn`,
minus the channels-first permute Block1D wraps it in).
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained FFN weights and return a native ttnn forward closure."""
    m = torch_module
    w1 = m.linear1.weight.detach().float()  # [ffn_dim, dim]
    b1 = m.linear1.bias.detach().float() if m.linear1.bias is not None else None
    w2 = m.linear2.weight.detach().float()  # [dim, ffn_dim]
    b2 = m.linear2.bias.detach().float() if m.linear2.bias is not None else None

    linear1_w = ttnn.from_torch(w1.t().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    linear1_b = (
        ttnn.from_torch(b1.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        if b1 is not None
        else None
    )
    linear2_w = ttnn.from_torch(w2.t().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    linear2_b = (
        ttnn.from_torch(b2.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        if b2 is not None
        else None
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(x, *args, **kwargs):
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        h = ttnn.matmul(x, linear1_w, compute_kernel_config=compute_config, memory_config=_DRAM)
        if linear1_b is not None:
            h = ttnn.add(h, linear1_b, memory_config=_DRAM)
        h = ttnn.gelu(h, memory_config=_DRAM)
        h = ttnn.matmul(h, linear2_w, compute_kernel_config=compute_config, memory_config=_DRAM)
        if linear2_b is not None:
            h = ttnn.add(h, linear2_b, memory_config=_DRAM)
        return h

    return forward


def f_f_n(*args, **kwargs):
    raise RuntimeError(
        "f_f_n requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
