# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `speech_connector` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_connector`, a
`vibevoice.modular.modeling_vibevoice.SpeechConnector`:

    x = self.fc1(features)          # Linear(input_dim, output_dim)
    x = self.norm(x)                # LlamaRMSNorm(output_dim, eps=1e-6)
    x = self.fc2(x)                 # Linear(output_dim, output_dim)
    return x

Channel-last (B, T, C) input; ported with `ttnn.matmul` for the two Linears
and native `ttnn.rms_norm` for the norm.
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained connector weights and return a native ttnn forward closure."""
    m = torch_module
    w1 = m.fc1.weight.detach().float()  # [output_dim, input_dim]
    b1 = m.fc1.bias.detach().float() if m.fc1.bias is not None else None
    w2 = m.fc2.weight.detach().float()  # [output_dim, output_dim]
    b2 = m.fc2.bias.detach().float() if m.fc2.bias is not None else None
    norm_weight = m.norm.weight.detach().float()
    eps = float(getattr(m.norm, "variance_epsilon", getattr(m.norm, "eps", 1e-6)))

    fc1_w = ttnn.from_torch(w1.t().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    fc1_b = (
        ttnn.from_torch(b1.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        if b1 is not None
        else None
    )
    fc2_w = ttnn.from_torch(w2.t().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    fc2_b = (
        ttnn.from_torch(b2.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        if b2 is not None
        else None
    )
    norm_w = ttnn.from_torch(
        norm_weight.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(features, *args, **kwargs):
        x = features
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        x = ttnn.matmul(x, fc1_w, compute_kernel_config=compute_config, memory_config=_DRAM)
        if fc1_b is not None:
            x = ttnn.add(x, fc1_b, memory_config=_DRAM)
        x = ttnn.rms_norm(x, epsilon=eps, weight=norm_w, memory_config=_DRAM)
        x = ttnn.matmul(x, fc2_w, compute_kernel_config=compute_config, memory_config=_DRAM)
        if fc2_b is not None:
            x = ttnn.add(x, fc2_b, memory_config=_DRAM)
        return x

    return forward


def speech_connector(*args, **kwargs):
    raise RuntimeError(
        "speech_connector requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
