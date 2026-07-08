# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `linear_activation` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.token_refiner.refiner_blocks.0.ff.net.0`,
a diffusers `LinearActivation`:

    hidden_states = self.proj(hidden_states)     # Linear(dim_in, dim_out)
    return self.activation(hidden_states)        # SiLU (here)

Input/output: (B, L, dim_in) -> (B, L, dim_out). Native ttnn: `matmul + bias`
followed by the detected activation (SiLU / GELU / ReLU / ...). Float32, HiFi4.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def _activation_fn(mod):
    name = type(mod).__name__.lower()
    if "silu" in name or "swish" in name:
        return lambda t: ttnn.silu(t)
    if "gelu" in name:
        approx = str(getattr(mod, "approximate", "none"))
        variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
        return lambda t: ttnn.gelu(t, variant=variant)
    if "relu" in name:
        return lambda t: ttnn.relu(t)
    if "mish" in name:
        return lambda t: ttnn.mish(t)
    if "sigmoid" in name:
        return lambda t: ttnn.sigmoid(t)
    if "tanh" in name:
        return lambda t: ttnn.tanh(t)
    return lambda t: t


def build(device, torch_module):
    """Bind the projection + activation and return a native ttnn forward."""
    m = torch_module
    proj = m.proj
    act = _activation_fn(m.activation)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    w = f32(proj.weight.detach().t())
    b = f32(proj.bias.detach().reshape(1, -1)) if proj.bias is not None else None

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(hidden_states, *args, **kwargs):
        x = _to_f32_device(hidden_states)
        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
        if b is not None:
            y = ttnn.add(y, b)
        return act(y)

    return forward


def linear_activation(*args, **kwargs):
    raise RuntimeError(
        "linear_activation requires build(device, torch_module) to bind the projection; "
        "the bare callable has no parameters."
    )
