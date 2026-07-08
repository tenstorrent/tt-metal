# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `pix_art_alpha_text_projection` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.time_text_embed.text_embedder`, a
diffusers `PixArtAlphaTextProjection`:

    hidden = self.linear_1(caption)     # Linear(in_features, hidden_size)
    hidden = self.act_1(hidden)         # SiLU (act_fn="silu" here)
    hidden = self.linear_2(hidden)      # Linear(hidden_size, out_features)
    return hidden

Input/output: (..., in_features) -> (..., out_features). Native ttnn: two
`matmul + bias` around the detected activation. Float32 with a HiFi4 config.
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
    return lambda t: t


def build(device, torch_module):
    """Bind the two projections + activation and return a native ttnn forward."""
    m = torch_module
    act = _activation_fn(m.act_1)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def lin(linear):
        w = f32(linear.weight.detach().t())
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    w1, b1 = lin(m.linear_1)
    w2, b2 = lin(m.linear_2)

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(caption, *args, **kwargs):
        x = _to_f32_device(caption)
        h = ttnn.matmul(x, w1, compute_kernel_config=compute_config)
        if b1 is not None:
            h = ttnn.add(h, b1)
        h = act(h)
        h = ttnn.matmul(h, w2, compute_kernel_config=compute_config)
        if b2 is not None:
            h = ttnn.add(h, b2)
        return h

    return forward


def pix_art_alpha_text_projection(*args, **kwargs):
    raise RuntimeError(
        "pix_art_alpha_text_projection requires build(device, torch_module) to bind the "
        "projections; the bare callable has no parameters."
    )
