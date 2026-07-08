# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_image_projection` of tencent/HunyuanVideo-1.5.

Reference submodule: `image_embedder`, a `HunyuanVideo15ImageProjection`:

    hidden = self.norm_in(image_embeds)    # LayerNorm(in_channels), affine
    hidden = self.linear_1(hidden)         # Linear(in_channels, in_channels)
    hidden = self.act_fn(hidden)           # GELU (erf)
    hidden = self.linear_2(hidden)         # Linear(in_channels, hidden_size)
    hidden = self.norm_out(hidden)         # LayerNorm(hidden_size), affine
    return hidden

Input/output: (B, L, in_channels) -> (B, L, hidden_size). Native ttnn: affine
`ttnn.layer_norm`, two `matmul + bias`, a `ttnn.gelu` (variant=Accurate, matching
torch `nn.GELU()`), and a final affine `ttnn.layer_norm`. Float32, HiFi4 config.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module):
    """Bind the two norms + two linears and return a native ttnn forward."""

    m = torch_module
    norm_in = m.norm_in
    norm_out = m.norm_out
    eps_in = float(getattr(norm_in, "eps", 1e-5))
    eps_out = float(getattr(norm_out, "eps", 1e-5))

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def norm_wb(norm):
        w = f32(norm.weight.detach().reshape(1, 1, -1)) if getattr(norm, "weight", None) is not None else None
        b = f32(norm.bias.detach().reshape(1, 1, -1)) if getattr(norm, "bias", None) is not None else None
        return w, b

    def lin_weights(linear):
        w = f32(linear.weight.detach().t())
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    nin_w, nin_b = norm_wb(norm_in)
    nout_w, nout_b = norm_wb(norm_out)
    w1, b1 = lin_weights(m.linear_1)
    w2, b2 = lin_weights(m.linear_2)

    approximate = str(getattr(m.act_fn, "approximate", "none"))
    gelu_variant = ttnn.GeluVariant.Tanh if approximate == "tanh" else ttnn.GeluVariant.Accurate

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _linear(x, w, b):
        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
        if b is not None:
            y = ttnn.add(y, b)
        return y

    def forward(image_embeds, *args, **kwargs):
        x = _to_f32_device(image_embeds)
        h = ttnn.layer_norm(x, epsilon=eps_in, weight=nin_w, bias=nin_b, compute_kernel_config=compute_config)
        h = _linear(h, w1, b1)
        h = ttnn.gelu(h, variant=gelu_variant)
        h = _linear(h, w2, b2)
        h = ttnn.layer_norm(h, epsilon=eps_out, weight=nout_w, bias=nout_b, compute_kernel_config=compute_config)
        return h

    return forward


def hunyuan_video15_image_projection(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_image_projection requires build(device, torch_module) "
        "to bind the norm/linear weights; the bare callable has no parameters."
    )
