# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_by_t5_text_projection` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder_2`, a `HunyuanVideo15ByT5TextProjection`:

    hidden = self.norm(encoder_hidden_states)   # LayerNorm(in_features), affine
    hidden = self.linear_1(hidden)              # Linear(in_features, hidden_size)
    hidden = self.act_fn(hidden)                # GELU (erf)
    hidden = self.linear_2(hidden)              # Linear(hidden_size, hidden_size)
    hidden = self.act_fn(hidden)                # GELU (erf)
    hidden = self.linear_3(hidden)              # Linear(hidden_size, out_features)
    return hidden

Input/output: (B, L, in_features) -> (B, L, out_features). Native ttnn: an affine
`ttnn.layer_norm` followed by three `matmul + bias` with `ttnn.gelu`
(variant=Accurate, matching torch `nn.GELU()`'s erf form) between the first two.
Float32 math with a HiFi4 config.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module):
    """Bind the norm + three linears and return a native ttnn forward."""

    m = torch_module
    norm = m.norm
    eps = float(getattr(norm, "eps", 1e-5))

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def lin_weights(linear):
        w = f32(linear.weight.detach().t())
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    norm_w = f32(norm.weight.detach().reshape(1, 1, -1)) if getattr(norm, "weight", None) is not None else None
    norm_b = f32(norm.bias.detach().reshape(1, 1, -1)) if getattr(norm, "bias", None) is not None else None

    w1, b1 = lin_weights(m.linear_1)
    w2, b2 = lin_weights(m.linear_2)
    w3, b3 = lin_weights(m.linear_3)

    # torch nn.GELU() defaults to the exact (erf) form.
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

    def forward(encoder_hidden_states, *args, **kwargs):
        x = _to_f32_device(encoder_hidden_states)
        h = ttnn.layer_norm(x, epsilon=eps, weight=norm_w, bias=norm_b, compute_kernel_config=compute_config)
        h = _linear(h, w1, b1)
        h = ttnn.gelu(h, variant=gelu_variant)
        h = _linear(h, w2, b2)
        h = ttnn.gelu(h, variant=gelu_variant)
        h = _linear(h, w3, b3)
        return h

    return forward


def hunyuan_video15_by_t5_text_projection(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_by_t5_text_projection requires build(device, torch_module) "
        "to bind the norm/linear weights; the bare callable has no parameters."
    )
