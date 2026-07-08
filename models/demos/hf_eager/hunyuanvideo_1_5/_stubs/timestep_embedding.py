# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timestep_embedding` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.time_text_embed.timestep_embedder`, a
diffusers `TimestepEmbedding`:

    linear_1 : Linear(256 -> 2048, bias=True)
    act      : SiLU
    linear_2 : Linear(2048 -> 2048, bias=True)
    cond_proj: None      (cond_proj_dim is None for this model)
    post_act : None

    forward(sample, condition=None):
        sample = linear_1(sample)
        sample = act(sample)          # SiLU
        sample = linear_2(sample)
        return sample

Inputs at test time:
    sample : (1, 64, 256) — PRIMARY (arrives as a ttnn tensor); `condition`
             has default None, is not a well-known arg, so the harness drops it.

Native ttnn strategy
--------------------
This is a plain 2-layer MLP: `matmul + bias -> SiLU -> matmul + bias`, the same
shape the sibling `combined_timestep_text_proj_embeddings` stub already ports
for this very `timestep_embedder`. Weights are transposed on host at build time
(`w = weight.t()` so `x @ w` matches `nn.Linear`), uploaded once as float32, and
all device math runs float32 under a HiFi4 kernel config for PCC headroom.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module):
    """Bind the two Linear layers; return a native ttnn forward."""
    m = torch_module

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def lin_weights(linear):
        w = f32(linear.weight.detach().t())  # (in, out)
        b = f32(linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
        return w, b

    w1, b1 = lin_weights(m.linear_1)
    w2, b2 = lin_weights(m.linear_2)

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(x, *args, **kwargs):
        # `condition` (cond_proj) is None for this model; the harness never
        # passes it, and any positional/keyword extras are ignored.
        h = _to_f32_device(x)
        h = ttnn.matmul(h, w1, compute_kernel_config=compute_config)
        if b1 is not None:
            h = ttnn.add(h, b1)
        h = ttnn.silu(h)
        h = ttnn.matmul(h, w2, compute_kernel_config=compute_config)
        if b2 is not None:
            h = ttnn.add(h, b2)
        return h

    return forward


def timestep_embedding(*args, **kwargs):
    raise RuntimeError(
        "timestep_embedding requires build(device, torch_module) to bind the "
        "linear_1/linear_2 weights; the bare callable has no parameters."
    )
