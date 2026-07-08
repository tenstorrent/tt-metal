# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_ada_norm` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.token_refiner.refiner_blocks.0.norm_out`,
a `HunyuanVideo15AdaNorm(in_features, 2 * in_features)`:

    temb = self.linear(self.nonlinearity(temb))     # SiLU then Linear(in, 2*in)
    gate_msa, gate_mlp = temb.chunk(2, dim=1)        # each (B, in)
    gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)   # (B, 1, in)
    return gate_msa, gate_mlp

Input at test time:
    temb : (B, in_features) — PRIMARY (arrives as a ttnn tensor)

Native ttnn strategy
--------------------
`chunk(linear(...), 2, dim=1)` splits the 2*in projection into the two gates;
output block i depends only on weight rows [i*in:(i+1)*in], so the split is done
on the host weight at build time (two (in, in) matmuls) — identical to the chunk
without slicing a device activation. Each gate is `unsqueeze(1)` -> (B, 1, in).
Float32 math with a HiFi4 config.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module):
    """Bind the AdaNorm linear (split into the two gate halves); native forward."""

    m = torch_module
    lin = m.linear  # Linear(in_features, 2 * in_features)
    in_features = int(lin.in_features)
    half = int(lin.out_features) // 2  # == in_features

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    w = lin.weight.detach()  # (2*half, in_features)
    b = lin.bias.detach() if lin.bias is not None else None

    w_msa = f32(w[:half, :].t())  # (in_features, half)
    w_mlp = f32(w[half:, :].t())
    b_msa = f32(b[:half].reshape(1, half)) if b is not None else None
    b_mlp = f32(b[half:].reshape(1, half)) if b is not None else None

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(temb, *args, **kwargs):
        t = _to_f32_device(temb)
        s = ttnn.silu(t)  # (B, in_features)

        gate_msa = ttnn.matmul(s, w_msa, compute_kernel_config=compute_config)
        gate_mlp = ttnn.matmul(s, w_mlp, compute_kernel_config=compute_config)
        if b_msa is not None:
            gate_msa = ttnn.add(gate_msa, b_msa)
            gate_mlp = ttnn.add(gate_mlp, b_mlp)

        b_dim = int(gate_msa.shape[0])
        gate_msa = ttnn.reshape(gate_msa, (b_dim, 1, half))  # unsqueeze(1)
        gate_mlp = ttnn.reshape(gate_mlp, (b_dim, 1, half))
        return gate_msa, gate_mlp

    return forward


def hunyuan_video15_ada_norm(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_ada_norm requires build(device, torch_module) to bind the "
        "linear weights; the bare callable has no parameters."
    )
