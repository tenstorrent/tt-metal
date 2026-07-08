# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_layer_norm_zero` of tencent/HunyuanVideo-1.5.

Reference submodule: `transformer_blocks.0.norm1`, a diffusers
`AdaLayerNormZero(hidden_size, norm_type="layer_norm")` built with
num_embeddings=None (so `self.emb is None`), i.e.:

    emb = self.linear(self.silu(emb))                         # Linear(C, 6C)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
    x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]   # LayerNorm(C), no affine
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

Inputs at test time:
    x   : (B, L, C) — primary tensor (arrives as a ttnn tensor)
    emb : (B, C)    — per-batch conditioning vector (arrives as a torch tensor)

Native ttnn strategy
--------------------
`chunk(linear(...), 6, dim=1)` splits the 6C projection into six (B, C) pieces;
output column block i depends only on weight rows [i*C:(i+1)*C]. We split the
host weight into six (C, C) matmuls at build time — identical to the chunk but
without slicing a device activation. Only `scale_msa`/`shift_msa` feed the
normalized `x` (index 0, the tensor per-component PCC compares); the remaining
four modulation vectors are returned unmodified so the full transformer block
gets its faithful 5-tuple. All device math is float32 with a HiFi4 config.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"

# chunk order of the 6C linear output (diffusers AdaLayerNormZero.forward).
_CHUNK_NAMES = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")


def build(device, torch_module):
    """Bind the AdaLayerNormZero weights and return a native ttnn forward."""

    m = torch_module
    lin = m.linear  # Linear(embedding_dim, 6 * embedding_dim)
    norm = m.norm  # LayerNorm(embedding_dim, elementwise_affine=False)

    embedding_dim = int(lin.out_features) // 6  # C
    eps = float(getattr(norm, "eps", 1e-6))

    norm_weight = getattr(norm, "weight", None)
    norm_bias = getattr(norm, "bias", None)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    w = lin.weight.detach()  # (6C, embedding_dim)
    b = lin.bias.detach() if lin.bias is not None else None

    # Split the 6C projection into six (embedding_dim, C) weight blocks + bias.
    w_blocks = []
    b_blocks = []
    for i in range(6):
        lo, hi = i * embedding_dim, (i + 1) * embedding_dim
        w_blocks.append(f32(w[lo:hi, :].t()))  # (embedding_dim, C)
        b_blocks.append(f32(b[lo:hi].reshape(1, embedding_dim)) if b is not None else None)

    ttnn_norm_w = f32(norm_weight.detach().reshape(1, 1, embedding_dim)) if norm_weight is not None else None
    ttnn_norm_b = f32(norm_bias.detach().reshape(1, 1, embedding_dim)) if norm_bias is not None else None

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(x, emb=None, *args, **kwargs):
        if emb is None:
            # tolerate positional emb: forward(x, timestep, class_labels, hidden_dtype, emb)
            if "emb" in kwargs:
                emb = kwargs["emb"]
            elif args:
                emb = args[-1]
        if emb is None:
            raise TypeError("ada_layer_norm_zero forward needs the `emb` conditioning tensor")

        x = _to_f32_device(x)
        e = _to_f32_device(emb)

        s = ttnn.silu(e)  # (B, C)
        parts = []
        for w_i, b_i in zip(w_blocks, b_blocks):
            p = ttnn.matmul(s, w_i, compute_kernel_config=compute_config)
            if b_i is not None:
                p = ttnn.add(p, b_i)
            parts.append(p)  # each (B, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts

        b_dim = int(scale_msa.shape[0])
        scale_msa_r = ttnn.reshape(scale_msa, (b_dim, 1, embedding_dim))
        shift_msa_r = ttnn.reshape(shift_msa, (b_dim, 1, embedding_dim))

        norm = ttnn.layer_norm(
            x, epsilon=eps, weight=ttnn_norm_w, bias=ttnn_norm_b, compute_kernel_config=compute_config
        )
        x_out = ttnn.add(ttnn.multiply(norm, ttnn.add(scale_msa_r, 1.0)), shift_msa_r)

        return x_out, gate_msa, shift_mlp, scale_mlp, gate_mlp

    return forward


def ada_layer_norm_zero(*args, **kwargs):
    raise RuntimeError(
        "ada_layer_norm_zero requires build(device, torch_module) to bind the "
        "linear/norm weights; the bare callable has no parameters."
    )
