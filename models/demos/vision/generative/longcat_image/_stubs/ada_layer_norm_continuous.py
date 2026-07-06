# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_layer_norm_continuous` (diffusers ``AdaLayerNormContinuous``)
for meituan-longcat/LongCat-Image — this is the model's final ``transformer.norm_out``.

Reference (diffusers ``AdaLayerNormContinuous.forward``)::

    emb = self.linear(self.silu(conditioning_embedding))   # Linear: cond_dim -> 2*embedding_dim
    scale, shift = torch.chunk(emb, 2, dim=1)               # each (B, embedding_dim)
    x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

For LongCat-Image's ``norm_out``:
  * ``linear`` : nn.Linear(3072 -> 6144, bias=True)
  * ``norm``   : LayerNorm(3072, eps=1e-6, elementwise_affine=False)  -> pure normalize, no affine
  * ``silu``   : SiLU

The whole thing runs natively in TTNN. ``chunk(emb, 2, dim=1)`` splits the linear's
output channels into the first half (scale) and second half (shift); this is identical
to splitting the linear WEIGHT ROWS ``[:C]`` / ``[C:]`` and biases at build time, which
lets us compute ``scale`` and ``shift`` with two independent matmuls and avoids any
runtime slice op.

Inputs at runtime (from the PCC harness):
  * ``x``                      : ttnn.Tensor, (B, N, embedding_dim), bf16, on device (primary).
  * ``conditioning_embedding`` : torch.Tensor, (B, cond_dim) (passed through as an extra kwarg);
                                 converted to a device ttnn.Tensor here.
"""

from __future__ import annotations

import torch

import ttnn


def _to_device_tile(t: torch.Tensor, device, *, dtype=ttnn.bfloat16):
    """torch -> ttnn on `device` in TILE layout (mesh-replicated on a mesh device)."""
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class _AdaLayerNormContinuous:
    """Native TTNN AdaLayerNormContinuous. Built once from the torch module; callable per-forward."""

    def __init__(self, device, torch_module):
        self.device = device

        lin = torch_module.linear
        w = lin.weight.detach().to(torch.float32)  # (2*C, C_cond)
        two_c = w.shape[0]
        self.embedding_dim = two_c // 2  # C
        c = self.embedding_dim

        # chunk(emb, 2, dim=1) == split output channels -> rows [:C] (scale) / [C:] (shift).
        # nn.Linear computes  y = x @ W.T ; ttnn.linear(a, b) computes a @ b, so pre-transpose
        # each weight half to (C_cond, C).
        w_scale = w[:c, :].transpose(0, 1).contiguous()  # (C_cond, C)
        w_shift = w[c:, :].transpose(0, 1).contiguous()  # (C_cond, C)
        self.w_scale = _to_device_tile(w_scale, device)
        self.w_shift = _to_device_tile(w_shift, device)

        if lin.bias is not None:
            b = lin.bias.detach().to(torch.float32)  # (2*C,)
            self.b_scale = _to_device_tile(b[:c].reshape(1, c), device)
            self.b_shift = _to_device_tile(b[c:].reshape(1, c), device)
        else:
            self.b_scale = None
            self.b_shift = None

        norm = torch_module.norm
        self.eps = float(getattr(norm, "eps", 1e-6))
        # elementwise_affine=False on LongCat's norm_out -> no learnable gamma/beta. Still
        # handle the affine case generically in case another checkpoint enables it.
        nw = getattr(norm, "weight", None)
        nb = getattr(norm, "bias", None)
        self.norm_weight = (
            _to_device_tile(nw.detach().to(torch.float32).reshape(-1), device) if nw is not None else None
        )
        self.norm_bias = _to_device_tile(nb.detach().to(torch.float32).reshape(-1), device) if nb is not None else None

    def _as_device_tensor(self, t):
        if isinstance(t, ttnn.Tensor):
            return t
        if isinstance(t, torch.Tensor):
            return _to_device_tile(t, self.device)
        raise TypeError(f"AdaLayerNormContinuous: unexpected input type {type(t)!r}")

    def __call__(self, x, conditioning_embedding=None, **_ignored):
        if conditioning_embedding is None:
            raise ValueError("AdaLayerNormContinuous requires `conditioning_embedding`")

        x = self._as_device_tensor(x)
        cond = self._as_device_tensor(conditioning_embedding)  # (B, C_cond)

        # emb = linear(silu(cond)) split into scale/shift via the two weight halves.
        silu_cond = ttnn.silu(cond)
        scale = ttnn.linear(silu_cond, self.w_scale, bias=self.b_scale)  # (B, C)
        shift = ttnn.linear(silu_cond, self.w_shift, bias=self.b_shift)  # (B, C)

        # (B, C) -> (B, 1, C) so the per-channel scale/shift broadcast over the sequence dim.
        b = scale.shape[0]
        c = self.embedding_dim
        scale = ttnn.reshape(scale, (b, 1, c))
        shift = ttnn.reshape(shift, (b, 1, c))

        normed = ttnn.layer_norm(x, epsilon=self.eps, weight=self.norm_weight, bias=self.norm_bias)

        one_plus_scale = ttnn.add(scale, 1.0)
        out = ttnn.mul(normed, one_plus_scale)  # broadcast (B,N,C) * (B,1,C)
        out = ttnn.add(out, shift)  # broadcast (B,N,C) + (B,1,C)
        return out


def build(device, torch_module):
    """PCC-harness entry point: construct the native TTNN port from the torch submodule."""
    return _AdaLayerNormContinuous(device, torch_module)
