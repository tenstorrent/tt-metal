# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_layer_norm_zero` (diffusers ``AdaLayerNormZero``) for
meituan-longcat/LongCat-Image — the ``norm1`` of each dual-stream transformer block.

Reference (diffusers ``AdaLayerNormZero.forward``, ``self.emb is None`` here)::

    emb = self.linear(self.silu(emb))                                       # Linear: C -> 6*C
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
    x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

For LongCat-Image's ``transformer_blocks.0.norm1``:
  * ``linear`` : nn.Linear(3072 -> 18432 = 6*3072, bias=True)
  * ``norm``   : LayerNorm(3072, eps=1e-6, elementwise_affine=False)  -> pure normalize, no affine
  * ``silu``   : SiLU

``chunk(emb, 6, dim=1)`` splits the linear's output channels into 6 equal blocks, which is
identical to splitting the linear WEIGHT ROWS into 6 blocks and biasing at build time; that
lets each chunk come from its own matmul with no runtime slice op. Only ``scale_msa`` (block 1)
and ``shift_msa`` (block 0) modulate ``x``; the remaining blocks are returned verbatim for the
downstream attention / MLP gating (the PCC harness compares output[0] == the modulated ``x``).

Runtime inputs (from the PCC harness):
  * ``x``   : ttnn.Tensor, (B, N, C), bf16, on device (primary).
  * ``emb`` : torch.Tensor, (B, C) (extra kwarg) -> converted to a device ttnn.Tensor here.
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


class _AdaLayerNormZero:
    """Native TTNN AdaLayerNormZero. Built once from the torch module; callable per-forward."""

    def __init__(self, device, torch_module):
        self.device = device

        norm = torch_module.norm
        ns = getattr(norm, "normalized_shape", None)
        lin = torch_module.linear
        w = lin.weight.detach().to(torch.float32)  # (num_chunks*C, C_cond)
        # embedding_dim C: prefer the norm's normalized dim; else infer from the linear.
        self.c = int(ns[-1]) if ns else int(w.shape[1])
        c = self.c
        self.num_chunks = int(w.shape[0] // c)  # 6 for AdaLayerNormZero

        # nn.Linear computes y = x @ W.T ; ttnn.linear(a, b) computes a @ b, so pre-transpose
        # each output-channel block to (C_cond, C).
        b = lin.bias.detach().to(torch.float32) if lin.bias is not None else None
        self.chunk_w = []
        self.chunk_b = []
        for i in range(self.num_chunks):
            wi = w[i * c : (i + 1) * c, :].transpose(0, 1).contiguous()  # (C_cond, C)
            self.chunk_w.append(_to_device_tile(wi, device))
            self.chunk_b.append(
                _to_device_tile(b[i * c : (i + 1) * c].reshape(1, c), device) if b is not None else None
            )

        self.eps = float(getattr(norm, "eps", 1e-6))
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
        raise TypeError(f"AdaLayerNormZero: unexpected input type {type(t)!r}")

    def __call__(self, x, emb=None, timestep=None, class_labels=None, hidden_dtype=None, **_ignored):
        if emb is None:
            raise ValueError("AdaLayerNormZero requires `emb` (this checkpoint has no internal embedder)")

        x = self._as_device_tensor(x)
        cond = self._as_device_tensor(emb)  # (B, C_cond)

        silu_emb = ttnn.silu(cond)
        chunks = [ttnn.linear(silu_emb, self.chunk_w[i], bias=self.chunk_b[i]) for i in range(self.num_chunks)]
        # chunk order == diffusers: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        shift_msa, scale_msa = chunks[0], chunks[1]

        b = x.shape[0]
        c = self.c
        scale_msa = ttnn.reshape(scale_msa, (b, 1, c))  # (B,1,C) broadcast over sequence
        shift_msa = ttnn.reshape(shift_msa, (b, 1, c))

        normed = ttnn.layer_norm(x, epsilon=self.eps, weight=self.norm_weight, bias=self.norm_bias)
        out = ttnn.mul(normed, ttnn.add(scale_msa, 1.0))  # norm(x) * (1 + scale_msa)
        out = ttnn.add(out, shift_msa)  # + shift_msa

        # Match torch return: (x, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (out, chunk2, chunk3, ...).
        return tuple([out] + chunks[2:])


def build(device, torch_module):
    """PCC-harness entry point: construct the native TTNN port from the torch submodule."""
    return _AdaLayerNormZero(device, torch_module)
