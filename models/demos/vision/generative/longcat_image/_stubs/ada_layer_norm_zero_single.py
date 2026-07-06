# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_layer_norm_zero_single` (diffusers ``AdaLayerNormZeroSingle``)
for meituan-longcat/LongCat-Image — the ``norm`` of each single-stream transformer block.

Reference (diffusers ``AdaLayerNormZeroSingle.forward``)::

    emb = self.linear(self.silu(emb))                         # Linear: C -> 3*C
    shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
    x = self.norm(x) * (1 + scale_msa)[:, None] + shift_msa[:, None]
    return x, gate_msa

For LongCat-Image's ``single_transformer_blocks.0.norm``:
  * ``linear`` : nn.Linear(3072 -> 9216 = 3*3072, bias=True)
  * ``norm``   : LayerNorm(3072, eps=1e-6, elementwise_affine=False)  -> pure normalize, no affine
  * ``silu``   : SiLU

Same construction as ``ada_layer_norm_zero`` but with 3 output-channel chunks instead of 6:
splitting the linear WEIGHT ROWS into 3 blocks == ``emb.chunk(3, dim=1)``. Only ``scale_msa``
(block 1) and ``shift_msa`` (block 0) modulate ``x``; ``gate_msa`` (block 2) is returned for the
downstream single-block gating (the PCC harness compares output[0] == the modulated ``x``).

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


class _AdaLayerNormZeroSingle:
    """Native TTNN AdaLayerNormZeroSingle. Built once from the torch module; callable per-forward."""

    def __init__(self, device, torch_module):
        self.device = device

        norm = torch_module.norm
        ns = getattr(norm, "normalized_shape", None)
        lin = torch_module.linear
        w = lin.weight.detach().to(torch.float32)  # (num_chunks*C, C_cond)
        self.c = int(ns[-1]) if ns else int(w.shape[1])
        c = self.c
        self.num_chunks = int(w.shape[0] // c)  # 3 for AdaLayerNormZeroSingle

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
        raise TypeError(f"AdaLayerNormZeroSingle: unexpected input type {type(t)!r}")

    def __call__(self, x, emb=None, **_ignored):
        if emb is None:
            raise ValueError("AdaLayerNormZeroSingle requires `emb`")

        x = self._as_device_tensor(x)
        cond = self._as_device_tensor(emb)  # (B, C_cond)

        silu_emb = ttnn.silu(cond)
        chunks = [ttnn.linear(silu_emb, self.chunk_w[i], bias=self.chunk_b[i]) for i in range(self.num_chunks)]
        # chunk order == diffusers: [shift_msa, scale_msa, gate_msa]
        shift_msa, scale_msa = chunks[0], chunks[1]

        b = x.shape[0]
        c = self.c
        scale_msa = ttnn.reshape(scale_msa, (b, 1, c))  # (B,1,C) broadcast over sequence
        shift_msa = ttnn.reshape(shift_msa, (b, 1, c))

        normed = ttnn.layer_norm(x, epsilon=self.eps, weight=self.norm_weight, bias=self.norm_bias)
        out = ttnn.mul(normed, ttnn.add(scale_msa, 1.0))  # norm(x) * (1 + scale_msa)
        out = ttnn.add(out, shift_msa)  # + shift_msa

        # Match torch return: (x, gate_msa) = (out, chunk2).
        return tuple([out] + chunks[2:])


def build(device, torch_module):
    """PCC-harness entry point: construct the native TTNN port from the torch submodule."""
    return _AdaLayerNormZeroSingle(device, torch_module)
