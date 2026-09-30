# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrization_list`
(`audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight`).

`ParametrizationList.forward()` takes no inputs: it reconstructs a `weight_norm`-parametrized
weight from the two tensors it owns, `original0` (g, `[out, 1, 1]`) and `original1` (v,
`[out, in, k]`):

    w[o] = g[o] * v[o] / ||v[o]||_2

i.e. the L2 norm is taken over every axis EXCEPT the output channel (`dim=0`). Both originals live
on the device, so the whole reconstruction is ttnn arithmetic over device-resident state.

Computed in float32. The norm sums 876 squares per output channel, and `g` here is **signed** --
23 of this layer's 1024 channels are negative -- so the identity is `||w|| == |g|`, not `== g`;
checking it the wrong way reads as a relative error of 2.0.

`weight_norm=` takes the ported `_WeightNorm` (`_stubs/weight_norm.py`) and delegates the
reconstruction to it, with `g`/`v` in their natural `[out, 1, 1]` / `[out, in, k]` shapes. That is
the real containment relation -- a `ParametrizationList` IS a list of parametrizations and this one
holds exactly that module -- so the composed pipeline runs the pair rather than two copies of the
same arithmetic. Left out, the reduction runs here against the flattened `[out, fan_in]` view.
"""

from __future__ import annotations

import torch

import ttnn


def _from_torch(t, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def build(device, torch_module):
    plist = torch_module
    weight_norm = plist[0]
    if int(weight_norm.dim) != 0:
        raise NotImplementedError(
            f"weight_norm over dim {weight_norm.dim} is not ported; only dim 0 (per-output-channel)"
        )

    v_torch = plist.original1.detach()
    g_torch = plist.original0.detach()
    shape = [int(s) for s in v_torch.shape]
    out_channels = shape[0]
    fan_in = int(v_torch.numel() // out_channels)

    v = _from_torch(v_torch.reshape(1, 1, out_channels, fan_in).contiguous(), device)
    g = _from_torch(g_torch.reshape(1, 1, out_channels, 1).contiguous(), device)
    # The same two originals in the shapes `_WeightNorm.forward(weight_g, weight_v)` takes.
    v_nat = _from_torch(v_torch.reshape(shape).contiguous(), device)
    g_nat = _from_torch(g_torch.reshape(out_channels, 1, 1).contiguous(), device)

    def parametrization_list(*args, weight_norm=None, **kwargs):
        if weight_norm is not None:
            return ttnn.reshape(weight_norm(g_nat, v_nat), shape)
        sq_sum = ttnn.sum(ttnn.multiply(v, v), dim=-1, keepdim=True)
        scale = ttnn.multiply(ttnn.rsqrt(sq_sum), g)
        return ttnn.reshape(ttnn.multiply(v, scale), shape)

    return parametrization_list
