# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `weight_norm` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.waveform_decoder.ups.0.parametrizations.weight.0`,
a `torch.nn.utils.parametrizations._WeightNorm` (dim=0). Its
`forward(weight_g, weight_v)` reparametrizes a weight tensor:

    weight = torch._weight_norm(weight_v, weight_g, dim)
           = weight_v * (weight_g / norm_except_dim(weight_v, 2, dim))

where `norm_except_dim(v, 2, dim)` is the L2 norm of `v` taken over EVERY axis
except `dim` (kept as size-1 axes). For dim=0 and v of shape (out, ...), the
norm has shape (out, 1, 1, ...).

Native ttnn: the norm is `sqrt(sum(v*v))` reduced over all axes but `dim`
(`ttnn.rsqrt` gives 1/‖v‖ directly), and the reparametrization is two broadcast
`ttnn.multiply`s. All in float32 for a clean PCC. `weight_g` is the primary
(ttnn) input; `weight_v` arrives as a torch tensor kwarg from the harness.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind the weight-norm `dim` and return a native ttnn forward closure."""
    dim = int(getattr(torch_module, "dim", 0))

    def _is_mesh(dev):
        try:
            if isinstance(dev, ttnn.MeshDevice):
                return True
        except AttributeError:
            pass
        return hasattr(dev, "get_device_ids") or hasattr(dev, "get_devices")

    def to_ttnn_f32(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t

        t = t.contiguous().float()
        if _is_mesh(device):
            try:
                return ttnn.as_tensor(
                    t,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
            except (AttributeError, TypeError):
                pass
        return ttnn.as_tensor(
            t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward(weight_g, weight_v=None, *args, **kwargs):
        if weight_v is None:
            # be tolerant to positional passing
            if args:
                weight_v = args[0]
            else:
                weight_v = kwargs.get("weight_v")
        g = to_ttnn_f32(weight_g)
        v = to_ttnn_f32(weight_v)

        rank = len(v.shape)
        d = dim % rank
        # ‖v‖ over every axis except `d`
        sq = ttnn.multiply(v, v)
        for ax in sorted((a for a in range(rank) if a != d), reverse=True):
            sq = ttnn.sum(sq, dim=ax, keepdim=True)
        inv_norm = ttnn.rsqrt(sq)  # 1/‖v‖, broadcastable over the non-`d` axes
        scale = ttnn.multiply(g, inv_norm)  # weight_g / ‖v‖
        return ttnn.multiply(v, scale)  # weight_v * (weight_g / ‖v‖)

    return forward


def weight_norm(*args, **kwargs):
    raise RuntimeError(
        "weight_norm requires build(device, torch_module) to bind the weight-norm dim; "
        "the bare callable has no configuration."
    )
