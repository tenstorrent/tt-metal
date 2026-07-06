# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timestep_embedding` (diffusers ``TimestepEmbedding``) for
meituan-longcat/LongCat-Image, submodule ``transformer.time_embed.timestep_embedder``.

``TimestepEmbedding.forward(sample[..., in_channels], condition=None)``::

    if condition is not None:  sample = sample + cond_proj(condition)   # cond_proj is None here
    sample = linear_1(sample)
    sample = act(sample)                                                # SiLU
    sample = linear_2(sample)
    # post_act is None
    return sample

This is exactly the ``timestep_embedder`` MLP the graduated
`long_cat_image_timestep_embeddings` parent runs (Linear -> SiLU -> Linear); here it
is the standalone submodule, so it takes the already-projected sinusoidal ``sample``
as input instead of the raw timestep. Runs fully fp32 (HiFi4 / fp32-accumulate).

The module is handed in via ``build(device, torch_module)`` — no model load — so unlike
the CPU fallback this needs no ``AutoModel`` path (this pipeline has no ``model_type``).
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
TILE = ttnn.TILE_LAYOUT


class _TimestepEmbedding:
    def __init__(self, device, mod):
        self.device = device
        self.mod = mod.eval() if hasattr(mod, "eval") else mod
        self._lin = {}
        self._compute = None

    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
        return self._compute

    def _to_f32(self, x):
        if isinstance(x, ttnn.Tensor):
            if x.layout != TILE:
                x = ttnn.to_layout(x, TILE)
            if x.dtype != F32:
                x = ttnn.typecast(x, F32)
            return x
        return ttnn.from_torch(x.to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)

    def _linear(self, x, tm):
        key = id(tm)
        if key not in self._lin:
            b = tm.bias
            self._lin[key] = (
                ttnn.from_torch(
                    tm.weight.detach().to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM
                ),
                ttnn.from_torch(
                    b.detach().reshape(1, -1).to(torch.float32),
                    dtype=F32,
                    layout=TILE,
                    device=self.device,
                    memory_config=DRAM,
                )
                if b is not None
                else None,
            )
        wt, bt = self._lin[key]
        return ttnn.linear(
            x, wt, bias=bt, transpose_b=True, memory_config=DRAM, dtype=F32, compute_kernel_config=self._ck()
        )

    def __call__(self, sample=None, condition=None, **_ignored):
        if sample is None:
            raise ValueError("timestep_embedding stub requires `sample`")
        te = self.mod
        x = self._to_f32(sample)
        cond_proj = getattr(te, "cond_proj", None)
        if cond_proj is not None and condition is not None:
            x = ttnn.add(x, self._linear(self._to_f32(condition), cond_proj))
        x = self._linear(x, te.linear_1)
        x = ttnn.silu(x)  # act == SiLU
        x = self._linear(x, te.linear_2)
        return x


def build(device, torch_module):
    """PCC-harness entry point: native TTNN diffusers TimestepEmbedding."""
    return _TimestepEmbedding(device, torch_module)
