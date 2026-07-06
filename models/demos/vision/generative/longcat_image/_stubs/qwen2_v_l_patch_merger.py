# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_patch_merger`
(transformers ``Qwen2_5_VLPatchMerger``) — the LongCat-Image Qwen2.5-VL vision
tower's patch merger (submodule ``text_encoder.model.visual.merger``).

Reference forward::

    x = mlp(ln_q(x).view(-1, hidden_size))

where ``ln_q`` is a RMSNorm over ``context_dim`` (= ln_q.weight[-1]),
``hidden_size = context_dim * spatial_merge_size**2``, and
``mlp = [Linear(hidden_size, hidden_size), GELU, Linear(hidden_size, dim)]``.
So: RMSNorm the context_dim, regroup ``spatial_merge_size**2`` tokens into one
row of ``hidden_size``, then a 2-layer GELU MLP that projects to the LM hidden
dim. (The vision tower does not fire on the text→image path, so this component
has no captured inputs; the PCC test drives it with a synthetic
``[rows, context_dim]`` tensor whose row count is a multiple of the merge
factor.)

Precision: fp32 activations, fp32 weights, HiFi4 / fp32-dest-acc. This is a tiny,
well-conditioned module (one norm + two linears + exact-erf GELU), so single-pass
HiFi4 clears PCC>=0.99 without the emulated-fp32 machinery the 28-layer decoder
stack needs. GELU uses the accurate (erf) mode to match ``nn.GELU()``'s default.
"""

from __future__ import annotations

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32 = ttnn.float32
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


class _Merger:
    def __init__(self, device, merger):
        self.device = device
        self.m = merger.eval() if hasattr(merger, "eval") else merger
        self.hidden_size = int(self.m.hidden_size)
        self.eps = float(getattr(self.m.ln_q, "variance_epsilon", 1e-6))
        self._compute = None
        self._lin = {}
        self._rms = None

    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        return self._compute

    def _to(self, t):
        if isinstance(t, ttnn.Tensor):
            if t.layout != TILE:
                t = ttnn.to_layout(t, TILE)
            if t.dtype != F32:
                t = ttnn.typecast(t, F32)
            return t
        return ttnn.from_torch(t.to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)

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

    def _rmsnorm(self, x):
        # Qwen2_5_VLRMSNorm over the last dim (context_dim), fp32.
        if self._rms is None:
            w = self.m.ln_q.weight.detach().reshape(1, -1).to(torch.float32)
            self._rms = ttnn.from_torch(w, dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)
        msq = ttnn.mean(ttnn.mul(x, x), dim=-1, keepdim=True, compute_kernel_config=self._ck())
        normed = ttnn.mul(x, ttnn.rsqrt(ttnn.add(msq, self.eps)))
        return ttnn.mul(normed, self._rms)

    def __call__(self, x=None, hidden_states=None, **_ignored):
        if x is None:
            x = hidden_states
        if x is None:
            raise ValueError("qwen2_v_l_patch_merger stub requires input `x`")
        x = self._to(x)  # [..., context_dim]
        h = self._rmsnorm(x)  # RMSNorm over context_dim
        # regroup into rows of hidden_size: view(-1, hidden_size). Reshape is a
        # row-major op, so drop to ROW_MAJOR, reshape, back to TILE.
        total = 1
        for d in h.shape:
            total *= int(d)
        rows = total // self.hidden_size
        h = ttnn.to_layout(h, RM)
        h = ttnn.reshape(h, [rows, self.hidden_size])
        h = ttnn.to_layout(h, TILE)
        h = self._linear(h, self.m.mlp[0])  # Linear(hidden, hidden) + bias
        h = ttnn.gelu(h, fast_and_approximate_mode=False)  # exact-erf GELU (nn.GELU default)
        h = self._linear(h, self.m.mlp[2])  # Linear(hidden, dim) + bias
        return h


def build(device, torch_module):
    """PCC-harness entry point: native TTNN Qwen2.5-VL patch merger."""
    return _Merger(device, torch_module)
