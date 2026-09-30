# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `time_embedding` (`acoustic_transformer.time_embedding`).

Sinusoidal embedding of the flow-matching timestep:

    emb = einsum("bi, j -> bj", t, inv_freq);   out = cat(cos(emb), sin(emb))

with `inv_freq = exp(-log(theta) * arange(dim // 2) / (dim // 2))`, dim 3072, theta 1e4. `inv_freq`
is a NON-PERSISTENT buffer -- rebuilt by the module, never loaded from the checkpoint -- and is
staged to the device here.

The einsum contracts `t`'s second axis, i.e. `emb[b, j] = (sum_i t[b, i]) * inv_freq[j]`. That row
sum is done explicitly before the outer product, so a `t` with more than one column gives the same
answer as the reference rather than only working for the `[B, 1]` shape the sampler happens to use.
"""

from __future__ import annotations

import torch

import ttnn


# `ttnn.linear`/`ttnn.matmul` on their DEFAULTS leave `fp32_dest_acc_en` off, so the accumulator
# rounds to bfloat16 at every step even when the activations are float32. The consumer of this
# stack resolves a top-1/top-2 margin of a few hundredths, and the audio path rounds onto 21
# levels 0.1 apart, so that rounding decides real codes. Every matmul below passes this.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _from_torch(t, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def build(device, torch_module):
    emb = torch_module
    inv_freq = _from_torch(emb.inv_freq.detach().reshape(1, -1).contiguous(), device)
    half = int(emb.inv_freq.shape[0])

    def time_embedding(t, **kwargs):
        batch = int(t.shape[0])
        widened = ttnn.typecast(ttnn.reshape(t, [1, 1, batch, int(t.shape[-1])]), ttnn.float32)
        phase = ttnn.matmul(
            ttnn.sum(widened, dim=-1, keepdim=True), inv_freq, compute_kernel_config=_COMPUTE
        )
        out = ttnn.concat([ttnn.cos(phase), ttnn.sin(phase)], dim=-1)
        return ttnn.reshape(out, [batch, 2 * half])

    return time_embedding
