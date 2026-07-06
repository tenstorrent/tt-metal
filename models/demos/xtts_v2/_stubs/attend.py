# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `attend` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_perceiver.layers.0.0.attend`, an
instance of `TTS.tts.layers.xtts.perceiver_encoder.Attend`. In eval it is a
plain (non-causal, dropout-free) scaled-dot-product attention over 4D operands
`(batch, heads, seq, dim)`:

    scale = q.shape[-1] ** -0.5
    sim   = einsum("b h i d, b h j d -> b h i j", q, k) * scale
    attn  = softmax(sim, dim=-1)
    out   = einsum("b h i j, b h j d -> b h i d", attn, v)

Captured shapes: q `[1, 8, 32, 64]`, k/v `[1, 8, 291, 64]`, mask `None`,
out `[1, 8, 32, 64]`. The module has no learned parameters.

Harness note: the PCC harness converts only the PRIMARY arg (`q`) to a ttnn
tensor; the extra operands (`k`, `v`) arrive as host torch tensors, so `build`
captures the device and this forward lifts them onto it before computing. The
whole similarity/softmax/aggregate path is native ttnn.
"""

from __future__ import annotations

import ttnn


def _to_ttnn(x, device):
    """Lift a host torch tensor onto `device` as a tiled bf16 ttnn tensor.

    ttnn tensors pass through unchanged; non-tensors (e.g. `mask=None`) too.
    """
    if isinstance(x, ttnn.Tensor):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return ttnn.as_tensor(
                x.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    except Exception:
        pass
    return x


def _attend(device, q, k, v, mask=None):
    q = _to_ttnn(q, device)
    k = _to_ttnn(k, device)
    v = _to_ttnn(v, device)

    scale = q.shape[-1] ** -0.5

    # sim = q @ k^T * scale  ->  [b, h, i, j]
    k_t = ttnn.transpose(k, -2, -1)
    sim = ttnn.matmul(q, k_t)
    sim = ttnn.multiply(sim, scale)

    # No key-padding mask and non-causal in the conditioning perceiver
    # (captured mask=None), so softmax runs over the full key axis.
    attn = ttnn.softmax(sim, dim=-1)

    # out = attn @ v  ->  [b, h, i, d]
    out = ttnn.matmul(attn, v)
    return out


def build(device, torch_module):
    """Return the native ttnn forward.

    `Attend` has no parameters, so `torch_module` is unused (kept to satisfy
    the harness's `build(device, torch_module)` contract). The device is
    captured so the forward can lift the host-side `k`/`v` operands.
    """

    def forward(q, k=None, v=None, mask=None, **_ignored):
        return _attend(device, q, k, v, mask)

    return forward


def attend(q, k=None, v=None, mask=None, **kwargs):
    """Bare module-level callable for harnesses that skip `build`.

    Falls back to the q tensor's own device for lifting host operands.
    """
    device = q.device() if isinstance(q, ttnn.Tensor) else None
    return _attend(device, q, k, v, mask)
