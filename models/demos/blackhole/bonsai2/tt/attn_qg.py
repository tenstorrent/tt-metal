# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-attention q/gate split without the [T, 24, 512] relayout.

The shared gated_attention_forward_ttnn reshapes the q projection [1, T, 12288] (per head: q 256 | gate 256) to
[1, T, 24, 512] and chunks it: on a 2048-token prefill chunk that reshape alone took 4.4 ms. reorder_q_proj stores the
dense q_proj with its columns as [all q heads | all gate heads] (whole tiles move, so the bf8 values are unchanged);
QGSplitTtnn, a ttnn stand-in for that module, then gives the shared code q as a slice reshaped to heads and the gate
as a flat slice."""


import ttnn

_REORDERED = set()


def reorder_q_proj(model):
    for layer in model.layers:
        if not layer.is_full_attention:
            continue
        attn = layer.attention
        w, cfg = attn.weights, attn.config
        H, D = cfg.num_heads, cfg.head_dim
        t = ttnn.to_torch(w.q_proj).float()  # [dim, H * 2D]
        dim = t.shape[0]
        t = t.reshape(dim, H, 2, D).permute(0, 2, 1, 3).reshape(dim, 2 * H * D).contiguous()
        new = ttnn.from_torch(
            t, dtype=w.q_proj.dtype, layout=ttnn.TILE_LAYOUT, device=attn.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(w.q_proj)
        object.__setattr__(w, "q_proj", new)
        _REORDERED.add(id(w))


def is_reordered(weights):
    return id(weights) in _REORDERED


class _QG:
    """The reordered q | gate projection output, standing in for the reshaped [B, T, H, 2D] tensor."""

    def __init__(self, t):
        self.t = t


class QGSplitTtnn:
    def __init__(self, weights, num_heads, head_dim, inner=None):
        self._w, self._H, self._D = weights, num_heads, head_dim
        self._inner = inner  # another stand-in to forward to (the ternary decode one), or ttnn

    def __getattr__(self, name):
        return getattr(self._inner if self._inner is not None else ttnn, name)

    def linear(self, a, b, *args, **kwargs):
        lin = (self._inner if self._inner is not None else ttnn).linear
        if b is self._w.q_proj:
            return _QG(lin(a, b, *args, **kwargs))
        return lin(a, b, *args, **kwargs)

    def reshape(self, t, shape, *args, **kwargs):
        if isinstance(t, _QG):
            return t
        return ttnn.reshape(t, shape, *args, **kwargs)

    def chunk(self, t, n, dim=-1):
        if not isinstance(t, _QG):
            return ttnn.chunk(t, n, dim=dim)
        x = t.t
        B, T, W = list(x.shape)
        half = W // 2
        q = ttnn.slice(x, [0, 0, 0], [B, T, half])
        g = ttnn.slice(x, [0, 0, half], [B, T, W])
        return ttnn.reshape(q, [B, T, self._H, self._D]), g

    def deallocate(self, t, *args, **kwargs):
        if isinstance(t, _QG):
            return ttnn.deallocate(t.t)
        return ttnn.deallocate(t, *args, **kwargs)
