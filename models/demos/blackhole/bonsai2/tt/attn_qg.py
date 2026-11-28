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


class HeadMajor:
    """A projection output written head-major [1, heads, T, D] (tprefill/linear2.py split=), standing in for the shared
    code's reshaped [B, T, heads, D] tensor: its reshape and transpose(1, 2) become no-ops and its per-head rms_norm
    runs on the head-major tensor (the same rows)."""

    def __init__(self, t, pending=None):
        self.t = t
        self.pending = pending  # a deferred per-head norm (+ RoPE): tprefill/qk_rope.QKNormRope, applied by apply_rope


class HeadsOut:
    """The SDPA output (head-major) standing in for concatenate_heads' [B, T, H D] (attention.py stand-in)."""

    def __init__(self, t):
        self.t = t


class SigGate:
    """sigmoid(gate), not computed yet."""

    def __init__(self, t):
        self.t = t


class GatedAttn:
    """concatenate_heads(attn) * sigmoid(gate), computed with the o projection's input rotation (rotate.py gate=)."""

    def __init__(self, attn, gate):
        self.attn, self.gate = attn, gate


class QGHeads:
    """The q projection as head-major q heads and the token-major gate (instead of one [q | gate] tensor)."""

    def __init__(self, q, gate):
        self.q, self.gate = q, gate


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
        if isinstance(t, (_QG, HeadMajor)):
            return t
        return ttnn.reshape(t, shape, *args, **kwargs)

    def rms_norm(self, t, *args, **kwargs):
        qkr = getattr(self._inner, "qkr", None) or {}
        if isinstance(t, HeadMajor) and id(kwargs.get("weight")) in qkr:
            return HeadMajor(t.t, pending=qkr[id(kwargs["weight"])])  # norm + RoPE together in apply_rope
        if isinstance(t, HeadMajor):
            y = ttnn.rms_norm(t.t, *args, **kwargs)
            ttnn.deallocate(t.t)
            return HeadMajor(y)
        return ttnn.rms_norm(t, *args, **kwargs)

    def transpose(self, t, *args, **kwargs):
        if isinstance(t, HeadMajor):
            return t if t.pending is not None else t.t  # already [B, heads, T, D]
        return ttnn.transpose(t, *args, **kwargs)

    def apply_rope(self, q, k, cos, sin, orig):
        """Stands in for the shared apply_rotary_pos_emb_ttnn: deferred head-major norms run fused with the RoPE."""
        if isinstance(q, HeadMajor) and q.pending is not None and isinstance(k, HeadMajor) and k.pending is not None:
            out = []
            for t in (q, k):
                out.append(t.pending(t.t, cos, sin))
                ttnn.deallocate(t.t)
            return tuple(out)
        return orig(q, k, cos, sin)

    def chunk(self, t, n, dim=-1):
        if not isinstance(t, _QG):
            return ttnn.chunk(t, n, dim=dim)
        if isinstance(t.t, QGHeads):
            return HeadMajor(t.t.q), t.t.gate
        x = t.t
        B, T, W = list(x.shape)
        half = W // 2
        q = ttnn.slice(x, [0, 0, 0], [B, T, half])
        g = ttnn.slice(x, [0, 0, half], [B, T, W])
        return ttnn.reshape(q, [B, T, self._H, self._D]), g

    def deallocate(self, t, *args, **kwargs):
        if isinstance(t, _QG):
            if isinstance(t.t, QGHeads):  # its q and gate live on as the chunk outputs
                return None
            return ttnn.deallocate(t.t)
        if isinstance(t, HeadMajor):
            return ttnn.deallocate(t.t)
        if isinstance(t, (HeadsOut, SigGate, GatedAttn)):
            return None  # consumed (and freed) by the fused gated rotation
        return ttnn.deallocate(t, *args, **kwargs)
