# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai's full-attention projections in their own ternary values for one-token decode: q (with its gate half),
k and v (one ternary pass, split into two outputs) on the rotated hidden state, o on the rotated attention output.
The converter kept every attention weight as is (no row or column reorder)."""

import numpy as np

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary import prerot
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, pack_records_inline
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation


def pack_attention(gguf, layer_num):
    t_q, s_q = gguf.ternary(f"blk.{layer_num}.attn_q.weight")
    t_k, s_k = gguf.ternary(f"blk.{layer_num}.attn_k.weight")
    t_v, s_v = gguf.ternary(f"blk.{layer_num}.attn_v.weight")
    t_o, s_o = gguf.ternary(f"blk.{layer_num}.attn_output.weight")
    rq, pq = pack_records_inline(t_q, s_q)
    t_kv = np.ascontiguousarray(np.concatenate([t_k, t_v], axis=1))
    s_kv = np.ascontiguousarray(np.concatenate([s_k, s_v], axis=0))
    rkv, pkv = pack_records_inline(t_kv, s_kv)
    ro, po = pack_records_inline(t_o, s_o)
    return {
        "q": (rq, pq, t_q.shape[0], t_q.shape[1]),
        "kv": (rkv, pkv, t_kv.shape[0], t_kv.shape[1], t_k.shape[1]),
        "o": (ro, po, t_o.shape[0], t_o.shape[1]),
    }


def qg_records(rec, ppr, heads, head_tiles):
    """q_proj records reordered from per head [q | gate] to [all q heads | all gate heads] (whole output tile columns
    move: attn_qg.reorder_q_proj's layout, for the prefill's q/gate split)."""
    r = rec.reshape(heads, 2, head_tiles, ppr * rec.shape[1])
    return np.ascontiguousarray(r.transpose(1, 0, 2, 3)).reshape(-1, rec.shape[1])


_PREFILL = {}  # id(dense weights) -> TernaryAttention with prefill ops (attention/prefill.py looks it up)


class TernaryAttention:
    """Called in the order gated_attention_forward_ttnn calls its linears: q, k, v, o."""

    def __init__(
        self, device, packed, rot_in: DecodeRotation, rot_out: DecodeRotation, prefill_rotations=None, heads=None
    ):
        self.q = TernaryDecodeLinear3.from_packed(device, *packed["q"])
        rec, ppr, K, N, split = packed["kv"]
        self.kv = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
        self.kv_split = split
        self.o = TernaryDecodeLinear3.from_packed(device, *packed["o"])
        self.rot_in, self.rot_out = rot_in, rot_out
        self._xr = None
        self._v = None
        self.pre_q = None
        if prefill_rotations is not None:
            # exact ternary prefill (bonsai2/tt/tprefill): q with its columns as [q heads | gate heads]
            from models.demos.blackhole.bonsai2.tt.tprefill.linear import TernaryPrefillLinear

            rq, pq, Kq, Nq = packed["q"]
            hd_tiles = Nq // heads // 2 // 32
            q_qg = TernaryDecodeLinear3.from_packed(device, qg_records(rq, pq, heads, hd_tiles), pq, Kq, Nq)
            self.pre_q = TernaryPrefillLinear.from_decode(q_qg)
            self.pre_kv = TernaryPrefillLinear.from_decode(self.kv)
            self.pre_o = TernaryPrefillLinear.from_decode(self.o)
            self.prot_in, self.prot_out = prefill_rotations[Kq], prefill_rotations[self.o.K]

    def project_q(self, x):
        lead = tuple(list(x.shape)[:-1])
        self._xr = prerot.take(x)  # k and v reuse it
        if self._xr is None:
            self._xr = self.rot_in(x)
        return self.q(self._xr, lead=lead)

    def project_k(self, x):
        lead = tuple(list(x.shape)[:-1])
        k, self._v = self.kv(self._xr, lead=lead, split=self.kv_split)
        ttnn.deallocate(self._xr)
        self._xr = None
        return k

    def project_v(self, x):
        v, self._v = self._v, None
        return v

    def project_o(self, a):
        lead = tuple(list(a.shape)[:-1])
        ar = self.rot_out(a)
        y = self.o(ar, lead=lead)
        ttnn.deallocate(ar)
        return y


class TernaryAttentionPrefillTtnn:
    """ttnn stand-in for the prefill attention (inside attn_qg.QGSplitTtnn): q, k, v, o in the exact ternary weights.
    q comes out as [q heads | gate heads], as the reordered dense q_proj did."""

    def __init__(self, weights, ternary: TernaryAttention):
        self._w, self._t = weights, ternary
        self._xr = self._v = None

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def linear(self, a, b, *args, **kwargs):
        w, t = self._w, self._t
        if b is w.q_proj:
            self._xr = t.prot_in(a)
            return t.pre_q(self._xr)
        if b is w.k_proj:
            kv = t.pre_kv(self._xr)
            ttnn.deallocate(self._xr)
            self._xr = None
            s, n = t.kv_split, list(kv.shape)
            k = ttnn.slice(kv, [0] * (len(n) - 1) + [0], n[:-1] + [s])
            self._v = ttnn.slice(kv, [0] * (len(n) - 1) + [s], n)
            ttnn.deallocate(kv)
            return k
        if b is w.v_proj:
            v, self._v = self._v, None
            return v
        if b is w.o_proj:
            ar = t.prot_out(a)
            y = t.pre_o(ar)
            ttnn.deallocate(ar)
            return y
        return ttnn.linear(a, b, *args, **kwargs)


def prefill_stand_in(weights, num_heads, head_dim):
    """The prefill ttnn stand-in for these attention weights, or None (no exact ternary prefill)."""
    t = _PREFILL.get(id(weights))
    if t is None:
        return None
    from models.demos.blackhole.bonsai2.tt.attn_qg import QGSplitTtnn

    return QGSplitTtnn(weights, num_heads, head_dim, inner=TernaryAttentionPrefillTtnn(weights, t))


class TernaryAttentionTtnn:
    """ttnn stand-in for gated_attention_forward_ttnn's module (decode only): its four projections go to the
    TernaryAttention, everything else is ttnn."""

    def __init__(self, weights, ternary: TernaryAttention):
        self._w, self._t = weights, ternary

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def linear(self, a, b, *args, **kwargs):
        w, t = self._w, self._t
        if b is w.q_proj:
            return t.project_q(a)
        if b is w.k_proj:
            return t.project_k(a)
        if b is w.v_proj:
            return t.project_v(a)
        if b is w.o_proj:
            return t.project_o(a)
        return ttnn.linear(a, b, *args, **kwargs)


def swap_in_ternary_attention(model, gguf, rotations, cache_dir=None, prefill_rotations=None) -> None:
    """Free each full-attention layer's dense q/k/v/o and give it a TernaryAttention (attention/decode.py)."""
    from pathlib import Path

    device = model.mesh_device
    for layer in model.layers:
        if not layer.is_full_attention:
            continue
        f = Path(cache_dir) / f"blk.{layer.layer_num}.attn.decode_inline.npz" if cache_dir else None
        if f is not None and f.exists():
            z = np.load(f)
            packed = {
                "q": (z["rq"], int(z["pq"]), int(z["kq"]), int(z["nq"])),
                "kv": (z["rkv"], int(z["pkv"]), int(z["kkv"]), int(z["nkv"]), int(z["skv"])),
                "o": (z["ro"], int(z["po"]), int(z["ko"]), int(z["no"])),
            }
        else:
            packed = pack_attention(gguf, layer.layer_num)
            if f is not None:
                q, kv, o = packed["q"], packed["kv"], packed["o"]
                np.savez(
                    f,
                    rq=q[0], pq=q[1], kq=q[2], nq=q[3],
                    rkv=kv[0], pkv=kv[1], kkv=kv[2], nkv=kv[3], skv=kv[4],
                    ro=o[0], po=o[1], ko=o[2], no=o[3],
                )  # fmt: skip
        attn = layer.attention
        w = attn.weights
        for t in (w.q_proj, w.k_proj, w.v_proj, w.o_proj):
            ttnn.deallocate(t)
        attn.ternary = TernaryAttention(
            device, packed, rotations[packed["q"][2]], rotations[packed["o"][2]],
            prefill_rotations=prefill_rotations, heads=attn.config.num_heads,
        )  # fmt: skip
        if prefill_rotations is not None:
            _PREFILL[id(w)] = attn.ternary
