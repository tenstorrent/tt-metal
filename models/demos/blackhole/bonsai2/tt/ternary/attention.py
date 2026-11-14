# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai's full-attention projections in their own ternary values for one-token decode: q (with its gate half),
k and v (one ternary pass, split into two outputs) on the rotated hidden state, o on the rotated attention output.
The converter kept every attention weight as is (no row or column reorder)."""

import os

import numpy as np

import ttnn
from models.demos.blackhole.bonsai2.tt.attn_qg import GatedAttn, HeadMajor, HeadsOut, QGHeads, SigGate
from models.demos.blackhole.bonsai2.tt.ternary import prerot
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, pack_records_inline
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation
from models.demos.blackhole.bonsai2.tt.tprefill.qk_rope import QKNormRope


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
        self.heads = heads
        self.head_dim = packed["q"][3] // heads // 2 if heads else None
        self.pre_q = None
        if prefill_rotations is not None:
            # exact ternary prefill (bonsai2/tt/tprefill): q with its columns as [q heads | gate heads]
            from models.demos.blackhole.bonsai2.tt.tprefill.linear import prefill_linear

            rq, pq, Kq, Nq = packed["q"]
            hd_tiles = Nq // heads // 2 // 32
            rq_qg = qg_records(rq, pq, heads, hd_tiles)
            q_qg = TernaryDecodeLinear3.from_packed(device, rq_qg, pq, Kq, Nq)
            self.pre_q = prefill_linear(q_qg, rq_qg)
            self.pre_hd = hd_tiles  # head size in tiles
            self.pre_q_tiles = Nq // 2 // 32  # q heads' tile columns (then the gate's)
            self.pre_kv = prefill_linear(self.kv, packed["kv"][0])
            self.pre_o = prefill_linear(self.o, packed["o"][0])
            self.prot_in, self.prot_out = prefill_rotations[Kq], prefill_rotations[self.o.K]

    def takes_norm(self):
        """Whether decode can run the layer's norm + rotation inside the q projection's op (prerot.put_norm); k / v
        then read x' from that op's L1 hub buffer (BONSAI_ATTN_NORM_FUSE=0: off)."""
        return os.environ.get("BONSAI_ATTN_NORM_FUSE", "1") == "1" and _ROT_FUSE() and self.heads is not None

    def _kvp(self):
        """This layer's K / V prefetch for the decode SDPA (stream/decode.py kv_prefetch), on the q projection."""
        from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

        kv = getattr(self, "kv_caches", None)
        if kv is None or not stream_decode.active() or stream_decode.STATE.kvp is None:
            return None
        return stream_decode.STATE.kv_prefetch(*kv)

    def project_q(self, x):
        lead = tuple(list(x.shape)[:-1])
        nrot = prerot.take_norm(x)
        kvp = self._kvp()
        if nrot is not None and self._hm_ok(lead):
            q, gate = self.q.rotated(nrot, x, lead=lead, qg_heads=self.heads, out_l1=True, kvp=kvp)
            if kvp is not None:
                from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

                stream_decode.STATE.kv_ready = True  # the SDPA takes its K / V from L1
            self._xr = self.q.last_xl1  # x' in every q core's L1 (k / v's cores are a subset)
            self._xr_local = True
            return _QGDec(q, gate)
        self._xr_local = False
        if nrot is not None:  # handed the norm but not fusing it: run it here
            xn, self._xr = nrot(x)
            ttnn.deallocate(xn)
        else:
            self._xr = prerot.take(x)  # k and v reuse it
        if self._xr is None:
            self._xr = self.rot_in(x)
        if self._hm_ok(lead):  # q head-major and the gate as their own tensors (no reshape / chunk / transpose)
            return _QGDec(*self.q(self._xr, lead=lead, qg_heads=self.heads, out_l1=_ROT_FUSE()))
        return self.q(self._xr, lead=lead)

    def project_k(self, x):
        lead = tuple(list(x.shape)[:-1])
        if self._hm_ok(lead):
            hk = self.kv_split // self.head_dim
            shape = [1, hk, 1, self.head_dim]
            local = getattr(self, "_xr_local", False)
            k, v = self.kv(self._xr, lead=lead, split=self.kv_split, shapes=(shape, shape), x_local=local)
            self._v = _HMDec(v)
            self.v_raw = v  # the k norm + RoPE op may copy it into the cache update's input (TernaryAttentionTtnn)
            k = _HMDec(k)
            if local:  # the shared hub buffer: not ours to free
                self._xr, self._xr_local = None, False
                return k
        else:
            k, self._v = self.kv(self._xr, lead=lead, split=self.kv_split)
        ttnn.deallocate(self._xr)
        self._xr = None
        return k

    def _hm_ok(self, lead):
        """Head-major decode outputs: one token, B = 1, the head size known (BONSAI_DECODE_HM=0: off)."""
        return (
            os.environ.get("BONSAI_DECODE_HM", "1") == "1"
            and self.heads is not None
            and tuple(lead)[:2] == (1, 1)
            and list(lead)[-1] == 1
        )

    def project_v(self, x):
        v, self._v = self._v, None
        return v

    def project_o(self, a):
        if isinstance(a, _GatedOut) and _ROT_FUSE():  # sigmoid(gate) * attn rotated inside the o projection's op
            lead = tuple(list(a.attn.shape)[:-1])
            if list(a.attn.shape)[-1] != self.o.K:  # token-major SDPA output (_TM): one token, as after the concat
                lead = (1, 1, 1)
            y = self.o.rotated(
                self.rot_out, a.gate, lead=lead, u=a.attn, sig=True, residual=prerot.take_residual(), out_l1=True
            )
            ttnn.deallocate(a.attn)
            ttnn.deallocate(a.gate)
            return y
        if isinstance(a, _GatedOut):  # sigmoid(gate) * attn, rotated in one op
            lead = tuple(list(a.attn.shape)[:-1])
            ar = self.rot_out(a.gate, u=a.attn, sig=True)
            ttnn.deallocate(a.attn)
            ttnn.deallocate(a.gate)
        else:
            lead = tuple(list(a.shape)[:-1])
            ar = self.rot_out(a)
        y = self.o(ar, lead=lead, residual=prerot.take_residual())  # the layer's residual add, if registered
        ttnn.deallocate(ar)
        return y


class _HeadsOutTransformer:
    """ttnn.transformer whose concatenate_heads defers (attn_qg.HeadsOut)."""

    def __getattr__(self, name):
        return getattr(ttnn.transformer, name)

    def concatenate_heads(self, x, *args, **kwargs):
        return HeadsOut(x) if not args and not kwargs else ttnn.transformer.concatenate_heads(x, *args, **kwargs)

    def chunked_scaled_dot_product_attention(self, q, k, v, page_table, *args, **kwargs):
        """The prefill's paged SDPA with q chunks of BONSAI_SDPA_Q (256) rows, k chunks of BONSAI_SDPA_K (128) and the
        approximate exp (BONSAI_SDPA_EXP_APPROX=1), fp32 DEST kept: the precise fp32 exp bounds the 128 / 128 default
        (T = 2048: 1.35 -> 0.71 ms at chunk start 0, 3.64 -> 1.67 ms at 2048). Chunk starts are multiples of 2048, so
        both chunk sizes divide them."""
        qc, kc = int(os.environ.get("BONSAI_SDPA_Q", "256")), int(os.environ.get("BONSAI_SDPA_K", "128"))
        T = list(q.shape)[-2]
        pc = kwargs.get("program_config")
        if pc is not None and T % qc == 0 and 2048 % qc == 0 and 2048 % kc == 0:
            kwargs["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=pc.compute_with_storage_grid_size,
                q_chunk_size=qc,
                k_chunk_size=kc,
                exp_approx_mode=os.environ.get("BONSAI_SDPA_EXP_APPROX", "1") == "1",
            )
        return ttnn.transformer.chunked_scaled_dot_product_attention(q, k, v, page_table, *args, **kwargs)


class TernaryAttentionPrefillTtnn:
    """ttnn stand-in for the prefill attention (inside attn_qg.QGSplitTtnn): q, k, v, o in the exact ternary weights.
    q comes out as [q heads | gate heads], as the reordered dense q_proj did."""

    def __init__(self, weights, ternary: TernaryAttention):
        self._w, self._t = weights, ternary
        self._xr = self._v = None

    # concatenate_heads(attn) * sigmoid(gate) -> o_proj: the three ops become markers and the o projection's input
    # rotation computes them (tprefill/rotate.py gate=), BONSAI_PREFILL_GATEROT=0 turns it off
    @property
    def transformer(self):
        if os.environ.get("BONSAI_PREFILL_GATEROT", "1") != "1":
            return ttnn.transformer
        return _HeadsOutTransformer()

    def sigmoid(self, x, *args, **kwargs):
        if os.environ.get("BONSAI_PREFILL_GATEROT", "1") == "1" and not args and not kwargs:
            return SigGate(x)
        return ttnn.sigmoid(x, *args, **kwargs)

    def multiply(self, a, b, *args, **kwargs):
        if isinstance(a, HeadsOut) and isinstance(b, SigGate):
            return GatedAttn(a.t, b.t)
        return ttnn.multiply(a, b, *args, **kwargs)

    @property
    def qkr(self):
        """{id(norm weight): QKNormRope} for the fused q / k norm + RoPE (attn_qg.QGSplitTtnn.apply_rope)."""
        return getattr(self._t, "pre_qkr", None)

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def linear(self, a, b, *args, **kwargs):
        w, t = self._w, self._t
        if b is w.q_proj:
            self._xr = prerot.take(a)  # the layer's fused norm + rotation (tprefill/norm_rot.py), else rotate here
            if self._xr is None:
                self._xr = t.prot_in(a)
            # q heads written head-major, the gate token-major (attn_qg.QGHeads; no reshape / transpose of q)
            q, gate = t.pre_q(self._xr, split=(t.pre_q_tiles, t.pre_hd, 0))
            return QGHeads(q, gate)
        if b is w.k_proj:
            k, v = t.pre_kv(self._xr, split=(t.kv_split // 32, t.pre_hd, t.pre_hd))  # both head-major
            ttnn.deallocate(self._xr)
            self._xr = None
            self._v = HeadMajor(v)
            return HeadMajor(k)
        if b is w.v_proj:
            v, self._v = self._v, None
            return v
        if b is w.o_proj:
            if isinstance(a, GatedAttn):
                ar = t.prot_out(a.attn, gate=a.gate)
                ttnn.deallocate(a.attn)
                ttnn.deallocate(a.gate)
            else:
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


class _NormRope:
    """Decode q or k awaiting its per-head norm, fused with the RoPE (tprefill/qk_rope.QKNormRope) in apply_rope;
    head_major once the shared code's transpose(1, 2) ran on it."""

    def __init__(self, t, op, head_major=False):
        self.t, self.op, self.head_major = t, op, head_major


class _HMDec:
    """A decode projection output written head-major [1, heads, 1, D] (decode_linear3 shapes= / qg_heads=), standing in
    for the shared code's [B, T, heads, D]: its reshape and transpose(1, 2) are no-ops."""

    def __init__(self, t):
        self.t = t


class _QGDec:
    """The decode q projection as (q head-major, gate [1, 1, heads D]): reshape is a no-op, chunk returns the two."""

    def __init__(self, q, gate):
        self.q, self.gate = q, gate


class _KVSh:
    """A token-major K / V decode shard (qk_rope.kv_shards) already in the layout paged_update_cache takes: the
    reshape and to_memory_config before the cache update pass it through."""

    def __init__(self, t):
        self.t = t


class _KVCache:
    """Decode K (before its norm + RoPE) and V for the fused cache write (qk_rope.QKNormRope cache=): the reshapes and
    to_memory_configs before the cache update pass it through; the two paged_update_cache calls (K's, then V's) run it
    once, with both caches (_Experimental)."""

    def __init__(self, k, cos, sin, v):
        self.k, self.cos, self.sin, self.v, self.kc, self.q = k, cos, sin, v, None, None


class _Experimental:
    """ttnn.experimental whose paged_update_cache takes _KVCache (K's call keeps the K cache, V's call runs the op)."""

    def __getattr__(self, name):
        return getattr(ttnn.experimental, name)

    def paged_update_cache(self, cache, x, *args, update_idxs_tensor=None, page_table=None, **kwargs):
        if not isinstance(x, _KVCache):
            return ttnn.experimental.paged_update_cache(
                cache, x, *args, update_idxs_tensor=update_idxs_tensor, page_table=page_table, **kwargs
            )
        assert not args and not kwargs and cache.dtype == ttnn.bfloat8_b, (args, kwargs, cache.dtype)
        if x.kc is None:
            x.kc = cache
            return None
        from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

        kv_l1 = None
        if stream_decode.active() and getattr(stream_decode.STATE, "kv_ready", False):
            p = stream_decode.STATE.kvp  # also the new row into the SDPA's prefetched K / V
            kv_l1 = (p["kvbuf"], p["maxr"], p["NG"], p["cores"])
        cache_args = (x.kc, cache, x.v, update_idxs_tensor, page_table)
        if x.q is not None:  # q's norm + RoPE (deferred by apply_rope) and k's in one op: q on cores 0 .., k after
            qparts, qt, nq = x.q
            kparts = x.k.op(x.k.t, x.cos, x.sin, cache=cache_args, kv_l1=kv_l1, core_offset=nq, parts=True)
            QKNormRope.run_parts(qparts, kparts)
            ttnn.deallocate(qt)
        else:
            x.k.op(x.k.t, x.cos, x.sin, cache=cache_args, kv_l1=kv_l1)
        ttnn.deallocate(x.k.t)
        ttnn.deallocate(x.v)
        return None


def _KV_CACHE_ON():
    return os.environ.get("BONSAI_KV_BF8") == "1" and os.environ.get("BONSAI_KV_CACHE_DIRECT", "1") == "1"


class _TM:
    """The decode SDPA's output, token-major [1, 1, heads, D]: the o projection's fused rotation reads it as is
    (rotate.py ROT_U_TM), so the transpose and concatenate_heads after the SDPA pass it through."""

    def __init__(self, t):
        self.t = t


def _TM_ON():
    return os.environ.get("BONSAI_ATTN_TM", "1") == "1" and _ROT_FUSE()


class _GatedOut:
    """Decode attn * sigmoid(gate), not computed yet (TernaryAttention.project_o rotates it in one op)."""

    def __init__(self, attn, gate):
        self.attn, self.gate = attn, gate


def _ROT_FUSE():
    return os.environ.get("BONSAI_ROT_FUSE", "1") == "1"


class _DecodeTransformer:
    """ttnn.transformer whose paged SDPA decode runs on the whole grid (BONSAI_DECODE_SDPA_GRID=0: the caller's):
    24 q / 4 kv heads, d = 256 at ~4100 positions: 8 x 8 85.3 us, 13 x 10 78.3 us (same chunks, same outputs)."""

    def __getattr__(self, name):
        return getattr(ttnn.transformer, name)

    def concatenate_heads(self, x, *args, **kwargs):
        if isinstance(x, _TM):
            return x
        if _ROT_FUSE() and not args:  # in L1: the o projection's fused rotation reads it during its weight stream
            kwargs.setdefault("memory_config", ttnn.L1_MEMORY_CONFIG)
        return ttnn.transformer.concatenate_heads(x, *args, **kwargs)

    def paged_scaled_dot_product_attention_decode(self, *args, **kwargs):
        pc = kwargs.get("program_config")
        from models.demos.blackhole.bonsai2.tt import core_ranges
        from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

        if os.environ.get("BONSAI_SDPA_DEC", "1") == "1" and _TM_ON() and len(args) == 3:
            out = _decode_sdpa(*args, **kwargs)
            if out is not None:
                return _TM(out)

        if pc is not None and stream_decode.active():  # the stream's receivers only
            rx = stream_decode.STATE.st.rx_order
            kwargs["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(len(rx) // 8, 8),
                sub_core_grids=core_ranges.rects(rx),
                q_chunk_size=pc.q_chunk_size,
                k_chunk_size=pc.k_chunk_size,
                exp_approx_mode=pc.exp_approx_mode,
            )
        elif pc is not None and os.environ.get("BONSAI_DECODE_SDPA_GRID", "1") == "1":
            kwargs["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=kwargs["page_table_tensor"].device().compute_with_storage_grid_size(),
                q_chunk_size=pc.q_chunk_size,
                k_chunk_size=pc.k_chunk_size,
                exp_approx_mode=pc.exp_approx_mode,
            )
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(*args, **kwargs)
        return _TM(out) if _TM_ON() else out


_SDPA = {}


def _decode_sdpa(q, k, v, cur_pos_tensor=None, page_table_tensor=None, scale=None, **kwargs):
    """The decode SDPA as one custom op (tt/sdpa_dec) on the stream's receivers (else the first cores of the grid):
    bf8 caches, 4 kv heads x 6 q heads, head_dim 256; None for any other shape (the stock op then)."""
    from models.demos.blackhole.bonsai2.tt.sdpa_dec.sdpa_dec import DecodeSDPA
    from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

    _, H, bs, D = list(k.shape)
    pf = stream_decode.active() and getattr(stream_decode.STATE, "kv_ready", False)
    Hq = 6 * H if pf else list(q.shape)[-2]  # (pf: q is the per-group hubs, [H * 32, D])
    if k.dtype != ttnn.bfloat8_b or D != 256 or Hq != 6 * H or page_table_tensor is None or cur_pos_tensor is None:
        return None
    device = k.device()
    if pf:
        # K / V prefetched into the SDPA cores' L1 by the q projection (stream/decode.py kv_prefetch)
        stream_decode.STATE.kv_ready = False
        p = stream_decode.STATE.kvp
        key = ("pf", tuple((c.x, c.y) for c in p["cores"]), H, p["maxr"], scale)
        if key not in _SDPA:
            _SDPA[key] = DecodeSDPA(device, p["cores"], H=H, Hq=Hq, D=D, scale=scale, maxr=p["maxr"])
        return _SDPA[key](q, k, v, cur_pos_tensor, page_table_tensor, pf=(p["kvbuf"], p["pinfo"]), q_hubs=p["hubs"])
    if stream_decode.active():
        cores = stream_decode.STATE.st.rx_order
    else:
        g = device.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(g.x * g.y)]
    cores = cores[: (len(cores) // H) * H]
    cap = list(page_table_tensor.shape)[-1] * bs
    key = (tuple((c.x, c.y) for c in cores), H, cap, scale)
    if key not in _SDPA:
        _SDPA[key] = DecodeSDPA(device, cores, H=H, Hq=Hq, D=D, max_positions=cap, scale=scale)
    return _SDPA[key](q, k, v, cur_pos_tensor, page_table_tensor)


class TernaryAttentionTtnn:
    """ttnn stand-in for gated_attention_forward_ttnn's module (decode only): its four projections go to the
    TernaryAttention; the q / k norms run fused with the RoPE (apply_rope, BONSAI_DECODE_QKR=0: off); everything else
    is ttnn."""

    def __init__(self, weights, ternary: TernaryAttention):
        self._w, self._t = weights, ternary

    def __getattr__(self, name):
        return getattr(ttnn, name)

    @property
    def transformer(self):
        return _DecodeTransformer()

    def rms_norm(self, t, *args, **kwargs):
        qkr = getattr(self._t, "pre_qkr", None) or {}
        w = kwargs.get("weight")
        if os.environ.get("BONSAI_DECODE_QKR", "1") == "1" and w is not None and id(w) in qkr and not args:
            if isinstance(t, _HMDec):
                return _NormRope(t.t, qkr[id(w)], head_major=True)
            return _NormRope(t, qkr[id(w)])
        if isinstance(t, _HMDec):
            return _HMDec(ttnn.rms_norm(t.t, *args, **kwargs))
        return ttnn.rms_norm(t, *args, **kwargs)

    def reshape(self, t, shape, *args, **kwargs):
        if isinstance(t, (_HMDec, _QGDec, _KVSh, _KVCache)):
            return t
        vmap = getattr(self, "_vmap", None)
        if vmap is not None and t is vmap[0]:
            self._vmap = None
            if isinstance(vmap[1], _KVCache):
                return vmap[1]  # V is read (and freed) by the cache write
            ttnn.deallocate(t)
            return _KVSh(vmap[1])
        if not args and not kwargs and list(t.shape) == list(shape):
            return t
        return ttnn.reshape(t, shape, *args, **kwargs)

    def chunk(self, t, n, dim=-1):
        if isinstance(t, _QGDec):
            return _HMDec(t.q), t.gate
        return ttnn.chunk(t, n, dim=dim)

    def transpose(self, t, *args, **kwargs):
        if isinstance(t, (_TM, _KVSh)):  # _KVSh: q, already token-major
            return t
        if isinstance(t, _NormRope):
            if t.head_major:
                return t
            y = ttnn.transpose(t.t, *args, **kwargs)  # [B, T, H, D] -> [B, H, T, D]: QKNormRope's head-major input
            ttnn.deallocate(t.t)
            return _NormRope(y, t.op, head_major=True)
        if isinstance(t, _HMDec):
            return t.t  # already [B, H, T, D]
        return ttnn.transpose(t, *args, **kwargs)

    def deallocate(self, t, *args, **kwargs):
        if isinstance(t, _QGDec):
            return None  # its q and gate live on as the chunk outputs
        if isinstance(t, (_NormRope, _HMDec)):
            return ttnn.deallocate(t.t)
        if isinstance(t, (SigGate, _GatedOut, _KVSh, _KVCache)):
            return None  # consumed (and freed) by project_o's gated rotation / persistent K / V shards
        return ttnn.deallocate(t, *args, **kwargs)

    # sigmoid(gate) * attn -> o_proj: the rotation before o computes the gated product (rotate.py sig=)
    def sigmoid(self, x, *args, **kwargs):
        if os.environ.get("BONSAI_DECODE_GATEROT", "1") == "1" and not args and not kwargs:
            return SigGate(x)
        return ttnn.sigmoid(x, *args, **kwargs)

    def multiply(self, a, b, *args, **kwargs):
        if isinstance(b, SigGate) and not args and not kwargs:
            return _GatedOut(a.t if isinstance(a, _TM) else a, b.t)
        return ttnn.multiply(a, b, *args, **kwargs)

    def apply_rope(self, q, k, cos, sin, orig):
        """Stands in for the shared apply_rotary_pos_emb_ttnn (attention/decode.py swaps it in)."""
        if isinstance(q, _NormRope) and q.head_major and isinstance(k, _NormRope) and k.head_major:
            from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

            if stream_decode.active():
                # q token-major in a height-sharded L1 tile row: the decode SDPA's q as is (the stock transpose would
                # run on the stream's sender cores too)
                from models.demos.blackhole.bonsai2.tt.tprefill.qk_rope import kv_shards

                _, Hq, _, Dq = list(q.t.shape)
                p = stream_decode.STATE.kvp
                if p is not None and getattr(stream_decode.STATE, "kv_ready", False):
                    # straight into the decode SDPA's per-group q hubs (sdpa_dec); with the cache write, one op with
                    # k's (run at the cache update: _Experimental)
                    qparts = q.op(q.t, cos, sin, kv_out=(p["qhub"], None, None), q_hub=p["hubs"], parts=True)
                    qo = _KVSh(p["qhub"])
                    v = getattr(self._t, "v_raw", None)
                    if v is not None and _KV_CACHE_ON() and os.environ.get("BONSAI_QK_ONE_OP", "1") == "1":
                        self._t.v_raw = None
                        kv = _KVCache(k, cos, sin, v)
                        kv.q = (qparts, q.t, Hq)
                        self._vmap = (v, kv)
                        return qo, kv
                    QKNormRope.run_parts(qparts)
                else:
                    (q_sh,) = kv_shards(q.t.device(), Hq, Dq, n=1)
                    q.op(q.t, cos, sin, kv_out=(q_sh, None, None))
                    qo = _KVSh(q_sh)
            else:
                qo = q.op(q.t, cos, sin)
            ttnn.deallocate(q.t)
            v = getattr(self._t, "v_raw", None)
            if v is not None and _KV_CACHE_ON():
                # K's norm + RoPE writes K and V straight into the bf8 caches, at the paged_update_cache calls
                self._t.v_raw = None
                kv = _KVCache(k, cos, sin, v)
                self._vmap = (v, kv)
                return qo, kv
            if v is not None and os.environ.get("BONSAI_KV_DIRECT", "1") == "1":
                # K and V straight into the paged cache update's token-major input (no reshape / shard ops)
                from models.demos.blackhole.bonsai2.tt.tprefill.qk_rope import kv_shards

                _, H, _, D = list(k.t.shape)
                k_sh, v_sh = kv_shards(k.t.device(), H, D)
                k.op(k.t, cos, sin, kv_out=(k_sh, v, v_sh))
                ttnn.deallocate(k.t)
                self._t.v_raw = None
                self._vmap = (v, v_sh)
                return qo, _KVSh(k_sh)
            ko = k.op(k.t, cos, sin)
            ttnn.deallocate(k.t)
            return qo, ko
        return orig(q, k, cos, sin)

    @property
    def experimental(self):
        return _Experimental()

    def to_memory_config(self, t, *args, **kwargs):
        if isinstance(t, _KVCache):
            return t
        if isinstance(t, _KVSh):
            return t.t
        return ttnn.to_memory_config(t, *args, **kwargs)

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
            if os.environ.get("BONSAI_PREFILL_QKR", "1") == "1":
                # q / k per-head norm + partial RoPE in one op (tprefill/qk_rope.py); consts built now, not in a trace
                from models.demos.blackhole.bonsai2.tt.tprefill.qk_rope import QKNormRope

                eps = attn.config.norm_eps
                attn.ternary.pre_qkr = {
                    id(nw): QKNormRope(device, ttnn.to_torch(nw).float().reshape(-1)[: attn.config.head_dim], eps)
                    for nw in (w.q_norm, w.k_norm)
                }
