# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 routed experts (reference/ds41_ref.py:_experts / _swiglu; inference MoE + Expert) on the SP x TP
mesh: DeepSeek 2D expert parallelism (deepseek_v3_d_p/tt/moe/README.md), dispatch groups = mesh columns, dispatch
along axis 0 (plan.md "Experts"), the pipeline of xing40_a4b_d_p/tt/experts.py:TtExperts.

Per expert e and token t routed to it (weight r = router[t, e], fp32):

    y_e(t) = w2 q(r * silu(min(w1 q(x), 10)) * clamp(w3 q(x), +-10))       q = fp8 act quant, blocks of 32 (tt/quant)

The routing weight multiplies h BEFORE its fp8 quantization (q(r h) != r q(h)), so the expert math needs each
dispatched row's own fp32 weight. Dispatch carries only the row, so the row carries it:

    x_aug [S/SP, H + 3 Egp] bf16 = [q(x) | p0 | p1 | p2]     q(x) bf16-exact; p0 + p1 + p2 = the fp32 routing weights of
                                                             this column's dispatch group (Eg experts, padded to Egp,
                                                             a multiple of 32), split into three bf16 pieces by
                                                             truncation (exact: 8 + 8 + 8 significant bits)
      -> masked_bincount + ttnn.bringup.offset_cumsum (axis 0) -> ttnn.bringup.dispatch (axis 0, ROW_MAJOR byte copy)
      -> per local expert le (global e = (c SP + r) epc + le, ExpertMapping column-major; j = r epc + le in the group):
           extract [cap, H + 3 Egp] -> x rows | piece columns
           r   = pieces @ Sel_j  (bf16 0/1 selector [3 Egp, 32]: one nonzero product per tile-dot, the three pieces in
                                  different K tiles, summed exactly in fp32 DEST, DS41-16) -> max over 32 equal columns
           g, u = ttnn.linear(x, W1^T / W3^T decoded, HiFi4 + fp32 acc, fp32 out)
           h   = silu(min(g, 10)) * clamp(u, +-10) * r            fp32 (SFPU)
           y   = ttnn.linear(fp8_qdq(h), W2^T decoded) -> bf16    (insert / combine are bf16-only)
           insert
      -> ttnn.bringup.combine (axis 0, init_zeros: the other group's slots read 0) -> [S/SP, K, H] bf16
      -> fp32 sum over the K slots (SFPU adds; weights already applied) = this group's partial [S/SP, H] fp32
      -> sum over the TP axis (the dispatch groups): gather_add (exact fp32, K-30) or reduce_scatter (setting
         EXPERTS_SUM), keep the chip's hidden columns -> experts_out [S/SP, H/TP] fp32

Weights (plan.md "Weights", DS41-10): the checkpoint's e2m1 codes and e8m0 scales, 0.53 B per value, never bfp8 / bfp4.
Per expert matrix W [N, K] the device holds W^T = T [K, N] as 8 nibble planes, P [K/8, N] int32, word (w, n) nibble i
= code(T[i K/8 + w, n]) (a lossless re-packing of the same codes at load), and the scales as bf16 2^e [K/32, N]. The
decode (bit-exact, probed on device against reference/weights.py:fp4_dequant) runs every chunk (setting EXPERTS_IMPL
'loop', components.yaml note: a temporary departure from agent rule 5 until the fork option decodes in the expert op):
    plane i: (P >> 4i) & 15 -> ttnn.gather from the 16-entry e2m1 table (EXPERTS_DECODE 'gather') or the composed
             arithmetic (0.5 c + 0.5 relu(c - 4) + relu(c - 6), sign from bit 3; 'arith')
    T = concat(planes, rows) * (Rep [K, K/32] @ S^T)  (the 0/1 Rep matmul repeats each scale over its 32 rows, exact)

The per-expert cap is the whole chunk (a token picks an expert once; the counts stay on the device), the flat
dispatch buffer K x the cap (all K of a token's experts may sit on one chip). expert counts / region offsets are
[1, E] (global ids, K-2); the dispatched buffer is never freed before combine (K-1).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule

from .common import DRAM, SP_AXIS, TP_AXIS, compute_config, tp_sum
from .quant import fp8_qdq
from .settings import settings

TILE = 32
PLANES = 8  # e2m1 codes per int32 word
BLOCK = 32  # e8m0 scale group along the expert's K
TRUNC_BF16 = -65536  # 0xFFFF0000 as int32: keep the bf16 part of an fp32's bits
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NAMES = ("w1", "w3", "w2")


def _ceil(a: int, b: int) -> int:
    return -(-a // b) * b


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


def _replicate(mesh, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def pack_planes(packed: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A routed expert matrix as stored (packed e2m1 [N, K/2] uint8, low nibble = even column; e8m0 [N, K/32]) ->
    (P [K/8, N] int32: word (w, n) nibble i = code of W[n, i K/8 + w]; S^T [K/32, N] fp32 = 2^e). Lossless."""
    n, half = packed.shape
    k = 2 * half
    assert k % (PLANES * BLOCK) == 0, f"K {k} is not a multiple of {PLANES * BLOCK}"
    b = packed.view(torch.uint8)
    codes = torch.stack([b & 15, b >> 4], dim=-1).reshape(n, k)  # [N, K] uint8, code of W[n, k]
    c = codes.view(n, PLANES, k // PLANES).permute(2, 0, 1)  # [K/8, N, 8]: (w, n, i) = code W[n, i K/8 + w]
    by = (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()  # [K/8, N, 4] bytes, nibble 2j low, 2j + 1 high
    words = by.view(torch.int32).reshape(k // PLANES, n)  # little endian: byte 0 = bits 0..7 = nibbles 0, 1
    st = scale.float().t().contiguous()  # [K/32, N]
    assert bool(torch.isfinite(st).all()) and float(st.min()) >= 2.0**-126, "e8m0 scale outside bf16 normals"
    return words, st


class TtExperts:
    def __init__(
        self,
        mesh,
        ck,
        layer: int,
        num_experts: int,
        hidden: int,
        inter: int,
        top_k: int,
        swiglu_limit: float,
        max_chunk: int,
        num_links: int = 1,
    ):
        """ck: reference/weights.Checkpoint (experts read one at a time, packed). max_chunk: the largest chunk (all
        mesh rows together)."""
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        E = int(num_experts)
        assert E % n == 0, f"{E} experts do not split over {n} chips"
        self.impl = settings.get("EXPERTS_IMPL")
        assert self.impl == "loop", self.impl
        self.decode_mode = settings.get("EXPERTS_DECODE")
        self.sum_mode = settings.get("EXPERTS_SUM")
        self.mesh, self.layer, self.num_links = mesh, layer, num_links
        self.E, self.K, self.H, self.I = E, int(top_k), int(hidden), int(inter)
        self.lim = float(swiglu_limit)
        self.dgs, self.ngroups, self.n = rows, cols, n
        self.epc, self.Eg = E // n, E // cols
        self.Egp = _ceil(self.Eg, TILE)
        self.W = self.H + 3 * self.Egp  # dispatched row: q(x) | three bf16 pieces of the group's routing weights
        assert self.H % TILE == 0 and self.I % (PLANES * BLOCK) == 0 and self.H % (PLANES * BLOCK) == 0
        self.max_chunk = int(max_chunk)
        assert self.max_chunk % (rows * TILE) == 0, f"chunk {max_chunk} is not {rows} x a multiple of {TILE}"
        self.cfg = compute_config(mesh)

        # dispatch table (groups, E + 1) sharded over columns; local -> global expert ids per chip
        self.dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh, ExpertMapping.create_dispatch_table(E, rows, cols), dispatch_axis=SP_AXIS
        )
        gtab = ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=self.epc, dispatch_group_size=rows, num_dispatch_groups=cols
        )  # [cols, rows, epc]
        gidx = ttnn.from_torch(
            gtab, mesh_mapper=get_ep_mesh_mapper(mesh), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, dtype=ttnn.uint32
        )
        self.gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)

        # selectors: local expert le of mesh row r is the group's expert j = r epc + le (the same on every column)
        sel = torch.zeros(rows, self.epc, 3 * self.Egp, TILE)
        for r in range(rows):
            for le in range(self.epc):
                j = r * self.epc + le
                for p in range(3):
                    sel[r, le, p * self.Egp + j, :] = 1.0
        self.sel = [
            ttnn.from_torch(
                sel[:, le : le + 1].contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=_mapper(mesh, (0, None)),
            )
            for le in range(self.epc)
        ]

        # decode constants: the e2m1 table per plane height, the scale-repeat matrices per K
        self.lut, self.rep = {}, {}
        for k in (self.H, self.I):
            h = k // PLANES
            lut = torch.zeros(1, 1, h, TILE)
            lut[..., :16] = torch.tensor(FP4_VALUES)
            self.lut[k] = _replicate(mesh, lut, ttnn.bfloat16)
            kb = _ceil(k // BLOCK, TILE)
            rep = torch.zeros(1, 1, k, kb)
            rep[0, 0, torch.arange(k), torch.arange(k) // BLOCK] = 1.0
            self.rep[k] = _replicate(mesh, rep, ttnn.bfloat16)

        # weights: per local slot le and matrix, one mesh tensor whose chip (r, c) shard is expert (c rows + r) epc + le
        self.planes = [dict() for _ in range(self.epc)]
        self.scales = [dict() for _ in range(self.epc)]
        for le in range(self.epc):
            host = {nm: ([None] * cols, [None] * cols) for nm in NAMES}
            for c in range(cols):
                ps = {nm: [] for nm in NAMES}
                ss = {nm: [] for nm in NAMES}
                for r in range(rows):
                    e = int(gtab[c, r, le])
                    p = ck.expert(layer, e)
                    for nm in NAMES:
                        w, s = pack_planes(*p[nm])
                        kb = _ceil(s.shape[0], TILE)
                        if kb != s.shape[0]:
                            s = torch.cat([s, s.new_zeros(kb - s.shape[0], s.shape[1])])
                        ps[nm].append(w)
                        ss[nm].append(s)
                for nm in NAMES:
                    host[nm][0][c] = torch.stack(ps[nm])  # [rows, K/8, N]
                    host[nm][1][c] = torch.stack(ss[nm])
            for nm in NAMES:
                pw = torch.stack(host[nm][0])  # [cols, rows, K/8, N]
                sw = torch.stack(host[nm][1])
                self.planes[le][nm] = ttnn.from_torch(
                    pw,
                    dtype=ttnn.int32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=DRAM,
                    mesh_mapper=get_ep_mesh_mapper(mesh),
                )
                self.scales[le][nm] = ttnn.from_torch(
                    sw,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=DRAM,
                    mesh_mapper=get_ep_mesh_mapper(mesh),
                )
        self._seq = {}
        self._modules(self.max_chunk // rows)  # the serving chunk's dispatch / combine sizes, built at load

    # ------------------------------------------------------------------ sizes (built once per rows-per-chip)
    def _modules(self, s: int):
        hit = self._seq.get(s)
        if hit is None:
            assert self.dgs * s <= self.max_chunk, f"chunk {self.dgs * s} > {self.max_chunk}"
            epc, metadata_len, max_buf, cap = compute_constants(s, self.E, self.K, self.n, self.dgs, self.K)
            assert epc == self.epc
            dispatch = TtDispatchModule(
                mesh_device=self.mesh,
                dispatch_group_size=self.dgs,
                experts_per_chip=epc,
                num_routed_experts=self.E,
                num_experts_per_tok=self.K,
                metadata_len=metadata_len,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=s,
                emb_dim=self.W,
                cluster_axis=SP_AXIS,
                num_links=self.num_links,
                topology=ttnn.Topology.Linear,
                subdevice_id=None,
            )
            combine = TtCombineModule(
                mesh_device=self.mesh,
                dispatch_group_size=self.dgs,
                num_dispatch_groups=self.ngroups,
                experts_per_chip=epc,
                num_experts_per_tok=self.K,
                seq_len_per_chip=s,
                cluster_axis=SP_AXIS,
                num_links=self.num_links,
                topology=ttnn.Topology.Linear,
                init_zeros=True,  # slots routed to the other groups' experts read as 0
            )
            hit = self._seq[s] = (dispatch, combine, cap)
        return hit

    # ------------------------------------------------------------------ weights
    def _decode(self, le: int, nm: str) -> ttnn.Tensor:
        """W^T [K, N] bf16 (exact) of local expert le's matrix nm, from its nibble planes and scales."""
        P = self.planes[le][nm]
        kp, nn = P.shape[-2], P.shape[-1]
        k = kp * PLANES
        planes = []
        for i in range(PLANES):
            sh = ttnn.bitwise_right_shift(P, 4 * i, memory_config=DRAM) if i else P
            if self.decode_mode == "gather":
                code = ttnn.bitwise_and(sh, 15, memory_config=DRAM)
                cu = ttnn.typecast(code, ttnn.uint32, memory_config=DRAM)
                v = ttnn.gather(self.lut[k], dim=-1, index=cu, memory_config=DRAM)
                for t in (code, cu):
                    ttnn.deallocate(t)
            else:
                c = ttnn.bitwise_and(sh, 7, memory_config=DRAM)
                sg = ttnn.bitwise_and(sh, 8, memory_config=DRAM)
                cf = ttnn.typecast(c, ttnn.float32, memory_config=DRAM)
                a = ttnn.relu(ttnn.add(cf, -4.0, memory_config=DRAM), memory_config=DRAM)
                b = ttnn.relu(ttnn.add(cf, -6.0, memory_config=DRAM), memory_config=DRAM)
                m = ttnn.add(ttnn.multiply(ttnn.add(cf, a, memory_config=DRAM), 0.5, memory_config=DRAM), b)
                sgf = ttnn.typecast(sg, ttnn.float32, memory_config=DRAM)
                sgn = ttnn.add(ttnn.multiply(sgf, -0.25, memory_config=DRAM), 1.0, memory_config=DRAM)
                v = ttnn.multiply(m, sgn, dtype=ttnn.bfloat16, memory_config=DRAM)
                for t in (c, sg, cf, a, b, m, sgf, sgn):
                    ttnn.deallocate(t)
            if i:
                ttnn.deallocate(sh)
            planes.append(v)
        T = ttnn.concat(planes, dim=-2, memory_config=DRAM)
        for t in planes:
            ttnn.deallocate(t)
        sf = ttnn.matmul(
            self.rep[k], self.scales[le][nm], dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, memory_config=DRAM
        )
        out = ttnn.multiply(T, sf, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(T)
        ttnn.deallocate(sf)
        assert (out.shape[-2], out.shape[-1]) == (k, nn)
        return out

    # ------------------------------------------------------------------ forward pieces
    def _pieces(self, dense: ttnn.Tensor, s: int) -> list[ttnn.Tensor]:
        """dense [1, 1, s, E] fp32 (any layout, TP-replicated) -> this column group's weights as three bf16 TILE
        pieces [1, 1, s, Egp] whose fp32 sum is the weight exactly."""
        d = dense if dense.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(dense, ttnn.ROW_MAJOR_LAYOUT)
        g = ttnn.mesh_partition(d, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM) if self.ngroups > 1 else d
        if self.Egp != self.Eg:
            gp = ttnn.pad(g, [(0, 0), (0, 0), (0, 0), (0, self.Egp - self.Eg)], 0.0)
            if g is not d:
                ttnn.deallocate(g)
            g = gp
        w = ttnn.to_layout(g, ttnn.TILE_LAYOUT, memory_config=DRAM)
        if g is not d and g is not w:
            ttnn.deallocate(g)
        if d is not dense:
            ttnn.deallocate(d)
        out, rest = [], w
        for p in range(3):
            if p < 2:
                bits = ttnn.bitcast(ttnn.bitcast(rest, ttnn.uint32), ttnn.int32)
                hb = ttnn.bitwise_and(bits, TRUNC_BF16, memory_config=DRAM)
                hi = ttnn.bitcast(ttnn.bitcast(hb, ttnn.uint32), ttnn.float32)
                nxt = ttnn.subtract(rest, hi, dtype=ttnn.float32, memory_config=DRAM)
            else:
                hi, nxt = rest, None
            out.append(ttnn.typecast(hi, ttnn.bfloat16, memory_config=DRAM))
            rest = nxt
        return out

    def _weight_col(self, wp: ttnn.Tensor, le: int) -> ttnn.Tensor:
        """Pieces [cap, 3 Egp] bf16 -> the row's fp32 routing weight for local expert le, [cap, 1]."""
        wc = ttnn.matmul(wp, self.sel[le], dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=DRAM)
        r = ttnn.max(wc, dim=-1, keepdim=True, memory_config=DRAM)
        ttnn.deallocate(wc)
        return r

    def _ffn(self, xt: ttnn.Tensor, r: ttnn.Tensor, le: int) -> ttnn.Tensor:
        """One expert over its extracted rows: xt [cap, H] bf16 (q(x)), r [cap, 1] fp32 -> [cap, H] bf16."""
        cfg, lim = self.cfg, self.lim
        w1 = self._decode(le, "w1")
        g = ttnn.linear(xt, w1, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM)
        ttnn.deallocate(w1)
        w3 = self._decode(le, "w3")
        u = ttnn.linear(xt, w3, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM)
        ttnn.deallocate(w3)
        if lim > 0:
            g2 = ttnn.minimum(g, lim, memory_config=DRAM)
            u2 = ttnn.clamp(u, -lim, lim, memory_config=DRAM)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            g, u = g2, u2
        sg = ttnn.silu(g, memory_config=DRAM)
        ttnn.deallocate(g)
        h = ttnn.multiply(sg, u, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(sg)
        ttnn.deallocate(u)
        hr = ttnn.multiply(h, r, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(h)
        hq = fp8_qdq(hr, dtype=ttnn.bfloat16)
        ttnn.deallocate(hr)
        w2 = self._decode(le, "w2")
        y = ttnn.linear(hq, w2, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM)
        ttnn.deallocate(w2)
        ttnn.deallocate(hq)
        yb = ttnn.typecast(y, ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(y)
        return ttnn.reshape(yb, (yb.shape[-2], yb.shape[-1]))  # insert takes a 2D slab

    def _loop(self, buf: ttnn.Tensor, counts, region_offsets, cap: int) -> ttnn.Tensor:
        """buf [max_buf, W] bf16 TILE -> expert outputs [max_buf, H] bf16 TILE (each expert's rows at its region)."""
        H = self.H
        out = ttnn.slice(buf, [0, 0], [buf.shape[0], H], memory_config=DRAM)  # bf16 TILE output slab
        for le in range(self.epc):
            tok = ttnn.experimental.deepseek_prefill.extract(
                buf, region_offsets, counts, self.gidx, local_expert_id=le, max_dispatched_tokens_per_expert=cap
            )
            xt = ttnn.slice(tok, [0, 0], [cap, H], memory_config=DRAM)
            wp = ttnn.slice(tok, [0, H], [cap, self.W], memory_config=DRAM)
            ttnn.deallocate(tok)
            r = self._weight_col(wp, le)
            ttnn.deallocate(wp)
            y = self._ffn(xt, r, le)
            ttnn.deallocate(xt)
            ttnn.deallocate(r)
            out = ttnn.experimental.deepseek_prefill.insert(
                out, y, region_offsets, counts, self.gidx, local_expert_id=le
            )
            ttnn.deallocate(y)
        return out

    def partial(self, x: ttnn.Tensor, dense: ttnn.Tensor, idx: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """This column group's experts over the chip's rows. x: ffn_norm [1, 1, s, H] fp32 TILE (rows split over axis
        0, replicated over axis 1); dense: routing [1, 1, s, E] fp32 (same placement); idx: the router's top-k ids
        [1, 1, s, K] (None: top-k of dense). Returns [1, 1, s, H] fp32 TILE (the group's sum, weights applied)."""
        s, K, H = x.shape[-2], self.K, self.H
        dispatch, combine, cap = self._modules(s)
        tmp = []
        if idx is None:
            dk = dense if dense.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(dense, ttnn.TILE_LAYOUT)
            _, idx = ttnn.topk(dk, k=K, dim=-1, largest=True, sorted=True)
            tmp.append(idx)
            if dk is not dense:
                ttnn.deallocate(dk)
        idx16 = idx if idx.dtype == ttnn.uint16 else ttnn.typecast(idx, ttnn.uint16)
        if idx16 is not idx:
            tmp.append(idx16)
        idx2 = ttnn.reshape(idx16, (s, K))
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(idx2, self.dispatch_table, self.E, K)
        offsets, counts, region_offsets, _ = ttnn.bringup.offset_cumsum(
            hist, cluster_axis=SP_AXIS, num_links=self.num_links, experts_per_chip=self.epc, memory_config=DRAM
        )
        ind = ttnn.reshape(ttnn.to_layout(idx2, ttnn.ROW_MAJOR_LAYOUT), (1, s, K))
        # the dispatched row: q(x) bf16 (exact) | the group's routing weights as three bf16 pieces
        qx = fp8_qdq(x, dtype=ttnn.bfloat16)
        pieces = self._pieces(dense, s)
        xa = ttnn.concat([qx] + pieces, dim=-1, memory_config=DRAM)
        for t in [qx] + pieces:
            ttnn.deallocate(t)
        xr = ttnn.to_layout(xa, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(xa)
        buf, meta = _dispatch(dispatch, ttnn.reshape(xr, (1, s, self.W)), ind, offsets, self.dispatch_table)
        ttnn.deallocate(xr)
        b2 = ttnn.reshape(buf, (buf.shape[-2], self.W))
        bt = ttnn.to_layout(b2, ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(buf)
        out = self._loop(bt, counts, region_offsets, cap)
        ttnn.deallocate(bt)
        out4 = ttnn.reshape(out, (1, 1, out.shape[-2], H))
        comb = _combine(combine, out4, meta, counts, region_offsets, s)  # [1, 1, s, K, H] bf16 ROW_MAJOR
        ttnn.deallocate(out)
        for t in [hist, offsets, counts, region_offsets, meta] + tmp:
            ttnn.deallocate(t)
        # fp32 sum over the K slots (the other groups' slots are zeros), on the SFPU
        ck = ttnn.permute(ttnn.reshape(comb, (1, s, K, H)), (0, 2, 1, 3), memory_config=DRAM)  # [1, K, s, H]
        ttnn.deallocate(comb)
        ct = ttnn.to_layout(ck, ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(ck)
        cf = ttnn.typecast(ct, ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(ct)
        acc = ttnn.slice(cf, [0, 0, 0, 0], [1, 1, s, H], memory_config=DRAM)
        for k in range(1, K):
            part = ttnn.slice(cf, [0, k, 0, 0], [1, k + 1, s, H], memory_config=DRAM)
            nxt = ttnn.add(acc, part, dtype=ttnn.float32, memory_config=DRAM)
            ttnn.deallocate(acc)
            ttnn.deallocate(part)
            acc = nxt
        ttnn.deallocate(cf)
        return acc

    def reduce(self, part: ttnn.Tensor) -> ttnn.Tensor:
        """Sum the column groups' partials [1, 1, s, H] fp32 over axis 1 and keep the chip's hidden columns
        [1, 1, s, H/TP] (gather_add: exact fp32 on the SFPU, K-30; reduce_scatter: one CCL, FPU)."""
        if self.ngroups == 1:
            return ttnn.clone(part, memory_config=DRAM)
        if self.sum_mode == "reduce_scatter":
            return ttnn.reduce_scatter(part, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        full = tp_sum(self.mesh, part, how="gather_add")
        out = ttnn.mesh_partition(full, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        ttnn.deallocate(full)
        return out

    def __call__(self, x: ttnn.Tensor, dense: ttnn.Tensor, idx: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """experts_out [1, 1, s, H/TP] fp32 (rows over axis 0, hidden columns over axis 1)."""
        part = self.partial(x, dense, idx)
        out = self.reduce(part)
        ttnn.deallocate(part)
        return out


def _dispatch(m, x, indices, offsets, table):
    """TtDispatchModule.forward on the bring-up fork ttnn.bringup.dispatch. Returns (buffer, metadata)."""
    return ttnn.bringup.dispatch(
        input_tensor=x,
        indices_tensor=indices,
        expert_offsets_tensor=offsets,
        expert_dispatch_table_tensor=table,
        dispatch_group_size=m.dispatch_group_size,
        experts_per_chip=m.experts_per_chip,
        num_routed_experts=m.num_routed_experts,
        num_experts_per_tok=m.num_experts_per_tok,
        metadata_len=m.metadata_len,
        max_dispatch_buffer_token_size=m.max_dispatch_buffer_token_size,
        cluster_axis=m.cluster_axis,
        num_links=m.num_links,
        topology=m.topology,
        fp8_output=m.fp8_output,
        subdevice_id=m.subdevice_id,
        num_workers_per_sender=m.num_workers_per_sender,
    )


def _combine(m, buf, meta, counts, region_offsets, seq_len):
    """TtCombineModule.forward on the bring-up fork ttnn.bringup.combine."""
    return ttnn.bringup.combine(
        buf,
        meta,
        counts,
        region_offsets,
        dispatch_group_size=m.dispatch_group_size,
        experts_per_chip=m.experts_per_chip,
        num_experts_per_tok=m.num_experts_per_tok,
        seq_len_per_chip=seq_len,
        cluster_axis=m.cluster_axis,
        num_links=m.num_links,
        topology=m.topology,
        memory_config=m.memory_config,
        init_zeros=m.init_zeros,
        use_fp8_combine=m.fp8_output,
    )


def build_experts(mesh, ck, cfg, layer: int, chunk: int) -> TtExperts:
    """The routed experts of one layer (reference Config: n_experts, topk_experts, dim, inter, swiglu_limit)."""
    return TtExperts(
        mesh,
        ck,
        layer,
        num_experts=cfg.n_experts,
        hidden=cfg.dim,
        inter=cfg.inter,
        top_k=cfg.topk_experts,
        swiglu_limit=cfg.swiglu_limit,
        max_chunk=chunk,
    )
