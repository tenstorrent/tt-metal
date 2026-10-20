# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 shared expert (reference/ds41_ref.py "shared_expert": _swiglu with lin=mm and no routing weight;
inference Expert.forward) on the SP x TP mesh, tensor parallel by the intermediate (plan.md "Shared expert"). Adapted
from models/demos/xing40_a4b_d_p/tt/mlp.py:TtDenseMLP with the model's fp8 act quant and clamps.

    shared_out = w2 q(silu(min(w1 q(x), 10)) * clamp(w3 q(x), +-10))      q = fp8 act quant, blocks of 32 (tt/quant)

Per chip (r, c), I/TP = 2304 / TP intermediate columns (1152 on 2x2, 576 on 4x4 / 8x4 / 2x4: multiples of 32, so the
fp8 blocks of h never straddle chips):

    x       ffn_norm [1, 1, S/SP, H] fp32 TILE (rows over axis 0, replicated over axis 1: tt/norm.py gather=True)
    q(x)    tt/quant.fp8_qdq, bf16 out (bit-exact; may be passed in when the experts already made it)
    g | u   one ttnn.linear by [W1^T_c | W3^T_c] [H, 2 I/TP] (column-parallel; bf16 weights, exact: e4m3 x 2^k),
            HiFi4 + fp32 accumulation, fp32 out (K-25)
    h       silu(min(g, 10)) * clamp(u, +-10), fp32 on the SFPU
    q(h)    fp8_qdq over the chip's own I/TP columns (the reference's blocks of 32 along I)
    partial ttnn.linear by W2^T_c [I/TP, H] (row-parallel), fp32 out: [1, 1, S/SP, H] fp32 (DS41-25: keep fp32)
    sum     over the TP axis, keep the chip's hidden columns (setting EXPERTS_SUM: gather_add = all_gather + SFPU add,
            exact fp32, K-30; reduce_scatter = one CCL). In the assembled model the partial is added to the routed
            experts' partial before that one exchange (``partial`` + TtExperts.reduce).

Output shared_out [1, 1, S/SP, H/TP] fp32 TILE (rows over axis 0, hidden columns over axis 1). No host work in the
forward; weights are laid out once at load.

SHARED_EXPERT_IMPL=flat (P.12, Marko Staletovic's K3_SE=flat, combine_rf/examples/tt_k3_token_split_moe.py
_build_shared_flat): the same per-chip slice run by ONE ttnn.bringup.flat_routed_expert call, the slice cut into
SE_PARTS always-on "experts" of I/TP / SE_PARTS columns that every own token goes to (counts [S/SP] each, regions
p S/SP; each part zero-padded to a multiple of 256 columns, the op's down-projection K block: 2x2 1152 -> 1280);
the weights (W1 | W3 | W2 of the chip's columns, as stored [N, K], weights_nk) laid out once at load in the
op's bank layout as bfloat8_b (the checkpoint's fp8 width, agent rule 7; bfp8 rounds e4m3 values more than 3 binades
below their 16-value group's largest, measured in tt/perf_shared_expert.py), the clamped SwiGLU in the op (limit 10),
the routed experts' precision (EXPERTS_PRECISION: glm = LoFi + bfp8 x / h tiles, full = MATMUL_FIDELITY + bf16
tiles), fp32 DEST, bf16 y [S/SP, H] TILE = the partial (the parts' sum when SE_PARTS > 1), in ccl.partial_dtype for
moe_add. Not hidden: one more program after the routed experts' (folding it into their program needs one weight
dtype for both: routed bfp4 vs shared bfp8; overlapping it with the dispatch needs sub-devices), but its x costs
nothing (the routed experts' gathered rows). P.12, 2x2, chunk 2048, layers 0 / 24 (tt/perf_shared_expert.py): device
1431 -> 250 us a layer; shared_out vs float64 (exact fp8 weights) rel 0.0030 linear, 0.020-0.022 flat (bias +1.3-1.4 %:
the bfp8 h / x tiles' packer rounding at LoFi, as the routed experts' glm path, DS41-74). x: with ``rows`` (EXPERTS_COMBINE=weighted hands over its row-major gathered q(x) of the column,
[R S/SP, H]) the op reads the chip's own rows by a load-time token index (no layout conversion here); without, the
chip's q(x) is converted to row major once. NUMERICS=official: q(x) as above, q(h) by the op's fp8_act_quant with unit
routing weights (that option reads a dispatch buffer, no token index).
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn

from . import ccl
from .common import DRAM, TP_AXIS, compute_config, sum_cols
from .quant import GROUP, act_q, native
from .settings import settings

TILE = 32
# SHARED_EXPERT_IMPL=flat: always-on experts the chip's intermediate slice is cut into (K3_SE_PARTS). 1 = the whole
# slice (2x2: 1152 columns, 36 tiles; 4x4 / 8x4 / 2x4: 576) as one expert: the op plans it (the gate/up split takes
# the 2304-wide routed experts at H 5120 too), y is the partial itself (no parts' sum) and the official path needs no
# repeated x. Measured in tt/perf_shared_expert.py (P.12).
SE_PARTS = 1
# the op's down projection runs K in blocks of 8 tiles (se6_dcompute.cpp: static_assert kt_d % 8 == 0), so an expert's
# intermediate is a multiple of 256: each part is zero-padded to it (2x2: 1152 -> 1280, +11 % work; 4x4 / 8x4 / 2x4:
# 576 -> 768, +33 %). Zero W1 / W3 rows give h = silu(min(0, 10)) * clamp(0) = 0 there, and zero W2 columns add nothing.
DOWN_K = 256
_DONE: dict = {}  # (mesh, down coordinators) -> the flat op's done words, shared by every layer's shared expert


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


class TtSharedExpert:
    def __init__(
        self,
        mesh,
        w1: torch.Tensor,
        w3: torch.Tensor,
        w2: torch.Tensor,
        swiglu_limit: float,
        chunk: int | None = None,
        impl: str | None = None,
        flat_opts: dict | None = None,
    ):
        """w1, w3 [I, H], w2 [H, I] (nn.Linear layout), fp32 values on the fp8 grid (bf16-exact). chunk: the chunk
        (all mesh rows; SHARED_EXPERT_IMPL=flat plans for chunk / SP rows a chip). impl: SHARED_EXPERT_IMPL value
        (None: the setting)."""
        tp = mesh.shape[TP_AXIS]
        inter, hidden = w1.shape
        assert w3.shape == (inter, hidden) and w2.shape == (hidden, inter), (w1.shape, w3.shape, w2.shape)
        assert inter % (TILE * tp) == 0 and (inter // tp) % GROUP == 0, f"intermediate {inter} over TP {tp}"
        assert hidden % (TILE * tp) == 0, f"hidden {hidden} over TP {tp}"
        for w in (w1, w3, w2):
            assert torch.equal(w.bfloat16().float(), w.float()), "fp8 weights must be bf16-exact"
        self.mesh, self.tp, self.H, self.I = mesh, tp, int(hidden), int(inter)
        self.ip = self.I // tp  # the chip's intermediate share
        self.lim = float(swiglu_limit)
        self.sum_mode = settings.get("EXPERTS_SUM")
        self.cfg = compute_config(mesh)
        self.impl = impl or settings.get("SHARED_EXPERT_IMPL")
        assert self.impl in ("linear", "flat"), self.impl
        self.reads_rows = False  # flat + native: takes the routed experts' gathered row-major q(x) (``rows``)
        if self.impl == "flat":
            assert chunk is not None, "SHARED_EXPERT_IMPL=flat needs the chunk (rows a chip)"
            self._load_flat(w1, w3, w2, int(chunk), flat_opts or {})
            return
        # [W1^T | W3^T] chip-major along the last dim: chip column c gets [W1^T[:, c ip:(c+1) ip] | W3^T[same]]
        w1t = w1.float().t().reshape(hidden, tp, self.ip)
        w3t = w3.float().t().reshape(hidden, tp, self.ip)
        w13 = torch.stack([w1t, w3t], dim=2).reshape(1, 1, hidden, 2 * self.I)
        self.w13 = ttnn.from_torch(
            w13.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=_mapper(mesh, (None, 3)),  # columns over axis 1, replicated over axis 0
        )
        self.w2 = ttnn.from_torch(
            w2.float().t().reshape(1, 1, inter, hidden).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=_mapper(mesh, (None, 2)),  # rows (the chip's intermediate) over axis 1
        )

    # ------------------------------------------------------------------ SHARED_EXPERT_IMPL=flat
    def _load_flat(self, w1, w3, w2, chunk: int, opts: dict):
        """The chip's intermediate slice as SE_PARTS always-on experts of the flat routed expert (bfp8 weights in its
        bank layout, built once), the load-time token index / counts / regions (and the official path's unit routing
        weights)."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        mesh, tp = self.mesh, self.tp
        R, C = tuple(mesh.shape)
        assert TP_AXIS == 1 and C == tp
        assert self.lim == 10.0, f"SHARED_EXPERT_IMPL=flat: swiglu_limit {self.lim} != the op's clamped_silu 10"
        assert chunk % (R * TILE) == 0, f"chunk {chunk} is not {R} x a multiple of {TILE}"
        P = SE_PARTS
        assert self.ip % (P * TILE) == 0 and (self.ip // P) % GROUP == 0, f"{self.ip} columns in {P} parts"
        self.S, self.P, self.ipart = chunk // R, P, self.ip // P
        self.ipad = -(-self.ipart // DOWN_K) * DOWN_K  # the part's width in the op (zero-padded)
        self.qdq = not native()
        self.precision = settings.get("EXPERTS_PRECISION")
        assert self.precision in ("full", "glm"), self.precision
        full = self.precision == "full"
        self.fidelity = int(
            (ttnn.MathFidelity.LoFi if not full else getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY"))).value
        )
        self.down_fp32 = True if not full else bool(settings.get("FP32_DEST_ACC"))
        x_bf16 = h_bf16 = full
        self.stochastic = False
        # (comparison only, tt/perf_shared_expert.py: other tile formats / fidelity / packer rounding than the setting's)
        x_bf16, h_bf16 = opts.get("x_bf16", x_bf16), opts.get("h_bf16", h_bf16)
        self.fidelity = int(opts.get("fidelity", self.fidelity))
        self.stochastic = bool(opts.get("stochastic", False))
        ip, q, qp = self.ip, self.ipart, self.ipad

        def pad(t, dim):  # (zero columns / rows up to the op's width)
            if qp == q:
                return t.float().contiguous()
            shape = list(t.shape)
            shape[dim] = qp - q
            return torch.cat([t.float(), torch.zeros(shape)], dim=dim).contiguous()

        def chip(c):  # mesh column c's slice as P experts (W1 [qp, H], W3 [qp, H], W2 [H, qp]: the stored [N, K])
            return [
                (
                    pad(w1[c * ip + p * q : c * ip + (p + 1) * q], 0),
                    pad(w3[c * ip + p * q : c * ip + (p + 1) * q], 0),
                    pad(w2[:, c * ip + p * q : c * ip + (p + 1) * q], 1),
                )
                for p in range(P)
            ]

        cols = [chip(c) for c in range(C)]
        self.fe = FlatRoutedExpert(
            mesh,
            [cols[d % C] for d in range(R * C)],  # (device r C + c holds column c's slice)
            m=max(256, self.S),
            H=self.H,
            I=qp,
            gids=[list(range(P)) for _ in range(R * C)],
            n_global=P,
            wdtype="bf8",
            act="clamped_silu",
            pin=1,
            x_bf16=x_bf16,
            h_bf16=h_bf16,
            weights_nk=True,
            routed=self.qdq,  # (official: the h options' plan for fp8_act_quant)
            check_exact=False,  # (bfp8 is the chosen format: it may round fp8 values, P.12)
        )
        lay = dict(self.fe.plan)
        key = (id(mesh), tuple(map(tuple, lay["coords"])))
        if key in _DONE:  # (zero words the down coordinators leave zeroed: one tensor for every layer)
            ttnn.deallocate(self.fe.done)
            self.fe.done = _DONE[key]
        else:
            _DONE[key] = self.fe.done
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def u32(t, mapper):
            return ttnn.from_torch(
                t.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=mapper,
            )

        S = self.S
        self.counts = u32(torch.tensor([[S] * P]), rep)
        self.regions = u32(torch.tensor([[p * S for p in range(P)]]), rep)
        # own x rows by token index: in the column's gathered rows [R S, H] (``rows``) mesh row r's are r S + t
        tok = torch.arange(S).repeat(P)
        self.tidx_rows = u32(
            torch.stack([r * S + tok for r in range(R)]), ttnn.ShardTensor2dMesh(mesh, (R, C), dims=(0, None))
        )
        self.tidx_own = u32(tok.reshape(1, -1), rep) if P > 1 else None
        self.reads_rows = not self.qdq
        self.ones = None
        if self.qdq:  # (official: h = q(1 * act): the op's fp8_act_quant reads the weight column of each part)
            ones = torch.zeros(P * S, TILE)
            for p in range(P):
                ones[p * S : (p + 1) * S, p] = 1.0
            self.ones = ttnn.from_torch(
                ones, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=rep
            )
        logger.info(
            f"SHARED_FLAT plan nu={lay['nu']} np={lay['np']} mt={lay['mt']} gate/up={len(lay['gu'])} down={lay['nd']} "
            f"parts={P} x {q} (padded to {qp}) rows={S} precision={self.precision} fidelity={self.fidelity} official={self.qdq}"
        )

    def _flat_partial(self, x: ttnn.Tensor, xq: ttnn.Tensor | None, rows: ttnn.Tensor | None) -> ttnn.Tensor:
        S, H, P = self.S, self.H, self.P
        assert x.shape[-2] == S, f"SHARED_EXPERT_IMPL=flat is built for {S} rows a chip, got {x.shape[-2]}"
        kw = dict(
            y_bf16=True,
            down_fp32=self.down_fp32,
            math_fidelity=self.fidelity,
            pack_stochastic_rounding=self.stochastic,
        )
        if rows is not None and not self.qdq:  # (the routed experts' gathered row-major q(x): own rows by index)
            y = self.fe(rows, self.counts, self.regions, token_index=self.tidx_rows, **kw)
        else:
            own = xq is None
            xq = act_q(x, dtype=ttnn.bfloat16) if own else xq
            xr = ttnn.to_layout(xq, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            if own and xq is not x:
                ttnn.deallocate(xq)
            xr2 = ttnn.reshape(xr, [S, H])
            if self.qdq:  # (a dispatch buffer: each part's rows at its region)
                xb = xr2 if P == 1 else ttnn.concat([xr2] * P, dim=0, memory_config=DRAM)
                y = self.fe(xb, self.counts, self.regions, routing_weights=self.ones, fp8_act_quant=True, **kw)
                if xb is not xr2:
                    ttnn.deallocate(xb)
            elif P == 1:
                y = self.fe(xr2, self.counts, self.regions, **kw)
            else:
                y = self.fe(xr2, self.counts, self.regions, token_index=self.tidx_own, **kw)
            ttnn.deallocate(xr)
        if rows is not None:
            ttnn.deallocate(rows)
        if P > 1:  # (the parts' outputs: disjoint intermediate slices of one MLP, summed in fp32)
            acc = ttnn.slice(y, [0, 0], [S, H], memory_config=DRAM)
            acc = ttnn.typecast(acc, ttnn.float32, memory_config=DRAM)
            for p in range(1, P):
                part = ttnn.slice(y, [p * S, 0], [(p + 1) * S, H], memory_config=DRAM)
                nxt = ttnn.add(acc, part, dtype=ttnn.float32, memory_config=DRAM)
                ttnn.deallocate(acc)
                ttnn.deallocate(part)
                acc = nxt
            ttnn.deallocate(y)
            y = acc
        out = ttnn.reshape(y, [1, 1, S, H])
        want = ccl.partial_dtype(self.mesh) if self.sum_mode == "gather_add" else ttnn.float32
        if out.dtype != want:  # (bf16 y; fp32 only off the shipped CCL_IMPL=fabric_bf16 path)
            o2 = ttnn.typecast(out, want, memory_config=DRAM)
            ttnn.deallocate(out)
            out = o2
        return out

    def partial(self, x: ttnn.Tensor, xq: ttnn.Tensor | None = None, rows: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """This chip's share of the output: x [1, 1, s, H] fp32 TILE (or its fp8 q-dq ``xq`` bf16, then x is unused)
        -> [1, 1, s, H] fp32 TILE partial (sum over the TP axis = shared_out). SHARED_EXPERT_IMPL=flat: ``rows`` =
        the routed experts' gathered row-major q(x) [R s, H] (consumed), the partial in ccl.partial_dtype."""
        if self.impl == "flat":
            return self._flat_partial(x, xq, rows)
        assert rows is None
        cfg, lim, ip = self.cfg, self.lim, self.ip
        own = xq is None
        if own:
            xq = act_q(x, dtype=ttnn.bfloat16)  # NUMERICS=native: a cast (x itself when ffn_norm wrote bf16)
        gu = ttnn.linear(xq, self.w13, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM)
        if own and xq is not x:
            ttnn.deallocate(xq)
        s = gu.shape[-2]
        g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, s, ip], memory_config=DRAM)
        u = ttnn.slice(gu, [0, 0, 0, ip], [1, 1, s, 2 * ip], memory_config=DRAM)
        ttnn.deallocate(gu)
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
        hq = act_q(h, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        y = ttnn.linear(hq, self.w2, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM)
        ttnn.deallocate(hq)
        return y

    def reduce(self, part: ttnn.Tensor) -> ttnn.Tensor:
        """Sum partials [1, 1, s, H] fp32 over axis 1, keep the chip's hidden columns [1, 1, s, H/TP]."""
        if self.tp == 1:
            return ttnn.clone(part, memory_config=DRAM)
        if part.dtype != ttnn.float32 and not (self.sum_mode == "gather_add" and ccl.bf16_sums()):
            pf = ttnn.typecast(part, ttnn.float32, memory_config=DRAM)
            out = self.reduce(pf)
            ttnn.deallocate(pf)
            return out
        if self.sum_mode == "reduce_scatter":
            return ttnn.reduce_scatter(part, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        return sum_cols(self.mesh, part)

    def __call__(self, x: ttnn.Tensor, xq: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """shared_out [1, 1, s, H/TP] fp32 (rows over axis 0, hidden columns over axis 1)."""
        part = self.partial(x, xq)
        out = self.reduce(part)
        ttnn.deallocate(part)
        return out


def build_shared_expert(
    mesh, ck, cfg, layer: int, chunk: int | None = None, impl: str | None = None, flat_opts: dict | None = None
) -> TtSharedExpert:
    """Layer ``layer``'s shared expert (fp8 weights dequantized at load, exact; reference Config.swiglu_limit).
    chunk: the chunk (all mesh rows; SHARED_EXPERT_IMPL=flat needs it); impl: SHARED_EXPERT_IMPL (None: setting)."""
    p = f"layers.{layer}.ffn.shared_experts."
    w = {n: ck.weight(p + n + ".weight", torch.float32) for n in ("w1", "w3", "w2")}
    return TtSharedExpert(
        mesh, w["w1"], w["w3"], w["w2"], cfg.swiglu_limit, chunk=chunk, impl=impl, flat_opts=flat_opts
    )
