# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 shared expert (reference/ds41_ref.py "shared_expert": _swiglu with lin=mm and no routing weight;
inference Expert.forward) on the SP x TP mesh, tensor parallel by the intermediate (plan.md "Shared expert").

    shared_out = w2 q(silu(min(w1 q(x), 10)) * clamp(w3 q(x), +-10))      q = fp8 act quant, blocks of 32 (released code)

The device does not emulate q (agent rule 17): x and h are the tiles the flat op multiplies. Per chip (r, c), I/TP =
2304 / TP intermediate columns (1152 on 2x2, 576 on 4x4 / 8x4 / 2x4), run by ONE ttnn.bringup.flat_routed_expert call
(P.12, Marko Staletovic's K3_SE=flat, combine_rf/examples/tt_k3_token_split_moe.py _build_shared_flat; FIX.9 deleted
the ttnn.linear chain): the slice cut into SE_PARTS always-on "experts" of I/TP / SE_PARTS columns that every own token
goes to (counts [S/SP] each, regions p S/SP; each part at its real width, no padded lanes: the op plans 18 / 36 tile
columns and a down K of any tile count since FIX.1, DS41-76 / DS41-115); the weights (W1 | W3 | W2 of the chip's columns, as stored [N, K],
weights_nk) laid out once at load in the op's bank layout as bfloat8_b (the checkpoint's fp8 width, agent rule 7; bfp8
rounds e4m3 values more than 3 binades below their 16-value group's largest, measured in tt/perf_shared_expert.py),
the clamped SwiGLU in the op (limit 10), the routed experts' precision (EXPERTS_PRECISION: glm = LoFi + bfp8 x / h
tiles, full = MATMUL_FIDELITY + bf16 tiles), fp32 DEST, bf16 y [S/SP, H] TILE = the partial (the parts' sum when
SE_PARTS > 1), in ccl.partial_dtype for moe_add.

    x       ffn_norm [1, 1, S/SP, H] (rows over axis 0, replicated over axis 1: tt/norm.py gather=True); with ``rows``
            (the routed experts' row-major gathered q(x) of the column, [R S/SP, H]) the op reads the chip's own rows
            by a load-time token index (no layout conversion here); without, the chip's x is converted to row major
    partial [1, 1, S/SP, H] (this chip's intermediate share)
    sum     over the TP axis, keep the chip's hidden columns (setting EXPERTS_SUM). In the assembled model the partial
            is added to the routed experts' partial before that one exchange (``partial`` + TtExperts.reduce).

Not hidden: one more program after the routed experts' (folding it into their program needs one weight dtype for
both: routed bfp4 vs shared bfp8; overlapping it with the dispatch needs sub-devices), but its x costs nothing (the
routed experts' gathered rows). P.12, 2x2, chunk 2048, layers 0 / 24 (tt/perf_shared_expert.py): device 1431 -> 250
us a layer vs the deleted linear chain; shared_out vs float64 (exact fp8 weights) rel 0.020-0.022 (bias +1.3-1.4 %:
the bfp8 h / x tiles' packer rounding at LoFi, as the routed experts' glm path, DS41-74 / DS41-77). Output shared_out
[1, 1, S/SP, H/TP] (rows over axis 0, hidden columns over axis 1). No host work in the forward.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn

from . import ccl
from .ccl import dealloc
from .common import DRAM, TP_AXIS, cast, fp32_acc, sum_cols
from .cores import from_torch, sub_cores, sub_cores_tiled, to_rm
from .settings import settings

TILE = 32
# always-on experts the chip's intermediate slice is cut into (K3_SE_PARTS). 1 = the whole
# slice (2x2: 1152 columns, 36 tiles; 4x4 / 8x4 / 2x4: 576) as one expert: the op plans it (the gate/up split takes
# the 2304-wide routed experts at H 5120 too), y is the partial itself (no parts' sum). Measured in tt/perf_shared_expert.py (P.12).
SE_PARTS = 1
_DONE: dict = {}  # (mesh, down coordinators) -> the flat op's done words, shared by every layer's shared expert


class TtSharedExpert:
    def __init__(
        self,
        mesh,
        w1: torch.Tensor,
        w3: torch.Tensor,
        w2: torch.Tensor,
        swiglu_limit: float,
        chunk: int,
        flat_opts: dict | None = None,
    ):
        """w1, w3 [I, H], w2 [H, I] (nn.Linear layout), fp32 values on the fp8 grid (bf16-exact). chunk: the chunk
        (all mesh rows; the op is planned for chunk / SP rows a chip). flat_opts: comparison-only overrides of the
        flat op's tile formats / fidelity / packer rounding (tt/perf_shared_expert.py)."""
        tp = mesh.shape[TP_AXIS]
        inter, hidden = w1.shape
        assert w3.shape == (inter, hidden) and w2.shape == (hidden, inter), (w1.shape, w3.shape, w2.shape)
        assert inter % (TILE * tp) == 0, f"intermediate {inter} over TP {tp}"
        assert hidden % (TILE * tp) == 0, f"hidden {hidden} over TP {tp}"
        for w in (w1, w3, w2):
            assert torch.equal(w.bfloat16().float(), w.float()), "fp8 weights must be bf16-exact"
        self.mesh, self.tp, self.H, self.I = mesh, tp, int(hidden), int(inter)
        self.ip = self.I // tp  # the chip's intermediate share
        self.lim = float(swiglu_limit)
        self.sum_mode = settings.get("EXPERTS_SUM")
        self.reads_rows = False  # takes the routed experts' gathered row-major x (``rows``)
        self._load_flat(w1, w3, w2, int(chunk), flat_opts or {})

    # ------------------------------------------------------------------ the flat op
    def _load_flat(self, w1, w3, w2, chunk: int, opts: dict):
        """The chip's intermediate slice as SE_PARTS always-on experts of the flat routed expert (bfp8 weights in its
        bank layout, built once), the load-time token index / counts / regions."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert
        from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import LEGACY

        mesh, tp = self.mesh, self.tp
        R, C = tuple(mesh.shape)
        assert TP_AXIS == 1 and C == tp
        assert self.lim == 10.0, f"shared expert: swiglu_limit {self.lim} != the op's clamped_silu 10"
        assert chunk % (R * TILE) == 0, f"chunk {chunk} is not {R} x a multiple of {TILE}"
        P = SE_PARTS
        assert self.ip % (P * TILE) == 0, f"{self.ip} columns in {P} parts"
        self.S, self.P, self.ipart = chunk // R, P, self.ip // P
        self.precision = settings.get("EXPERTS_PRECISION")
        assert self.precision in ("full", "glm"), self.precision
        full = self.precision == "full"
        self.fidelity = int(
            (ttnn.MathFidelity.LoFi if not full else getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY"))).value
        )
        self.down_fp32 = fp32_acc(needed=True)  # the routed experts' choice (P.13: their down keeps fp32 DEST)
        x_bf16 = h_bf16 = full
        self.stochastic = False
        # (comparison only, tt/perf_shared_expert.py: other tile formats / fidelity / packer rounding than the setting's)
        x_bf16, h_bf16 = opts.get("x_bf16", x_bf16), opts.get("h_bf16", h_bf16)
        self.fidelity = int(opts.get("fidelity", self.fidelity))
        self.stochastic = bool(opts.get("stochastic", False))
        ip, q = self.ip, self.ipart

        def chip(c):  # mesh column c's slice as P experts (W1 [q, H], W3 [q, H], W2 [H, q]: the stored [N, K])
            return [
                (
                    w1[c * ip + p * q : c * ip + (p + 1) * q].float().contiguous(),
                    w3[c * ip + p * q : c * ip + (p + 1) * q].float().contiguous(),
                    w2[:, c * ip + p * q : c * ip + (p + 1) * q].float().contiguous(),
                )
                for p in range(P)
            ]

        cols = [chip(c) for c in range(C)]
        self.fe = FlatRoutedExpert(
            mesh,
            [cols[d % C] for d in range(R * C)],  # (device r C + c holds column c's slice)
            m=max(256, self.S),
            H=self.H,
            I=q,
            gids=[list(range(P)) for _ in range(R * C)],
            n_global=P,
            wdtype="bf8",
            act="clamped_silu",
            pin=1,
            x_bf16=x_bf16,
            h_bf16=h_bf16,
            weights_nk=True,
            check_exact=False,  # (bfp8 is the chosen format: it may round fp8 values, P.12)
            # EXPERTS_LAYOUT (as the routed experts): legacy = the plan before the op's new defaults; marko = his
            # defaults, every row of the chip's buffer an expert's (one always-on expert), one x helper
            layout=LEGACY if settings.get("EXPERTS_LAYOUT") == "legacy" else dict(rows_hint=self.S, x_helpers=1),
        )
        lay = dict(self.fe.plan)
        key = (id(mesh), tuple(map(tuple, lay["coords"])))
        if key in _DONE:  # (zero words the down coordinators leave zeroed: one tensor for every layer)
            dealloc(self.fe.done)
            self.fe.done = _DONE[key]
        else:
            _DONE[key] = self.fe.done
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def u32(t, mapper):
            return from_torch(
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
        self.reads_rows = True
        logger.info(
            f"SHARED_FLAT plan nu={lay['nu']} np={lay['np']} mt={lay['mt']} gate/up={len(lay['gu'])} down={lay['nd']} "
            f"parts={P} x {q} rows={S} precision={self.precision} fidelity={self.fidelity}"
        )

    def _flat_partial(self, x: ttnn.Tensor, xq: ttnn.Tensor | None, rows: ttnn.Tensor | None) -> ttnn.Tensor:
        S, H, P = self.S, self.H, self.P
        assert x.shape[-2] == S, f"the shared expert is built for {S} rows a chip, got {x.shape[-2]}"
        kw = dict(
            y_bf16=True,
            down_fp32=self.down_fp32,
            math_fidelity=self.fidelity,
            pack_stochastic_rounding=self.stochastic,
        )
        if rows is not None:  # (the routed experts' gathered row-major x: own rows by index)
            y = self.fe(rows, self.counts, self.regions, token_index=self.tidx_rows, **kw)
        else:
            own = xq is None
            xq = cast(x, ttnn.bfloat16) if own else xq
            xr = to_rm(xq, memory_config=DRAM)
            if own and xq is not x:
                dealloc(xq)
            xr2 = ttnn.reshape(xr, [S, H], sub_core_grids=sub_cores(xr))
            if P == 1:
                y = self.fe(xr2, self.counts, self.regions, **kw)
            else:
                y = self.fe(xr2, self.counts, self.regions, token_index=self.tidx_own, **kw)
            dealloc(xr)
        if rows is not None:
            dealloc(rows)
        if P > 1:  # (the parts' outputs: disjoint intermediate slices of one MLP, summed in fp32)
            acc = ttnn.slice(y, [0, 0], [S, H], memory_config=DRAM, sub_core_grids=sub_cores(y))
            acc = ttnn.typecast(acc, ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(acc))
            for p in range(1, P):
                part = ttnn.slice(y, [p * S, 0], [(p + 1) * S, H], memory_config=DRAM, sub_core_grids=sub_cores(y))
                nxt = ttnn.add(acc, part, dtype=ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores(acc))
                dealloc(acc)
                dealloc(part)
                acc = nxt
            dealloc(y)
            y = acc
        out = ttnn.reshape(y, [1, 1, S, H], sub_core_grids=sub_cores(y))
        want = ccl.partial_dtype(self.mesh) if self.sum_mode == "gather_add" else ttnn.float32
        if out.dtype != want:  # (bf16 y; fp32 only off the shipped CCL_IMPL=fabric_bf16 path)
            o2 = ttnn.typecast(out, want, memory_config=DRAM, sub_core_grids=sub_cores_tiled(out))
            dealloc(out)
            out = o2
        return out

    def partial(self, x: ttnn.Tensor, xq: ttnn.Tensor | None = None, rows: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """This chip's share of the output: x [1, 1, s, H] fp32 TILE (or its bf16 cast ``xq``, then x is unused)
        -> [1, 1, s, H] TILE partial in ccl.partial_dtype (sum over the TP axis = shared_out). ``rows`` = the routed
        experts' gathered row-major x [R s, H] (consumed)."""
        return self._flat_partial(x, xq, rows)

    def reduce(self, part: ttnn.Tensor) -> ttnn.Tensor:
        """Sum partials [1, 1, s, H] fp32 over axis 1, keep the chip's hidden columns [1, 1, s, H/TP]."""
        if self.tp == 1:
            return ttnn.clone(part, memory_config=DRAM)
        if part.dtype != ttnn.float32 and not (self.sum_mode == "gather_add" and ccl.bf16_sums()):
            pf = ttnn.typecast(part, ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(part))
            out = self.reduce(pf)
            dealloc(pf)
            return out
        if self.sum_mode == "reduce_scatter":
            return ttnn.reduce_scatter(part, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        return sum_cols(self.mesh, part)

    def __call__(self, x: ttnn.Tensor, xq: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """shared_out [1, 1, s, H/TP] fp32 (rows over axis 0, hidden columns over axis 1)."""
        part = self.partial(x, xq)
        out = self.reduce(part)
        dealloc(part)
        return out


def build_shared_expert(mesh, ck, cfg, layer: int, chunk: int, flat_opts: dict | None = None) -> TtSharedExpert:
    """Layer ``layer``'s shared expert (fp8 weights dequantized at load, exact; reference Config.swiglu_limit).
    chunk: the chunk (all mesh rows)."""
    p = f"layers.{layer}.ffn.shared_experts."
    w = {n: ck.weight(p + n + ".weight", torch.float32) for n in ("w1", "w3", "w2")}
    return TtSharedExpert(mesh, w["w1"], w["w3"], w["w2"], cfg.swiglu_limit, chunk=chunk, flat_opts=flat_opts)
