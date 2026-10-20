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
           g | u = ttnn.linear(x, [W1; W3], transpose_b, HiFi4 + fp32 acc, fp32 out)   (bfp8 / bfp4 [N, K] at load;
                                                                fp4_decode: two linears on per-chunk decoded bf16 W^T)
           h   = silu(min(g, 10)) * clamp(u, +-10) * r            fp32 (SFPU)
           y   = ttnn.linear(fp8_qdq(h), W2, transpose_b) -> bf16  (insert / combine are bf16-only)
           insert
      -> ttnn.bringup.combine (axis 0, init_zeros: the other group's slots read 0) -> [S/SP, K, H] bf16
      -> fp32 sum over the K slots (SFPU adds; weights already applied) = this group's partial [S/SP, H] fp32
      -> sum over the TP axis (the dispatch groups): gather_add (exact fp32, K-30) or reduce_scatter (setting
         EXPERTS_SUM), keep the chip's hidden columns -> experts_out [S/SP, H/TP] fp32

Weights (setting EXPERTS_IMPL, DS41-10 / DS41-35):
  bfp8            (max precision) every routed expert's w1 / w3 / w2 converted ONCE at load: the checkpoint's
                  e2m1 x e8m0 values (reference/weights.py:fp4_dequant, exact in bf16) uploaded as bfloat8_b in the
                  stored [N, K] orientation (ttnn.linear transpose_b), w1 | w3 fused to one [2I, H] matrix per expert.
                  bfp8 holds every value exactly there (one exponent per 16 values along K, inside one e8m0 block;
                  asserted per tensor at load, `_bfp8_exact`); W^T would not (DS41-35). 1.06 B per value: 0.84 GiB
                  per layer per chip at 24 experts per chip (this run's scaled 2x2 and galaxy_4x4). No per-chunk
                  weight work.
  bfp4 (default)  (P.1; the default since P.10: the checkpoint's fp4 width, agent rule 7) the bfp8 path with
                  bfloat4_b weights: same [N, K] orientation, transpose_b,
                  fused w1 | w3, converted once at load. Lossy: bfloat4_b keeps 3 magnitude bits under the group's
                  exponent (round to nearest even, saturating; `_bfp4_round` = ttnn's host packing), so an e2m1 value
                  more than 2 binades below its group's largest, or needing its 2nd bit there, moves. The fraction of
                  values changed and the worst per-matrix relative change are logged per layer at load (`bfp4_stats`).
                  0.5625 B per value: 0.44 GiB per layer per chip at 24 experts per chip.
  fp4_decode      the previous path: the checkpoint's packed codes kept on the device (0.53 B per value) and decoded
                  to a bf16 W^T for every expert in every chunk (a per-chunk weight rebuild, against agent rule 5;
                  kept for comparison only). Per matrix W [N, K] the device holds W^T = T [K, N] as 8 nibble planes,
                  P [K/8, N] int32, word (w, n) nibble i = code(T[i K/8 + w, n]), and the scales as bf16 2^e [K/32, N];
    plane i: (P >> 4i) & 15 -> ttnn.gather from the 16-entry e2m1 table (EXPERTS_DECODE 'gather') or the composed
             arithmetic (0.5 c + 0.5 relu(c - 4) + relu(c - 6), sign from bit 3; 'arith')
    T = concat(planes, rows) * (Rep [K, K/32] @ S^T)  (the 0/1 Rep matmul repeats each scale over its 32 rows, exact)
Kernel (setting EXPERTS_KERNEL, bfp8 / bfp4 weights):
  loop     the per-local-expert pipeline above (extract -> ttnn.linear -> SFPU -> fp8_qdq -> ttnn.linear -> insert).
  unified  ONE ttnn.bringup.unified_routed_expert_moe program over every local expert (P.2), on the dispatched buffer:
             x      = buf[:, :H] (q(x), bf16 ROW_MAJOR)
             r_rows = pieces @ SelAll (bf16 0/1 [3 Egp, 32] per mesh row r, column le = local expert le's three piece
                      columns r epc + le: one nonzero product per tile-dot, exact, DS41-16) -> fp32 ROW_MAJOR
                      [rows, 32], row i column le = row i's routing weight for local expert le
             op(x, routing_weights=r_rows, fp8_act_quant=True, weights_transposed=True, ClampedSiluGlu,
                high_precision, HiFi4 + fp32 dest): per local expert le and routed row i,
                y_i = w2 q(r_i,le silu(min(w1 x_i, 10)) clamp(w3 x_i, +-10)); w1 / w3 / w2 as stored ([N, K], the
                same block-float tensors the loop uses, w1 and w3 separate), q in-kernel per 32 (ue8m0, e4m3 RNE,
                bit-exact with tt/quant.fp8_qdq on exact inputs: the fork's tests/unit/test_ds41_h_options.py)
           -> the same combine / sum / reduce.
  flat     ONE ttnn.bringup.flat_routed_expert program (Marko Staletovic's flat spatially pipelined expert, P.3:
           16 DRAM bank readers stream every local expert's weights once into the gate/up cores, x relays tilize the
           dispatched rows and multicast them, down cores write y at each expert's region), on the dispatched buffer:
             x      = buf (q(x) | routing pieces, bf16 ROW_MAJOR; the op reads each row's first H values, hidden=H)
             r_rows = pieces @ SelAll as for unified, kept Float32 TILE [rows, 32]
             op(x, routing_weights=r_rows, fp8_act_quant=True, weights_nk, clamped_silu, x / h / y bf16, fp32 DEST
                (gate/up, down_fp32), fidelity MATMUL_FIDELITY; EXPERTS_PRECISION=glm: LoFi, x / h / y bfp8 tiles,
                fp32 DEST down, P.4): per local expert le and routed row i,
                y_i = w2 q(r_i,le silu(min(w1 x_i, 10)) clamp(w3 x_i, +-10)), w1 / w3 / w2 laid out once at load in
                the op's bank layout from the stored [N, K] matrices (every tile the transpose of the x @ W tile, so
                bfp8 shares its exponents along K: exact, asserted at load, DS41-35), q in-kernel per 32 (ue8m0, e4m3
                RNE; the fork's tests/unit/test_ds41_h_options.py: bit-exact on exact GEMMs)
           -> the same combine / sum / reduce.

The per-expert cap is the whole chunk (a token picks an expert once; the counts stay on the device), the flat
dispatch buffer K x the cap (all K of a token's experts may sit on one chip). expert counts / region offsets are
[1, E] (global ids, K-2); the dispatched buffer is never freed before combine (K-1).
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v41_flash_d_p.reference.weights import fp4_dequant

from . import ccl
from .common import DRAM, SP_AXIS, TP_AXIS, compute_config, sum_cols
from .quant import act_q, native
from .settings import settings

TILE = 32
PLANES = 8  # e2m1 codes per int32 word
BLOCK = 32  # e8m0 scale group along the expert's K
TRUNC_BF16 = -65536  # 0xFFFF0000 as int32: keep the bf16 part of an fp32's bits
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NAMES = ("w1", "w3", "w2")
BFP8_GROUP = 16  # bfloat8_b: one shared exponent per 16 values of a tile face row (along the last dim)
BFP8_MANT = 7  # magnitude bits under the shared exponent (the leading one included)
BFP4_MANT = 3  # bfloat4_b: the same 16-value groups, 3 magnitude bits
BLOCK_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}  # EXPERTS_IMPL values loaded once as block floats


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


def _bfp8_exact(t: torch.Tensor) -> bool:
    """True when bfloat8_b holds every value of t (bf16 / fp32, last dim a multiple of 16) exactly: within each group
    of 16 values along the last dim, every value is an integer multiple of 2^(E - 6), E = the group's largest binary
    exponent (7 magnitude bits under the shared exponent; zeros are exact)."""
    g = t.float().reshape(-1, BFP8_GROUP)
    amax = g.abs().amax(dim=-1, keepdim=True)
    _, e = torch.frexp(amax)  # amax = m 2^e, m in [0.5, 1): leading bit 2^(e - 1)
    q = torch.ldexp(g, BFP8_MANT - e)  # the 7-bit grid: leading bit -> 2^6
    return bool((q == q.round()).all()) and bool((q.abs() < 2**BFP8_MANT).all())


def _bfp4_round(t: torch.Tensor) -> torch.Tensor:
    """t (last dim a multiple of 16) as bfloat4_b holds it, fp32 (ttnn's host packing, blockfloat_common.cpp
    convert_u32_to_bfp, round to nearest even): per 16 values along the last dim, E = the group's largest binary
    exponent; each value -> round(v / 2^(E - 2)) clamped to +-7, times 2^(E - 2). Checked against ttnn's own packing
    by tt/perf_experts.py:test_bfp4_round_matches_ttnn."""
    g = t.float().reshape(-1, BFP8_GROUP)
    _, e = torch.frexp(g.abs().amax(dim=-1, keepdim=True))  # amax = m 2^e, m in [0.5, 1): unit 2^(e - 3)
    s = torch.ldexp(torch.ones_like(e, dtype=torch.float64), BFP4_MANT - e)  # powers of two, exact in fp64
    up, down = s.float(), (1.0 / s).float()
    assert bool(torch.isfinite(up).all() and torch.isfinite(down).all() and (up > 0).all() and (down > 0).all())
    lim = 2**BFP4_MANT - 1
    return (g * up).round_().clamp_(-lim, lim).mul_(down).reshape(t.shape)


def _bfp4_change(w: torch.Tensor) -> tuple[int, torch.Tensor]:
    """w [..., N, K]: (values bfloat4_b changes, per-matrix relative change ||q(w) - w|| / ||w|| over the last two
    dims, fp32)."""
    wf = w.float()
    d = _bfp4_round(w).sub_(wf)
    rel = torch.linalg.vector_norm(d, dim=(-2, -1)) / torch.linalg.vector_norm(wf, dim=(-2, -1)).clamp_min(1e-30)
    return int(torch.count_nonzero(d)), rel


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
        impl: str | None = None,
        combine: str | None = None,
        combine_acc: str | None = None,
        combine_planner: str | None = None,
    ):
        """ck: reference/weights.Checkpoint (experts read one at a time, packed). max_chunk: the largest chunk (all
        mesh rows together). impl: EXPERTS_IMPL value, combine: EXPERTS_COMBINE value, combine_acc: COMBINE_ACC value,
        combine_planner: COMBINE_PLANNER value
        (None: the setting)."""
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        E = int(num_experts)
        assert E % n == 0, f"{E} experts do not split over {n} chips"
        self.impl = impl or settings.get("EXPERTS_IMPL")
        assert self.impl in ("bfp8", "bfp4", "fp4_decode"), self.impl
        self.decode_mode = settings.get("EXPERTS_DECODE")
        self.kernel = settings.get("EXPERTS_KERNEL")
        assert self.kernel in ("loop", "unified", "flat"), self.kernel
        assert self.kernel == "loop" or self.impl in BLOCK_DTYPES, f"EXPERTS_KERNEL={self.kernel} needs block weights"
        self.sum_mode = settings.get("EXPERTS_SUM")
        # EXPERTS_PRECISION (P.4): full = MATMUL_FIDELITY + bf16 x / h / y tiles; glm = GLM-5.3's flat expert (LoFi
        # expert matmuls, bfp8 x / h / y tiles, fp32 DEST down). The selector matmuls keep self.cfg in both.
        self.precision = settings.get("EXPERTS_PRECISION")
        assert self.precision in ("full", "glm"), self.precision
        # EXPERTS_COMBINE (P.8): separate = dispatch / flat expert (routing weight + q(h) inside) / combine / K-slot sum;
        # weighted = the flat expert + combine_rf's weighted fp32 combine in one program (tt/weighted_combine.py)
        self.combine_mode = combine or settings.get("EXPERTS_COMBINE")
        assert self.combine_mode in ("separate", "weighted"), self.combine_mode
        assert (
            self.combine_mode == "separate" or self.kernel == "flat"
        ), "EXPERTS_COMBINE=weighted needs EXPERTS_KERNEL=flat"
        self.weighted = self.combine_mode == "weighted"
        # COMBINE_ACC (P.9): the weighted combine's per-token running sums in L1, fp32 (2x2: 2 passes of 512 home
        # tokens) or bf16 (one pass: the expert weights stream once a layer)
        self.combine_acc = combine_acc or settings.get("COMBINE_ACC")
        assert self.combine_acc in ("fp32", "bf16"), self.combine_acc
        # COMBINE_PLANNER (P.11): the weighted combine's planner on one core per pass, or the column multi-core
        # planner (bit-identical tables)
        self.combine_planner = combine_planner or settings.get("COMBINE_PLANNER")
        assert self.combine_planner in ("one", "multi"), self.combine_planner
        # NUMERICS=native (P.8): no fp8 q(x) / q(h) (rule 17); the expert ops then run without fp8_act_quant
        self.qdq = not native()
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
        # the expert matmuls' own config (glm: LoFi). The routing-weight selector matmuls are no expert matmuls and
        # stay on self.cfg (their exactness, DS41-16 / DS41-23, was shown at MATMUL_FIDELITY, never at LoFi)
        self.fidelity = (
            ttnn.MathFidelity.LoFi
            if self.precision == "glm"
            else getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY"))
        )
        self.ecfg = (
            ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=bool(settings.get("FP32_DEST_ACC")),
                packer_l1_acc=False,
            )
            if self.precision == "glm"
            else self.cfg
        )
        self.bfp4_stats = None  # EXPERTS_IMPL=bfp4: what bfloat4_b changes in this layer's experts (_load_block)

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

        if self.kernel in ("unified", "flat"):
            # the fused ops bake ClampedSiluGlu's limit 10 (unified_routed_expert_ffn_types.hpp; flat's se3_compute.cpp
            # SE_ACT 3, ClampedSiluGluConfigDsV4)
            assert self.lim == 10.0, f"EXPERTS_KERNEL={self.kernel}: swiglu_limit {self.lim} != the op's 10"
            # the routing-weight rows carry one column per local expert in one tile (24 per chip on 2x2 / 4x4, 12 on 8x4)
            assert (
                self.epc <= TILE
            ), f"EXPERTS_KERNEL={self.kernel}: {self.epc} experts per chip > {TILE} weight columns"
            # one selector for every local expert: column le of mesh row r = the group's expert j = r epc + le
            sel_all = torch.zeros(rows, 1, 3 * self.Egp, TILE)
            for r in range(rows):
                for le in range(self.epc):
                    for p in range(3):
                        sel_all[r, 0, p * self.Egp + r * self.epc + le, le] = 1.0
            self.sel_all = ttnn.from_torch(
                sel_all,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=_mapper(mesh, (0, None)),
            )
        self._seq = {}
        if not self.weighted:
            self._modules(self.max_chunk // rows)  # the serving chunk's dispatch / combine sizes, built at load
        if self.impl in BLOCK_DTYPES:
            self._load_block(ck, gtab)
        else:
            self._load_fp4(ck, gtab)

    def _load_fp4(self, ck, gtab):
        """fp4_decode: per local slot le and matrix, nibble planes + scales (one mesh tensor whose chip (r, c) shard
        is expert (c rows + r) epc + le), and the decode constants."""
        rows, cols, mesh, layer = self.dgs, self.ngroups, self.mesh, self.layer
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

    def _load_block(self, ck, gtab):
        """bfp8 / bfp4: per local slot le, W13 [2I, H] (w1 rows | w3 rows) and W2 [H, I] bfloat8_b / bfloat4_b as
        stored ([N, K], used with transpose_b), one mesh tensor each whose chip (r, c) shard is expert
        (c rows + r) epc + le. Built once.

        Orientation matters (DS41-35): bfloat8_b shares one exponent per 16 values along the last dim. Along K (as
        stored) a group lies inside one 32-wide e8m0 block, so it holds e2m1 values (0.5 .. 6, 4 bits) under one
        scale: always exact. Along N (W^T) a group spans 16 output rows with their own scales, and a few w2 groups
        (layer 0: 2-3 of 737,280 per matrix) mix 2^-3 and 2^-8 scales: not exact.

        bfp4 (lossy, owner pick): the same orientation (a group still shares one e8m0 scale); `bfp4_stats` records
        the fraction of this layer's expert values bfloat4_b changes and the worst per-matrix relative change."""
        rows, cols, mesh, layer, H, I = self.dgs, self.ngroups, self.mesh, self.layer, self.H, self.I
        dtype = BLOCK_DTYPES[self.impl]
        flat = self.kernel == "flat"
        flat_w = [[None] * self.epc for _ in range(rows * cols)]  # flat: [device r cols + c][le] = (W1, W3, W2) stored
        self.w13, self.w2 = [], []
        self.w1, self.w3 = [], []  # EXPERTS_KERNEL=unified: w1 and w3 as separate [I, H] tensors (the op's gate / up)
        changed = total = 0
        worst = (0.0, None)
        for le in range(self.epc):
            w13 = torch.empty(cols, rows, 2 * I, H, dtype=torch.bfloat16)
            w2 = torch.empty(cols, rows, H, I, dtype=torch.bfloat16)
            for c in range(cols):
                for r in range(rows):
                    p = ck.expert(layer, int(gtab[c, r, le]))
                    w = {nm: fp4_dequant(*p[nm]) for nm in NAMES}  # [N, K] fp32, e2m1 x 2^e (exact in bf16)
                    assert w["w1"].shape == (I, H) and w["w2"].shape == (H, I), (w["w1"].shape, w["w2"].shape)
                    w13[c, r, :I] = w["w1"]
                    w13[c, r, I:] = w["w3"]
                    w2[c, r] = w["w2"]
                    for t in w.values():
                        assert bool(torch.equal(t.to(torch.bfloat16).float(), t)), "e2m1 x 2^e not bf16-exact"
                    if flat:  # (the flat op's layout is built from these below, one device's experts at a time)
                        flat_w[r * cols + c][le] = (w13[c, r, :I], w13[c, r, I:], w2[c, r])
            if self.impl == "bfp8":
                for t, nm in ((w13, "w1 | w3"), (w2, "w2")):
                    assert _bfp8_exact(t), f"layer {layer} slot {le} {nm}: bfloat8_b would round a value (DS41-35)"
            else:
                for mats in ((w13[:, :, :I], "w1"), (w13[:, :, I:], "w3"), (w2, "w2")):
                    n_ch, rel = _bfp4_change(mats[0])
                    changed, total = changed + n_ch, total + mats[0].numel()
                    v, i = rel.flatten().max(dim=0)
                    if float(v) > worst[0]:
                        c, r = divmod(int(i), rows)
                        worst = (float(v), (int(gtab[c, r, le]), mats[1]))
            pairs = [(w13, self.w13), (w2, self.w2)]
            if self.kernel == "unified":
                pairs = [(w13[:, :, :I].contiguous(), self.w1), (w13[:, :, I:].contiguous(), self.w3), (w2, self.w2)]
            if flat:
                pairs = []  # (no per-expert device tensors: the flat op's bank layout below)
            for t, out in pairs:
                out.append(
                    ttnn.from_torch(
                        t,
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=DRAM,
                        mesh_mapper=get_ep_mesh_mapper(mesh),
                    )
                )
        if flat:
            self._load_flat(flat_w, gtab)
        if self.impl == "bfp4":
            self.bfp4_stats = {
                "layer": layer,
                "changed_frac": changed / total,
                "worst_rel": worst[0],
                "worst_matrix": worst[1],
            }
            logger.info(
                f"EXPERTS_BFP4 layer={layer} values_changed={changed}/{total} ({100 * changed / total:.2f} %) "
                f"worst_matrix_rel={worst[0]:.4g} (expert, matrix)={worst[1]}"
            )

    def _load_flat(self, flat_w, gtab):
        """EXPERTS_KERNEL=flat: every local expert's (W1, W3, W2) as stored ([N, K], bf16-exact e2m1 x 2^e) in the
        flat op's per-core bank layout (FlatRoutedExpert, weights_nk: each tile the transpose of x @ W's tile, so a bfp8
        tile shares its exponents along K: exact, the groups _load_block asserts on [N, K]; bfp4: ttnn's lossy packing
        of the same groups, the values EXPERTS_IMPL=bfp4 gives the other kernels). Built once per layer; the plan is the
        op's for this chip's grid (16-column units, 32-row sub-blocks with the h options)."""
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        rows, cols = self.dgs, self.ngroups
        gids = [[int(gtab[c, r, le]) for le in range(self.epc)] for r in range(rows) for c in range(cols)]
        n_global = self.E
        if self.weighted:
            # EXPERTS_COMBINE=weighted: the column's planner numbers the column's experts from 0 (chip r holds
            # [r epc, (r + 1) epc) of its column's Eg; ExpertMapping column-major), counts are [1, Eg]; an expert sees at
            # most every token of its column once (R x the chip's rows)
            gids = [[r * self.epc + le for le in range(self.epc)] for r in range(rows) for c in range(cols)]
            n_global = self.Eg
            cap = self.max_chunk
        else:
            _, _, cap = self._modules(self.max_chunk // rows)
        self.flat_cap = max(256, _ceil(cap, TILE))  # (the op is built for >= 256 rows per expert)
        self.fe = FlatRoutedExpert(
            self.mesh,
            flat_w,
            m=self.flat_cap,
            H=self.H,
            I=self.I,
            gids=gids,
            n_global=n_global,
            wdtype={"bfp8": "bf8", "bfp4": "bf4"}[self.impl],
            act="clamped_silu",
            pin=1,
            # EXPERTS_PRECISION=glm: bfp8 x / h tiles, as GLM-5.3 runs the op (GLM_AG_XBF16 / _HBF16 default 0)
            x_bf16=self.precision == "full",
            h_bf16=self.precision == "full",
            weights_nk=True,
            routed=not self.weighted,  # (the h options' plan: separate; weighted runs neither h option)
            # (no second exactness pass: a laid-out tile row is 32 consecutive K values of one stored row, the same
            # 16-value groups _bfp8_exact asserted on the [N, K] matrices above)
            check_exact=False,
        )
        logger.info(
            f"EXPERTS_FLAT layer={self.layer} plan nu={self.fe.plan['nu']} np={self.fe.plan['np']} "
            f"mt={self.fe.plan['mt']} gate/up={len(self.fe.plan['gu'])} down={self.fe.plan['nd']} cap={self.flat_cap} "
            f"precision={self.precision} combine={self.combine_mode}"
        )
        self.wc = None
        if self.weighted:
            from .weighted_combine import weighted_combine

            self.wc = weighted_combine(
                self.mesh,
                dict(self.fe.plan),
                H=self.H,
                I=self.I,
                epc=self.epc,
                K=self.K,
                S=self.max_chunk // rows,
                out_dtype=ccl.partial_dtype(self.mesh) if self.sum_mode == "gather_add" else ttnn.float32,
                x_bf16=self.precision == "full",
                h_bf16=self.precision == "full",
                acc_dtype=ttnn.bfloat16 if self.combine_acc == "bf16" else ttnn.float32,
                planner=self.combine_planner,
            )
            # COMBINE_ACC=bf16 exists to run the chunk's home tokens in ONE pass (the weights streamed once a layer)
            assert self.combine_acc != "bf16" or self.wc.P == 1, (
                f"COMBINE_ACC=bf16: {self.wc.S} home tokens per chip need {self.wc.P} passes on this grid "
                f"({self.wc.A} accumulators)"
            )
            logger.info(
                f"EXPERTS_WEIGHTED layer={self.layer} COMBINE_ACC={self.combine_acc} passes={self.wc.P} x {self.wc.S_pass} "
                f"COMBINE_PLANNER={self.combine_planner} ({self.wc.planner_desc})"
            )

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
        """fp4_decode: W^T [K, N] bf16 (exact) of local expert le's matrix nm, from its nibble planes and scales."""
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
        cfg, lim, I = self.ecfg, self.lim, self.I
        if self.impl in BLOCK_DTYPES:
            gu = ttnn.linear(
                xt, self.w13[le], transpose_b=True, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM
            )
            lead = list(gu.shape)[:-1]  # [.., cap]: the weight's [1, 1] batch dims lead the output
            z = [0] * len(lead)
            g = ttnn.slice(gu, z + [0], lead + [I], memory_config=DRAM)
            u = ttnn.slice(gu, z + [I], lead + [2 * I], memory_config=DRAM)
            ttnn.deallocate(gu)
        else:
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
        hq = act_q(hr, dtype=ttnn.bfloat16)  # NUMERICS=native: a cast
        ttnn.deallocate(hr)
        if self.impl in BLOCK_DTYPES:
            y = ttnn.linear(
                hq, self.w2[le], transpose_b=True, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM
            )
        else:
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

    def _unified(self, buf: ttnn.Tensor, counts, region_offsets, cap: int) -> ttnn.Tensor:
        """EXPERTS_KERNEL=unified. buf [max_buf, W] bf16 ROW_MAJOR (the dispatched rows: q(x) | routing pieces) ->
        expert outputs [max_buf, H] bf16 TILE (each expert's rows at its region; a fresh buffer, buf untouched)."""
        n, H = buf.shape[-2], self.H
        xr = ttnn.slice(buf, [0, 0], [n, H], memory_config=DRAM)  # q(x), ROW_MAJOR bf16
        wp = ttnn.slice(buf, [0, H], [n, self.W], memory_config=DRAM)
        wpt = ttnn.to_layout(wp, ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(wp)
        # row i column le = row i's fp32 routing weight for local expert le (exact: one product per tile-dot)
        rw = ttnn.matmul(
            ttnn.reshape(wpt, (1, 1, n, 3 * self.Egp)),
            self.sel_all,
            dtype=ttnn.float32,
            compute_kernel_config=self.cfg,
            memory_config=DRAM,
        )
        ttnn.deallocate(wpt)
        rwr = ttnn.to_layout(ttnn.reshape(rw, (n, TILE)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(rw)
        out = ttnn.bringup.unified_routed_expert_moe(
            xr,
            region_offsets,
            counts,
            self.gidx,
            self.w1,
            self.w3,
            self.w2,
            max_dispatched_tokens_per_expert=cap,
            compute_kernel_config=self.ecfg,
            activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
            high_precision=True,
            routing_weights=rwr,
            fp8_act_quant=self.qdq,
            weights_transposed=True,
        )
        ttnn.deallocate(xr)
        ttnn.deallocate(rwr)
        return out

    def _routing_rows(self, buf: ttnn.Tensor) -> ttnn.Tensor:
        """buf [n, W] bf16 ROW_MAJOR -> [n, 32] fp32 TILE: row i column le = row i's routing weight for local expert le
        (its three bf16 pieces summed exactly by one 0/1 selector matmul, one product per tile-dot, DS41-16)."""
        n = buf.shape[-2]
        wp = ttnn.slice(buf, [0, self.H], [n, self.W], memory_config=DRAM)
        wpt = ttnn.to_layout(wp, ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(wp)
        rw = ttnn.matmul(
            ttnn.reshape(wpt, (1, 1, n, 3 * self.Egp)),
            self.sel_all,
            dtype=ttnn.float32,
            compute_kernel_config=self.cfg,
            memory_config=DRAM,
        )
        ttnn.deallocate(wpt)
        return ttnn.reshape(rw, (n, TILE))

    def _flat(self, buf: ttnn.Tensor, counts, region_offsets) -> ttnn.Tensor:
        """EXPERTS_KERNEL=flat. buf [max_buf, W] bf16 ROW_MAJOR (the dispatched rows: q(x) | routing pieces) -> expert
        outputs [max_buf, H] bf16 TILE (each expert's rows at its region; a fresh buffer, buf untouched)."""
        rw = self._routing_rows(buf)
        out = self.fe(
            buf,
            counts,
            region_offsets,
            # glm: bfp8 y TILE (combine reads it), fp32 DEST down as GLM_AG_DOWN_FP32's default, LoFi
            y_bf16=self.precision == "full",
            down_fp32=True if self.precision == "glm" else bool(settings.get("FP32_DEST_ACC")),
            math_fidelity=int(self.fidelity.value),
            routing_weights=rw,
            fp8_act_quant=self.qdq,
            hidden=self.H,
        )
        ttnn.deallocate(rw)
        return out

    def _weighted(self, x: ttnn.Tensor, dense: ttnn.Tensor, idx: ttnn.Tensor | None) -> ttnn.Tensor:
        """EXPERTS_COMBINE=weighted (tt/weighted_combine.py): q(x) of the mesh row gathered over the column, the
        column's top-k ids and bf16 weights, then the flat expert + weighted fp32 combine in one program per pass.
        Returns this column group's partial [1, 1, s, H] (fp32; bf16 under CCL_IMPL=fabric_bf16)."""
        s, K = x.shape[-2], self.K
        wc = self.wc
        assert s == wc.S, f"EXPERTS_COMBINE=weighted is built for {wc.S} rows per chip, got {s}"
        qx = act_q(x, dtype=ttnn.bfloat16)  # NUMERICS=native: a cast (x itself when ffn_norm wrote bf16)
        xg = wc.gather_x(qx)
        if qx is not x:
            ttnn.deallocate(qx)
        dk = dense if dense.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(dense, ttnn.TILE_LAYOUT, memory_config=DRAM)
        if idx is None:  # (the routing's K nonzero weights and their ids; order is free: the combine sums them)
            wts, ids = ttnn.topk(dk, k=K, dim=-1, largest=True, sorted=True)
        else:
            ids = idx
            wts = ttnn.gather(dk, dim=-1, index=idx)
        if dk is not dense:
            ttnn.deallocate(dk)
        ids16 = ids if ids.dtype == ttnn.uint16 else ttnn.typecast(ids, ttnn.uint16, memory_config=DRAM)
        ids_g, wts_g = wc.routing(ids16, wts)
        for t in {id(t): t for t in (ids16, wts) + ((ids,) if idx is None else ())}.values():
            if t is not idx:
                ttnn.deallocate(t)
        out = wc(self.fe, xg, ids_g, wts_g, int(self.fidelity.value))
        for t in (xg, ids_g, wts_g):
            ttnn.deallocate(t)
        return out

    def partial(self, x: ttnn.Tensor, dense: ttnn.Tensor, idx: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """This column group's experts over the chip's rows. x: ffn_norm [1, 1, s, H] fp32 TILE (rows split over axis
        0, replicated over axis 1); dense: routing [1, 1, s, E] fp32 (same placement); idx: the router's top-k ids
        [1, 1, s, K] (None: top-k of dense). Returns [1, 1, s, H] fp32 TILE (the group's sum, weights applied)."""
        if self.weighted:
            return self._weighted(x, dense, idx)
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
        qx = act_q(x, dtype=ttnn.bfloat16)  # NUMERICS=native: a cast (x itself when ffn_norm wrote bf16)
        pieces = self._pieces(dense, s)
        xa = ttnn.concat([qx] + pieces, dim=-1, memory_config=DRAM)
        for t in ([qx] if qx is not x else []) + pieces:
            ttnn.deallocate(t)
        xr = ttnn.to_layout(xa, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(xa)
        buf, meta = _dispatch(dispatch, ttnn.reshape(xr, (1, s, self.W)), ind, offsets, self.dispatch_table)
        ttnn.deallocate(xr)
        b2 = ttnn.reshape(buf, (buf.shape[-2], self.W))
        if self.kernel == "unified":
            out = self._unified(b2, counts, region_offsets, cap)
            ttnn.deallocate(buf)
        elif self.kernel == "flat":
            out = self._flat(b2, counts, region_offsets)
            ttnn.deallocate(buf)
        else:
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
        return sum_cols(self.mesh, part)

    def free(self):
        """Release the device weights (tests that build several layers' or impls' experts in one process)."""
        mats = [getattr(self, "w13", []), getattr(self, "w2", []), getattr(self, "w1", []), getattr(self, "w3", [])]
        for d in getattr(self, "planes", []) + getattr(self, "scales", []):
            mats.append(list(d.values()))
        for ts in mats:
            for t in ts:
                ttnn.deallocate(t)
        self.w13, self.w2, self.planes, self.scales = [], [], [], []
        self.w1, self.w3 = [], []
        fe = getattr(self, "fe", None)
        if fe is not None:  # the flat op's bank-layout weights, expert ids and done words
            for t in (fe.w_gu, fe.w_d, fe.w_rd, fe.gidx, fe.done):
                if t is not None:
                    ttnn.deallocate(t)
            self.fe = None

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


def build_experts(
    mesh,
    ck,
    cfg,
    layer: int,
    chunk: int,
    impl: str | None = None,
    combine: str | None = None,
    combine_acc: str | None = None,
    combine_planner: str | None = None,
) -> TtExperts:
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
        impl=impl,
        combine=combine,
        combine_acc=combine_acc,
        combine_planner=combine_planner,
    )
