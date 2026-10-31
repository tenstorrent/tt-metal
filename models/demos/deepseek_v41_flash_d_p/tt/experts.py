# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 routed experts (reference/ds41_ref.py:_experts / _swiglu; inference MoE + Expert) on the SP x TP
mesh: DeepSeek 2D expert parallelism (deepseek_v3_d_p/tt/moe/README.md), dispatch groups = mesh columns (plan.md
"Experts").

Per expert e and token t routed to it (weight r = router[t, e]):

    y_e(t) = w2 q(r * silu(min(w1 q(x), 10)) * clamp(w3 q(x), +-10))       q = fp8 act quant, blocks of 32 (released code)

The device does not emulate q (agent rule 17): x is the bf16 the matmuls read, and the routing weight moves from h to
the combine (w2 is linear: w2 (r h) = r (w2 h)). ONE pipeline (FIX.9 deleted the separate dispatch / combine chain, the
routing-weight pieces, the loop / unified / fp4_decode kernels):

    x (ffn_norm, bf16) of the mesh row gathered over the column (axis 0): xg [R S/SP, H] bf16 ROW_MAJOR
    the column's top-k ids and bf16 weights (the router's own, INDEXER_GLUE=on, or top-k / gather of dense)
    -> per pass (tt/weighted_combine.py): ttnn.bringup.flat_routed_expert_with_program = Marko Staletovic's flat
       spatially pipelined expert (DRAM bank readers stream every local expert's weights once, x relays tilize and
       multicast the rows they read by token index, clamped_silu, EXPERTS_PRECISION: glm = LoFi + bfp8 x / h / y
       tiles, full = MATMUL_FIDELITY + bf16 tiles; fp32 DEST gate/up and down) + combine_rf's router-forwarded
       weighted combine: each expert row goes to its token's home chip in the column, is multiplied there by its
       routing weight and summed per token (COMBINE_ACC, fixed det order: bit-identical calls)
    -> this column group's partial [S/SP, H] (bf16 under CCL_IMPL=fabric_bf16, else fp32)
    -> sum over the TP axis (the dispatch groups) by the reduce-scatter (EXPERTS_SUM; in the assembled model after
       moe_add has added the shared expert's partial), the chip's hidden columns -> experts_out [S/SP, H/TP]

Weights (setting EXPERTS_IMPL, DS41-10 / DS41-35), every routed expert's w1 / w3 / w2 laid out ONCE at load in the flat
op's bank layout from the stored [N, K] matrices (weights_nk; reference/weights.py:fp4_dequant, e2m1 x 2^e, exact in
bf16):
  bfp4 (default)  (P.1; the default since P.10: the checkpoint's fp4 width, agent rule 7) bfloat4_b: lossy, 3
                  magnitude bits under each 16-value group's exponent (round to nearest even, saturating;
                  `_bfp4_round` = ttnn's host packing). The fraction of values changed and the worst per-matrix
                  relative change are logged per layer at load (`bfp4_stats`). 0.5625 B per value.
  bfp8            bfloat8_b: exact for every e2m1 x e8m0 value in the [N, K] orientation (one exponent per 16 values
                  along K, inside one e8m0 block; asserted at load, `_bfp8_exact`); W^T would not be (DS41-35).
No per-chunk weight work. The counts / plans / tables stay on the device (the column planner, combine_plan).
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v41_flash_d_p.reference.weights import fp4_dequant

from . import ccl
from .ccl import dealloc
from .common import DRAM, TP_AXIS, cast, sum_cols
from .cores import sub_cores, sub_cores_tiled
from .settings import settings

TILE = 32
NAMES = ("w1", "w3", "w2")
BFP8_GROUP = 16  # bfloat8_b: one shared exponent per 16 values of a tile face row (along the last dim)
BFP8_MANT = 7  # magnitude bits under the shared exponent (the leading one included)
BFP4_MANT = 3  # bfloat4_b: the same 16-value groups, 3 magnitude bits
BLOCK_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}  # EXPERTS_IMPL values (the weights' block format)


def _ceil(a: int, b: int) -> int:
    return -(-a // b) * b


def _bfp8_exact(t: torch.Tensor) -> bool:
    """True when bfloat8_b holds every value of t (bf16 / fp32, last dim a multiple of 16) exactly: within each group
    of 16 values along the last dim, every value is an integer multiple of 2^(E - 6), E = the group's largest binary
    exponent (7 magnitude bits under the shared exponent; zeros are exact)."""
    g = t.float().reshape(-1, BFP8_GROUP)
    amax = torch.maximum(g.amax(dim=-1, keepdim=True), g.amin(dim=-1, keepdim=True).neg_())  # max |v| per group
    _, e = torch.frexp(amax)  # amax = m 2^e, m in [0.5, 1): leading bit 2^(e - 1)
    q = torch.ldexp(g, BFP8_MANT - e)  # the 7-bit grid: leading bit -> 2^6
    return bool((q == q.round()).all()) and bool((q.abs() < 2**BFP8_MANT).all())


def _bfp4_round(t: torch.Tensor) -> torch.Tensor:
    """t (last dim a multiple of 16) as bfloat4_b holds it, fp32 (ttnn's host packing, blockfloat_common.cpp
    convert_u32_to_bfp, round to nearest even): per 16 values along the last dim, E = the group's largest binary
    exponent; each value -> round(v / 2^(E - 2)) clamped to +-7, times 2^(E - 2). Checked against ttnn's own packing
    by tt/perf_experts.py:test_bfp4_round_matches_ttnn."""
    g = t.float().reshape(-1, BFP8_GROUP)
    # max |v| of each group = max(max v, -min v) (no |g| copy of the whole tensor)
    amax = torch.maximum(g.amax(dim=-1, keepdim=True), g.amin(dim=-1, keepdim=True).neg_())
    _, e = torch.frexp(amax)  # amax = m 2^e, m in [0.5, 1): unit 2^(e - 3)
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


def _bf16_exact(t: torch.Tensor) -> bool:
    """True when bf16 holds every value of the fp32 tensor t exactly (the low 16 bits of each value are zero) and every
    value is finite: = torch.equal(t.to(bf16).float(), t) for finite t, one pass with no copy."""
    assert t.dtype == torch.float32 and t.is_contiguous()
    low = t.view(torch.int16)[..., 0::2]  # (little endian: the low half of each fp32 word)
    lo, hi = torch.aminmax(t)  # (NaN propagates; isfinite over the whole tensor took 0.28 s per stack)
    return int(torch.count_nonzero(low)) == 0 and bool(torch.isfinite(lo)) and bool(torch.isfinite(hi))


READ_WORKERS = 8  # threads copying one slot's stored experts out of the checkpoint maps (_host_weights)
_READ_POOL = None


def _read_pool() -> ThreadPoolExecutor:
    global _READ_POOL
    if _READ_POOL is None:
        _READ_POOL = ThreadPoolExecutor(READ_WORKERS, thread_name_prefix="ds41-expert-read")
    return _READ_POOL


def expert_cache_dir(spec) -> Path:
    """The routed experts' weight cache of a run (EXPERTS_WEIGHT_CACHE): the spec's tt_cache (art, the mesh, -scaled,
    the cache version) / experts-<checkpoint revision>."""
    rev = str(spec.get("hf.revision") or "norev")[:12]
    return Path(spec.tt_cache()) / f"experts-{rev}"


def expert_cache_prefix(cache_dir, ck, layer: int, impl: str, mesh, n_experts: int) -> str | None:
    """The FlatRoutedExpert cache_prefix of one layer's routed experts, or None (no cache_dir, EXPERTS_WEIGHT_CACHE=off):
    <cache_dir>/<rows>x<cols>/L<layer>.<impl>.<fingerprint>, the fingerprint a hash of the layer's stored expert
    tensors as the checkpoint index places them (file, dtype, shape, offset, bytes: the real and the scaled checkpoint
    differ). The fork's file names add the block dtype and the hash of the layout plan."""
    if cache_dir is None or settings.get("EXPERTS_WEIGHT_CACHE") != "on":
        return None
    rows, cols = tuple(mesh.shape)
    h = hashlib.sha1()
    for e in range(int(n_experts)):
        for nm in NAMES:
            for part in ("weight", "scale"):
                name = f"layers.{layer}.ffn.experts.{e}.{nm}.{part}"
                dt, shape, off, nb = ck.info(name)
                h.update(repr((name, os.path.basename(ck.weight_map[name]), str(dt), shape, off, nb)).encode())
    return str(Path(cache_dir) / f"{rows}x{cols}" / f"L{layer:02d}.{impl}.{h.hexdigest()[:12]}")


class _CacheMarker:
    """<prefix>.done.json, written after FlatRoutedExpert has dumped a layer's laid-out weights: the files (and their
    sizes) it wrote and the layer's bfp4_stats. The fork dumps straight to the final names, so a process killed while
    dumping leaves a truncated file: at construction every <prefix>.flat.* file the marker does not list with its size
    is deleted (the fork then misses and repacks)."""

    def __init__(self, prefix: str | None):
        self.prefix, self.stats = prefix, None
        if prefix is None:
            return
        self.path = Path(prefix + ".done.json")
        rec = {}
        if self.path.is_file():
            try:
                rec = json.loads(self.path.read_text())
            except (OSError, ValueError):
                rec = {}
        files = rec.get("files", {})
        for f in self._files():
            if files.get(f.name) != f.stat().st_size:
                logger.warning(f"EXPERTS_CACHE dropping {f} (not in the completion marker {self.path.name})")
                f.unlink()
        self.stats = rec.get("stats")

    def _files(self):
        stem = Path(self.prefix)
        return sorted(stem.parent.glob(f"{stem.name}.flat.*")) if stem.parent.is_dir() else []

    def write(self, stats):
        rec = {"files": {f.name: f.stat().st_size for f in self._files()}, "stats": stats}
        tmp = self.path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, self.path)
        self.stats = stats


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
        impl: str | None = None,
        combine_acc: str | None = None,
        combine_planner: str | None = None,
        cache_dir=None,
    ):
        """ck: reference/weights.Checkpoint (experts read one at a time, packed). max_chunk: the chunk (all mesh rows
        together). impl: EXPERTS_IMPL value, combine_acc: COMBINE_ACC value, combine_planner: COMBINE_PLANNER value
        (None: the setting). cache_dir: the laid-out weight cache (expert_cache_dir; None: no cache)."""
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        E = int(num_experts)
        assert E % n == 0, f"{E} experts do not split over {n} chips"
        self.impl = impl or settings.get("EXPERTS_IMPL")
        assert self.impl in BLOCK_DTYPES, self.impl
        self.sum_mode = settings.get("EXPERTS_SUM")
        # EXPERTS_PRECISION (P.4): full = MATMUL_FIDELITY + bf16 x / h / y tiles; glm = GLM-5.3's flat expert (LoFi
        # expert matmuls, bfp8 x / h / y tiles, fp32 DEST down)
        self.precision = settings.get("EXPERTS_PRECISION")
        assert self.precision in ("full", "glm"), self.precision
        # COMBINE_ACC (P.9): the weighted combine's per-token running sums in L1, fp32 (2x2: 2 passes of 512 home
        # tokens) or bf16 (one pass: the expert weights stream once a layer)
        self.combine_acc = combine_acc or settings.get("COMBINE_ACC")
        assert self.combine_acc in ("fp32", "bf16"), self.combine_acc
        # COMBINE_PLANNER (P.11): the weighted combine's planner on one core per pass, or the column multi-core
        # planner (bit-identical tables)
        self.combine_planner = combine_planner or settings.get("COMBINE_PLANNER")
        assert self.combine_planner in ("one", "multi"), self.combine_planner
        self.mesh, self.layer = mesh, layer
        self.E, self.K, self.H, self.I = E, int(top_k), int(hidden), int(inter)
        self.lim = float(swiglu_limit)
        self.dgs, self.ngroups, self.n = rows, cols, n
        self.epc, self.Eg = E // n, E // cols
        assert self.H % TILE == 0 and self.I % TILE == 0
        self.max_chunk = int(max_chunk)
        assert self.max_chunk % (rows * TILE) == 0, f"chunk {max_chunk} is not {rows} x a multiple of {TILE}"
        # the flat op bakes ClampedSiluGlu's limit 10 (se3_compute.cpp SE_ACT 3, ClampedSiluGluConfigDsV4)
        assert self.lim == 10.0, f"swiglu_limit {self.lim} != the flat op's clamped_silu 10"
        self.fidelity = (
            ttnn.MathFidelity.LoFi
            if self.precision == "glm"
            else getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY"))
        )
        self.cache_dir = cache_dir
        self.bfp4_stats = None  # EXPERTS_IMPL=bfp4: what bfloat4_b changes in this layer's experts (_load_block)
        # local -> global expert ids per chip [cols, rows, epc] (ExpertMapping column-major: chip (r, c) holds
        # (c rows + r) epc ..): which checkpoint experts each chip's weights hold
        gtab = ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=self.epc, dispatch_group_size=rows, num_dispatch_groups=cols
        )
        # SHARED_EXPERT (P.12): the gathered row-major q(x) of the column kept for the shared expert, which reads its
        # own rows from it by token index (take_rows); set by the assembled model's block
        self.keep_rows, self._rows = False, None
        self._load_block(ck, gtab)

    def _load_block(self, ck, gtab):
        """Per local slot le, every chip's W1 / W3 [I, H] and W2 [H, I] as stored ([N, K]), checked (bfp8: exact) or
        measured (bfp4: `bfp4_stats`) in the 16-value groups along K the flat op's tiles share, then laid out in the op's
        bank layout (_load_flat). Built once; with a weight cache (EXPERTS_WEIGHT_CACHE, FIX.4) only on a miss: the
        fork calls _host_weights only when its laid-out files are missing.

        Orientation matters (DS41-35): a block-float group shares one exponent per 16 values along the last dim. Along
        K (as stored) a group lies inside one 32-wide e8m0 block, so bfloat8_b holds its e2m1 values (0.5 .. 6, 4
        bits) under one scale: always exact. Along N (W^T) a group spans 16 output rows with their own scales.

        bfp4 (lossy, owner pick): the same orientation (a group still shares one e8m0 scale); `bfp4_stats` records
        the fraction of this layer's expert values bfloat4_b changes and the worst per-matrix relative change."""
        import time

        t0 = time.time()
        prefix = expert_cache_prefix(self.cache_dir, ck, self.layer, self.impl, self.mesh, self.E)
        marker = _CacheMarker(prefix)
        self._host_s = None
        self._load_flat(lambda: self._host_weights(ck, gtab), gtab, prefix)
        hit = self._host_s is None
        if prefix is not None:
            if hit:
                self.bfp4_stats = marker.stats
            else:
                marker.write(self.bfp4_stats)
        if self.impl == "bfp4" and self.bfp4_stats is not None:
            st = self.bfp4_stats
            logger.info(
                f"EXPERTS_BFP4 layer={self.layer} values_changed={st['changed']}/{st['total']} "
                f"({100 * st['changed_frac']:.2f} %) worst_matrix_rel={st['worst_rel']:.4g} "
                f"(expert, matrix)={st['worst_matrix']}{' (cached)' if hit else ''}"
            )
        logger.info(
            f"EXPERTS_LOAD layer={self.layer} impl={self.impl} cache={'hit' if hit else 'miss' if prefix else 'off'} "
            f"total={time.time() - t0:.1f} s host_weights={0.0 if hit else self._host_s:.1f} s"
        )

    def _host_weights(self, ck, gtab):
        """The weights FlatRoutedExpert lays out: [device r cols + c][le] = (W1, W3, W2) as stored, bf16 (exact), each
        slot's 32-chip stack dequantized in one go (fp4_dequant over the stacked packed bytes: the same per-element
        arithmetic), checked once per stacked tensor: bf16-exact (the fp32 values' low 16 bits zero) and, bfp8, held
        exactly by bfloat8_b (_bfp8_exact); bfp4: `bfp4_stats` from the fp32 stacks (no copies)."""
        import time

        t0 = time.time()
        rows, cols, layer, H, I = self.dgs, self.ngroups, self.layer, self.H, self.I
        n = rows * cols
        flat_w = [[None] * self.epc for _ in range(n)]
        changed = total = 0
        worst = (0.0, None)
        pool = _read_pool()
        for le in range(self.epc):
            ids = [int(gtab[c, r, le]) for c in range(cols) for r in range(rows)]  # chip order (c, r)
            parts = list(pool.map(lambda e: ck.expert(layer, e), ids))
            w = {}
            for nm in NAMES:
                packed = torch.stack([p[nm][0] for p in parts])  # [n, N, K/2] uint8
                scale = torch.stack([p[nm][1] for p in parts])  # [n, N, K/32] e8m0
                N_ = packed.shape[1]
                w[nm] = fp4_dequant(packed.reshape(n * N_, -1), scale.reshape(n * N_, -1)).view(n, N_, -1)
                assert _bf16_exact(w[nm]), f"layer {layer} slot {le} {nm}: e2m1 x 2^e not bf16-exact"
            del parts
            assert w["w1"].shape[1:] == (I, H) and w["w2"].shape[1:] == (H, I), (w["w1"].shape, w["w2"].shape)
            if self.impl == "bfp8":
                for nm in NAMES:
                    assert _bfp8_exact(w[nm]), f"layer {layer} slot {le} {nm}: bfloat8_b would round a value (DS41-35)"
            else:
                for nm in NAMES:
                    n_ch, rel = _bfp4_change(w[nm])
                    changed, total = changed + n_ch, total + w[nm].numel()
                    v, i = rel.max(dim=0)
                    if float(v) > worst[0]:
                        c, r = divmod(int(i), rows)
                        worst = (float(v), (int(gtab[c, r, le]), nm))
            w13 = torch.empty(n, 2 * I, H, dtype=torch.bfloat16)
            w13[:, :I] = w.pop("w1")
            w13[:, I:] = w.pop("w3")
            w2 = w.pop("w2").to(torch.bfloat16)
            for k in range(n):
                c, r = divmod(k, rows)
                flat_w[r * cols + c][le] = (w13[k, :I], w13[k, I:], w2[k])
        if self.impl == "bfp4":
            self.bfp4_stats = {
                "layer": layer,
                "changed": changed,
                "total": total,
                "changed_frac": changed / total,
                "worst_rel": worst[0],
                "worst_matrix": list(worst[1]),
            }
        self._host_s = time.time() - t0
        return flat_w

    def _load_flat(self, flat_w, gtab, cache_prefix=None):
        """Every local expert's (W1, W3, W2) as stored ([N, K], bf16-exact e2m1 x 2^e) in the flat op's per-core bank
        layout (FlatRoutedExpert, weights_nk: each tile the transpose of x @ W's tile, so a bfp8 tile shares its
        exponents along K: exact, the groups _load_block asserts on [N, K]; bfp4: ttnn's lossy packing of the same
        groups). Built once per layer; the plan is the op's for this chip's grid (16-column units). The column's
        planner numbers the column's experts from 0 (chip r holds [r epc, (r + 1) epc) of its column's Eg;
        ExpertMapping column-major), counts are [1, Eg]; an expert sees at most every token of its column once."""
        from .weighted_combine import weighted_combine

        rows = self.dgs
        self.fe = self.flat_op(flat_w, cache_prefix)
        rows_ = self.max_chunk // rows
        self.wc = weighted_combine(
            self.mesh,
            dict(self.fe.plan),
            H=self.H,
            I=self.I,
            epc=self.epc,
            K=self.K,
            S=rows_,
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

    def flat_op(self, flat_w, cache_prefix=None):
        """The layer's FlatRoutedExpert: the weights laid out once (or loaded from the weight cache). Also used by
        tt/moe_block_adapter.py (MOE_IMPL=block): the same arguments, so the same cache files."""
        from ttnn.bringup.combine_rf.sparse_combine_rf import L1_MAX_WRITERS
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert
        from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import LEGACY

        rows, cols = self.dgs, self.ngroups
        gids = [[r * self.epc + le for le in range(self.epc)] for r in range(rows) for c in range(cols)]
        self.flat_cap = max(256, _ceil(self.max_chunk, TILE))  # (the op is built for >= 256 rows per expert)
        # EXPERTS_LAYOUT: legacy = the flat op's plan before its DRISC-only / even-split defaults (the cached
        # weight layout, FIX.4); marko = his defaults + moe_block.layout_of's rows_hint / x_helpers
        self.flat_layout = (
            dict(LEGACY)
            if settings.get("EXPERTS_LAYOUT") == "legacy"
            else dict(rows_hint=round(self.max_chunk * self.K / self.E), x_helpers=1)
        )
        fe = FlatRoutedExpert(
            self.mesh,
            flat_w,  # (a callable: called only when the weight cache misses)
            m=self.flat_cap,
            H=self.H,
            I=self.I,
            gids=gids,
            n_global=self.Eg,
            wdtype={"bfp8": "bf8", "bfp4": "bf4"}[self.impl],
            act="clamped_silu",
            pin=1,
            # EXPERTS_PRECISION=glm: bfp8 x / h tiles, as GLM-5.3 runs the op (GLM_AG_XBF16 / _HBF16 default 0)
            x_bf16=self.precision == "full",
            h_bf16=self.precision == "full",
            weights_nk=True,
            # (no second exactness pass: a laid-out tile row is 32 consecutive K values of one stored row, the same
            # 16-value groups _bfp8_exact asserted on the [N, K] matrices above)
            check_exact=False,
            cache_prefix=cache_prefix,
            layout=self.flat_layout,
            # (at most the combine's L1 hand-off writers: a Galaxy chip with its east DRAM readers in column 7 has 40
            # down cores under the legacy layout, column 6 has 32; capped, both get the 32-core down split and the same
            # weight cache key. The marko layout's even split already keeps at most 32 down cores and no reader tails:
            # a no-op there)
            max_y_writers=L1_MAX_WRITERS,
        )
        logger.info(
            f"EXPERTS_FLAT layer={self.layer} plan nu={fe.plan['nu']} np={fe.plan['np']} "
            f"mt={fe.plan['mt']} gate/up={len(fe.plan['gu'])} down={fe.plan['nd']} cap={self.flat_cap} "
            f"precision={self.precision}"
        )
        return fe

    def partial(
        self, x: ttnn.Tensor, dense: ttnn.Tensor | None, idx: ttnn.Tensor | None = None, wts: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        """This column group's experts over the chip's rows (tt/weighted_combine.py): q(x) of the mesh row gathered
        over the column, the column's top-k ids and bf16 weights, then the flat expert + weighted combine in one
        program per pass. x: ffn_norm [1, 1, s, H] (rows split over axis 0, replicated over axis 1; bf16 or fp32);
        dense: routing [1, 1, s, E] fp32 (same placement; may be None when idx and wts are given); idx: the router's
        top-k ids [1, 1, s, K] (None: top-k of dense); wts: the router's weights of idx [1, 1, s, K] (INDEXER_GLUE=on).
        Returns [1, 1, s, H] (the group's sum, weights applied; bf16 under CCL_IMPL=fabric_bf16, else fp32)."""
        s, K = x.shape[-2], self.K
        wc = self.wc
        assert s == wc.S, f"the experts are built for {wc.S} rows per chip, got {s}"
        qx = cast(x, ttnn.bfloat16)  # (x itself when ffn_norm wrote bf16)
        xg = wc.gather_x(qx)
        if qx is not x:
            dealloc(qx)
        given = wts is not None and idx is not None  # INDEXER_GLUE=on (P.16): the router's own weights, as they are
        if given:
            ids = idx
        else:
            dk = (
                dense
                if dense.layout == ttnn.TILE_LAYOUT
                else ttnn.to_layout(dense, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(dense))
            )
            if idx is None:  # (the routing's K nonzero weights and their ids; order is free: the combine sums them)
                wts, ids = ttnn.topk(dk, k=K, dim=-1, largest=True, sorted=True)
            else:
                ids = idx
                wts = ttnn.gather(dk, dim=-1, index=idx)  # = the router's wts (dense is their scatter)
            if dk is not dense:
                dealloc(dk)
        ids16 = (
            ids
            if ids.dtype == ttnn.uint16
            else ttnn.typecast(ids, ttnn.uint16, memory_config=DRAM, sub_core_grids=sub_cores_tiled(ids))
        )
        ids_g, wts_g = wc.routing(ids16, wts)
        passed = (ids_g, wts_g)  # (routing hands its inputs through on a one-row mesh / bf16 weights: freed after)
        for t in {id(t): t for t in (ids16, wts) + ((ids,) if idx is None else ())}.values():
            if t is not idx and not (given and t is wts) and not any(t is p for p in passed):
                dealloc(t)  # (the router's outputs: the block frees them)
        out = wc(self.fe, xg, ids_g, wts_g, int(self.fidelity.value))
        for t in passed:
            if t is not idx and not (given and t is wts):
                dealloc(t)
        if self.keep_rows:  # (the shared expert's x: take_rows hands it over, the shared expert frees it)
            if self._rows is not None:
                dealloc(self._rows)
            self._rows = xg
        else:
            dealloc(xg)
        return out

    def take_rows(self) -> ttnn.Tensor | None:
        """The last call's gathered row-major q(x) of the column [R s, H] bf16 (keep_rows), handed over once
        (the caller frees it); None when there is none."""
        rows, self._rows = self._rows, None
        return rows

    def reduce(self, part: ttnn.Tensor, fp32: bool = True) -> ttnn.Tensor:
        """Sum the column groups' partials [1, 1, s, H] fp32 over axis 1 and keep the chip's hidden columns
        [1, 1, s, H/TP] (gather_add: exact fp32 on the SFPU, K-30; reduce_scatter: one CCL, FPU). ``fp32`` False
        (GLUE_FOLD=on, P.19: the residual's mhc_post takes bf16): the bf16 reduce-scatter's result as is."""
        if self.ngroups == 1:
            return ttnn.clone(part, memory_config=DRAM)
        if self.sum_mode == "reduce_scatter":
            return ttnn.reduce_scatter(part, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        return sum_cols(self.mesh, part, fp32=fp32)

    def free(self):
        """Release the device weights (tests that build several layers' or impls' experts in one process)."""
        fe = getattr(self, "fe", None)
        if fe is not None:  # the flat op's bank-layout weights, expert ids and done words
            for t in (fe.w_gu, fe.w_d, fe.w_rd, fe.gidx, fe.done):
                if t is not None:
                    dealloc(t)
            self.fe = None

    def __call__(self, x: ttnn.Tensor, dense: ttnn.Tensor, idx: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """experts_out [1, 1, s, H/TP] fp32 (rows over axis 0, hidden columns over axis 1)."""
        part = self.partial(x, dense, idx)
        out = self.reduce(part)
        dealloc(part)
        return out


def build_experts(
    mesh,
    ck,
    cfg,
    layer: int,
    chunk: int,
    impl: str | None = None,
    combine_acc: str | None = None,
    combine_planner: str | None = None,
    cache_dir=None,
) -> TtExperts:
    """The routed experts of one layer (reference Config: n_experts, topk_experts, dim, inter, swiglu_limit); cache_dir:
    the laid-out weight cache (expert_cache_dir(spec); None: none)."""
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
        combine_acc=combine_acc,
        combine_planner=combine_planner,
        cache_dir=cache_dir,
    )
