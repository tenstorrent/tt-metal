# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC hyper-connections for GLM-5.3-Flash, on top of the DeepSeek-V4 implementation.

GLM-5.3-Flash's ``Glm5NextTextHyperConnection`` is the DeepSeek-V4 parametrization exactly:
unweighted RMSNorm over the 4 packed streams, one fused projection ``fn [24, 4*4096]``, then
``pre = sigmoid(a_pre*raw + b) + eps``, ``post = 2*sigmoid(a_post*raw + b)``,
``comb = sinkhorn(softmax(a_res*raw + b) + eps)`` with one column normalisation followed by
``iters-1`` (row, column) pairs. ``TtMHCWrap`` runs that in fp32 with the fused
``mhc_split_sinkhorn`` kernel, so this module only adapts the configuration and the tensor
convention.

Tensor convention (from ``tt_mhc``): residual streams are packed on the last dim,
``[1, 1, T, 4*4096]`` fp32 with stream ``i`` at columns ``[i*4096, (i+1)*4096)`` and the
``T = B*S`` tokens down the rows. HF keeps ``[B, S, 4, 4096]``; the conversion is a reshape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _compute_kernel_config  # noqa: F401  (re-exported)
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand  # noqa: F401  (re-exported: tt/embedding.py)
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap, W  # noqa: F401  (re-exported)
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _cols as _composite_cols
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _mix as _composite_mix
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _project as _composite_project
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _streams as _composite_streams
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _upload, build_consts  # noqa: F401  (re-exported)
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import MhcPostMix
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import MhcPreBlock
from models.demos.glm53_flash.tt.config import GLM53FlashConfig

TILE = 32
# K-split of the decode projection: the [T,16384] x [16384,32] contraction has a single output tile,
# so the only parallelism is over K; 32 batches of 512 put it on 32 cores
KS_DECODE = 32
# "fast" (TtMHCFast, this module) | "wrap" (the DeepSeek composite TtMHCWrap, the reference path)
# fast (the exact custom-op decode path; default since 2026-10-21: on the 8x4 it halves the decode step, 75.5 vs
# 144.1 ms at 32 users, at the same 45-layer chain PCC, and the 16 K / 64 K needle runs stay token-identical to HF)
# with the composite wrap on the prefill rows (MHC_PREFILL_IMPL); wrap = the composite everywhere (A/B)
MHC_IMPL = os.environ.get("GLM53_MHC_IMPL", "fast")
# TtMHCFast residual update: 0 = one HiFi4 matmul (bf16-rounded operands, ~1e-3 rel rms per wrap); 3 (or 1) =
# exact: both operands split into 3 bf16 parts, the 6 significant products in one stacked-K matmul (3e-9);
# 2 = the cheaper 2-way split, 3 products (3e-6). See TtMHCFast._exact_mm.
MHC_EXACT = {"0": 0, "1": 3, "2": 2, "3": 3}[os.environ.get("GLM53_MHC_EXACT", "0")]
# 1/2: the fused exact fp32 post-block kernel custom_ops/mhc_post_mix (a model-local ttnn.generic_op: one program
# replacing the coefficient extraction, the block-diagonal matmul and the addcmul; exact like the composite
# chain); 2 = the v2 kernel (resident operands, x8 coefficient tiles, SFPU FMAs) reading the bf16 sublayer output
# directly, so the post-block is the single op
MHC_POST_KERNEL = int(os.environ.get("GLM53_MHC_POST_KERNEL", "2"))
# prefill (T > 32) pre-block: 1 = the composite's projection and exact fp32 mix, op for op (default; the
# accuracy gate); 0 = the earlier stream-major n-split matmul + fast_reduce_nc path (A/B only)
MHC_PREFILL_EXACT = os.environ.get("GLM53_MHC_PREFILL_EXACT", "1") == "1"
# decode (T <= 32) pre-block: 1 = exact fp32 reductions of the K-split partials (ttnn.sum, SFPU) instead of
# fast_reduce_nc (bf16 rounding) and h through the post-mix kernel's pre-mix mode + the composite's typecast +
# rms_norm (default, needs GLM53_MHC_POST_KERNEL=2); 0 = fast_reduce_nc + the P@x matmul + sharded norm (A/B)
MHC_DECODE_EXACT = os.environ.get("GLM53_MHC_DECODE_EXACT", "1") == "1"
# 1: the fused exact pre-block custom op (custom_ops/mhc_pre_block) for decode wraps with the fused norm: one
# program replaces the projection, Sinkhorn, mix and norm ops and hands the post-block its lane-layout
# coefficients (needs GLM53_MHC_POST_KERNEL=2)
# (2: the exact SFPU projection variant of the op)
# Defaults since 2026-10-28: the fused pre-block (PRE_BLOCK=1) with the fused Sinkhorn and the "extras" splits passed
# the 45-layer 8x4 logits gate (pcc 0.9952, top-1 93.0 %, top-5 95.5 % vs 0.9946 / 93.0 / 93.1 for the former defaults;
# PRE_BLOCK=1 alone failed it at 0.8455) and one e2e run: traced decode 61.5 ms (16.3 tok/s per user at 32 users, 520
# aggregate) vs 70.35 ms, warm 128-token prefill 866 vs 905 ms, chain pcc 0.947 vs 0.862 (_logs/mesh/sched_validate.log,
# logits_8x4_mhcknobs + e2e_8x4_B4_mhcknobs). The former values stay selectable through the same variables.
MHC_PRE_BLOCK = int(os.environ.get("GLM53_MHC_PRE_BLOCK", "1"))
# GLM53_MHC_PREFILL_IMPL=wrap (with GLM53_MHC_IMPL=fast): the composite wrap serves the prefill rows, the fast path
# the decode tile (A/B of the prefill numerics; see TtMHCSplit)
MHC_PREFILL_IMPL = os.environ.get("GLM53_MHC_PREFILL_IMPL", "wrap" if MHC_IMPL == "fast" else "")
# A3 (decode ledger): the sublayer's input RMSNorm in the exact decode pre-block. "sharded" = typecast to bf16,
# interleaved_to_sharded into the width-sharded L1 layout over the 8x4 worker grid, the sharded multicore
# rms_norm (program config pc_norm), sharded_to_interleaved back (4 ops, 32 cores); "sharded_cast" = the same
# with the fp32 -> bf16 cast folded into the reshard (3 ops; A/B of the reshard kernel's rounding); "single" =
# the composite's typecast + interleaved rms_norm (2 ops, ONE core at T <= 32: 46 us per wrap, the pre-A3 path)
MHC_NORM = os.environ.get(
    "GLM53_MHC_NORM", "sharded"
)  # the 32-core norm (A3); single = the pre-A3 one-core norm. Default
# since 2026-10-26: chip-0 A/B PCC 0.999998 vs single (traced wrap 246 -> 210 us), block decoder + e2e, 8x4 e2e 80.6 ms at
# 32 users, logits gate 0.9948 / top-1 95.3 %, 64 K needle 32/32
NORM_MODES = ("sharded", "sharded_cast", "single")
assert MHC_NORM in NORM_MODES, MHC_NORM
# The norm mode of a fast-path call inside a prefill scope (``GLMMesh.prefill_scope``: ``decoder_layer.prefill_forward``
# and the MTP prefill enter it). A 128-token prompt on 8 rows is 16-row slabs = one tile, which ``TtMHCSplit.select``
# routes to the fast path, so the sharded norm's 4 ops ran inside the traced prefill: measured on the 8x4
# (2026-10-26, 45 layers, 4 users/row, 128-token prompt, one run each) GLM53_MHC_NORM=sharded -> decode 79.75 ms /
# TTFT 297 ms, =single -> decode 83.06 ms / TTFT 266 ms. "single" (default): prefill-scope calls take the one-core
# norm, everything else the configured GLM53_MHC_NORM (``effective_norm_mode``); "same": the configured mode
# everywhere (A/B). ``decoder_layer._hc_call_impl`` passes ``prefill=self.mesh.in_prefill``.
MHC_PREFILL_NORM = os.environ.get("GLM53_MHC_PREFILL_NORM", "single")
assert MHC_PREFILL_NORM in ("single", "same"), MHC_PREFILL_NORM


def effective_norm_mode(configured: str, *, prefill: bool = False, override: str | None = None) -> str:
    """The norm mode one fast-path call runs: an explicit per-call ``override`` wins; otherwise ``single`` inside a
    prefill scope (``MHC_PREFILL_NORM == "single"``), the ``configured`` mode (the instance's) outside. Pure function of
    its arguments and the module knob: a captured trace replays the mode it was captured in."""
    if override is not None:
        assert override in NORM_MODES, override
        return override
    assert configured in NORM_MODES, configured
    if prefill and MHC_PREFILL_NORM == "single":
        return "single"
    return configured


# A4 (decode ledger): the parametrization (pre / post / Sinkhorn comb). "kernel" = the fused mhc_split_sinkhorn op
# (one program per token tile: all T <= 32 users' 4x4 matrices sit in ONE tile and every normalisation is one
# matmul + recip + mul on that tile, 52 us on one core); "batched" = that tile sequence as composite ttnn ops
# (parametrize_graph: each op sees all users at once; 131 ops at iters=20). The composite cannot beat the
# single-program kernel at one tile of data (op count is the cost there), so it is the A/B / reference path,
# not the default; the lever on the 52 us is the kernel's per-tile sequence (ledger A4, a kernel change).
MHC_SINKHORN = os.environ.get("GLM53_MHC_SINKHORN", "kernel")
assert MHC_SINKHORN in ("kernel", "batched"), MHC_SINKHORN
# ---- mHC roofline knobs (_agents/mhc_roofline/STATUS.md); every default is today's path
# A9: the two K-split reductions of the composite decode projection (part [1,32,32,32] and ssp [1,32,32,1], summed
# over the 32 batches). "exact" = ttnn.sum(dim=1) (SFPU fp32; transpose + reduce + transpose [+ fill_pad]: 7 ops for
# both, 25 us); "fast" = ttnn.experimental.fast_reduce_nc (2 ops, ~4 us): its FPU add unpacks every fp32 partial through
# SrcA at 19 bits, a 2^-11 relative rounding per partial (~4.9e-4 rel on the mixes, the comb coefficients ~1e-5 abs;
# tests/test_mhc_roofline_host.py). The composite decode pre-block only: the fused pre-block (A1) sums its partials
# exactly on the SFPU and has none of these ops. Honoured only with GLM53_MHC_DECODE_EXACT=1 (=0 already uses the fast
# reduce).
MHC_KSUM = os.environ.get("GLM53_MHC_KSUM", "exact")
assert MHC_KSUM in ("exact", "fast"), MHC_KSUM
# A4 inside the fused pre-block: "llk" = the mhc_split_sinkhorn tile sequence (one LLK op + CB round trip per step, ~130
# ops on the coordinator core, ~52 us per wrap); "fused" = one dest pass per normalisation (matmul, [+eps], recip,
# m copied in, SFPU multiply: 2*iters + 3 passes, ~20 us expected). A bounded rounding change (fp32 SFPU products where
# the LLK sequence multiplied through 19-bit FPU operands): logits gate required before it becomes the default.
MHC_PRE_SINKHORN = os.environ.get("GLM53_MHC_PRE_SINKHORN", "fused")
assert MHC_PRE_SINKHORN in ("llk", "fused"), MHC_PRE_SINKHORN
# the fused pre-block's work split: "cols" = 128 column tiles over the grid (two on each of the first 8 cores of a 12x10
# grid: they bound the projection phase); "extras" = a two-column core projects only its first column, the second's 4
# stream tiles are projected by 4 single-column non-leader cores (12x10: critical path 10 -> 7 cost units; 13x10 / 8x8
# unchanged by construction). Same partial sums, grouped differently: bit-level agreement is not guaranteed (fp32 adds
# in a different order), PCC-level is.
MHC_PRE_SPLIT = os.environ.get("GLM53_MHC_PRE_SPLIT", "extras")
assert MHC_PRE_SPLIT in ("cols", "extras"), MHC_PRE_SPLIT
# the post-mix kernel's work split: "tile" = one (token-tile, column-tile) position per unit (128 over 120 cores: 8
# cores take two); "extras" = those 8 cores' second positions split by output stream over 32 one-position cores
# (critical path 12 -> 9 cost units; every output tile is still computed by exactly one core: bit-identical)
MHC_POST_SPLIT = os.environ.get("GLM53_MHC_POST_SPLIT", "extras")
assert MHC_POST_SPLIT in ("tile", "extras"), MHC_POST_SPLIT
# D13: the residual-stream layout over the TP columns. "replicated" = every column chip holds the packed [B, 4*C]
# streams (today); "colsharded" = each column chip holds its C/4 columns of every stream (design only: see
# _agents/mhc_roofline/STATUS.md D13 -- after A1 the wrap's cost is the serial Sinkhorn + the gather tree, not the
# elementwise 2 MB, and the two collectives it adds per wrap outweigh the /4; TtMHCFast refuses it at construction)
MHC_LAYOUT = os.environ.get("GLM53_MHC_LAYOUT", "replicated")
assert MHC_LAYOUT in ("replicated", "colsharded"), MHC_LAYOUT


@dataclass(frozen=True)
class MHCKernelConfig:
    """The fields ``TtMHCWrap`` reads (``MHCConfig`` in the DeepSeek-V4 reference)."""

    dim: int
    n: int
    sinkhorn_iters: int
    eps: float
    norm_eps: float

    @property
    def mix_hc(self) -> int:
        return (2 + self.n) * self.n


def mhc_kernel_config(cfg: GLM53FlashConfig) -> MHCKernelConfig:
    # HF: input_norm = UnweightedRMSNorm(eps=rms_norm_eps); hc_eps is the stochasticity eps.
    return MHCKernelConfig(
        dim=cfg.hidden_size,
        n=cfg.hc_mult,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )


def build_mhc(
    device, cfg: GLM53FlashConfig, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, impl: str | None = None
):
    """One hyper-connection (attention or FFN) from the raw checkpoint tensors
    ``hc_*_fn [24, 16384]`` (bf16), ``hc_*_base [24]`` and ``hc_*_scale [3]`` (fp32).
    ``impl``: "fast" (``TtMHCFast``) or "wrap" (``TtMHCWrap``) returns exactly that class (the unit tests); ``None``
    (the model) takes ``GLM53_MHC_IMPL`` and, with ``GLM53_MHC_PREFILL_IMPL=wrap``, the ``TtMHCSplit`` pair."""
    kcfg = mhc_kernel_config(cfg)
    assert tuple(fn.shape) == (kcfg.mix_hc, cfg.hc_mult * cfg.hidden_size), fn.shape
    assert base.numel() == kcfg.mix_hc and scale.numel() == 3
    explicit = impl is not None
    impl = impl or MHC_IMPL
    cls = {"fast": TtMHCFast, "wrap": TtMHCWrap}[impl]
    hc = cls(device, kcfg, fn.float(), base.float(), scale.float(), dtype=ttnn.float32)
    if not explicit and impl == "fast" and MHC_PREFILL_IMPL == "wrap":
        # A/B: composite wrap for prefill rows, fast path for decode (one tile of rows)
        return TtMHCSplit(hc, TtMHCWrap(device, kcfg, fn.float(), base.float(), scale.float(), dtype=ttnn.float32))
    return hc


class TtMHCSplit:
    """Two hyper-connection implementations behind one object: ``decode`` (fast path) for a single tile of
    rows (the decode batch), ``prefill`` (the composite wrap) for anything taller. ``decoder_layer._hc_call_impl``
    resolves the implementation per call through ``select``; ``supports_collapse`` is the decode path's
    (the collapse is decode-only). ``GLM53_MHC_PREFILL_IMPL=wrap`` with ``GLM53_MHC_IMPL=fast``."""

    def __init__(self, decode, prefill):
        self.decode, self.prefill = decode, prefill

    def select(self, x: ttnn.Tensor):
        return self.decode if x.padded_shape[-2] == TILE else self.prefill

    @property
    def supports_collapse(self) -> bool:
        return bool(getattr(self.decode, "supports_collapse", False))


def streams_hf_to_packed(x: torch.Tensor) -> torch.Tensor:
    """HF ``[B, S, n, C]`` -> packed ``[1, 1, B*S, n*C]`` (host side, tests only)."""
    B, S, n, C = x.shape
    return x.reshape(1, 1, B * S, n * C)


def streams_packed_to_hf(x: torch.Tensor, batch: int, n: int) -> torch.Tensor:
    """Packed ``[1, 1, B*S, n*C]`` -> HF ``[B, S, n, C]`` (host side, tests only)."""
    T, nC = x.shape[-2], x.shape[-1]
    return x.reshape(batch, T // batch, n, nC // n)


# ----------------------------------------------------------------------------------------------
# TtMHCFast: the TtMHCWrap arithmetic without the data-movement composites
# ----------------------------------------------------------------------------------------------
def _rows_up(T: int) -> int:
    return -(-T // TILE) * TILE


def _view_rows(t: ttnn.Tensor, rows: int) -> ttnn.Tensor:
    """Re-declare the logical row count of a ``[1,1,T,Wd]`` TILE tensor inside its padded tile rows
    (zero-copy; ``rows`` and ``T`` round up to the same padded height). Widening (``T`` -> 32)
    makes the pad rows ordinary rows so every stage runs whole tiles with no padding bookkeeping;
    narrowing restores the contract shape. Both go through ``ttnn.reshape``'s explicit-padded-shape
    view (no device op)."""
    T, Wd = t.shape[-2], t.shape[-1]
    if rows == T and len(t.shape) == 4:
        return t
    Tp, Wp = t.padded_shape[-2], t.padded_shape[-1]
    assert _rows_up(rows) == Tp, (rows, Tp)
    return ttnn.reshape(t, ttnn.Shape([1, 1, rows, Wd]), ttnn.Shape([1, 1, Tp, Wp]))


def _view_tile_width(t: ttnn.Tensor) -> ttnn.Tensor:
    """A ``[Tp,k]`` kernel output with ``k <= 32`` as ``[1,1,Tp,32]``: the whole tile incl. its (finite)
    pad columns, so the elementwise ops below see no padding (view)."""
    Tp = t.shape[-2]
    assert t.padded_shape[-1] == W and Tp % TILE == 0, (t.shape, t.padded_shape)
    return ttnn.reshape(t, ttnn.Shape([1, 1, Tp, W]), ttnn.Shape([1, 1, Tp, W]))


# ----------------------------------------------------------------------------------------------
# The parametrization (pre / post / Sinkhorn comb) as composite ops: GLM53_MHC_SINKHORN=batched
# ----------------------------------------------------------------------------------------------
# build_consts' tile order (the host <-> kernel contract of mhc_split_sinkhorn)
CONST_TILES = ("sel_pre", "sel_post", "sel_comb", "base_pre", "base_post", "base_comb", "rb", "cb")
# the kernel caps the comb logits at 80 before exp (no inf/inf; a no-op for any realistic logit)
EXP_CAP = 80.0


def const_tile_specs(consts_t: torch.Tensor) -> dict[str, tuple[torch.Tensor, tuple[int, ...]]]:
    """The 8 ``build_consts`` tiles ``[8, 32, 32]`` as the batched graph's operands, by ``CONST_TILES`` name:
    the selection / RB / CB tiles whole ``[1,1,32,32]``; the base tiles (one bias row replicated down the tile,
    the kernel's tile + tile add) as the single row ``[1,1,1,32]`` the composite adds with a row broadcast, so
    the graph runs at any token count."""
    out = {}
    for i, name in enumerate(CONST_TILES):
        t = consts_t[i]
        out[name] = (t[:1], (1, 1, 1, W)) if name.startswith("base_") else (t, (1, 1, W, W))
    return out


def parametrize_graph(ops, mixes, k: dict, iters: int, eps: float):
    """The ``mhc_split_sinkhorn`` kernel's tile sequence as a backend-independent graph over a whole token
    tile: every op sees all ``T <= 32`` users at once, the Sinkhorn of their ``n x n`` matrices being the
    matmul form ``m / (m @ K + eps)`` on the packed ``[T, n*n]`` rows (``K`` = RB row-sum / CB col-sum
    broadcast, identity on the pad columns). One column normalisation of ``softmax + eps``, then
    ``iters - 1`` (row, column) pairs, ``eps`` in every denominator but the softmax's; ``pre = sigmoid(.) +
    eps``, ``post = 2 sigmoid(.)``.

    ``ops``: ``_TtnnOps`` (the device path) or ``_TorchOps`` (``parametrize_torch``: the host reference of
    tests/test_mhc_host_reference.py runs the very op order the device runs). ``mixes`` ``[.., T, 32]``
    (columns past mix_hc zero), ``k`` the 8 constant tiles of ``build_consts`` by ``CONST_TILES`` name.
    Returns (pre, post, comb) as full ``[.., T, 32]`` tiles (the first n / n / n*n columns are the
    coefficients). Op count at iters=20: 125 (comb) + 3 (pre) + 3 (post) = 131."""

    def normalize(m, kt, use_eps):
        s = ops.mm(m, kt)  # every entry's row / column sum, broadcast in place
        r = ops.recip_eps(s, eps) if use_eps else ops.recip(s)
        ops.free(s)
        out = ops.mul(m, r)
        ops.free(m)
        ops.free(r)
        return out

    z = ops.mm(mixes, k["sel_comb"])  # a_res * raw_comb, left-aligned
    lg = ops.add(z, k["base_comb"])
    ops.free(z)
    m = ops.exp_capped(lg, EXP_CAP)  # in place
    m = normalize(m, k["rb"], False)  # row softmax (no eps)
    m2 = ops.add_scalar(m, eps)
    ops.free(m)
    m = normalize(m2, k["cb"], True)  # the first column normalisation
    for _ in range(iters - 1):
        m = normalize(m, k["rb"], True)
        m = normalize(m, k["cb"], True)
    comb = m
    z = ops.mm(mixes, k["sel_pre"])
    s = ops.add_sigmoid(z, k["base_pre"])
    ops.free(z)
    pre = ops.add_scalar(s, eps)
    ops.free(s)
    z = ops.mm(mixes, k["sel_post"])
    s = ops.add_sigmoid(z, k["base_post"])
    ops.free(z)
    post = ops.mul_scalar(s, 2.0)
    ops.free(s)
    return pre, post, comb


class _TorchOps:
    """torch backend of ``parametrize_graph`` (host reference; ``count`` = device ops the graph would issue)."""

    def __init__(self):
        self.count = 0

    def _n(self, t):
        self.count += 1
        return t

    def mm(self, a, b):
        return self._n(a @ b)

    def add(self, a, b):
        return self._n(a + b)

    def add_scalar(self, a, c):
        return self._n(a + c)

    def mul(self, a, b):
        return self._n(a * b)

    def mul_scalar(self, a, c):
        return self._n(a * c)

    def recip(self, s):
        return self._n(torch.reciprocal(s))

    def recip_eps(self, s, eps):
        return self._n(torch.reciprocal(s + eps))

    def exp_capped(self, x, cap):
        self.count += 2  # clamp + exp
        return torch.exp(torch.clamp(x, max=cap))

    def add_sigmoid(self, z, b):
        return self._n(torch.sigmoid(z + b))

    def free(self, t):
        pass


class _TtnnOps:
    """ttnn backend: fp32 HiFi4 ops on ``[1,1,Tp,32]`` tiles (the kernel's FPU / SFPU sequences: HiFi4 matmul,
    SFPU recip / exp / sigmoid), temporaries deallocated as the graph consumes them (trace-safe: no host ops,
    every tensor transient within the step)."""

    def __init__(self, ckc):
        self.ckc = ckc
        self.count = 0

    def mm(self, a, b):
        self.count += 1
        return ttnn.matmul(a, b, compute_kernel_config=self.ckc)

    def add(self, a, b):
        self.count += 1
        return ttnn.add(a, b)

    def add_scalar(self, a, c):
        self.count += 1
        return ttnn.add(a, c)

    def mul(self, a, b):
        self.count += 1
        return ttnn.multiply(a, b)

    def mul_scalar(self, a, c):
        self.count += 1
        return ttnn.multiply(a, c)

    def recip(self, s):
        self.count += 1
        return ttnn.reciprocal(s)

    def recip_eps(self, s, eps):
        self.count += 1  # add + recip in DST, like the kernel's OP_RECIP_EPS
        return ttnn.add(s, eps, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RECIP)])

    def exp_capped(self, x, cap):
        self.count += 2
        ttnn.clamp(x, min=None, max=cap, output_tensor=x)
        return ttnn.exp(x, output_tensor=x)

    def add_sigmoid(self, z, b):
        self.count += 1
        return ttnn.add(z, b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)])

    def free(self, t):
        ttnn.deallocate(t)


def parametrize_torch(mixes: torch.Tensor, consts: torch.Tensor, iters: int, eps: float):
    """Host mirror of the batched device path: ``mixes [T, w]`` (``w <= 32``, zero-extended to the tile width),
    ``consts [8, 32, 32]`` from ``build_consts`` -> (pre, post, comb) ``[T, 32]`` tiles in ``mixes``' dtype and the
    op count the device graph issues."""
    T, w = mixes.shape
    assert w <= W, mixes.shape
    mx = torch.zeros(T, W, dtype=mixes.dtype)
    mx[:, :w] = mixes
    k = {name: t.reshape(shape[-2], shape[-1]).to(mixes.dtype) for name, (t, shape) in const_tile_specs(consts).items()}
    ops = _TorchOps()
    pre, post, comb = parametrize_graph(ops, mx, k, iters, eps)
    return pre, post, comb, ops.count


class TtMHCFast(LightweightModule):
    """mHC-wrapped sublayer with the ``TtMHCWrap`` semantics (fp32 tensors, HiFi4, the fused Sinkhorn
    kernel) and the composite stages rebuilt so that a decode step issues only compute ops (matmul,
    broadcast elementwise, reduce) on zero-copy views: no slice / untilize / tilize / concat / repeat /
    fill_pad. The one layout fact: a packed ``[1,1,T,n*C]`` TILE tensor whose rows fit one tile row
    (``T <= 32``) has exactly the tile order of the stream-major ``[1,n,T,C]`` tensor (stream ``i`` =
    tiles ``[i*C/32, (i+1)*C/32)``), of its ``[(i,t), c]`` matrix form ``[n*T, C]`` and of the K-split
    ``[1,KS,T,n*C/KS]`` tensor, so all three are ``ttnn.experimental.view`` reinterpretations.

    Decode step (``T <= 32`` logical rows, widened to the 32-row tile by ``_view_rows``), per wrap:

    pre-block (13 ops incl. the norm):
      1 ``part = xk @ fn_k``: K-split (KS=32 batches of 512) batched matmul, one core per batch;
      2 ``mixes_un = fast_reduce_nc(part)``;  3 ``xx = xk * xk``;  4 ``ssp = sum(xx, W)`` (32 cores);
      5 ``ss = fast_reduce_nc(ssp)``;  6 ``rs = rsqrt(ss + eps*nC)`` (one add with an rsqrt activation;
      ``sqrt(nC)`` is folded into ``fn_k``);  7 ``mixes = mixes_un * rs``;  8 ``mhc_split_sinkhorn``;
      9 ``v = pre_tile @ S_pre``, 10 ``P = v * I_pre``: the block-diagonal ``P[t, (i,t')] = delta(t,t')
      pre[t,i]`` ``[T, n*T]``;  11 ``h = P @ x_sm[(i,t), c]`` written bf16 width-sharded over 32 cores;
      12 ``rms_norm`` (sharded multicore program config, the sublayer's input norm, bf16 in/out like the
      composite path's typecast + norm);  13 ``sharded_to_interleaved`` (the sublayers take interleaved).
    post-block (8 ops; +1 ``where`` guard at ``T < 32``):
      1 ``y32 = typecast(y)``;  2-3 ``comb_c[j,i,t] = comb[t,i,j]`` and 4-5 ``post_c[j,t]`` as ``[.., T, 1]``
      columns by one mask multiply + one ``sum`` over W each (exact);  6 ``A = I_tile * comb_c``: the
      block-diagonal ``A[(j,t), (i,t')] = delta(t,t') comb[t,i,j]`` ``[n*T, n*T]`` (view of the
      ``[n,n,T,T]`` product);  7 ``mix = A @ x_sm[(i,t), c]``;  8 ``out = mix + post_c * y`` (one
      broadcast ``addcmul`` on the ``[1,n,T,C]`` view, which IS the packed output).

    Numerics (``_logs/optmhc_probe{2,3,4,5}.log``, vs float64): fp32 elementwise ops and ``ttnn.sum`` are
    exact (3e-8); the HiFi4 fp32 matmul rounds its operands to bf16 (9e-4 rel rms on the projection at
    K=n*C, the same op the composite path uses), so the block-diagonal mixes ``h`` and ``mix`` carry
    ~4e-4..1.2e-3 rel rms (``h`` is bf16 for the sublayer anyway; ``mix`` lands on the fp32 residual,
    where the composite path's elementwise chain is exact); ``fast_reduce_nc`` rounds to bf16 (4e-4 on the
    partial sums). The block-diagonal contraction runs over the pad rows too (``0 * NaN = NaN``), hence
    the contract: the input streams' pad rows are finite (host uploads zero-pad, the embedding gathers
    pad-token rows) and ``y``'s pad rows are zeroed by the ``where`` guard; the output's pad rows are
    finite garbage, not zeros.

    Exact decode pre-block (``MHC_DECODE_EXACT`` with ``post_kernel == 2``, the default; 18 ops per wrap with
    the fused norm, 16 before A3): the projection as above but with exact fp32 ``ttnn.sum`` chains for the two
    K-split reductions (7 device ops for both), ``mhc_split_sinkhorn`` (one program: pre / post / comb for every
    user of the token tile), ``h_mix`` (the post-mix kernel's pre-mix mode, exact fp32 ``h = sum_i pre_i x_i``),
    then the norm as ``_norm_decode``: ``typecast(h, bf16)``, ``interleaved_to_sharded`` (width-sharded over the
    8x4 grid, 4 tiles per core), ``rms_norm`` (sharded multicore program config), ``sharded_to_interleaved``
    (``GLM53_MHC_NORM``; ``single`` = the composite's interleaved ``rms_norm`` on one core, the pre-A3 path).
    ``GLM53_MHC_SINKHORN=batched`` runs the Sinkhorn program's tile sequence as 131 composite ops instead
    (``parametrize_graph``; A/B only).

    Prefill (``T > 32``): the stream-major form costs one tile-block transpose in and one out per wrap
    (``[T/32, n, 32, C] <-> [n, T/32, 32, C]``), the projection splits over the ``n`` streams, ``h`` is a
    broadcast multiply + ``fast_reduce_nc``, the residual update the exact fp32 broadcast ``addcmul``
    chain over the packed column slices (the block-diagonal matrices do not scale with T), and the norm
    the interleaved multicore one (``T/32`` tile rows).
    """

    fuses_norm = True  # forward(norm=(weight, eps)) applies the sublayer's input RMSNorm itself
    supports_collapse = True  # forward(collapse=True) emits the LM head's mean over the new streams

    def __init__(
        self,
        device,
        cfg,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        dtype=ttnn.float32,
        exact: int | None = None,
        post_kernel: bool | None = None,
        norm_mode: str | None = None,
        sinkhorn_mode: str | None = None,
        ksum_mode: str | None = None,
        pre_sinkhorn: str | None = None,
        pre_split: str | None = None,
        post_split: str | None = None,
        layout: str | None = None,
    ):
        assert dtype == ttnn.float32, f"TtMHCFast is fp32-only (fused op requires FLOAT32), got {dtype}"
        layout = MHC_LAYOUT if layout is None else str(layout)
        if layout != "replicated":
            raise NotImplementedError(
                f"GLM53_MHC_LAYOUT={layout}: the column-sharded residual (D13) is a design, not a path "
                "(_agents/mhc_roofline/STATUS.md); the residual contract through the embedding, the sublayer input "
                "assembly and the head collapse is not wired"
            )
        self.layout = layout
        self.device = device
        self.cfg = cfg
        self.exact = MHC_EXACT if exact is None else int(exact)  # residual update: 0 | 2 | 3 bf16 parts
        assert self.exact in (0, 2, 3), self.exact
        self.post_kernel = MHC_POST_KERNEL if post_kernel is None else int(post_kernel)  # 0 | 1 | 2 (version)
        assert self.post_kernel in (0, 1, 2), self.post_kernel
        # A3 / A4 knobs (module defaults from GLM53_MHC_NORM / GLM53_MHC_SINKHORN; constructor overrides for A/Bs)
        self.norm_mode = MHC_NORM if norm_mode is None else str(norm_mode)
        assert self.norm_mode in NORM_MODES, self.norm_mode
        self.sinkhorn_mode = MHC_SINKHORN if sinkhorn_mode is None else str(sinkhorn_mode)
        assert self.sinkhorn_mode in ("kernel", "batched"), self.sinkhorn_mode
        # roofline knobs (module defaults; constructor overrides for the A/Bs)
        self.ksum_mode = MHC_KSUM if ksum_mode is None else str(ksum_mode)
        self.pre_sinkhorn = MHC_PRE_SINKHORN if pre_sinkhorn is None else str(pre_sinkhorn)
        self.pre_split = MHC_PRE_SPLIT if pre_split is None else str(pre_split)
        self.post_split = MHC_POST_SPLIT if post_split is None else str(post_split)
        assert self.ksum_mode in ("exact", "fast"), self.ksum_mode
        assert self.pre_sinkhorn in MhcPreBlock.SINKHORN_MODES, self.pre_sinkhorn
        assert self.pre_split in MhcPreBlock.SPLIT_MODES, self.pre_split
        assert self.post_split in ("tile", "extras"), self.post_split
        self.n = int(cfg.n)
        # the fused post-block as a model-local generic op (custom_ops/mhc_post_mix); output allocated per call
        self.post_mix = (
            MhcPostMix(self.n, version=self.post_kernel, split=self.post_split if self.post_kernel == 2 else "tile")
            if self.post_kernel
            else None
        )
        # the fused pre-block (built on the first decode call with the fused norm: it holds the norm weight)
        self.pre_block_on = MHC_PRE_BLOCK != 0 and self.post_kernel == 2
        self._fn_raw = fn.float() if self.pre_block_on else None
        self.pre_block = None
        self._pre_block_norm = None
        self.C = int(cfg.dim)
        self.nC = self.n * self.C
        self.iters = int(cfg.sinkhorn_iters)
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.ckc = _compute_kernel_config()
        n, C, nC = self.n, self.C, self.nC
        mix_hc = (2 + n) * n
        assert mix_hc <= W and tuple(fn.shape) == (mix_hc, nC), fn.shape
        assert C % TILE == 0 and nC % KS_DECODE == 0 and (nC // KS_DECODE) % TILE == 0, (C, nC)
        assert C % (32 * TILE) == 0, f"the sharded norm splits C={C} over 32 cores in whole tiles"
        self.ks = KS_DECODE
        # projection weight, transposed, zero-padded to a full tile width (the kernel reads the first
        # mix_hc columns of its 32-wide input) and scaled by sqrt(nC): mixes = (x @ fn_T) * sqrt(nC) *
        # rsqrt(sum x^2 + eps*nC) = RMSNorm(x) @ fn_T. K-split views for decode (KS batches) and prefill
        # (n batches = the streams of the stream-major tensor)
        fn_t = torch.zeros(nC, W, dtype=torch.float32)
        fn_t[:, :mix_hc] = fn.float().t() * (nC**0.5)
        self.fn_k = _upload(device, fn_t, (1, self.ks, nC // self.ks, W), dtype)
        self.fn_s = ttnn.reshape(self.fn_k, [1, n, C, W])
        # prefill (T > 32): the composite's own projection weight and ops, so the Sinkhorn input is
        # bit-identical to TtMHCWrap's (accuracy gate: the prefill path must match the composite)
        self.fn_T = _upload(device, fn.float().t(), (1, 1, nC, mix_hc), dtype)
        self.eps_nc = self.norm_eps * nC
        consts_t = build_consts(cfg, scale, base)
        self.consts = _upload(device, consts_t, (8, W, W), dtype)
        # the batched (composite) parametrization reads the same 8 constant tiles as separate [1,1,32,32] tensors
        self.const_tiles = (
            {name: _upload(device, t, shape, dtype) for name, (t, shape) in const_tile_specs(consts_t).items()}
            if self.sinkhorn_mode == "batched"
            else None
        )
        # the K-split matmul's single output tile per batch: one core per batch (32 batches)
        self.pc_ksplit = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        )
        # coefficient masks: kernel output tile [T, 32] (k real columns) times mask[.., 1, 32], summed
        # over W -> one [.., T, 1] column per batch entry. pre -> [1, n(i)]; post -> [1, n(j)];
        # comb -> [n(j), n(i)] (comb[i, j] is column i*n + j)
        m_pre = torch.zeros(1, n, 1, W)
        m_post = torch.zeros(1, n, 1, W)
        m_comb = torch.zeros(n, n, 1, W)
        for i in range(n):
            m_pre[0, i, 0, i] = 1.0
            m_post[0, i, 0, i] = 1.0
            for j in range(n):
                m_comb[j, i, 0, i * n + j] = 1.0
        self.m_pre = _upload(device, m_pre, (1, n, 1, W), dtype)
        self.m_post = _upload(device, m_post, (1, n, 1, W), dtype)
        self.m_comb = _upload(device, m_comb, (n, n, 1, W), dtype)
        # collapse mode (the LM head's unweighted mean over the n new streams, folded into the
        # residual update): cbar_i = mean_j comb[i,j] -> [1, n(i)], pbar = mean_j post_j -> [1, 1]
        m_comb_mean = torch.zeros(1, n, 1, W)
        m_post_mean = torch.zeros(1, 1, 1, W)
        for i in range(n):
            m_post_mean[0, 0, 0, i] = 1.0 / n
            for j in range(n):
                m_comb_mean[0, i, 0, i * n + j] = 1.0 / n
        self.m_comb_mean = _upload(device, m_comb_mean, (1, n, 1, W), dtype)
        self.m_post_mean = _upload(device, m_post_mean, (1, 1, 1, W), dtype)
        # decode block-diagonal forms (T == 32 rows): P[t, (i,t')] = delta(t,t') pre[t,i] =
        # (pre_tile @ S_pre) * I_pre with S_pre[k, (i,t')] = delta(k,i); A[(j,t), (i,t')] =
        # delta(t,t') comb[t,i,j] = I_tile * comb_c[j,i,t] (broadcast over t', viewed as [n*T, n*T])
        Tt = TILE
        s_pre = torch.zeros(W, n * Tt)
        for i in range(n):
            s_pre[i, i * Tt : (i + 1) * Tt] = 1.0
        self.s_pre = _upload(device, s_pre, (1, 1, W, n * Tt), dtype)
        self.i_pre = _upload(device, torch.eye(Tt).repeat(1, n), (1, 1, Tt, n * Tt), dtype)
        self.i_tile = _upload(device, torch.eye(Tt), (1, 1, Tt, Tt), dtype)
        # pad-row guard for T < 32: y's pad rows -> 0 (the contraction above would spread a NaN)
        self._rowmask: dict[int, ttnn.Tensor] = {}
        # h: [T, n*T] @ [n*T, C] with the output bf16 width-sharded over 32 cores (4 tiles each) for the
        # sharded multicore RMSNorm; the sublayers take interleaved, hence the sharded_to_interleaved
        self.h_grid = ttnn.CoreGrid(y=4, x=8)
        self.h_mem = ttnn.create_sharded_memory_config(
            [Tt, C // 32],
            self.h_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.pc_h = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=n,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=C // (32 * TILE),
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.pc_norm = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            subblock_w=4,
            block_h=1,
            block_w=C // (32 * TILE),
            inplace=False,
        )
        # mix: [n*T, n*T] @ [n*T, C]
        self.pc_mix = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=n,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=n,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        # prefill (T > 32): the n-split projection has T/32 x n output tiles
        self.pc_proj_prefill = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        )

    # ------------------------------------------------------------------ layout helpers
    def _stream_major(self, x32: ttnn.Tensor) -> tuple[ttnn.Tensor, bool]:
        """``[1,1,Tp,nC]`` -> ``[1,n,Tp,C]``; returns (tensor, owned): a view for one tile row,
        a transposed copy otherwise."""
        Tp = x32.shape[-2]
        if Tp == TILE:
            return ttnn.experimental.view(x32, [1, self.n, Tp, self.C]), False
        blocks = ttnn.experimental.view(x32, [Tp // TILE, self.n, TILE, self.C])
        t = ttnn.transpose(blocks, 0, 1)  # [n, Tp/32, 32, C]: tile-block copy
        return ttnn.reshape(t, [1, self.n, Tp, self.C]), True

    def _packed(self, sm: ttnn.Tensor) -> ttnn.Tensor:
        """``[1,n,Tp,C]`` -> ``[1,1,Tp,nC]`` (the inverse of ``_stream_major``)."""
        Tp = sm.shape[-2]
        if Tp == TILE:
            return ttnn.experimental.view(sm, [1, 1, Tp, self.nC])
        blocks = ttnn.reshape(sm, [self.n, Tp // TILE, TILE, self.C])
        t = ttnn.transpose(blocks, 0, 1)  # [Tp/32, n, 32, C]
        ttnn.deallocate(sm)
        return ttnn.experimental.view(t, [1, 1, Tp, self.nC])

    def _coef(self, t: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
        """Kernel output ``[Tp,k]`` -> ``[R0, R1, Tp, 1]`` coefficient columns selected by ``mask``
        ``[R0, R1, 1, 32]`` (2 exact ops: broadcast multiply, sum over W)."""
        prod = ttnn.multiply(_view_tile_width(t), mask)  # [R0,R1,Tp,32]
        c = ttnn.sum(prod, dim=-1, keepdim=True, compute_kernel_config=self.ckc)
        ttnn.deallocate(prod)
        return c

    def _reduce_batches(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """``[1,K,Tp,w]`` -> ``[1,1,Tp,w]`` sum over the K-split batches: exact fp32 (``ttnn.sum``, SFPU; it
        transposes, 15 us on these tiny tensors) or ``fast_reduce_nc`` (A9, ``ksum_mode="fast"`` or
        ``MHC_DECODE_EXACT=0``: the FPU add unpacks every partial through the 19-bit SrcA, ~4.9e-4 rel on the sum)."""
        if MHC_DECODE_EXACT and self.ksum_mode == "exact":
            return ttnn.sum(t, dim=1, keepdim=True, compute_kernel_config=self.ckc)
        return ttnn.experimental.fast_reduce_nc(t, dims=[1], compute_kernel_config=self.ckc)

    def _rowmask_for(self, T: int) -> ttnn.Tensor:
        """``[1,1,32,1]`` fp32: 1 on the ``T`` real rows, 0 on the pad rows (built once per T)."""
        if T not in self._rowmask:
            m = torch.zeros(1, 1, TILE, 1)
            m[0, 0, :T, 0] = 1.0
            self._rowmask[T] = _upload(self.device, m, (1, 1, TILE, 1), ttnn.float32)
        return self._rowmask[T]

    def _bf16_parts(self, t: ttnn.Tensor, parts: int) -> list[ttnn.Tensor]:
        """fp32 ``t`` -> ``parts`` bf16 tensors whose sum is ``t`` to 8*parts mantissa bits (3 parts: all
        24, i.e. exact): typecast to bf16, widen back, subtract, repeat on the remainder (3*parts - 2 ops)."""
        out, rem = [], t
        for k in range(parts):
            hi = ttnn.typecast(rem, ttnn.bfloat16)
            out.append(hi)
            if k < parts - 1:
                hi32 = ttnn.typecast(hi, ttnn.float32)
                nxt = ttnn.subtract(rem, hi32)
                ttnn.deallocate(hi32)
                if k > 0:
                    ttnn.deallocate(rem)
                rem = nxt
        if parts > 1:
            ttnn.deallocate(rem)
        return out

    def _exact_mm(self, coef: ttnn.Tensor, x: ttnn.Tensor) -> ttnn.Tensor:
        """``coef [1,1,M,K] @ x [1,1,K,N]`` with the composite path's fp32 precision: the HiFi4 matmul is
        exact for bf16-exact operands, so both are split into ``self.exact`` bf16 parts (hi, mid, lo) and
        the products whose part indices sum to < parts (the ones above 2^-8*parts) are stacked along K
        into ONE bf16 matmul with fp32 accumulation. 3 parts: 6 products, 3e-9 rel rms vs float64; 2 parts:
        3 products, 3e-6 (`_logs/optmhc_probe6.log`)."""
        parts = self.exact
        pairs = [(a, b) for a in range(parts) for b in range(parts) if a + b < parts]
        cp = self._bf16_parts(coef, parts)  # tiny
        xp = self._bf16_parts(x, parts)  # the residual: 3*parts - 2 ops on 2 MB
        in0 = ttnn.concat([cp[a] for a, b in pairs], dim=-1)  # [M, K*len(pairs)]
        in1 = ttnn.concat([xp[b] for a, b in pairs], dim=-2)  # [K*len(pairs), N] bf16
        for t in cp + xp:
            ttnn.deallocate(t)
        kt = coef.shape[-1] // TILE * len(pairs)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=kt,
            out_subblock_h=1 if coef.shape[-2] == TILE else 2,
            out_subblock_w=2,
            per_core_M=coef.shape[-2] // TILE,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        out = ttnn.matmul(in0, in1, compute_kernel_config=self.ckc, program_config=pc, dtype=ttnn.float32)
        ttnn.deallocate(in0)
        ttnn.deallocate(in1)
        return out

    # ------------------------------------------------------------------ stages
    def _project(self, x32: ttnn.Tensor, x_sm: ttnn.Tensor) -> ttnn.Tensor:
        """``[1,1,Tp,nC]`` -> mixes ``[1,1,Tp,32]`` = RMSNorm(x) @ fn_T (zero columns past mix_hc)."""
        Tp = x32.shape[-2]
        if Tp != TILE and MHC_PREFILL_EXACT:
            # prefill: the composite's projection, op for op (matmul + sum of squares + rsqrt scale), so
            # mixes is bit-identical to TtMHCWrap.project; [1,1,Tp,24] (one tile wide)
            return _composite_project(x32, self.fn_T, self.norm_eps, self.ckc)
        if Tp == TILE:
            xk, fn_k, pc = ttnn.experimental.view(x32, [1, self.ks, Tp, self.nC // self.ks]), self.fn_k, self.pc_ksplit
        else:
            xk, fn_k, pc = x_sm, self.fn_s, self.pc_proj_prefill
        part = ttnn.matmul(xk, fn_k, compute_kernel_config=self.ckc, program_config=pc)  # [1,K,Tp,32]
        xx = ttnn.multiply(xk, xk)
        ssp = ttnn.sum(xx, dim=-1, keepdim=True, compute_kernel_config=self.ckc)  # [1,K,Tp,1]
        ttnn.deallocate(xx)
        mixes_un = self._reduce_batches(part)  # [1,1,Tp,32]
        # fast_reduce_nc keeps the padded width: re-declare the [Tp,1] column (view) so the scale
        # below is a column broadcast, not an elementwise product with rsqrt(eps) pad columns
        ss = ttnn.reshape(self._reduce_batches(ssp), ttnn.Shape([1, 1, Tp, 1]), ttnn.Shape([1, 1, Tp, W]))
        ttnn.deallocate(part)
        ttnn.deallocate(ssp)
        rs = ttnn.add(ss, self.eps_nc, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT)])
        ttnn.deallocate(ss)
        mixes = ttnn.multiply(mixes_un, rs)
        ttnn.deallocate(mixes_un)
        ttnn.deallocate(rs)
        return mixes

    def _collapse(self, x32: ttnn.Tensor, x_sm: ttnn.Tensor, pre: ttnn.Tensor, norm, norm_mode: str | None = None):
        """h ``[1,1,Tp,C]`` = sum_i pre_i x_i (fp32 interleaved), or with ``norm = (weight, eps)`` the
        sublayer's input ``RMSNorm(h)`` in bf16 interleaved (``norm_mode``: this call's A3 mode, default the
        instance's)."""
        Tp = x32.shape[-2]
        if Tp == TILE and MHC_DECODE_EXACT and self.post_kernel == 2:
            # exact fp32 h = sum_i pre_i x_i through the post-mix kernel's pre-mix mode (the P@x matmul rounds
            # its operands), then the composite's typecast + rms_norm so the sublayer input matches it
            h = self.post_mix.allocate_h(x32)
            self.post_mix.h_mix(x32, _view_rows(pre, Tp), h)
            if norm is None:
                return h
            return self._norm_decode(h, norm, norm_mode)
        if Tp == TILE:
            v = ttnn.matmul(_view_tile_width(pre), self.s_pre, compute_kernel_config=self.ckc)  # [1,1,Tp,nTp]
            P = ttnn.multiply(v, self.i_pre)
            ttnn.deallocate(v)
            res_v = ttnn.reshape(x_sm, [1, 1, self.n * Tp, self.C])  # [(i,t), c]
            if norm is None:
                h = ttnn.matmul(P, res_v, compute_kernel_config=self.ckc)
                ttnn.deallocate(P)
                return h
            weight, eps = norm
            h_sh = ttnn.matmul(
                P,
                res_v,
                compute_kernel_config=self.ckc,
                program_config=self.pc_h,
                memory_config=self.h_mem,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(P)
            hn = ttnn.rms_norm(
                h_sh,
                epsilon=eps,
                weight=weight,
                program_config=self.pc_norm,
                memory_config=self.h_mem,
                compute_kernel_config=self.ckc,
            )
            ttnn.deallocate(h_sh)
            h = ttnn.sharded_to_interleaved(hn, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hn)
            return h
        if MHC_PREFILL_EXACT:
            # prefill: the composite's exact fp32 mix (multiply + addcmul over stream slices; fast_reduce_nc
            # would round the partial products to bf16)
            pre4 = ttnn.reshape(pre, [1, 1, Tp, self.n])
            h = _composite_mix(_composite_streams(x32, self.n), _composite_cols(pre4, self.n))
        else:
            pre_c = self._coef(pre, self.m_pre)  # [1,n,Tp,1]
            prod = ttnn.multiply(x_sm, pre_c)  # [1,n,Tp,C]
            ttnn.deallocate(pre_c)
            h = ttnn.experimental.fast_reduce_nc(prod, dims=[1], compute_kernel_config=self.ckc)
            ttnn.deallocate(prod)
        if norm is None:
            return h
        weight, eps = norm
        hb = ttnn.typecast(h, ttnn.bfloat16)
        ttnn.deallocate(h)
        hn = ttnn.rms_norm(hb, epsilon=eps, weight=weight, compute_kernel_config=self.ckc)
        ttnn.deallocate(hb)
        return hn

    def _norm_decode(self, h: ttnn.Tensor, norm, norm_mode: str | None = None) -> ttnn.Tensor:
        """The sublayer's input RMSNorm of the exact decode ``h`` ``[1,1,32,C]`` fp32 interleaved -> bf16 interleaved
        (A3). ``sharded``: typecast, reshard into ``h_mem`` (width-sharded bf16, 4 tiles per core on the 8x4 grid),
        the sharded multicore ``rms_norm`` (``pc_norm``), reshard back to DRAM interleaved for the sublayer: 4 ops,
        32 cores. ``sharded_cast``: the cast folded into the reshard (``interleaved_to_sharded(output_dtype)``,
        3 ops). ``single``: the composite's typecast + interleaved ``rms_norm`` (2 ops on ONE core at this height).
        All three consume ``h``. ``h_mem`` / ``pc_norm`` are built once in ``__init__``: the decode tile is always
        the 32-row tile (``_view_rows`` widens every T <= 32), so there is one shape per wrap; nothing here
        touches the host (trace-safe), the L1 shards are transient within the step. ``norm_mode`` overrides the
        instance's mode for this call (``forward(norm_mode=...)`` / the prefill-scope selection)."""
        weight, eps = norm
        mode = self.norm_mode if norm_mode is None else norm_mode
        assert mode in NORM_MODES, mode
        if mode == "single":
            hb = ttnn.typecast(h, ttnn.bfloat16)
            ttnn.deallocate(h)
            hn = ttnn.rms_norm(hb, epsilon=eps, weight=weight, compute_kernel_config=self.ckc)
            ttnn.deallocate(hb)
            return hn
        assert h.padded_shape[-2] == TILE and h.padded_shape[-1] == self.C, h.padded_shape
        if mode == "sharded_cast":
            hs = ttnn.interleaved_to_sharded(h, self.h_mem, output_dtype=ttnn.bfloat16)
            ttnn.deallocate(h)
        else:
            hb = ttnn.typecast(h, ttnn.bfloat16)
            ttnn.deallocate(h)
            hs = ttnn.interleaved_to_sharded(hb, self.h_mem)
            ttnn.deallocate(hb)
        hn = ttnn.rms_norm(
            hs,
            epsilon=eps,
            weight=weight,
            program_config=self.pc_norm,
            memory_config=self.h_mem,
            compute_kernel_config=self.ckc,
        )
        ttnn.deallocate(hs)
        out = ttnn.sharded_to_interleaved(hn, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hn)
        return out

    def _parametrize(self, mixes: ttnn.Tensor):
        """``mixes [1,1,Tp,32]`` (or ``[1,1,Tp,mix_hc]``) -> (pre ``[Tp,n]``, post ``[Tp,n]``, comb ``[Tp,n*n]``) fp32,
        one tile wide (padded): the fused ``mhc_split_sinkhorn`` program (``kernel``) or its tile sequence as
        composite ops over the whole token tile (``batched``, ``parametrize_graph``; the outputs are the full
        ``[1,1,Tp,32]`` tiles re-declared to the kernel's logical widths, zero-copy)."""
        if self.sinkhorn_mode == "kernel":
            return ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
                mixes, self.consts, self.n, self.iters, self.eps
            )
        Tp = mixes.shape[-2]
        ops = _TtnnOps(self.ckc)
        pre, post, comb = parametrize_graph(ops, _view_tile_width(mixes), self.const_tiles, self.iters, self.eps)
        n = self.n

        def narrow(t, k):
            return ttnn.reshape(t, ttnn.Shape([1, 1, Tp, k]), ttnn.Shape([1, 1, Tp, W]))

        return narrow(pre, n), narrow(post, n), narrow(comb, n * n)

    def _fused_pre_block(self, norm):
        """The fused pre-block op for this wrap's norm weight (built once; the weight is read back to the host).
        The build uploads device tensors and reads one back, so the first fused call must be an eager one (the
        decoder's warm-up step before ``capture_decode_trace``), never a call inside a trace capture."""
        weight, eps = norm
        if self.pre_block is None or self._pre_block_norm is not weight:
            # the norm weight is replicated over the mesh: read shard 0 (a plain to_torch of a mesh tensor needs a composer)
            shards = ttnn.get_device_tensors(weight)
            w = ttnn.to_torch(shards[0] if shards else weight).float().reshape(-1)
            self.pre_block = MhcPreBlock(
                self.device,
                self._fn_raw,
                self.consts,
                w,
                n=self.n,
                C=self.C,
                sinkhorn_iters=self.iters,
                eps=self.eps,
                norm_eps=self.norm_eps,
                rms_eps=float(eps),
                exact=MHC_PRE_BLOCK == 2,
                sinkhorn=self.pre_sinkhorn,
                split=self.pre_split,
            )
            self._pre_block_norm = weight
        return self.pre_block

    def _pre(self, x: ttnn.Tensor, norm=None, norm_mode: str | None = None):
        """x ``[1,1,T,nC]`` -> (h ``[1,1,T,C]``, post ``[Tp,n]``, comb ``[Tp,n*n]``, x32, x_sm, owned).
        ``norm_mode``: this call's A3 norm mode (default the instance's; the fused pre-block norms in-kernel)."""
        T = x.shape[-2]
        x32 = _view_rows(x, _rows_up(T))
        if self.pre_block_on and norm is not None and x32.shape[-2] == TILE:
            # one program: projection + Sinkhorn + mix + norm; post / comb come out in the post kernel's lane layout
            pb = self._fused_pre_block(norm)
            hn, post, comb = pb.allocate_outputs(x32)
            pb(x32, hn, post, comb)
            return _view_rows(hn, T), post, comb, x32, None, False
        if x32.shape[-2] != TILE and self.post_kernel and MHC_PREFILL_EXACT:
            x_sm, owned = None, False  # prefill through the composite projection/mix + kernel post-block: unused
        else:
            x_sm, owned = self._stream_major(x32)
        mixes = self._project(x32, x_sm)
        pre, post, comb = self._parametrize(mixes)
        ttnn.deallocate(mixes)
        h32 = self._collapse(x32, x_sm, pre, norm, norm_mode)
        ttnn.deallocate(pre)
        return _view_rows(h32, T), post, comb, x32, x_sm, owned

    def _post(
        self, y: ttnn.Tensor, x32: ttnn.Tensor, x_sm: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor, collapse=False
    ):
        """y ``[1,1,T,C]`` fp32 (F's output), x32 / x_sm the residual streams -> ``[1,1,T,nC]``:
        new_j = post_j * y + sum_i comb[i,j] * x_i. ``collapse`` (decode shapes): return the LM
        head's unweighted mean over the n new streams ``[1,1,T,C]`` instead (same ops: the mean
        masks fold ``1/n sum_j`` into the coefficient columns, so ``mean_j new_j = sum_i cbar_i x_i
        + pbar y`` is one ``[T, n*T]`` block-diagonal matmul plus the ``addcmul``)."""
        T = y.shape[-2]
        Tp = x32.shape[-2]
        n = self.n
        y32 = _view_rows(y, Tp)
        if self.post_kernel:
            # one program on the packed streams (any Tp): exact fp32, row-local (no pad-row guard needed)
            out = self.post_mix.allocate_output(x32, collapse)
            self.post_mix(x32, y32, _view_rows(post, Tp), _view_rows(comb, Tp), out, collapse)
            return _view_rows(out, T)
        if Tp == TILE:
            if T != Tp:
                y32 = ttnn.where(self._rowmask_for(T), y32, 0.0)
            res_v = ttnn.reshape(x_sm, [1, 1, n * Tp, self.C])
            if collapse:
                cbar = self._coef(comb, self.m_comb_mean)  # [1, n(i), Tp, 1]
                pbar = self._coef(post, self.m_post_mean)  # [1, 1, Tp, 1]
                Pb = ttnn.multiply(self.i_tile, cbar)  # [1,n,Tp,Tp]: delta(t,t') cbar[t,i]
                ttnn.deallocate(cbar)
                Pb2 = ttnn.experimental.view(Pb, [1, 1, Tp, n * Tp])
                if self.exact:
                    mix = self._exact_mm(Pb2, res_v)  # [1,1,Tp,C]
                else:
                    mix = ttnn.matmul(Pb2, res_v, compute_kernel_config=self.ckc)
                ttnn.deallocate(Pb)
                out = ttnn.addcmul(mix, y32, pbar, output_tensor=mix)
                ttnn.deallocate(pbar)
                if T != Tp:
                    ttnn.deallocate(y32)
                return _view_rows(out, T)
            comb_c = self._coef(comb, self.m_comb)  # [n(j), n(i), Tp, 1]
            post_c = self._coef(post, self.m_post)  # [1, n(j), Tp, 1]
            A = ttnn.multiply(self.i_tile, comb_c)  # [n,n,Tp,Tp]: delta(t,t') comb[t,i,j]
            ttnn.deallocate(comb_c)
            A2 = ttnn.experimental.view(A, [1, 1, n * Tp, n * Tp])
            if self.exact:
                mix = self._exact_mm(A2, res_v)  # [(j,t), c], fp32-exact
            else:
                mix = ttnn.matmul(A2, res_v, compute_kernel_config=self.ckc, program_config=self.pc_mix)
            ttnn.deallocate(A)
            mix_sm = ttnn.reshape(mix, [1, n, Tp, self.C])
            out_sm = ttnn.addcmul(mix_sm, y32, post_c, output_tensor=mix_sm)  # + post_j * y
            ttnn.deallocate(post_c)
            if T != Tp:
                ttnn.deallocate(y32)
        else:
            assert not collapse, "collapse mode is for decode shapes (T <= 32)"
            comb_c = self._coef(comb, self.m_comb)  # [n(j), n(i), Tp, 1]
            post_c = self._coef(post, self.m_post)  # [1, n(j), Tp, 1]
            out_sm = ttnn.multiply(y32, post_c)  # [1,n,Tp,C]: post_j * y
            ttnn.deallocate(post_c)
            for i in range(n):
                x_i = ttnn.slice(x32, [0, 0, 0, i * self.C], [1, 1, Tp, (i + 1) * self.C])  # [1,1,Tp,C]
                c_i = ttnn.slice(comb_c, [0, i, 0, 0], [n, i + 1, Tp, 1])  # [n(j),1,Tp,1]
                c_i = ttnn.experimental.view(c_i, [1, n, Tp, 1])
                out_sm = ttnn.addcmul(out_sm, x_i, c_i, output_tensor=out_sm)
                ttnn.deallocate(x_i)
                ttnn.deallocate(c_i)
            ttnn.deallocate(comb_c)
        return _view_rows(self._packed(out_sm), T)

    # ------------------------------------------------------------------ TtMHCWrap-compatible API
    def project(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``[1,1,T,nC]`` -> mixes ``[1,1,T,32]`` (columns ``[mix_hc, 32)`` are zero)."""
        T = x.shape[-2]
        x32 = _view_rows(x, _rows_up(T))
        if x32.shape[-2] != TILE and self.post_kernel and MHC_PREFILL_EXACT:
            x_sm, owned = None, False  # prefill through the composite projection/mix + kernel post-block: unused
        else:
            x_sm, owned = self._stream_major(x32)
        mixes = self._project(x32, x_sm)
        if owned:
            ttnn.deallocate(x_sm)
        return _view_rows(mixes, T)

    def hc_pre(self, x: ttnn.Tensor):
        """``[1,1,T,nC]`` -> (h ``[1,1,T,C]``, post ``[1,1,T,n]``, comb ``[1,1,T,n*n]``)."""
        T = x.shape[-2]
        h, post, comb, _, x_sm, owned = self._pre(x)
        if owned:
            ttnn.deallocate(x_sm)
        r4 = lambda t: _view_rows(ttnn.reshape(t, [1, 1, t.shape[-2], t.shape[-1]]), T)
        return h, r4(post), r4(comb)

    def hc_post(
        self, x: ttnn.Tensor, residual: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor, collapse: bool = False
    ) -> ttnn.Tensor:
        """x: F's output ``[1,1,T,C]`` fp32; residual ``[1,1,T,nC]``; post / comb as ``hc_pre`` returns them;
        ``collapse``: the mean over the n new streams ``[1,1,T,C]`` (see ``_post``)."""
        T = residual.shape[-2]
        res32 = _view_rows(residual, _rows_up(T))
        res_sm, owned = self._stream_major(res32)
        Tp = res_sm.shape[-2]
        out = self._post(x, res32, res_sm, _view_rows(post, Tp), _view_rows(comb, Tp), collapse=collapse)
        if owned:
            ttnn.deallocate(res_sm)
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        sublayer,
        sublayer_dtype=None,
        norm=None,
        collapse: bool = False,
        *,
        norm_mode: str | None = None,
        prefill: bool = False,
    ) -> ttnn.Tensor:
        """x: ``[1,1,T,nC]``; sublayer: ``[1,1,T,C]`` -> ``[1,1,T,C]``. Returns ``[1,1,T,nC]``.
        ``sublayer_dtype`` casts the collapsed hidden going in and back to ``x``'s dtype coming out;
        ``norm = (weight [1,1,1,C] bf16, eps)`` applies the sublayer's input RMSNorm to the collapsed
        hidden inside the pre-block (bf16 out; ``sublayer_dtype`` must then be bf16); ``collapse``
        returns the LM head's mean over the n new streams ``[1,1,T,C]`` fp32 instead of the streams
        (the last layer's FFN wrap feeding the head; decode shapes only).
        ``norm_mode`` (``sharded`` / ``sharded_cast`` / ``single``) overrides the instance's A3 norm mode for
        this call; ``prefill`` marks a call inside a prefill scope (``decoder_layer._hc_call_impl`` passes
        ``mesh.in_prefill``): without an override such a call takes the one-core norm (``effective_norm_mode``,
        ``GLM53_MHC_PREFILL_NORM``), so a one-tile prefill slab routed here by ``TtMHCSplit.select`` does not pay
        the sharded norm's reshards inside the traced prefill."""
        if norm is not None:
            assert sublayer_dtype == ttnn.bfloat16, "the fused norm emits bf16"
        mode = effective_norm_mode(self.norm_mode, prefill=prefill, override=norm_mode)
        h, post, comb, x32, x_sm, owned = self._pre(x, norm, mode)
        if sublayer_dtype is None or norm is not None:
            cast = h
        else:
            cast = ttnn.typecast(h, sublayer_dtype)
            ttnn.deallocate(h)
        out = sublayer(cast)
        if sublayer_dtype is None or self.post_kernel == 2:
            y = out  # the v2 post-block kernel reads the bf16 sublayer output directly (widened on unpack)
            if cast is not h and cast is not out:
                ttnn.deallocate(cast)
        else:
            y = ttnn.typecast(out, x.dtype)
            ttnn.deallocate(out)
            ttnn.deallocate(cast)
        new = self._post(y, x32, x_sm, post, comb, collapse=collapse)
        ttnn.deallocate(y)
        ttnn.deallocate(post)
        ttnn.deallocate(comb)
        if owned:
            ttnn.deallocate(x_sm)
        return new
