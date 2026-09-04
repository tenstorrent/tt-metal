#!/usr/bin/env python3
"""laneMR — host-side TRUE-MATH golden + ULP-contract leg for the 2^32 streamer.

This turns the laneMK/laneMQ 2-way (sem-vs-hand *equivalence*) galaxy sweep into a
3-WAY check that also asks: is each certified leg *correct* vs the true-math oracle,
not merely equal-to-the-expert? The golden is computed HOST-SIDE (CPU torch) for the
same raw uint32 inputs a chunk streamed to the device, so it rides along for free on
the same device pass with NO 16GB retention: per chunk we fold a per-leg running
max-ULP, an out-of-tolerance counter, and the FIRST out-of-tolerance witness.

Design (faithful reuse, NOT a reinvention):
  * The op->true-math map and every dispatch constant are lifted verbatim from the
    authoritative in-repo golden ``helpers.golden_generators.UnarySFPUGolden`` (the
    exact oracle the harness itself grades sem AND hand against). The selftest
    cross-checks this vectorized golden BYTE-FOR-BYTE against that scalar golden.
  * The output-format pipeline (bf16 input truncation for the Float32/dest_acc=No
    dst, bf16 output rounding, NaN->+inf, FTZ below 2^-126) mirrors
    ``UnarySFPUGolden.__call__`` for the exact config the laneMK streamer nodes use
    (InputOutputFormat(Float32, Float32), DestAccumulation.No -> 16-bit bf16 dst).
  * The ULP metric is the bf16 bit-distance from ``extract_accuracy.compute_ulp_bitdistance``
    (tt-polynomial-fitter); vendored here (galaxy consume venv has no ttpoly on path)
    and asserted identical to the fitter's own function in the selftest.
  * The accuracy contract is the harness's own ``passed_test`` tolerance: the Float32
    default (atol=0.05, rtol=0.05) or the op's ``CUSTOM_TOLERANCES`` override.

Honesty: an out-of-tolerance witness at an out-of-DOMAIN input (erfinv |x|>=1, or a
non-finite input) is NOT a bug -- the witness record carries the input's classification
so the report can say "licensed vs bug" precisely. Ops with no honest torch reference
are marked checkable=False with a reason rather than faked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is always present in the harness env
    torch = None


# ─────────────────────────────────────────────────────────────────────────────
# ULP bit-distance (bf16) — vendored verbatim from tt-polynomial-fitter
# extract_accuracy.compute_ulp_bitdistance (the 'bf16' branch) + _ordered_float_bits.
# "how many representable bf16 values apart are the reference and the device output."
# The selftest asserts this equals the fitter's own function element-for-element.
# ─────────────────────────────────────────────────────────────────────────────
def _ordered_bf16_bits(bits16: np.ndarray) -> np.ndarray:
    """Map bf16 bit patterns (as int32 in [0,0xFFFF]) to a monotone signed ordering."""
    b = np.asarray(bits16, dtype=np.uint64)
    sign = np.uint64(0x8000)
    vmask = np.uint64(0xFFFF)
    mag = sign - np.uint64(1)
    b = np.where((b & mag) == 0, np.uint64(0), b)  # +0 == -0
    neg = (b & sign) != 0
    ordered = np.where(neg, (~b) & vmask, b | sign)
    return ordered.astype(np.int64)


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even a value (any precision) into a 16-bit bf16 code."""
    x32 = np.asarray(x, dtype=np.float64).astype(np.float32)
    bits = x32.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + bias) & np.uint32(0xFFFF0000)
    return (rounded >> np.uint32(16)).astype(np.int32)


def bf16_bitdistance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Integer bf16 ULP distance |ord(RN_bf16(true)) - ord(RN_bf16(pred))| per element."""
    ref = _ordered_bf16_bits(_to_bf16_bits(y_true))
    approx = _ordered_bf16_bits(_to_bf16_bits(y_pred))
    return np.abs(ref - approx).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch constants — lifted verbatim from sfpu_dispatch_constants.py /
# UnarySFPUGolden (kept local so the streamer needs no harness import at runtime;
# the selftest pins them against the live harness values).
# ─────────────────────────────────────────────────────────────────────────────
RPOW_BASE = 2.0
FMOD_DIVISOR = 2.0
REMAINDER_DIVISOR = 2.0
UNARY_POWER_EXP = 2.0
RDIV_VALUE = 2.0
CLAMP_MIN = -1.0
CLAMP_MAX = 1.0
SOFTSHRINK_LAMBDA = 0.5
HARDSHRINK_LAMBDA = 0.5
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0
XIELU_ALPHA_P = 1.0
XIELU_ALPHA_N = 1.0
XIELU_BETA = 0.5

BF16_TINY = 2.0**-126  # FTZ threshold for Float16_b / Float32 (finfo.tiny)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized true-math op bodies. Each takes an fp32 numpy array (the value the SFPU
# actually sees after bf16 truncation) and returns an fp64 numpy array (high precision,
# pre-output-rounding). These mirror UnarySFPUGolden's per-element methods exactly.
# ─────────────────────────────────────────────────────────────────────────────
def _t(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float64))


def _np(t):
    return t.detach().numpy().astype(np.float64)


def _erf(x):
    return _np(torch.erf(_t(x)))


def _erfc(x):
    return _np(torch.erfc(_t(x)))


def _erfinv(x):
    return _np(torch.erfinv(_t(x)))


def _gelu_exact(x):
    # GeluAppx golden == exact (erf) gelu, matching UnarySFPUGolden._gelu.
    return _np(torch.nn.functional.gelu(_t(x)))


def _sigmoid(x):
    return _np(torch.sigmoid(_t(x)))


def _hardtanh(x):
    return _np(torch.clamp(_t(x), CLAMP_MIN, CLAMP_MAX))


def _rpow(x):
    return _np(torch.pow(torch.tensor(RPOW_BASE, dtype=torch.float64), _t(x)))


def _fmod(x):
    return _np(torch.fmod(_t(x), torch.tensor(FMOD_DIVISOR, dtype=torch.float64)))


def _remainder(x):
    return _np(
        torch.remainder(_t(x), torch.tensor(REMAINDER_DIVISOR, dtype=torch.float64))
    )


def _unary_power(x):
    return _np(torch.pow(_t(x), UNARY_POWER_EXP))


def _rdiv(x):
    return RDIV_VALUE / np.asarray(x, dtype=np.float64)


def _add1(x):
    return np.asarray(x, dtype=np.float64) + 1.0


def _sign(x):
    return _np(torch.sign(_t(x)))


def _signbit(x):
    xf = np.asarray(x, dtype=np.float32)
    return (np.signbit(xf)).astype(np.float64)


def _heaviside(x):
    xf = np.asarray(x, dtype=np.float64)
    return np.where(xf < 0.0, 0.0, np.where(xf > 0.0, 1.0, 0.5))


def _cbrt(x):
    return _np(torch.sign(_t(x)) * torch.abs(_t(x)).pow(1.0 / 3.0))


def _expm1(x):
    return _np(torch.expm1(_t(x)))


def _softplus(x):
    return _np(
        torch.nn.functional.softplus(
            _t(x), beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD
        )
    )


def _softsign(x):
    return _np(torch.nn.functional.softsign(_t(x)))


def _softshrink(x):
    return _np(torch.nn.functional.softshrink(_t(x), lambd=SOFTSHRINK_LAMBDA))


def _hardshrink(x):
    return _np(torch.nn.functional.hardshrink(_t(x), lambd=HARDSHRINK_LAMBDA))


def _hardmish(x):
    return _np(_t(x) * torch.clamp(0.5 * _t(x) + 1.0, 0.0, 1.0))


def _xielu(x):
    xf = np.asarray(x, dtype=np.float64)
    beta_x = XIELU_BETA * xf
    pos = XIELU_ALPHA_P * xf * xf + beta_x
    neg = XIELU_ALPHA_N * (np.expm1(xf) - xf) + beta_x
    return np.where(xf > 0.0, pos, neg)


def _tanh_derivative_lut(x):
    # The LICENSED LUT contract (UnarySFPUGolden._tanh_derivative_lut): 1 - t^2 where t
    # is the raw 3-region SFPLUT, NOT accurate tanh. This is the kernel's own design
    # contract (the header documents catastrophic cancellation past |x|~3.4).
    a = np.abs(np.asarray(x, dtype=np.float64))
    t = np.where(a < 1.0, 0.90625 * a, np.where(a < 2.0, 0.09375 * a + 0.8125, 1.0))
    return 1.0 - t * t


def _tanh_derivative_true(x):
    # The TRUE math tanh'(x) = sech^2(x) = 1 - tanh^2, computed stably as 1/cosh^2.
    return _np(1.0 / torch.cosh(_t(x)) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Output-format pipeline for the streamer's Float32/Float32, dest_acc=No config
# (dst_format = Float16_b). Mirrors UnarySFPUGolden.__call__ for that case.
# ─────────────────────────────────────────────────────────────────────────────
def bf16_truncate(u32: np.ndarray) -> np.ndarray:
    """Input as the SFPU sees it: Float32 operand truncated to bf16 (& 0xFFFF0000)."""
    return (u32.astype(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _round_bf16_as_f32(hp: np.ndarray) -> np.ndarray:
    """RNE round an fp64 array to bf16, returned as fp32 values (torch bfloat16 cast)."""
    if torch is not None:
        t = (
            torch.from_numpy(np.asarray(hp, dtype=np.float32))
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        return t.numpy()
    bits = np.asarray(hp, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def format_golden_f32_noacc(hp: np.ndarray) -> np.ndarray:
    """hp (fp64 true math) -> the fp32-container reference the device should output.

    bf16-round -> NaN->+inf (dst=Float16_b, data=Float32 falls in __call__'s default
    convert_nan_to_inf case) -> FTZ below 2^-126. Returns fp32.
    """
    y = _round_bf16_as_f32(hp).astype(np.float32)
    y = np.where(np.isnan(y), np.float32(np.inf), y)  # convert_nan_to_inf: NaN -> +inf
    y = np.where(np.abs(y.astype(np.float64)) < BF16_TINY, np.float32(0.0), y)  # FTZ
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Op registry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GoldenSpec:
    op: str  # streamer op key (matches ops31.tsv / idmap)
    math: Optional[Callable[[np.ndarray], np.ndarray]]  # fp32-in -> fp64 true math
    kind: str = "f32_noacc"  # 'f32_noacc' | 'int32' | 'unsupported'
    atol: float = 0.05  # harness passed_test Float32 default
    rtol: float = 0.05
    domain: Optional[tuple] = (
        None  # (lo, hi) finite in-domain interval, else None=all reals
    )
    note: str = ""
    checkable: bool = True


# Divergent priority ops (the correctness question actually matters for these).
_DIVERGENT = [
    GoldenSpec("erf-fresh", _erf, note="torch.erf; all reals"),
    GoldenSpec("erfc-fresh", _erfc, note="torch.erfc; all reals"),
    GoldenSpec(
        "erfinv-fresh",
        _erfinv,
        domain=(-1.0, 1.0),
        note="torch.erfinv; DOMAIN (-1,1) — |x|>=1 is out-of-domain, +-inf/nan expected",
    ),
    GoldenSpec(
        "geluappx-fresh",
        _gelu_exact,
        atol=0.13,
        rtol=0.05,
        note="exact (erf) gelu; GeluAppx is a LICENSED 6-segment LUT approx (CUSTOM_TOLERANCES 0.13/0.05)",
    ),
    GoldenSpec(
        "sigmoidlut-fresh",
        _sigmoid,
        atol=0.05,
        rtol=0.05,
        note="torch.sigmoid; production is a LUT6 approx (contract 0.05/0.05)",
    ),
    GoldenSpec(
        "tanhderivlut-fresh",
        _tanh_derivative_lut,
        note=(
            "LICENSED LUT contract 1-t_lut^2 (NOT accurate sech^2); see tanhderiv-true "
            "column for the true-math distance the LUT intentionally trades away"
        ),
    ),
    GoldenSpec(
        "hardtanh-fresh",
        _hardtanh,
        note="clamp(x,-1,1); EXACT piecewise-linear (expect 0 ULP)",
    ),
    GoldenSpec("rpow", _rpow, note="2.0**x; inf/nan inputs -> inf/nan (band7)"),
    GoldenSpec(
        "fmod-fresh", _fmod, note="fmod(x,2.0); inf/nan inputs special (band15)"
    ),
]

# Bit-exact ops (sem==hand over all 2^32 — one leg vs torch confirms 'correct AND
# matches expert'). All Float32/dest_acc=No unless noted.
_BITEXACT = [
    GoldenSpec("add1", _add1, note="x+1; EXACT (expect 0 ULP)"),
    GoldenSpec("sign", _sign, note="sign(x); EXACT"),
    GoldenSpec("signbit", _signbit, note="signbit(x)->0/1; EXACT"),
    GoldenSpec("heaviside-fresh", _heaviside, note="0/0.5/1; EXACT"),
    GoldenSpec("cbrt-fresh", _cbrt, note="sign(x)|x|^(1/3)"),
    GoldenSpec("expm1cw-fresh", _expm1, note="expm1(x)"),
    GoldenSpec("softplus-fresh", _softplus, note="softplus beta=1 thr=20"),
    GoldenSpec("softsign-fresh", _softsign, note="x/(1+|x|)"),
    GoldenSpec("softshrink-fresh", _softshrink, note="softshrink lambda=0.5"),
    GoldenSpec("hardshrink-fresh", _hardshrink, note="hardshrink lambda=0.5"),
    GoldenSpec("hardmish-fresh", _hardmish, note="x*clamp(0.5x+1,0,1)"),
    GoldenSpec("xielu-fresh", _xielu, note="xielu beta=0.5 alpha_p=alpha_n=1"),
    GoldenSpec("rdiv", _rdiv, note="2.0/x"),
    GoldenSpec("remainder-fresh", _remainder, note="remainder(x,2.0)"),
    GoldenSpec("unarypower-fresh", _unary_power, note="x**2"),
]

# Ops with no honest single torch reference on this streamer config (stated, not faked).
_UNSUPPORTED = {
    "absint32": "integer abs on the int32 view — golden is exact-int, not a float ULP; "
    "int32 leg checkable via kind=int32 (abs(int)) but not modeled here yet",
    "bitwisenot": "integer ~ on the int32 view — exact-int, not float ULP",
    "unaryshift-fresh": "integer shift — exact-int, not float ULP",
    "comp": "NotEqualZero predicate (0/1) — exact boolean, trivially correct; no ULP surface",
    "eqz-fresh": "EqualZero predicate (0/1) — exact boolean; no ULP surface",
    "unarycomp-fresh": "unary compare predicate (0/1) — exact boolean; no ULP surface",
    "castfp32tofp16a": "Float32/dest_acc=Yes cast (fp32 dst, distinct format path) — "
    "bit-lattice cast proven exact by laneCT 2^32; not a torch-ULP surface",
}

REGISTRY: dict[str, GoldenSpec] = {}
for _s in _DIVERGENT + _BITEXACT:
    REGISTRY[_s.op] = _s
for _op, _why in _UNSUPPORTED.items():
    REGISTRY[_op] = GoldenSpec(
        _op, None, kind="unsupported", note=_why, checkable=False
    )


def get_spec(op: str) -> Optional[GoldenSpec]:
    return REGISTRY.get(op)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming correctness accumulator — one per leg, folded chunk by chunk, no retention.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrectnessAccumulator:
    spec: GoldenSpec
    patterns: int = 0
    max_ulp: float = 0.0
    max_ulp_input_u32: int = -1
    n_out_of_tol: int = 0
    first_witness_u32: int = -1
    first_witness_class: str = ""
    first_witness_dev: float = 0.0
    first_witness_golden: float = 0.0
    # tanhderiv extra: distance to the TRUE math (sech^2), reported alongside the LUT contract.
    max_ulp_true: float = 0.0

    def _classify(self, u32: int, xf: float) -> str:
        if not math.isfinite(xf):
            return "nonfinite-input"
        if self.spec.domain is not None:
            lo, hi = self.spec.domain
            if not (lo < xf < hi):
                return "out-of-domain"
        return "in-domain"

    def update(self, chunk_start: int, valid_count: int, dev_bytes: bytes) -> None:
        """Fold one chunk: dev_bytes = valid_count fp32 little-endian device outputs."""
        with np.errstate(all="ignore"):
            self._update(chunk_start, valid_count, dev_bytes)

    def _update(self, chunk_start: int, valid_count: int, dev_bytes: bytes) -> None:
        if valid_count <= 0:
            return
        u32 = np.arange(chunk_start, chunk_start + valid_count, dtype=np.uint32)
        xin = bf16_truncate(u32)  # what the SFPU actually sees
        hp = self.spec.math(xin)  # fp64 true math
        golden = format_golden_f32_noacc(hp)  # fp32-container reference
        dev = np.frombuffer(dev_bytes[: valid_count * 4], dtype="<f4").astype(
            np.float32
        )

        ulp = bf16_bitdistance(golden, dev)
        # tolerance check on the (bf16) values, matching passed_test isclose + equal_nan.
        g = golden.astype(np.float64)
        d = dev.astype(np.float64)
        both_nan = np.isnan(g) & np.isnan(d)
        both_inf = np.isinf(g) & np.isinf(d) & (np.sign(g) == np.sign(d))
        close = np.abs(d - g) <= (self.spec.atol + self.spec.rtol * np.abs(g))
        within = close | both_nan | both_inf
        out = ~within & np.isfinite(
            ulp
        )  # nonfinite golden handled by both_nan/both_inf

        # running max ULP + its input
        if ulp.size:
            i = int(np.nanargmax(np.where(np.isfinite(ulp), ulp, -1.0)))
            if ulp[i] > self.max_ulp:
                self.max_ulp = float(ulp[i])
                self.max_ulp_input_u32 = int(u32[i])

        # out-of-tolerance count + first witness
        n_out = int(np.count_nonzero(out))
        if n_out:
            self.n_out_of_tol += n_out
            if self.first_witness_u32 < 0:
                j = int(np.argmax(out))
                self.first_witness_u32 = int(u32[j])
                self.first_witness_class = self._classify(int(u32[j]), float(xin[j]))
                self.first_witness_dev = float(dev[j])
                self.first_witness_golden = float(golden[j])

        # tanhderiv: also track distance to the TRUE sech^2 (reporting the licensed gap)
        if self.spec.op == "tanhderivlut-fresh":
            true_hp = _tanh_derivative_true(xin)
            true_g = format_golden_f32_noacc(true_hp)
            ulp_true = bf16_bitdistance(true_g, dev)
            if ulp_true.size:
                m = float(np.nanmax(np.where(np.isfinite(ulp_true), ulp_true, -1.0)))
                if m > self.max_ulp_true:
                    self.max_ulp_true = m

        self.patterns += valid_count

    def result_line(self, leg: str) -> str:
        w = self.first_witness_u32
        extra = (
            f",max_ulp_true_sech2={self.max_ulp_true:.0f}"
            if self.spec.op == "tanhderivlut-fresh"
            else ""
        )
        return (
            f"LANEMR_CORRECTNESS,leg={leg},op={self.spec.op},patterns={self.patterns},"
            f"max_bf16_ulp={self.max_ulp:.0f},max_ulp_input=0x{max(self.max_ulp_input_u32,0):08x},"
            f"n_out_of_tol={self.n_out_of_tol},"
            f"within_contract={self.n_out_of_tol == 0},"
            f"first_witness=0x{max(w,0):08x},first_witness_class={self.first_witness_class or '-'},"
            f"witness_dev={self.first_witness_dev!r},witness_golden={self.first_witness_golden!r},"
            f"atol={self.spec.atol},rtol={self.spec.rtol}{extra}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# binarypow (laneMQ two-operand streamer) — the highest-interest divergent op.
# Joint J in [0,2^32): base16 = J>>16, exp16 = J&0xFFFF (raw bf16 patterns). The
# device writes pow(base, exp) to the EVEN tile of each tile pair (sfpu_binary_test.cpp
# `call(tile, tile+1, tile)`); odd tiles stay the 0xA5 clear sentinel. Output Float16_b
# (2 bytes), dest_acc=No, contract atol=rtol=0.05 (passed_test Float16_b default).
# Golden mirrors BinarySFPUGolden._pow: (base_fp32 ** exp_fp32) -> bf16.
# ─────────────────────────────────────────────────────────────────────────────
BINARY_POW_ATOL = 0.05
BINARY_POW_RTOL = 0.05
_ELEMS_PER_TILE = 1024


def _bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Raw bf16 bit patterns (uint16) -> fp32 values (shift into the fp32 high half)."""
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def binary_pow_golden_bf16(base16: np.ndarray, exp16: np.ndarray) -> np.ndarray:
    """pow(base, exp) per BinarySFPUGolden._pow: (a_fp32 ** b_fp32) rounded to bf16, as fp32."""
    a = _bf16_bits_to_f32(base16).astype(np.float64)
    b = _bf16_bits_to_f32(exp16).astype(np.float64)
    with np.errstate(all="ignore"):
        hp = np.power(
            a, b
        )  # fp64 pow (a,b are exact bf16 values); matches (fp32**fp32) target
    return _round_bf16_as_f32(hp).astype(np.float32)


@dataclass
class BinaryPowAccumulator:
    """Streaming correctness for binarypow: device even-tile outputs vs torch.pow (bf16)."""

    atol: float = BINARY_POW_ATOL
    rtol: float = BINARY_POW_RTOL
    joints: int = 0
    max_ulp: float = 0.0
    max_ulp_joint: int = -1
    n_out_of_tol: int = 0
    first_witness_joint: int = -1
    first_witness_class: str = ""
    first_witness_dev: float = 0.0
    first_witness_golden: float = 0.0

    def update(
        self, dispatch_start: int, pairs: int, result_region_bytes: bytes
    ) -> None:
        with np.errstate(all="ignore"):
            self._update(dispatch_start, pairs, result_region_bytes)

    def _update(self, dispatch_start: int, pairs: int, res: bytes) -> None:
        tile_bytes = _ELEMS_PER_TILE * 2  # bf16 tile = 1024 * 2 bytes
        for p in range(pairs):
            joint0 = dispatch_start + p * _ELEMS_PER_TILE
            base16 = (joint0 >> 16) & 0xFFFF
            lo = joint0 & 0xFFFF
            exp16 = (np.arange(_ELEMS_PER_TILE, dtype=np.uint32) + lo).astype(np.uint16)
            base_arr = np.full(_ELEMS_PER_TILE, base16, dtype=np.uint16)
            golden = binary_pow_golden_bf16(base_arr, exp16)  # fp32 (bf16-valued)

            even_off = (2 * p) * tile_bytes  # output lives in the EVEN tile of the pair
            dev16 = np.frombuffer(res[even_off : even_off + tile_bytes], dtype="<u2")
            if dev16.size < _ELEMS_PER_TILE:
                raise ValueError(
                    f"binarypow dispatch@{dispatch_start} pair {p}: short result region"
                )
            dev = _bf16_bits_to_f32(dev16.astype(np.uint32)).astype(np.float32)

            ulp = bf16_bitdistance(golden, dev)
            g = golden.astype(np.float64)
            d = dev.astype(np.float64)
            both_nan = np.isnan(g) & np.isnan(d)
            both_inf = np.isinf(g) & np.isinf(d) & (np.sign(g) == np.sign(d))
            close = np.abs(d - g) <= (self.atol + self.rtol * np.abs(g))
            within = close | both_nan | both_inf
            out = ~within & np.isfinite(ulp)

            i = int(np.nanargmax(np.where(np.isfinite(ulp), ulp, -1.0)))
            if ulp[i] > self.max_ulp:
                self.max_ulp = float(ulp[i])
                self.max_ulp_joint = joint0 + i
            n_out = int(np.count_nonzero(out))
            if n_out:
                self.n_out_of_tol += n_out
                if self.first_witness_joint < 0:
                    j = int(np.argmax(out))
                    self.first_witness_joint = joint0 + j
                    b_f = float(_bf16_bits_to_f32(base_arr[j : j + 1])[0])
                    self.first_witness_class = (
                        "base<=0 (pow via exp(b*log a) -> nan/inf, expected)"
                        if b_f <= 0.0
                        else ("nonfinite-base" if not math.isfinite(b_f) else "base>0")
                    )
                    self.first_witness_dev = float(dev[j])
                    self.first_witness_golden = float(golden[j])
            self.joints += _ELEMS_PER_TILE

    def result_line(self, leg: str) -> str:
        w = max(self.first_witness_joint, 0)
        return (
            f"LANEMR_CORRECTNESS,leg={leg},op=binarypow,joints={self.joints},"
            f"max_bf16_ulp={self.max_ulp:.0f},max_ulp_joint=0x{max(self.max_ulp_joint,0):08x},"
            f"n_out_of_tol={self.n_out_of_tol},within_contract={self.n_out_of_tol == 0},"
            f"first_witness=0x{w:08x},first_witness_class={self.first_witness_class or '-'},"
            f"witness_dev={self.first_witness_dev!r},witness_golden={self.first_witness_golden!r},"
            f"atol={self.atol},rtol={self.rtol}"
        )
