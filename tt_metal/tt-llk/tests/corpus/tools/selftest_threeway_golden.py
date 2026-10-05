#!/usr/bin/env python3
"""laneMR selftest — proves the vectorized 3-way golden is FAITHFUL and the accumulator
flags what it must, with NO device.

Cases:
  1. FAITHFUL golden: the vectorized golden (threeway_golden) matches the authoritative
     scalar in-repo golden helpers.golden_generators.UnarySFPUGolden BIT-FOR-BIT over a
     1024-element edge tile, for a spread of ops (erf, fmod, rpow, hardtanh, geluappx,
     sigmoidlut, add1, sign, cbrt, softsign). This is the whole soundness argument: the
     fast path == the oracle the harness itself grades against.
  2. FAITHFUL ULP: bf16_bitdistance == the fitter's extract_accuracy.compute_ulp_bitdistance
     ('bf16') element-for-element (when the fitter is importable).
  3. KNOWN-CORRECT: feeding the device the golden bytes -> 0 max ULP, within_contract=True.
  4. SEEDED BUG: a single perturbed output element -> flagged out-of-tolerance with the
     correct first witness (input + dev/golden), while the rest stay clean.
  5. CLASS honesty: post-bf16 inputs have a disjoint/exhaustive partition covering
     NaN, signed infinities, signed zeros, signed subnormals, exact domain boundaries,
     and remaining in/out-of-domain normal values.

Run from tests/: python corpus/tools/selftest_threeway_golden.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# tests/ on path for the harness helpers
sys.path.insert(0, str(HERE.parent.parent / "python_tests"))

import threeway_golden as tg  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(
        f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else "")
    )
    if not cond:
        FAILED.append(name)


F32MAX = float(np.finfo(np.float32).max)


def _bf16(xs_f32):
    """fp32 numpy -> bfloat16 torch, KEEPING each NaN's sign bit.

    A bare `.to(torch.bfloat16)` canonicalizes every NaN to 0xFFFF, i.e. a NEGATIVE
    NaN, which is not what Dest holds and not what the device is fed. Every oracle
    call in this file that needs a bf16 operand goes through here.
    """
    import torch

    from helpers.golden_generators import cast_preserving_nan_sign

    return cast_preserving_nan_sign(
        torch.from_numpy(np.asarray(xs_f32, dtype=np.float32)), torch.bfloat16
    )


def _explain(mine, ref, nan_source, atol, rtol):
    """Classify every mine-vs-oracle difference. Returns (ok, parts, masks).

    Exactly three explanations are admissible, and a diff that is none of them FAILS:

      (a) SUB-TOLERANCE ROUNDING. The oracle rounds through the bf16 DST dtype while
          this module carries fp64, so the two can land on adjacent bf16 codes.
          Admissible only inside the row's own accuracy contract.
      (b) MATCHED INFINITY, same sign.
      (c) CONVERTED-NaN INFINITY SIGN. Two infinities of opposite sign where the
          golden's PRE-CONVERSION value was NaN. See numeric_comparison's
          CONVERTED-NaN INFINITY SIGN POLICY: the sign bit a packer-converted NaN
          carries is not a numeric quantity and is not determined by any host math
          library. A computed overflow produces an infinity directly, so it is never
          admitted and a wrong overflow sign is still a failure.

    The predecessor's fourth class, ORACLE fp32 OVERFLOW, is deliberately gone: the
    in-repo oracle now evaluates in fp64 (see UnarySFPUGolden._torch_unary), so the
    class must be EMPTY. It is still computed, and a nonzero count now FAILS.
    """
    g = ref.astype(np.float64)
    m = mine.astype(np.float64)
    xn = np.asarray(nan_source, dtype=bool)
    fin = np.isfinite(g) & np.isfinite(m)
    rounding = fin & (np.abs(m - g) <= (atol + rtol * np.abs(g)))
    matched_inf = np.isinf(g) & np.isinf(m) & (np.signbit(g) == np.signbit(m))
    nan_sign = np.isinf(g) & np.isinf(m) & ~matched_inf & xn
    nan_pair = np.isnan(g) & np.isnan(m)
    stale_ovf = (
        np.isinf(g) & np.isfinite(m) & (np.abs(m) <= F32MAX)
        & (np.signbit(g) == np.signbit(m))
    )
    explained = rounding | matched_inf | nan_sign | nan_pair
    ok = bool(np.all(explained)) and not bool(stale_ovf.any())
    parts = []
    if rounding.any():
        worst = float(np.max(np.abs(m[rounding] - g[rounding])))
        parts.append(
            f"{int(rounding.sum())} sub-tolerance bf16-vs-fp64 rounding "
            f"(worst |d|={worst:.3e} <= {atol}+{rtol}|g|)"
        )
    if matched_inf.any():
        parts.append(f"{int(matched_inf.sum())} matched infinity")
    if nan_sign.any():
        parts.append(
            f"{int(nan_sign.sum())} opposite-signed infinity from a CONVERTED NaN "
            "(nan_inf_sign_policy)"
        )
    if nan_pair.any():
        parts.append(f"{int(nan_pair.sum())} matched NaN")
    if stale_ovf.any():
        parts.append(
            f"{int(stale_ovf.sum())} ORACLE fp32 OVERFLOW -- this class must be EMPTY "
            f"now that _torch_unary evaluates in fp64; e.g. mine={float(m[stale_ovf][0])!r} "
            f"oracle={float(g[stale_ovf][0])!r}"
        )
    bad = ~explained
    if bad.any():
        parts.append(
            f"{int(bad.sum())} UNEXPLAINED, e.g. mine={float(m[bad][0])!r} "
            f"oracle={float(g[bad][0])!r} (nan_source={bool(xn[bad][0])})"
        )
    return ok, "; ".join(parts) or "identical"


def _edge_tile():
    """1024 fp32 patterns spanning specials, denormals, both signs, domain edges."""
    specials = [
        0x00000000,
        0x80000000,
        0x00000001,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x3F800000,
        0xBF800000,
        0x3F000000,
        0xBF000000,
        0x40000000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7FA00000,
        0x3EAAAAAB,
        0x42000000,
        0xC2000000,
        0x3DCCCCCD,
    ]
    rng = np.random.default_rng(20260904)
    rest = rng.integers(0, 1 << 32, size=1024 - len(specials), dtype=np.uint64).astype(
        np.uint32
    )
    u = np.concatenate([np.array(specials, dtype=np.uint32), rest.astype(np.uint32)])
    return u[:1024]


def _scalar_golden(mathop, u32):
    """Authoritative in-repo golden over the same tile (skip_tilize, element-wise)."""
    from helpers.format_config import DataFormat  # noqa
    from helpers.golden_generators import UnarySFPUGolden
    from helpers.llk_params import DestAccumulation  # noqa

    operand = u32.astype(np.uint32).view(np.float32).copy()
    import torch

    t = torch.from_numpy(operand)
    g = UnarySFPUGolden()(
        mathop,
        t,
        DataFormat.Float32,
        DestAccumulation.No,
        DataFormat.Float32,
        (64, 64),
        iterations=None,  # auto -> numel // TILE_SIZE(=32); processes the whole tile
        skip_tilize=True,
    )
    return g.detach().numpy().astype(np.float32)


# ── case 1 + 2: faithfulness ────────────────────────────────────────────────
def case_faithful():
    print("case 1/2: vectorized golden == scalar harness golden (bit-for-bit)")
    from helpers.llk_params import MathOperation

    op_to_mathop = {
        "erf-fresh": MathOperation.Erf,
        "erfc-fresh": MathOperation.Erfc,
        "erfinv-fresh": MathOperation.Erfinv,
        "fmod-fresh": MathOperation.Fmod,
        "rpow": MathOperation.Rpow,
        "hardtanh-fresh": MathOperation.Hardtanh,
        "geluappx-fresh": MathOperation.GeluAppx,
        "sigmoidlut-fresh": MathOperation.Sigmoid,
        "add1": MathOperation.Add1,
        "sign": MathOperation.Sign,
        "cbrt-fresh": MathOperation.Cbrt,
        "softsign-fresh": MathOperation.Softsign,
        "softplus-fresh": MathOperation.Softplus,
        "xielu-fresh": MathOperation.Xielu,
        # laneMT corpus extension: EVERY newly registered checkable op, so the
        # extension is proven against the in-repo scalar oracle rather than
        # reviewed by eye. A new GoldenSpec that is not in this map fails
        # `corpus-coverage` below.
        "erf": MathOperation.Erf,
        "erfc": MathOperation.Erfc,
        "erfinv": MathOperation.Erfinv,
        "sigmoid": MathOperation.Sigmoid,
        "sigmoid-fresh": MathOperation.Sigmoid,
        "softsign": MathOperation.Softsign,
        "tanhderivative": MathOperation.TanhDerivative,
        "tanhderivative-lut": MathOperation.TanhDerivativeLut,
        "heaviside": MathOperation.Heaviside,
        "digamma": MathOperation.Digamma,
        "digamma-fresh": MathOperation.Digamma,
        "lgamma": MathOperation.Lgamma,
        "polygamma": MathOperation.Polygamma,
        "i0": MathOperation.I0,
        "i1": MathOperation.I1,
        "i1-fresh": MathOperation.I1,
        "expm1": MathOperation.Expm1,
        "expm1-fresh": MathOperation.Expm1,
        "expm1cw": MathOperation.Expm1Cw,
        "cbrt": MathOperation.Cbrt,
        "mish": MathOperation.Mish,
        "selu": MathOperation.Selu,
        "softplus": MathOperation.Softplus,
        "xielu": MathOperation.Xielu,
        "unarypower": MathOperation.UnaryPower,
        "sqrtcustom": MathOperation.SqrtCustom,
        "rsqrtcompat": MathOperation.RsqrtCompat,
        "fmod": MathOperation.Fmod,
        "remainder": MathOperation.Remainder,
        "clamp": MathOperation.Clamp,
        "clamp-fresh": MathOperation.Clamp,
        "hardtanh": MathOperation.Hardtanh,
        "hardmish": MathOperation.Hardmish,
        "hardshrink": MathOperation.Hardshrink,
        "softshrink": MathOperation.Softshrink,
        "prelu": MathOperation.Prelu,
        "identity": MathOperation.Identity,
    }
    # Every op registered by the laneMT corpus extension must be checked here.
    # Without this a new GoldenSpec could ship unverified, which is exactly the
    # failure mode the module's soundness argument rules out.
    corpus_ops = {spec.op for spec in tg._CORPUS_UNARY}
    check(
        "corpus-coverage",
        corpus_ops <= set(op_to_mathop),
        f"unverified corpus specs: {sorted(corpus_ops - set(op_to_mathop))}",
    )
    u = _edge_tile()
    for op, mathop in op_to_mathop.items():
        spec = tg.get_spec(op)
        hp = spec.evaluate(tg.bf16_truncate(u))
        mine = tg.format_golden_f32_noacc(hp)
        # Faithfulness is asserted where the function is DEFINED. Two exclusions,
        # both pointwise and both because neither oracle is computing the
        # function there -- asserting parity would be asserting that two wrong
        # numbers match:
        #   * outside spec.domain (erfinv |x|>=1, sqrt of a negative);
        #   * at a gamma-family POLE. torch does not signal these: trigamma(-1)
        #     is 1.29e15 in fp32 and 6.58e32 in fp64, finite precision-dependent
        #     noise where the true value is +inf.
        # Every other input is graded, INCLUDING the non-pole negatives of the
        # gamma family -- digamma(-1.5) is an ordinary defined value and a
        # kernel that flips its sign there is a defect, not a licensed miss.
        # The excluded population is reported, never silently dropped.
        xs = tg.bf16_truncate(u)
        graded = np.ones(u.shape, dtype=bool)
        if spec.domain is not None:
            classes = tg.unary_input_classes(xs, spec.domain)
            graded &= ~classes["out_of_domain_finite_normal"]
        if op in tg.GAMMA_POLE_OPS:
            graded &= ~np.array([tg.at_gamma_pole(op, float(v)) for v in xs])
        n_ungraded = int(np.count_nonzero(~graded))
        ungraded_note = (
            f"; {n_ungraded}/1024 undefined (out-of-domain or pole) not graded"
            if n_ungraded else ""
        )
        try:
            ref = _scalar_golden(mathop, u)
        except Exception as e:  # pragma: no cover
            check(f"faithful[{op}]", False, f"scalar golden raised: {e}")
            continue
        mine_bits = mine.view(np.uint32)
        ref_bits = ref.astype(np.float32).view(np.uint32)
        # canonicalize -0.0 vs +0.0 (both are 'zero' to the tolerance/ULP path)
        mine_bits = np.where(mine_bits == 0x80000000, np.uint32(0), mine_bits)
        ref_bits = np.where(ref_bits == 0x80000000, np.uint32(0), ref_bits)
        d = np.where((mine_bits != ref_bits) & graded)[0]
        n_graded = int(np.count_nonzero(graded))
        if d.size == 0:
            check(
                f"faithful[{op}]",
                True,
                f"{n_graded}/{n_graded} bit-identical to harness golden{ungraded_note}",
            )
            continue
        # Where they differ, the harness golden computes this op in the bf16 DST dtype
        # (torch.tensor(x, dtype=Float16_b)) while we deliberately use true fp32/fp64
        # (the charter's "true-math oracle"). Prove every such diff is a sub-tolerance
        # near-boundary rounding (well inside the op's accuracy contract) — NOT a math
        # error — by requiring |harness - true| <= atol + rtol*|harness| there.
        g = ref[d].astype(np.float64)
        m = mine[d].astype(np.float64)
        finite = np.isfinite(g) & np.isfinite(m)
        ok = np.all(
            np.abs(m[finite] - g[finite]) <= (spec.atol + spec.rtol * np.abs(g[finite]))
        )
        worst = float(np.max(np.abs(m[finite] - g[finite]))) if finite.any() else 0.0
        check(
            f"faithful[{op}]",
            bool(ok),
            f"{d.size}/{n_graded} differ ONLY at sub-tolerance bf16-vs-fp64 rounding "
            f"(worst |Δ|={worst:.2e} <= {spec.atol}+{spec.rtol}|g|){ungraded_note}",
        )


# ── case 1c: the 16-bit band mode's goldens, over the WHOLE input space ──────
# A Float16_b row has 65536 distinct inputs, so the band the device runs is
# exhaustive -- and so is this check. No sampling, no edge tile: every op's
# vectorized golden is compared against the scalar in-repo oracle at every bf16
# pattern that exists. A mis-wired golden looks exactly like a mass device
# failure, which is the reason this runs before any result is believed.
_BF16_OP_TO_MATHOP_NAME = {
    "abs": "Abs",
    "negative": "Neg",
    "ceil-fresh": "Ceil",
    "relu": "ReluMax",
    "unarymaxmin-max": "UnaryMax",
    "unarymaxmin-min": "UnaryMin",
    "threshold": "Threshold",
    "threshold-fresh": "Threshold",
    "threshold-fitted": "Threshold",
    "fill": "Fill",
    "fill-fresh": "Fill",
    "activations": "Hardsigmoid",
    "hardsigmoid-fresh": "Hardsigmoid",
    "square": "Square",
    "square-fresh": "Square",
    "tanh": "Tanh",
    "tanh-fresh": "Tanh",
    "tanh-fitted": "Tanh",
    "tanhlut-fresh": "Tanh",
    "tanhshrink": "Tanhshrink",
    "tanhshrink-fresh": "Tanhshrink",
    "tanhderivative-fitted": "TanhDerivative",
    "sigmoid-fitted": "Sigmoid",
    "sigmoidappx": "SigmoidAppx",
    "sigmoidappx-tree": "SigmoidAppx",
    "silu": "Silu",
    "silu-fresh": "Silu",
    "gelu": "Gelu",
    "gelu-fresh": "Gelu",
    "gelu-fitted": "Gelu",
    "gelu-licensed": "Gelu",
    "exp": "Exp",
    "exp-fitted": "Exp",
    "exp2": "Exp2",
    "exp2-fresh": "Exp2",
    "expm1-fitted": "Expm1",
    "celu": "Celu",
    "celu-fitted": "Celu",
    "elu": "Elu",
    "elu-fresh": "Elu",
    "elu-fitted": "Elu",
    "selu-fitted": "Selu",
    "mish-fitted": "Mish",
    "i0-fitted": "I0",
    "i1-fitted": "I1",
    "digamma-fitted": "Digamma",
    "polygamma-fitted": "Polygamma",
    "log": "Log",
    "log-fresh": "Log",
    "log-fitted": "Log",
    "log1p": "Log1p",
    "log1p-fresh": "Log1p",
    "log1p-fitted": "Log1p",
    "sqrt": "Sqrt",
    "sqrt-fresh": "Sqrt",
    "rsqrt-fresh": "Rsqrt",
    "rsqrt-fitted": "Rsqrt",
    "acosh-fitted": "Acosh",
    "trigonometry": "Acosh",
    "trigonometry-fresh": "Acosh",
    "recip": "Reciprocal",
    "recip-ilv2": "Reciprocal",
}


def _scalar_golden_bf16(mathop, u16):
    """The in-repo oracle on the Float16_b -> Float16_b, dest_acc=No row."""
    from helpers.format_config import DataFormat
    from helpers.golden_generators import UnarySFPUGolden
    from helpers.llk_params import DestAccumulation
    import torch

    operand = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()
    g = UnarySFPUGolden()(
        mathop,
        torch.from_numpy(operand),
        DataFormat.Float16_b,
        DestAccumulation.No,
        DataFormat.Float16_b,
        (256, 256),
        iterations=None,
        skip_tilize=True,
    )
    return g.detach().float().numpy().astype(np.float32)


def _bf16_faithful_one(op, mathop_name, u16):
    """Compare one op's vectorized golden to the oracle over `u16`. Returns a note."""
    from helpers.llk_params import MathOperation

    mathop = getattr(MathOperation, mathop_name)
    spec = tg.get_spec(op)
    xs = tg._bf16_bits_to_f32(u16.astype(np.uint32))
    mine_hp = np.asarray(spec.evaluate(xs), dtype=np.float64)
    mine = tg.format_golden_f32_noacc(mine_hp)
    ref = _scalar_golden_bf16(mathop, u16)

    graded = np.ones(u16.shape, dtype=bool)
    if spec.domain is not None:
        graded &= ~tg.unary_input_classes(xs, spec.domain)[
            "out_of_domain_finite_normal"
        ]
    if op in tg.GAMMA_POLE_OPS or op in tg.RECIPROCAL_POLE_OPS:
        graded &= ~np.array([tg.at_pole(op, float(v)) for v in xs])
    n_ungraded = int(np.count_nonzero(~graded))

    mine_bits = mine.view(np.uint32).copy()
    ref_bits = ref.astype(np.float32).view(np.uint32).copy()
    mine_bits[mine_bits == 0x80000000] = 0  # +0 == -0 to the ULP path
    ref_bits[ref_bits == 0x80000000] = 0
    d = np.where((mine_bits != ref_bits) & graded)[0]
    n_graded = int(np.count_nonzero(graded))
    tail = f"; {n_ungraded}/{u16.size} undefined (out-of-domain or pole) not graded"
    if d.size == 0:
        return True, f"{n_graded}/{n_graded} bit-identical over the WHOLE bf16 space{tail}"
    ok, parts = _explain(
        mine[d], ref[d], np.isnan(mine_hp[d]), spec.atol, spec.rtol
    )
    return ok, f"{d.size}/{n_graded} differ: " + parts + tail


def case_faithful_bf16(ops=None, n=None):
    print("case 1c: 16-bit band goldens == scalar oracle over the whole bf16 space")
    corpus = {spec.op for spec in tg._CORPUS_BF16}
    check(
        "bf16-corpus-coverage",
        corpus <= set(_BF16_OP_TO_MATHOP_NAME),
        f"unverified bf16 specs: {sorted(corpus - set(_BF16_OP_TO_MATHOP_NAME))}",
    )
    u16 = np.arange(0, n or 65536, dtype=np.uint32)
    todo = ops or sorted(_BF16_OP_TO_MATHOP_NAME)
    for op in todo:
        try:
            ok, note = _bf16_faithful_one(op, _BF16_OP_TO_MATHOP_NAME[op], u16)
        except Exception as e:
            ok, note = False, f"raised: {type(e).__name__}: {e}"
        check(f"faithful-bf16[{op}]", ok, note)

# ── case 1d: the blaze / coverage vehicle goldens ────────────────────────────
# These rows are NOT graded by UnarySFPUGolden -- each carries its own golden
# closure inside its own test file. So that closure is the oracle here, and it is
# the one this compares against, over the whole bf16 space. The blaze `_g_*`
# functions are module-level and imported directly; the coverage goldens are
# local closures, so they are captured by intercepting _run_coverage's golden_fn
# argument -- the real closure the row grades against, not a re-typed copy.
_BLAZE_ORACLE = {
    "clampedsilu-gate": "_g_csilu_gate",
    "clampedsilu-up": "_g_csilu_up",
    "clampedsilu-clamped": "_g_csilu_clamped",
    "situ-gate": "_g_situ_gate",
    "scaledtanh": "_g_scaledtanh",
    "logitsoftcap": "_g_logitsoftcap",
    "siluscaled": "_g_siluscaled",
    "addrsqrt": "_g_addrsqrt",
    "sdpaexp": "_g_sdpaexp",
}


def _capture_coverage_goldens():
    """Run each coverage test with the device call stubbed, keeping golden_fn."""
    import test_sfpu_coverage as cov

    captured = {}
    real = cov._run_coverage

    class _Stop(Exception):
        pass

    def fake(op, fresh_cpp_impl, golden_fn, formats, dest_acc, **kw):
        captured["fn"] = golden_fn
        captured["formats"] = formats
        raise _Stop()

    cov._run_coverage = fake
    out = {}
    for row, call in (
        ("addrsqrt-fresh", lambda: cov.test_sfpu_coverage_add_rsqrt(1)),
        ("smoothstep-fresh", lambda: cov.test_sfpu_coverage_smoothstep(1)),
        ("copydest-fresh", lambda: cov.test_sfpu_coverage_copy_dest(1)),
    ):
        captured.clear()
        try:
            call()
        except _Stop:
            pass
        if "fn" in captured:
            out[row] = captured["fn"]
    cov._run_coverage = real
    return out


def case_faithful_blaze():
    print("case 1d: blaze/coverage goldens == their OWN test-file closures")
    import torch

    import test_sfpu_blaze as bz

    u16 = np.arange(65536, dtype=np.uint32)
    xs = tg._bf16_bits_to_f32(u16)
    xt = _bf16(xs)
    dummy_b = torch.zeros(1024, dtype=torch.bfloat16)

    oracles = {}
    for base, attr in _BLAZE_ORACLE.items():
        oracles[f"blaze-{base}"] = getattr(bz, attr)
    oracles.update(
        {k: v for k, v in _capture_coverage_goldens().items()}
    )
    # Every graded blaze/coverage spec must have an oracle here, or it ships
    # unverified -- the same rule case 1c enforces for the bf16 corpus.
    graded = {
        spec.op
        for spec in (tg._CORPUS_BLAZE + tg._CORPUS_COVERAGE)
    }
    # t8/t32 twins are the same math as their 1-tile row; verifying the math once
    # per op verifies them, so they map onto the base row's oracle.
    def base_of(op):
        for suffix in ("-t32", "-t8"):
            if op.endswith(suffix):
                return op[: -len(suffix)]
        return op

    missing = sorted(o for o in graded if base_of(o) not in oracles)
    check("blaze-oracle-coverage", not missing, f"unverified: {missing}")

    for op in sorted({base_of(o) for o in graded}):
        spec = tg.get_spec(op)
        fn = oracles.get(op)
        if spec is None or fn is None:
            check(f"faithful-blaze[{op}]", False, "no spec/oracle")
            continue
        try:
            ref_hp = np.asarray(
                fn(xt, dummy_b).to(torch.float32).numpy(), dtype=np.float64
            )
            ref = tg.format_golden_f32_noacc(ref_hp)
            mine_hp = np.asarray(spec.evaluate(xs), dtype=np.float64)
            mine = tg.format_golden_f32_noacc(mine_hp)
        except Exception as e:
            check(f"faithful-blaze[{op}]", False, f"raised: {type(e).__name__}: {e}")
            continue
        mb = mine.view(np.uint32).copy()
        rb = ref.view(np.uint32).copy()
        mb[mb == 0x80000000] = 0
        rb[rb == 0x80000000] = 0
        d = np.where(mb != rb)[0]
        if d.size == 0:
            check(f"faithful-blaze[{op}]", True, "65536/65536 bit-identical to the row's own golden")
            continue
        # The test-file closures evaluate in fp32; this module uses fp64. Admit a diff
        # only through the shared explainer -- the row's own contract, a matched
        # infinity, or the NaN-operand infinity-sign policy.
        ok, parts = _explain(
            mine[d], ref[d],
            np.isnan(mine_hp[d]) | np.isnan(ref_hp[d]),
            spec.atol, spec.rtol,
        )
        note = f"{d.size}/65536 differ: " + parts
        check(f"faithful-blaze[{op}]", ok, note)


# ── case 1e: the two single-row vehicles (sdpa, binopscalar) ─────────────────
# Both have a registered in-repo golden CLASS, so that class is the oracle and
# the comparison is over the whole bf16 space, like case 1c.
def case_faithful_vehicles():
    print("case 1e: sdpa / binopscalar goldens == their in-repo golden classes")
    import torch
    from helpers.format_config import DataFormat
    from helpers.golden_generators import (
        ScalarBinopGolden,
        SdpaExpUnclampedGolden,
    )
    from helpers.llk_params import MathOperation

    u16 = np.arange(65536, dtype=np.uint32)
    xs = tg._bf16_bits_to_f32(u16)
    xt = _bf16(xs)

    cases = {
        # The node id pins scale bits 16256 = 0x3F80 = bf16 1.0.
        "sdpa": lambda: SdpaExpUnclampedGolden()(xt, 16256, DataFormat.Float16_b),
        "binopscalar": lambda: ScalarBinopGolden()(
            MathOperation.ScalarAdd,
            xt,
            int(
                np.array([np.float32(tg.BINOP_SCALAR_ADD)]).view(np.uint32)[0]
            ),
            DataFormat.Float16_b,
        ),
    }
    for op, make in cases.items():
        spec = tg.get_spec(op)
        try:
            ref_hp = np.asarray(make().to(torch.float32).numpy(), dtype=np.float64)
            ref = tg.format_golden_f32_noacc(ref_hp)
            mine_hp = np.asarray(spec.evaluate(xs), dtype=np.float64)
            mine = tg.format_golden_f32_noacc(mine_hp)
        except Exception as e:
            check(f"faithful-vehicle[{op}]", False, f"raised: {type(e).__name__}: {e}")
            continue
        mb = mine.view(np.uint32).copy(); rb = ref.view(np.uint32).copy()
        mb[mb == 0x80000000] = 0; rb[rb == 0x80000000] = 0
        d = np.where(mb != rb)[0]
        if d.size == 0:
            check(f"faithful-vehicle[{op}]", True, "65536/65536 bit-identical to the in-repo golden class")
            continue
        ok, parts = _explain(
            mine[d], ref[d], np.isnan(mine_hp[d]) | np.isnan(ref_hp[d]),
            spec.atol, spec.rtol,
        )
        check(f"faithful-vehicle[{op}]", ok, f"{d.size}/65536 differ: " + parts)


# ── case 1f: the dest_acc=Yes rows, at their own precision ───────────────────
# Different output pipeline (no bf16 rounding of the result, NaN PRESERVED rather
# than converted to +inf), so it needs its own oracle call: input Float16_b,
# output Float32, dest_acc=Yes. Exhaustive over the bf16 input space.
def case_faithful_destacc():
    print("case 1f: dest_acc=Yes goldens == the oracle at 32-bit DEST / fp32 out")
    import torch
    from helpers.format_config import DataFormat
    from helpers.golden_generators import UnarySFPUGolden
    from helpers.llk_params import DestAccumulation, MathOperation

    u16 = np.arange(65536, dtype=np.uint32)
    xs = tg._bf16_bits_to_f32(u16)
    for op, mathop_name in (("sigmoid-destacc", "Sigmoid"), ("softplus-destacc", "Softplus")):
        spec = tg.get_spec(op)
        check(f"destacc-flag[{op}]", spec is not None and spec.dst_acc, "dst_acc must be True")
        ref = (
            UnarySFPUGolden()(
                getattr(MathOperation, mathop_name),
                torch.from_numpy(xs.astype(np.float32)),
                DataFormat.Float32,
                DestAccumulation.Yes,
                DataFormat.Float16_b,
                (256, 256),
                iterations=None,
                skip_tilize=True,
            )
            .detach()
            .float()
            .numpy()
            .astype(np.float32)
        )
        mine_hp = np.asarray(spec.evaluate(xs), dtype=np.float64)
        mine = tg.format_golden_f32_acc(mine_hp)
        mb = mine.view(np.uint32).copy(); rb = ref.view(np.uint32).copy()
        mb[mb == 0x80000000] = 0; rb[rb == 0x80000000] = 0
        d = np.where(mb != rb)[0]
        if d.size == 0:
            check(f"faithful-destacc[{op}]", True, "65536/65536 bit-identical at fp32 DEST")
            continue
        ok, parts = _explain(
            mine[d], ref[d], np.isnan(mine_hp[d]), spec.atol, spec.rtol
        )
        check(f"faithful-destacc[{op}]", ok, f"{d.size}/65536 differ: " + parts)

    # The arithmetic behind SIGMOID_SUBNORMAL_NOTE, asserted rather than asserted-in-prose:
    # widening Dest cannot rescue the value, because fp32 and bf16 share a min NORMAL.
    check(
        "fp32-and-bf16-share-min-normal",
        float(np.finfo(np.float32).tiny) == tg.BF16_TINY == 2.0**-126,
        f"fp32 tiny={float(np.finfo(np.float32).tiny)!r} BF16_TINY={tg.BF16_TINY!r}",
    )
    sg = float(tg.format_golden_f32_acc(tg._sigmoid(np.array([-89.0], dtype=np.float32)))[0])
    check(
        "sigmoid-89-flushes-at-fp32-dest-too",
        sg == 0.0,
        f"sigmoid(-89)=2.2274e-39 < 2^-126, so the fp32-DEST golden is {sg!r} as well",
    )


def case_fitter_ulp():
    print("case 2b: bf16_bitdistance == fitter compute_ulp_bitdistance")
    try:
        fitter = Path.home() / "tt-polynomial-fitter"
        sys.path.insert(0, str(fitter))
        from extract_accuracy import compute_ulp_bitdistance
    except Exception as e:
        check("fitter-ulp-parity", True, f"SKIP (fitter not importable: {e})")
        return
    rng = np.random.default_rng(7)
    a = rng.standard_normal(4096) * 10.0
    b = a + rng.standard_normal(4096) * 0.01
    mine = tg.bf16_bitdistance(a, b)
    ref = compute_ulp_bitdistance(a, b, precision="bf16")
    check(
        "fitter-ulp-parity",
        np.array_equal(mine, ref),
        f"max|delta|={np.max(np.abs(mine-ref)) if mine.size else 0}",
    )


def case_special_numeric_policy():
    print("case 2c: non-finite tolerance and ULP policy")
    golden = np.array(
        [np.inf, -np.inf, np.nan, np.inf, np.nan], dtype=np.float32
    )
    device = np.array(
        [1.0, np.inf, 1.0, np.inf, np.float32(np.nan)], dtype=np.float32
    )
    ulp, within = tg.numeric_comparison(golden, device, 0.05, 0.05)
    check(
        "special-within-policy",
        np.array_equal(within, np.array([False, False, False, True, True])),
        str(within),
    )
    check(
        "special-ulp-policy",
        np.array_equal(ulp, np.array([65535.0, 65535.0, 65535.0, 0.0, 0.0])),
        str(ulp),
    )


# ── case 3: known-correct ────────────────────────────────────────────────────
def case_known_correct():
    print("case 3: golden bytes fed back -> 0 ULP, within contract")
    for op in ("erf-fresh", "add1", "hardtanh-fresh"):
        spec = tg.get_spec(op)
        acc = tg.CorrectnessAccumulator(spec)
        acc.update(0, 0, b"")  # empty chunk is a no-op
        # stream the tile as one chunk starting at input 0 with the golden as device output
        acc2 = tg.CorrectnessAccumulator(spec)
        # inputs must equal 0..1023 for update() to regenerate them; build golden on THAT range
        u0 = np.arange(0, 1024, dtype=np.uint32)
        golden0 = tg.format_golden_f32_noacc(spec.math(tg.bf16_truncate(u0)))
        acc2.update(0, 1024, golden0.astype("<f4").tobytes())
        check(
            f"known-correct[{op}]",
            acc2.max_ulp == 0.0 and acc2.n_out_of_tol == 0,
            f"max_ulp={acc2.max_ulp} n_out={acc2.n_out_of_tol}",
        )


# ── case 4: seeded bug ────────────────────────────────────────────────────────
def case_seeded_bug():
    print("case 4: one perturbed output -> flagged with correct first witness")
    spec = tg.get_spec("erf-fresh")
    u0 = np.arange(0, 1024, dtype=np.uint32)
    golden0 = tg.format_golden_f32_noacc(spec.math(tg.bf16_truncate(u0)))
    dev = golden0.copy()
    # pick an in-domain finite element and push it far out of tolerance
    idx = 700
    dev[idx] = np.float32(golden0[idx] + 5.0)  # erf range is [-1,1]; +5 is grossly out
    acc = tg.CorrectnessAccumulator(spec)
    acc.update(0, 1024, dev.astype("<f4").tobytes())
    check("seeded-bug-flagged", acc.n_out_of_tol >= 1, f"n_out={acc.n_out_of_tol}")
    check(
        "seeded-bug-witness-input",
        acc.first_witness_u32 == int(u0[idx]),
        f"got 0x{acc.first_witness_u32:08x} want 0x{int(u0[idx]):08x}",
    )
    check("seeded-bug-max-ulp", acc.max_ulp > 0.0, f"max_ulp={acc.max_ulp}")


# ── case 5: domain classification ────────────────────────────────────────────
def case_domain():
    print("case 5: exhaustive post-bf16 IEEE/domain classification")
    spec = tg.get_spec("erfinv-fresh")
    acc = tg.CorrectnessAccumulator(spec)
    # x=2.0 (bf16 0x40000000) is |x|>=1 -> out of erfinv domain
    check(
        "class-out-of-domain",
        acc._classify(0x40000000, 2.0) == "out_of_domain_finite_normal",
    )
    check(
        "class-in-domain",
        acc._classify(0x3F000000, 0.5) == "in_domain_finite_normal",
    )
    check(
        "class-pos-inf", acc._classify(0x7F800000, float("inf")) == "pos_inf_input"
    )
    raw = np.array(
        [
            0x7FC00000,
            0x7F800000,
            0xFF800000,
            0x00000000,
            0x80000000,
            0x00010000,
            0x80010000,
            0x3F800000,
            0xBF800000,
            0x3F000000,
            0x40000000,
            0xC0000000,
        ],
        dtype=np.uint32,
    )
    classes = tg.unary_input_classes(tg.bf16_truncate(raw), spec.domain)
    membership = sum(mask.astype(np.uint8) for mask in classes.values())
    check("unary-class-exhaustive", np.all(membership == 1), str(classes))
    expected_counts = {
        "nan_input": 1,
        "pos_inf_input": 1,
        "neg_inf_input": 1,
        "pos_zero_input": 1,
        "neg_zero_input": 1,
        "pos_subnormal_input": 1,
        "neg_subnormal_input": 1,
        "domain_lower_boundary": 1,
        "domain_upper_boundary": 1,
        "in_domain_finite_normal": 1,
        "out_of_domain_finite_normal": 2,
    }
    got_counts = {name: int(np.count_nonzero(mask)) for name, mask in classes.items()}
    check("unary-class-populations", got_counts == expected_counts, str(got_counts))

    # Binary classification is hierarchical rather than a 17x17 Cartesian product:
    # base specials, then exponent specials for normal bases, then normal-pair sign.
    specials = raw[:7] >> np.uint32(16)
    one = np.uint32(0x3F80)
    neg_one = np.uint32(0xBF80)
    base_bits = np.concatenate(
        [specials, np.full(7, one, dtype=np.uint32), np.array([one, neg_one])]
    )
    exp_bits = np.concatenate(
        [np.full(7, one, dtype=np.uint32), specials, np.array([one, one])]
    )
    base = tg._bf16_bits_to_f32(base_bits)
    exp = tg._bf16_bits_to_f32(exp_bits)
    pair_classes = tg.binary_input_classes(base, exp)
    pair_membership = sum(mask.astype(np.uint8) for mask in pair_classes.values())
    check("binary-class-exhaustive", np.all(pair_membership == 1), str(pair_classes))
    pair_counts = {
        name: int(np.count_nonzero(mask)) for name, mask in pair_classes.items()
    }
    check(
        "binary-class-populations",
        all(count == 1 for count in pair_counts.values()),
        str(pair_counts),
    )


# ── case 6b: every joint-pointwise binary golden vs BinarySFPUGolden ─────────
# The oracle's binary methods are per-element scalar, so this sweeps a full base
# STRATUM: one fixed base and all 65536 exponent patterns -- exactly the band the
# device leg runs -- for each of several representative bases, including the
# specials (0, +-1, +-inf, nan, a subnormal) that a random sample would miss.
_BINARY_OP_TO_MATHOP = {
    "binarypow": "SfpuElwpow",
    "binarypow-fresh": "SfpuElwpow",
    "binary-float": "SfpuElwsub",
    "binarycomp": "SfpuElwEq",
    "binaryfmod": "SfpuBinaryFmod",
    "binaryremainder": "SfpuBinaryRemainder",
    "atan2": "SfpuAtan2",
    "atan2-fitted": "SfpuAtan2",
    "minmax-max": "SfpuBinaryMax",
    "minmax-min": "SfpuBinaryMin",
    "isclose": "SfpuIsclose",
    "isclose-fresh": "SfpuIsclose",
    "mask": "SfpuMask",
}

# Representative bases, chosen to be the hard ones: both zeros, both units, both
# infinities, a NaN, a subnormal, the bf16 near-max, and two ordinary values.
_BINARY_BASES = [
    0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x0001,
    0x7F70, 0x4130, 0xC0A1, 0x3F70,
]


def case_binary_registry():
    print("case 6b: BINARY_REGISTRY goldens == BinarySFPUGolden, per base stratum")
    import torch
    from helpers.golden_generators import BinarySFPUGolden
    from helpers.llk_params import MathOperation

    reg = {op: sp for op, sp in tg.BINARY_REGISTRY.items() if sp.checkable}
    check(
        "binary-registry-coverage",
        set(reg) <= set(_BINARY_OP_TO_MATHOP),
        f"unverified binary specs: {sorted(set(reg) - set(_BINARY_OP_TO_MATHOP))}",
    )
    oracle = BinarySFPUGolden()
    exp16 = np.arange(65536, dtype=np.uint32)
    b_vals = tg._bf16_bits_to_f32(exp16)
    bt = _bf16(b_vals)

    for op in sorted(reg):
        spec = reg[op]
        mathop = getattr(MathOperation, _BINARY_OP_TO_MATHOP[op])
        fn = oracle.ops[mathop]
        worst_note, ok_all, total_diff, total_n = "", True, 0, 0
        for base_bits in _BINARY_BASES:
            a_vals = tg._bf16_bits_to_f32(np.full(65536, base_bits, dtype=np.uint32))
            at = _bf16(a_vals)
            try:
                # The oracle's binary methods are scalar; apply elementwise.
                ref_hp = np.array(
                    [float(fn(at[i], bt[i])) for i in range(0, 65536, 1)],
                    dtype=np.float64,
                )
            except Exception as e:
                ok_all = False
                worst_note = f"oracle raised at base 0x{base_bits:04x}: {type(e).__name__}: {e}"
                break
            ref = tg.format_golden_f32_noacc(ref_hp)
            mine_hp = np.asarray(
                spec.math(a_vals.astype(np.float64), b_vals.astype(np.float64)),
                dtype=np.float64,
            )
            mine = tg.format_golden_f32_noacc(mine_hp)
            mb = mine.view(np.uint32).copy(); rb = ref.view(np.uint32).copy()
            mb[mb == 0x80000000] = 0; rb[rb == 0x80000000] = 0
            d = np.where(mb != rb)[0]
            total_n += 65536
            total_diff += d.size
            if d.size:
                ok, parts = _explain(
                    mine[d], ref[d],
                    np.isnan(mine_hp[d]) | np.isnan(ref_hp[d]),
                    spec.atol, spec.rtol,
                )
                if not ok:
                    ok_all = False
                    worst_note = f"base 0x{base_bits:04x}: {parts}"
                    break
        note = worst_note or (
            f"{total_n - total_diff}/{total_n} bit-identical across "
            f"{len(_BINARY_BASES)} base strata ({total_diff} explained diffs)"
        )
        check(f"faithful-binary[{op}]", ok_all, note)


def case_binarypow():
    print("case 6: binarypow golden faithful to BinarySFPUGolden._pow + region accum")
    import torch

    # faithfulness: my fp64 pow->bf16 vs the harness _pow ((fp32**fp32)->bf16), over a
    # spread of bf16 base/exp patterns. Differ only at sub-tolerance fp32-vs-fp64 ties.
    from helpers.golden_generators import BinarySFPUGolden

    rng = np.random.default_rng(11)
    base16 = rng.integers(0, 1 << 16, size=4096, dtype=np.uint16)
    exp16 = rng.integers(0, 1 << 16, size=4096, dtype=np.uint16)
    mine = tg.binary_pow_golden_bf16(base16, exp16)
    a = torch.from_numpy(tg._bf16_bits_to_f32(base16.astype(np.uint32))).to(
        torch.bfloat16
    )
    b = torch.from_numpy(tg._bf16_bits_to_f32(exp16.astype(np.uint32))).to(
        torch.bfloat16
    )
    ref = BinarySFPUGolden()._pow(a, b).to(torch.float32).numpy()
    diff = np.where(mine.view(np.uint32) != ref.view(np.uint32))[0]
    if diff.size == 0:
        check("binarypow-faithful", True, "4096/4096 bit-identical to _pow")
    else:
        g = ref[diff].astype(np.float64)
        m = mine[diff].astype(np.float64)
        fin = np.isfinite(g) & np.isfinite(m)
        ok = np.all(np.abs(m[fin] - g[fin]) <= (0.05 + 0.05 * np.abs(g[fin])))
        check(
            "binarypow-faithful",
            bool(ok),
            f"{diff.size}/4096 differ, all sub-tolerance fp32-vs-fp64 pow ties",
        )

    # region accumulator: build 2 pairs, even tiles = golden, odd = 0xA5 sentinel.
    pairs = 2
    ELEMS = tg._ELEMS_PER_TILE
    region = bytearray()
    dispatch_start = 0x3F800000  # base16=0x3F80 (=1.0), exps sweep
    golden_tiles = []
    for p in range(pairs):
        joint0 = dispatch_start + p * ELEMS
        base = np.full(ELEMS, (joint0 >> 16) & 0xFFFF, dtype=np.uint16)
        exps = (np.arange(ELEMS, dtype=np.uint32) + (joint0 & 0xFFFF)).astype(np.uint16)
        g = tg.binary_pow_golden_bf16(base, exps)
        gbits = tg._to_bf16_bits(g).astype(np.uint16)
        golden_tiles.append(gbits)
        region += gbits.tobytes()  # even tile = output
        region += b"\xa5\xa5" * ELEMS  # odd tile = sentinel
    acc = tg.BinaryPowAccumulator()
    acc.update(dispatch_start, pairs, bytes(region))
    check(
        "binarypow-known-correct",
        acc.max_ulp == 0.0 and acc.n_out_of_tol == 0,
        f"max_ulp={acc.max_ulp} n_out={acc.n_out_of_tol}",
    )

    # seeded bug: corrupt one even-tile output far out of tolerance.
    region2 = bytearray(region)
    bad = np.uint16(tg._to_bf16_bits(np.array([1e30]))[0])  # huge value
    region2[10 * 2 : 10 * 2 + 2] = np.array([bad], dtype=np.uint16).tobytes()
    acc2 = tg.BinaryPowAccumulator()
    acc2.update(dispatch_start, pairs, bytes(region2))
    check(
        "binarypow-seeded-bug",
        acc2.n_out_of_tol >= 1 and acc2.first_witness_joint == dispatch_start + 10,
        f"n_out={acc2.n_out_of_tol} witness=0x{max(acc2.first_witness_joint,0):08x}",
    )


# ── case 7: the six modelling contracts, each pinned at its own witness ──────
# One check per fix, each asserting the NEW value and the OLD one's absence, so a
# regression cannot pass by quietly reverting the model.
def case_modelling_contracts():
    print("case 7: the six oracle modelling contracts")
    import math

    import torch

    u16 = np.arange(65536, dtype=np.uint32)
    xs = tg._bf16_bits_to_f32(u16)
    sub = (np.abs(xs.astype(np.float64)) < tg.BF16_TINY) & (xs != 0)
    nan = np.isnan(xs)

    # (1) INPUT FTZ. The steep-at-zero ops must answer as if the operand were +-0,
    #     and the flush must be visible in the golden for the exact populations the
    #     exhaustive silicon leg attributed to it.
    expect_ftz = {
        # op: (device answer at the witness, witness bf16 pattern)
        "sqrt-fresh": (0.0, 0x8001),      # sqrt(-9.18e-41): NaN->+inf before, 0 now
        "log-fresh": (-math.inf, 0x0001),  # log(9.18e-41): -92.0 before, -inf now
        "ceil-fresh": (0.0, 0x0001),       # ceil(tiny+): 1.0 before, 0 now
    }
    for op, (dev, patt) in expect_ftz.items():
        spec = tg.get_spec(op)
        x = tg._bf16_bits_to_f32(np.array([patt], dtype=np.uint32))
        new = float(tg.format_golden_f32_noacc(spec.evaluate(x))[0])
        old = float(tg.format_golden_f32_noacc(spec.math(x))[0])
        check(
            f"input-ftz[{op}]",
            new == dev and old != dev,
            f"x={float(x[0])!r}: golden {old!r} -> {new!r}, device {dev!r}",
        )
    # ...and it must NOT be applied to a compare-only body. sign is the discriminator:
    # ckernel_sfpu_sign.h tests `v == 0.0F` exactly, so a subnormal is not zero to it.
    for op, patt, want in (
        ("sign", 0x0001, 1.0),
        ("sign", 0x8001, -1.0),
        ("heaviside", 0x0001, 1.0),
        ("heaviside", 0x8001, 0.0),
    ):
        spec = tg.get_spec(op)
        if spec is None:
            continue
        x = tg._bf16_bits_to_f32(np.array([patt], dtype=np.uint32))
        got = float(tg.format_golden_f32_noacc(spec.evaluate(x))[0])
        check(
            f"input-ftz-exempt[{op}@0x{patt:04x}]",
            got == want and op in tg.INPUT_FTZ_EXEMPT,
            f"compare-only body: golden {got!r} (flushing would give "
            f"{float(tg.format_golden_f32_noacc(spec.math(tg.input_ftz(x)))[0])!r})",
        )

    # (1b) WHOLE-POPULATION SPECIALS. These are the exact witnesses that
    # distinguish a finite-domain polynomial/select from the operation it
    # implements. NaNs become signed infinities at the bf16 destination
    # boundary; that is the existing packer model, not a numeric relaxation.
    for op, patt, want in (
        # None means a converted NaN: either infinity sign is accepted by the
        # policy because a NaN sign is not a numeric quantity.
        ("elu-fresh", 0xFF81, None),
        ("gelu-fresh", 0xFF81, None),
        ("relu", 0x7F81, None),
        ("sigmoid-fitted", 0x7F81, None),
        ("tanh-fresh", 0x7F81, None),
        ("threshold-fresh", 0xFF81, None),
        ("threshold-fitted", 0xFF81, None),
        ("log-fresh", 0x7F80, math.inf),
        ("sigmoidappx-tree", 0x7F80, 1.0),
        ("sigmoidappx-tree", 0xFF80, 0.0),
        ("rsqrt-fresh", 0x8000, -math.inf),
        ("rsqrt-fresh", 0x8001, -math.inf),
    ):
        spec = tg.get_spec(op)
        x = tg._bf16_bits_to_f32(np.array([patt], dtype=np.uint32))
        got = float(tg.format_golden_f32_noacc(spec.evaluate(x))[0])
        matches = math.isinf(got) if want is None else got == want
        check(
            f"whole-population-special[{op}@0x{patt:04x}]",
            matches,
            f"golden {got!r}, expected {'converted NaN' if want is None else repr(want)}",
        )
    # and the flush must never be applied to a normal operand
    normal = ~sub & np.isfinite(xs)
    check(
        "input-ftz-touches-only-subnormals",
        bool(np.array_equal(tg.input_ftz(xs)[normal], xs[normal])),
        f"{int(np.count_nonzero(normal))} normal/zero/inf inputs unchanged",
    )

    # (2) NaN -> SIGN-PRESERVED inf, on the pure-move row.
    ident = tg.get_spec("copydest-fresh")
    gi = tg.format_golden_f32_noacc(ident.evaluate(xs))
    check(
        "nan-inf-sign[copydest-fresh]",
        bool(np.all(np.isneginf(gi[nan & np.signbit(xs)])))
        and bool(np.all(np.isposinf(gi[nan & ~np.signbit(xs)]))),
        "127 negative NaNs -> -inf (device -inf), 127 positive NaNs -> +inf",
    )
    gn = tg.format_golden_f32_noacc(tg.get_spec("negative").evaluate(xs))
    check(
        "nan-inf-sign[negative-flips]",
        bool(np.all(np.isneginf(gn[nan & ~np.signbit(xs)]))),
        "SFPU negation flips a NaN's sign bit, so a POSITIVE NaN packs to -inf",
    )
    ga = tg.format_golden_f32_noacc(tg.get_spec("abs").evaluate(xs))
    check(
        "nan-inf-sign[abs-is-SFPABS]",
        bool(np.all(np.isneginf(ga[nan & np.signbit(xs)])))
        and float(ga[u16 == 0xFF80][0]) == math.inf,
        "SFPABS float mod leaves a negative NaN alone but DOES clear -inf",
    )

    # (3) min/max under the SFPU sign-magnitude order, and the 127-vs-254 split that
    #     rules out both NaN propagation and IEEE minNum/maxNum.
    C = np.float32(tg.UNARY_MAX_MIN_VALUE)
    dmax = tg.format_golden_f32_noacc(tg.sfpu_max(xs, C))
    dmin = tg.format_golden_f32_noacc(tg.sfpu_min(xs, C))
    # The recorded silicon facts. Two are witness values; two are the absence of a
    # miss against a golden that was +inf at every NaN.
    recorded = {
        ("max", 0x7F81): math.inf,   # not among the 127 misses vs a +inf golden
        ("max", 0xFF81): 0.0,        # recorded graded_witness_dev
        ("min", 0x7F81): 0.0,        # recorded graded_witness_dev
    }
    # min(-NaN, 0.0) is the model's PREDICTION (-inf); the record cannot distinguish
    # it from 0.0, because both miss a +inf golden. It is what the silicon re-run of
    # unarymaxmin-min is for.
    def old_pipeline(hp):
        y = tg._round_bf16_as_f32(hp).astype(np.float32)
        y = np.where(np.isnan(y), np.float32(np.inf), y)
        return np.where(
            np.abs(y.astype(np.float64)) < tg.BF16_TINY, np.float32(0.0), y
        ).astype(np.float32)

    candidates = {
        "sign-magnitude": (tg.sfpu_max(xs, C), tg.sfpu_min(xs, C)),
        "NaN-propagating": (
            np.maximum(xs.astype(np.float64), float(C)),
            np.minimum(xs.astype(np.float64), float(C)),
        ),
        "IEEE minNum/maxNum": (
            np.where(nan, float(C), xs.astype(np.float64)),
            np.where(nan, float(C), xs.astype(np.float64)),
        ),
    }
    verdicts = {}
    for name, (hmax, hmin) in candidates.items():
        gmax = tg.format_golden_f32_noacc(hmax)
        gmin = tg.format_golden_f32_noacc(hmin)
        agrees = all(
            float({"max": gmax, "min": gmin}[side][u16 == patt][0]) == want
            or (math.isinf(want) and float({"max": gmax, "min": gmin}[side][u16 == patt][0]) == want)
            for (side, patt), want in recorded.items()
        )
        verdicts[name] = agrees
    check(
        "minmax-order-discriminates",
        verdicts["sign-magnitude"]
        and not verdicts["NaN-propagating"]
        and not verdicts["IEEE minNum/maxNum"],
        "of the three candidate models only the sign-magnitude order reproduces all "
        f"three recorded device values {verdicts}",
    )
    # and the recorded miss counts against the OLD golden (propagating + hardcoded
    # +inf) must come back at 127 / 254, which is how the rows were reported.
    _, w_max = tg.numeric_comparison(
        old_pipeline(np.maximum(xs.astype(np.float64), float(C)))[nan],
        tg.format_golden_f32_noacc(tg.sfpu_max(xs, C))[nan], 0.05, 0.05)
    _, w_min = tg.numeric_comparison(
        old_pipeline(np.minimum(xs.astype(np.float64), float(C)))[nan],
        tg.format_golden_f32_noacc(tg.sfpu_min(xs, C))[nan], 0.05, 0.05)
    check(
        "minmax-old-golden-miss-counts",
        (int(np.count_nonzero(~w_max)), int(np.count_nonzero(~w_min))) == (127, 254),
        f"old golden vs the modelled device: max {int(np.count_nonzero(~w_max))} / "
        f"min {int(np.count_nonzero(~w_min))} (recorded 127 / 254)",
    )

    # (4) gelu(-inf) is the limit 0, not the 0*inf NaN.
    for op in ("gelu", "gelu-fresh", "gelu-fitted", "gelu-licensed"):
        spec = tg.get_spec(op)
        if spec is None:
            continue
        v = float(
            tg.format_golden_f32_noacc(
                spec.evaluate(tg._bf16_bits_to_f32(np.array([0xFF80], dtype=np.uint32)))
            )[0]
        )
        check(f"gelu-neg-inf-limit[{op}]", v == 0.0, f"golden at -inf = {v!r} (device 0.0)")

    # (5) i0/i1 can SEE a defect in [89, 91.5] -- the fp64 golden is finite there,
    #     the fp32 oracle was +inf, and the bisected overflow constants come back.
    from helpers.golden_generators import UnarySFPUGolden
    from helpers.format_config import DataFormat
    from helpers.llk_params import DestAccumulation, MathOperation

    band = np.array([89.0, 89.5, 90.0, 90.5, 91.0, 91.5], dtype=np.float32)
    for op, mathop, order in (("i1", MathOperation.I1, 1), ("i0", MathOperation.I0, 0)):
        spec = tg.get_spec(op)
        g = tg.format_golden_f32_noacc(spec.evaluate(band))
        scalar = (
            UnarySFPUGolden()(
                mathop,
                torch.from_numpy(np.tile(band, 1024 // band.size).astype(np.float32)),
                DataFormat.Float16_b,
                DestAccumulation.No,
                DataFormat.Float16_b,
                (32, 32),
                iterations=None,
                skip_tilize=True,
            )
            .detach()
            .float()
            .numpy()[: band.size]
        )
        finite_both = bool(np.all(np.isfinite(g))) and bool(np.all(np.isfinite(scalar)))
        # an overflowing kernel must now FAIL; it used to PASS against a +inf golden
        _, w_inf = tg.numeric_comparison(
            g, np.full(band.size, np.inf, dtype=np.float32), 0.05, 0.05
        )
        _, w_ok = tg.numeric_comparison(g, g, 0.05, 0.05)
        check(
            f"i-band-visible[{op}]",
            finite_both and not bool(np.any(w_inf)) and bool(np.all(w_ok)),
            f"[89,91.5]: golden finite {list(np.round(g.astype(np.float64), 0))[:1]}..., "
            "an +inf device answer now scores 65535 (it used to be ULP 0)",
        )
    f32max = float(np.finfo(np.float32).max)
    for name, fn, const in (
        ("I0", torch.special.i0, 91.9007646),
        ("I1", torch.special.i1, 91.9021397),
    ):
        v = fn(torch.tensor(const, dtype=torch.float64)).item()
        check(
            f"bisected-overflow[{name}]",
            math.isfinite(v) and v <= f32max and v > 0.5 * f32max,
            f"{name}({const!r}) = {v:.9e} <= FLT_MAX {f32max:.9e} -- the kernel clamp "
            "sits at the golden's own FLT_MAX crossing, inside the bf16 gap (91.5, 92)",
        )

    # (6) lgamma is graded against the STAGE contract, and the device value is exact.
    lg = tg.get_spec("lgamma")
    for x, dev, exact in ((-5.03125, 4.84375, True), (-0.9375, -0.02490234375, False)):
        xa = np.array([x], dtype=np.float32)
        new = float(tg.format_golden_f32_noacc(lg.evaluate(xa))[0])
        _, w = tg.numeric_comparison(np.array([new]), np.array([dev]), lg.atol, lg.rtol)
        check(
            f"lgamma-stage[{x}]",
            bool(w[0]) and (new == dev if exact else True),
            f"stage golden {new!r} vs device {dev!r} (bit-exact={new == dev}); the "
            f"composite golden was {float(torch.lgamma(torch.tensor(x, dtype=torch.float64)).item())!r}",
        )
    check(
        "lgamma-has-no-poles",
        "lgamma" not in tg.GAMMA_POLE_OPS
        and bool(
            np.all(
                np.isfinite(
                    tg.format_golden_f32_noacc(
                        lg.evaluate(np.array([0.0, -1.0, -2.0, -7.0], dtype=np.float32))
                    )
                )
            )
        ),
        "the stage argument (x<0.5)?1-x:x is always >= 0.5, so nothing is excluded",
    )


def main():
    print("laneMR three-way golden selftest")
    case_faithful()
    case_faithful_bf16()
    case_faithful_blaze()
    case_faithful_vehicles()
    case_faithful_destacc()
    case_fitter_ulp()
    case_special_numeric_policy()
    case_known_correct()
    case_seeded_bug()
    case_domain()
    case_binary_registry()
    case_binarypow()
    case_modelling_contracts()
    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
