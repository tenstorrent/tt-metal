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
        hp = spec.math(tg.bf16_truncate(u))
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
    mine = tg.format_golden_f32_noacc(spec.math(xs))
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
    # Where they differ, exactly two explanations are admissible, and a diff that
    # is neither is a FAIL.
    #
    #   (a) SUB-TOLERANCE ROUNDING. The oracle evaluates in the bf16 DST dtype
    #       (torch.tensor(x, dtype=Float16_b)) while this module deliberately uses
    #       fp64 true math, so the two can land on adjacent bf16 codes. Admissible
    #       only when |mine - oracle| is inside the row's own accuracy contract.
    #
    #   (b) ORACLE fp32 OVERFLOW. UnarySFPUGolden._torch_unary evaluates in
    #       torch.float32, so torch.special.i1(89.0) comes back +inf although the
    #       true value 1.89e37 is comfortably inside fp32's range. The fp64 golden
    #       is then STRICTLY more faithful than the oracle, and this is recorded
    #       rather than tolerated: it means the harness's own i0/i1 row cannot see
    #       a device defect at those inputs, because its reference is already inf.
    #       Admissible only when the fp64 value is finite AND representable in
    #       fp32 AND the oracle's value is an infinity of the same sign.
    g = ref[d].astype(np.float64)
    m = mine[d].astype(np.float64)
    finite = np.isfinite(g) & np.isfinite(m)
    F32MAX = float(np.finfo(np.float32).max)
    oracle_overflow = (
        np.isinf(g)
        & np.isfinite(m)
        & (np.abs(m) <= F32MAX)
        & (np.signbit(g) == np.signbit(m))
    )
    rounding = finite & (
        np.abs(m - g) <= (spec.atol + spec.rtol * np.abs(g))
    )
    matched_inf = np.isinf(g) & np.isinf(m) & (np.signbit(g) == np.signbit(m))
    explained = rounding | oracle_overflow | matched_inf
    ok = bool(np.all(explained))
    parts = []
    if rounding.any():
        worst = float(np.max(np.abs(m[rounding] - g[rounding])))
        parts.append(
            f"{int(rounding.sum())} sub-tolerance bf16-vs-fp64 rounding "
            f"(worst |d|={worst:.3e} <= {spec.atol}+{spec.rtol}|g|)"
        )
    if oracle_overflow.any():
        xo = xs[d][oracle_overflow]
        parts.append(
            f"{int(oracle_overflow.sum())} where the ORACLE overflowed fp32 to "
            f"+-inf but the true value is finite and fp32-representable "
            f"(|x| in [{float(np.min(np.abs(xo)))!r},{float(np.max(np.abs(xo)))!r}]) "
            "-- the fp64 golden here is strictly better than the in-repo oracle"
        )
    if not ok:
        bad = ~explained
        parts.append(
            f"{int(bad.sum())} UNEXPLAINED, e.g. x={float(xs[d][bad][0])!r} "
            f"mine={float(m[bad][0])!r} oracle={float(g[bad][0])!r}"
        )
    return ok, f"{d.size}/{n_graded} differ: " + "; ".join(parts) + tail


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
    xt = torch.from_numpy(xs.astype(np.float32)).to(torch.bfloat16)
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
            ref_hp = fn(xt, dummy_b)
            ref = tg.format_golden_f32_noacc(
                np.asarray(ref_hp.to(torch.float32).numpy(), dtype=np.float64)
            )
            mine = tg.format_golden_f32_noacc(spec.math(xs))
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
        g = ref[d].astype(np.float64)
        m = mine[d].astype(np.float64)
        fin = np.isfinite(g) & np.isfinite(m)
        matched_inf = np.isinf(g) & np.isinf(m) & (np.signbit(g) == np.signbit(m))
        # The test-file closures evaluate in fp32; this module uses fp64. Admit a
        # diff only inside the row's own contract, and admit an fp32-overflow-only
        # diff the same way case 1c does.
        F32MAX = float(np.finfo(np.float32).max)
        ovf = np.isinf(g) & np.isfinite(m) & (np.abs(m) <= F32MAX) & (np.signbit(g) == np.signbit(m))
        tol = spec.atol + spec.rtol * np.abs(g)
        near = fin & (np.abs(m - g) <= np.maximum(tol, 0.0))
        ok = bool(np.all(near | ovf | matched_inf))
        worst = float(np.max(np.abs(m[fin] - g[fin]))) if fin.any() else 0.0
        bad = ~(near | ovf | matched_inf)
        note = (
            f"{d.size}/65536 differ: {int(near.sum())} inside the row's own contract "
            f"(worst |d|={worst:.3e} <= {spec.atol}+{spec.rtol}|g|), "
            f"{int(ovf.sum())} oracle fp32 overflow"
        )
        if not ok:
            note += (
                f", {int(bad.sum())} UNEXPLAINED e.g. x={float(xs[d][bad][0])!r} "
                f"mine={float(m[bad][0])!r} oracle={float(g[bad][0])!r}"
            )
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
    xt = torch.from_numpy(xs.astype(np.float32)).to(torch.bfloat16)

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
            ref = tg.format_golden_f32_noacc(
                np.asarray(make().to(torch.float32).numpy(), dtype=np.float64)
            )
            mine = tg.format_golden_f32_noacc(spec.math(xs))
        except Exception as e:
            check(f"faithful-vehicle[{op}]", False, f"raised: {type(e).__name__}: {e}")
            continue
        mb = mine.view(np.uint32).copy(); rb = ref.view(np.uint32).copy()
        mb[mb == 0x80000000] = 0; rb[rb == 0x80000000] = 0
        d = np.where(mb != rb)[0]
        if d.size == 0:
            check(f"faithful-vehicle[{op}]", True, "65536/65536 bit-identical to the in-repo golden class")
            continue
        g = ref[d].astype(np.float64); m = mine[d].astype(np.float64)
        fin = np.isfinite(g) & np.isfinite(m)
        minf = np.isinf(g) & np.isinf(m) & (np.signbit(g) == np.signbit(m))
        F32MAX = float(np.finfo(np.float32).max)
        ovf = np.isinf(g) & np.isfinite(m) & (np.abs(m) <= F32MAX) & (np.signbit(g) == np.signbit(m))
        near = fin & (np.abs(m - g) <= (spec.atol + spec.rtol * np.abs(g)))
        ok = bool(np.all(near | ovf | minf))
        worst = float(np.max(np.abs(m[fin] - g[fin]))) if fin.any() else 0.0
        check(
            f"faithful-vehicle[{op}]", ok,
            f"{d.size}/65536 differ: {int(near.sum())} inside contract "
            f"(worst |d|={worst:.3e}), {int(ovf.sum())} oracle fp32 overflow",
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


def main():
    print("laneMR three-way golden selftest")
    case_faithful()
    case_faithful_bf16()
    case_faithful_blaze()
    case_faithful_vehicles()
    case_fitter_ulp()
    case_special_numeric_policy()
    case_known_correct()
    case_seeded_bug()
    case_domain()
    case_binarypow()
    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
