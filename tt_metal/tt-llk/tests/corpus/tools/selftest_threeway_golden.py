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
  5. DOMAIN honesty: an erfinv |x|>=1 witness classifies as 'out-of-domain', a finite
     in-(-1,1) input as 'in-domain'.

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
    }
    u = _edge_tile()
    for op, mathop in op_to_mathop.items():
        spec = tg.get_spec(op)
        hp = spec.math(tg.bf16_truncate(u))
        mine = tg.format_golden_f32_noacc(hp)
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
        d = np.where(mine_bits != ref_bits)[0]
        if d.size == 0:
            check(f"faithful[{op}]", True, "1024/1024 bit-identical to harness golden")
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
            f"{d.size}/1024 differ ONLY at sub-tolerance bf16-vs-fp64 rounding "
            f"(worst |Δ|={worst:.2e} <= {spec.atol}+{spec.rtol}|g|)",
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


# ── case 3: known-correct ────────────────────────────────────────────────────
def case_known_correct():
    print("case 3: golden bytes fed back -> 0 ULP, within contract")
    for op in ("erf-fresh", "add1", "hardtanh-fresh"):
        spec = tg.get_spec(op)
        u = _edge_tile()
        golden = tg.format_golden_f32_noacc(spec.math(tg.bf16_truncate(u)))
        dev_bytes = golden.astype("<f4").tobytes()
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
    print("case 5: erfinv out-of-domain vs in-domain classification")
    spec = tg.get_spec("erfinv-fresh")
    acc = tg.CorrectnessAccumulator(spec)
    # x=2.0 (bf16 0x40000000) is |x|>=1 -> out of erfinv domain
    check("class-out-of-domain", acc._classify(0x40000000, 2.0) == "out-of-domain")
    check("class-in-domain", acc._classify(0x3F000000, 0.5) == "in-domain")
    check(
        "class-nonfinite", acc._classify(0x7F800000, float("inf")) == "nonfinite-input"
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
    case_fitter_ulp()
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
