#!/usr/bin/env python3
"""Host proof for the positive-domain MulInt32 limb-2 source contract.

Run from ``tt_metal/tt-llk/tests`` with a dedicated proof environment::

    python3 -m venv .venv-formal
    .venv-formal/bin/pip install -r corpus/tools/requirements-formal.txt
    .venv-formal/bin/python corpus/selftest_mul_int32_limb2_contract.py
"""

import sys

try:
    import z3
except ImportError as exc:
    raise SystemExit(
        "z3-solver is required; install corpus/tools/requirements-formal.txt"
    ) from exc


FAILS = []


def check(name, condition, detail=""):
    if condition:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


# Model both source representations, including their shared final signed-to-SM
# conversion.  A 23-bit operand has no sign bit, so it excludes the distinct
# SM32 negative-zero encoding while still including ordinary zero.
a23, b23 = z3.BitVecs("a23 b23", 23)
a_raw = z3.ZeroExt(9, a23)
b_raw = z3.ZeroExt(9, b23)


def sm32_to_int32(value):
    magnitude = z3.ZeroExt(1, z3.Extract(30, 0, value))
    return z3.If(z3.Extract(31, 31, value) == 1, -magnitude, magnitude)


def int32_to_sm32(value):
    sign = z3.Extract(31, 31, value)
    magnitude = z3.If(sign == 1, -value, value)
    return z3.Concat(sign, z3.Extract(30, 0, magnitude))


def limb2(lhs, rhs):
    product64 = z3.ZeroExt(32, lhs) * z3.ZeroExt(32, rhs)
    lo = z3.ZeroExt(9, z3.Extract(22, 0, product64))
    hi = z3.Extract(54, 23, product64)
    return lo + (hi << 23)


old_product = limb2(sm32_to_int32(a_raw), sm32_to_int32(b_raw))
new_product = limb2(a_raw, b_raw)
expected32 = z3.Extract(
    31, 0, z3.ZeroExt(32, a_raw) * z3.ZeroExt(32, b_raw)
)
solver = z3.Solver()
solver.add(
    z3.Or(
        old_product != new_product,
        new_product != expected32,
        int32_to_sm32(old_product) != int32_to_sm32(new_product),
    )
)
check(
    "raw-positive and decoded-input pipelines are identical for all 23-bit operands",
    solver.check() == z3.unsat,
)

# The measured row's full representation proof: positive SM32 is raw unsigned
# magnitude, and its product remains below INT_MAX, so the retained vInt->SM32
# store round-trips the ordinary mathematical product.  Include zero even
# though random stimuli start at one, plus both upper-bound corners.
a32, b32 = z3.BitVecs("a32 b32", 32)
domain = z3.And(
    z3.ULE(a32, z3.BitVecVal(40000, 32)),
    z3.ULE(b32, z3.BitVecVal(40000, 32)),
)
product64 = z3.ZeroExt(32, a32) * z3.ZeroExt(32, b32)
solver = z3.Solver()
solver.add(domain, z3.UGE(product64, z3.BitVecVal(1 << 31, 64)))
check("row product never sets the signed bit", solver.check() == z3.unsat)


def encode_sm32(value):
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return 0x80000000 | ((-value) & 0x7FFFFFFF)
    return value


check(
    "retained INT_MIN store encoding is SM32 negative zero",
    encode_sm32(0x80000000) == 0x80000000,
)


for lhs, rhs in (
    (0, 0),
    (0, 40000),
    (1, 40000),
    (39999, 40000),
    (40000, 40000),
    (1 << 22, 512),       # exactly INT_MIN in the low 32-bit result
    ((1 << 22) + 1, 513), # bit 31 set with a nonzero magnitude
    ((1 << 22) + 1, (1 << 22) + 3), # product exceeds 2^32
):
    product = lhs * rhs
    lo = product & ((1 << 23) - 1)
    hi = product >> 23
    low32 = product & 0xFFFFFFFF
    recombined = (lo + (hi << 23)) & 0xFFFFFFFF
    check(
        f"directed edge {lhs}*{rhs}",
        recombined == low32
        and encode_sm32(recombined) == encode_sm32(low32),
    )

if FAILS:
    print(f"SELFTEST FAILURES: {len(FAILS)}", file=sys.stderr)
    raise SystemExit(1)
print("SELFTEST PASS: mul_int32 limb-2 contract")
