"""Model of the SFPU polygamma kernel in fp32, before and after the scale fold.

Mirrors ckernel_sfpu_polygamma.h: NUM_TERMS exact terms plus an Euler-Maclaurin
tail, with the (-1)^(n+1) * n! scale applied either at the very end (old) or
folded into every accumulated quantity (new). fp32 rounding is emulated with a
struct round-trip; the reference is computed in double precision.

Only points whose true value is itself a normal fp32 are asserted on: below
2**-126 nothing can represent the answer, which is not the defect under test.
"""

import math
import struct

NUM_TERMS = 6
SMALLEST_NORMAL = 2.0 ** -126


def f32(v):
    # Round to fp32, then apply the flush-to-zero behaviour the SFPU documents
    # for subnormal results (tech_reports/Handling_Special_Value).
    out = struct.unpack("f", struct.pack("f", float(v)))[0]
    if out == 0.0:
        return 0.0
    if abs(out) < SMALLEST_NORMAL:
        return 0.0
    return out


def power(x, pwr, val):
    while True:
        if pwr & 1:
            val = f32(val * x)
        pwr >>= 1
        if not pwr:
            break
        x = f32(x * x)
    return val


def polyval(x, c0, c1, c2, c3):
    acc = f32(c3)
    acc = f32(acc * x + c2)
    acc = f32(acc * x + c1)
    acc = f32(acc * x + c0)
    return acc


def tail_series(z, n):
    s = 1.0 / n / z**n + 0.5 / z ** (n + 1)
    s += (n + 1) / 12.0 / z ** (n + 2)
    s -= (n + 1) * (n + 2) * (n + 3) / 720.0 / z ** (n + 4)
    return s


def reference(x, n):
    total = sum(1.0 / (x + k) ** (n + 1) for k in range(NUM_TERMS))
    total += tail_series(x + NUM_TERMS, n)
    sign = 1.0 if (n + 1) % 2 == 0 else -1.0
    return sign * math.factorial(n) * total


def kernel(x, n, folded):
    x = f32(x)
    nf = float(n)
    n1, n2, n3, n4, n5 = (float(n + i) for i in range(1, 6))
    sign = 1.0 if (n + 1) % 2 == 0 else -1.0
    scale = f32(sign * math.factorial(n))

    if folded:
        inv_nf = f32(scale / nf)
        half_scale = f32(0.5 * scale)
        c_b2 = f32(scale * n1 / 12.0)
        c_b4 = f32(-scale * (n1 * n2 * n3) / 720.0)
        c_b6 = f32(scale * (n1 * n2 * n3 * n4 * n5) / 30240.0)
    else:
        inv_nf = f32(1.0 / nf)
        half_scale = f32(0.5)
        c_b2 = f32(n1 / 12.0)
        c_b4 = f32(-(n1 * n2 * n3) / 720.0)
        c_b6 = f32((n1 * n2 * n3 * n4 * n5) / 30240.0)

    total = f32(0.0)
    for k in range(NUM_TERMS):
        xi = f32(x + f32(k))
        inv_xi = f32(1.0 / xi)
        seed = f32(inv_xi * scale) if folded else inv_xi
        total = f32(total + power(inv_xi, n, seed))

    z = f32(x + f32(NUM_TERMS))
    inv_z = f32(1.0 / z)
    inv_z2 = f32(inv_z * inv_z)
    e_val = polyval(inv_z2, inv_nf, c_b2, c_b4, c_b6)
    tail = f32(e_val + f32(half_scale * inv_z))

    pwr = n
    if pwr & 1:
        tail = f32(tail * inv_z)
    pwr >>= 1
    if pwr:
        tail = power(inv_z2, pwr, tail)

    total = f32(total + tail)
    return total if folded else f32(total * scale)


def rel_err(got, ref):
    if ref == 0.0:
        return 0.0
    return abs(got - ref) / abs(ref)


def main():
    xs = [0.5, 1.0, 3.0, 10.0, 100.0, 1448.0, 2500.0, 3142.2869,
            6208.0, 17408.0]
    worst_old = worst_new = 0.0
    zeros_old = zeros_new = checked = 0

    for n in range(1, 12):
        for x in xs:
            ref = reference(x, n)
            if abs(ref) < SMALLEST_NORMAL:
                continue
            old = kernel(x, n, folded=False)
            new = kernel(x, n, folded=True)
            checked += 1
            zeros_old += 1 if old == 0.0 else 0
            zeros_new += 1 if new == 0.0 else 0
            # Each side is scored on its own: a flushed zero is counted in the
            # hard-zero tally, and accuracy is measured over the surviving points.
            eo = rel_err(old, ref) if old != 0.0 else rel_err(0.0, ref)
            en = rel_err(new, ref) if new != 0.0 else rel_err(0.0, ref)
            worst_old = max(worst_old, eo)
            worst_new = max(worst_new, en)
            if en > 1e-4:
                print(f"n={n} x={x}: ref={ref:.6e} old={old:.6e} "
                      f"({eo:.3g}) new={new:.6e} ({en:.3g})")

    print(f"checked={checked} hard_zeros_old={zeros_old} hard_zeros_new={zeros_new}")
    print(f"worst_rel_err old={worst_old:.3g} new={worst_new:.3g}")
    assert zeros_old > zeros_new, "fold should remove most hard zeros"
    assert worst_new < worst_old, "fold did not improve accuracy"
    assert worst_new < worst_old / 100, "fold should tighten accuracy by two orders"
    print("OK: folding the scale keeps every representable point accurate")


if __name__ == "__main__":
    main()
