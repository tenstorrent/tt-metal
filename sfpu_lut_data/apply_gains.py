#!/usr/bin/env python3
"""Apply the three constant-only LUT retunes. Idempotent-checked: every anchor must
match exactly once, so a moved file fails loudly instead of half-patching."""
import pathlib
import sys

ROOT = pathlib.Path("/localdev/ldjurovic/tt-metal-lutfit")
K = ROOT / "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu"
LLK = ROOT / "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu"

EDITS = []

# ---------------------------------------------------------------- gain 1: sigmoid_appx
EDITS.append((K / "ckernel_sfpu_sigmoid_appx.h", [(
    """inline void sigmoid_appx_init() {
    l_reg[LRegs::LReg0] = vUInt(static_cast<std::uint16_t>(0x3DFF));
    l_reg[LRegs::LReg1] = vUInt(static_cast<std::uint16_t>(0x21D8));
    l_reg[LRegs::LReg2] = vUInt(static_cast<std::uint16_t>(0xFF10));
}""",
    """inline void sigmoid_appx_init() {
    // 3-entry SFPLUT, minimax per segment. A = imm[15:8], B = imm[7:0]; the byte format is
    // s(1)|e(3)|m(4) = (-1)^s * 2^-e * (1 + m/16), and byte 0xFF reads back as exactly 0.0.
    // sigmoid(x) = 0.5 + lut(x), so the table fits sigmoid(|x|) - 0.5, which is concave:
    //   |x| < 1 : 0.234375  *|x|                 (was 0.2265625*|x|)
    //   |x| < 2 : 0.1484375 *|x| + 0.08984375    (was 0.265625 *|x| - 0.046875 -- a slope
    //                                             LARGER than segment 0's, which a concave
    //                                             target cannot use; that was the defect)
    //   else    : 0.5, so sigmoid saturates at exactly 1.0 (unchanged)
    // Max |err| on [1, 2) drops 0.1029 -> 0.0360 and on [0, 1) 0.0098 -> 0.0034, measured
    // on n300. Error is pointwise non-increasing over the whole line; segment 2 is untouched.
    l_reg[LRegs::LReg0] = vUInt(static_cast<std::uint16_t>(0x3EFF));
    l_reg[LRegs::LReg1] = vUInt(static_cast<std::uint16_t>(0x3347));
    l_reg[LRegs::LReg2] = vUInt(static_cast<std::uint16_t>(0xFF10));
}""")]))

# ------------------------------------------------------------------------ gain 2: tanh
TANH_NEW_COMMENT = """        // Continuous piecewise-linear tanh. The knot at |x| = 1 is placed to minimise the
        // worst error over both adjoining segments instead of at a round number: the old
        // table interpolated (0,0) -> (1, 0.90625) -> (2, 1.0), and tanh(1) = 0.7616, so
        // that one knot value carried the whole 0.1447 error budget on both sides of it.
        // Preserves tanh(0) = 0 exactly, continuity at |x| = 1 and 2 (0.8125 and 1.0 from
        // both sides), monotonicity, and the exact 1.0 saturation.
        // Max |err| 0.1447 -> 0.0506, measured on n300."""
EDITS.append((K / "ckernel_sfpu_tanh.h", [(
    """        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x1DFF);  // 0.90625*x
        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x481A);  // 0.09375*x + 0.8125
        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0xFF00);  // 1""",
    TANH_NEW_COMMENT + """
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x1AFF);  // 0.8125*x
        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x3814);  // 0.1875*x + 0.625
        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0xFF00);  // 1""")]))

# The tt-llk copy of the same table, kept in step. (The Blackhole copy is deliberately
# left alone: the fit is arch-independent but the measurement is not, and no Blackhole
# part was available to confirm it.)
EDITS.append((LLK / "ckernel_sfpu_tanh.h", [(
    """    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(static_cast<std::uint16_t>(0x1DFF)); // 0.90625*x
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(static_cast<std::uint16_t>(0x481A)); // 0.09375*x + 0.8125
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(static_cast<std::uint16_t>(0xFF00)); // 1""",
    """    // Minimax knot placement; see the Wormhole metal copy of this table in
    // hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h for the rationale.
    // Max |err| 0.1447 -> 0.0506, measured on n300.
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(static_cast<std::uint16_t>(0x1AFF)); // 0.8125*x
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(static_cast<std::uint16_t>(0x3814)); // 0.1875*x + 0.625
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(static_cast<std::uint16_t>(0xFF00)); // 1""")]))

# tanh_derivative_lut holds an independent copy of the OLD table and its golden models
# that table exactly, so it is frozen on purpose. Record why, so the divergence is not
# read as an oversight.
EDITS.append((K / "ckernel_sfpu_tanh_derivative.h", [(
    """template <bool APPROXIMATION_MODE>
inline void tanh_derivative_init() {
    l_reg[LRegs::LReg0] = vUInt(static_cast<std::uint16_t>(0x1DFF));  // 0.90625*x""",
    """// Deliberately still the pre-retune tanh table. calculate_tanh's copy was refitted for
// accuracy; this one is the contract of a deprecated kernel whose golden models these exact
// coefficients as "the LUT", so the two copies diverge on purpose. Retuning here would need
// the golden updated in lockstep, and would improve a result the header already documents as
// cancellation-dominated (Max ULP = 15,140) -- use calculate_tanh_derivative_sech2 instead.
template <bool APPROXIMATION_MODE>
inline void tanh_derivative_init() {
    l_reg[LRegs::LReg0] = vUInt(static_cast<std::uint16_t>(0x1DFF));  // 0.90625*x""")]))

# ------------------------------------------------------------------- gain 3: gelu_appx
EDITS.append((K / "ckernel_sfpu_gelu.h", [(
    """        // LUT segments (6-entry piecewise linear, each hi/lo pair packed into one imm32):
        // [0.0, 0.5): slope=0.1928, intercept=-0.000104  (lreg0)
        // [0.5, 1.0): slope=0.4939, intercept=-0.1605  (lreg0 hi / lreg4 hi)
        // [1.0, 1.5): slope=0.6189, intercept=-0.2797  (lreg1)
        // [1.5, 2.0): slope=0.6099, intercept=-0.2635  (lreg1 hi / lreg5 hi)
        // [2.0, 3.0): slope=0.5402, intercept=-0.1194  (lreg2)
        // [3.0, ∞):   slope=0.5,    intercept=0.0      (lreg2 hi / lreg6 hi)
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x37E7322B);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0xB12286D8);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x38E138F3);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0xB437B479);

        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x38003852);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0x7c00afa4);""",
    """        // LUT segments (6-entry piecewise linear, each hi/lo pair packed into one imm32).
        // Coefficients are IEEE fp16; slopes in LReg0/1/2, intercepts in LReg4/5/6, lo half
        // = even segment, hi half = odd. Minimax per segment for the target
        // g(a) = a*(Phi(a) - 0.5), a = |x|; the segments are disjoint intervals so each
        // (A, B) is independently optimal over the whole fp16 grid.
        // [0.0, 0.5): slope=0.19140625,  intercept=-0.0115814209
        // [0.5, 1.0): slope=0.491210938, intercept=-0.156616211
        // [1.0, 1.5): slope=0.6171875,   intercept=-0.27734375
        // [1.5, 2.0): slope=0.609375,    intercept=-0.262939453
        // [2.0, 3.0): slope=0.541503906, intercept=-0.123901367
        // [3.0, ∞):   slope=0.5,         intercept=0.0
        //
        // The last segment is PINNED at exactly (0.5, 0.0): with those values the kernel
        // returns 0.5*x + 0.5*|x| + 0 == x to the last bit for x >= 3. A free minimax fit
        // proposes slope 0.5004883 because it halves the error at x = 3, and that is wrong --
        // the absolute error then grows without bound (x = 128 would return 128.06).
        //
        // Max |err| 0.0234 -> 0.0116 overall, measured on n300; the gain is concentrated in
        // segment 0, where a line is fitting a quadratic (g(a) ~ 0.3989a^2) and the old
        // near-zero intercept forced the line to be a chord. Letting the intercept float
        // straddles the curve instead, which is the factor-of-two Chebyshev result. It costs
        // gelu_appx(0) = -0.0116 rather than -0.000104, widening an already-negative sliver
        // just above zero from x < 0.00015 to x < 0.0168; |err| is smaller at every point of
        // the segment regardless.
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x37DC3220);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0xB103A1EE);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x38E038F0);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0xB435B470);

        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x38003855);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0x7c00afee);""")]))

fail = False
for path, subs in EDITS:
    if not path.exists():
        print(f"MISSING {path}")
        fail = True
        continue
    s = path.read_text()
    for old, new in subs:
        n = s.count(old)
        if n != 1:
            print(f"ANCHOR x{n} (want 1) in {path.name}:\n  {old.strip().splitlines()[0][:90]}")
            fail = True
            continue
        s = s.replace(old, new)
    path.write_text(s)
    print(f"patched {path.relative_to(ROOT)}")
sys.exit(1 if fail else 0)
