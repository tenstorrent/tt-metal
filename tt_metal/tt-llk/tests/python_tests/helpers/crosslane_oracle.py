# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-lane arsenal ORACLE (lane FB, 2026-08-21).

Host-side reference semantics for every cross-lane consensus primitive over
the honest Tensix SFPU lane geometry: 32 lanes = 4 subvector rows x 8
columns, one LReg spanning 4 consecutive Dst rows x 8 (even or odd) columns
(SFPLOAD.md: Row = (Addr & ~3) + Lane/8, Column = (Lane & 7) * 2 (+1)).

DERIVATION DISCIPLINE: every instruction-level model below is transcribed
from ~/sfpi-uplift/tt-isa-documentation functional models (BlackholeA0
unless noted; SFPTRANSP.md and SFPCONFIG.md are shared with WormholeB0):

  - SFPSHFT2.md   : COPY4 / SUBVEC_CHAINED_COPY4 / SUBVEC_SHFLROR1_AND_COPY4
                    / SUBVEC_SHFLROR1 / SUBVEC_SHFLSHR1 (BH fixed; WH lane-0
                    of each row is UnpredictableValue -- hardware bug)
  - SFPTRANSP.md  : Transpose4(0) + Transpose4(4) -- 4x4x8 tensor, swaps the
                    two length-4 axes; NOT a lane-grid transpose
  - SFPSWAP.md    : VDGetsMin mask table (mods 1..8, undocumented mod 9),
                    SignMagIsSmaller total order, the equal-value swap quirk,
                    ENABLE_DEST_INDEX companion swap, EXCHANGE_SRCB_SRCC
  - SFPCONFIG.md  : first-8-lane vertical broadcast, VD=11..14 LReg writes,
                    VD=15 LaneConfig write/OR/AND/XOR, MOD1_IMM16_IS_LANE_MASK
                    (mask bit (Lane & 7) * 2), imm-form default constants
  - VectorUnit.md : IsLaneEnabled (ROW_MASK per column + LaneFlags)
  - SFPLOAD.md / SFPSTORE.md : mod0 formats INT32=4, UINT16=6, LO16_ONLY=14,
                    HI16_ONLY=15 partial-register semantics

Docs are the PRIOR; the pinned craq sims are the ORACLE for shipped claims.
Any divergence between this module and a pinned sim is a reportable finding
(tt-isa-documentation standing mandate).

Everything is pure Python over u32 lattices (lists of 32 ints) -- exact by
construction, no numpy/torch dependency, importable from both pytest and the
gate script.
"""

from __future__ import annotations

import struct

LANES = 32
ROWW = 8   # lanes per subvector row
NROWS = 4  # subvector rows

M32 = 0xFFFFFFFF

# ---------------------------------------------------------------------------
# lane geometry helpers
# ---------------------------------------------------------------------------


def lane_row(lane: int) -> int:
    """Subvector row of a lane (SFPLOAD.md: Row = ... + Lane / 8)."""
    return lane // ROWW


def lane_col(lane: int) -> int:
    """Column of a lane (SFPLOAD.md: Column = (Lane & 7) * 2 [+1])."""
    return lane & 7


def dst_mapping(addr: int, lane: int) -> tuple[int, int]:
    """SFPLOAD/SFPSTORE lane -> (Dst row, Dst column) for a given Addr.

    SFPLOAD.md functional model:
        Row    = (Addr & ~3) + Lane / 8
        Column = (Lane & 7) * 2, +1 if (Addr & 2) or DEST_RD_COL_EXCHANGE
    """
    row = (addr & ~3) + lane // 8
    col = (lane & 7) * 2 + (1 if addr & 2 else 0)
    return row, col


def vconst_tileid() -> list[int]:
    """vConstTileId == 2 * lane_id (sfpi.h CREG_IDX_TILEID; lane-EX proof,
    re-verified on-sim by the lane tracer's lanetag calibration)."""
    return [2 * lane for lane in range(LANES)]


def lane_id_sentinels(tag: int = 0) -> list[int]:
    """Distinct per-lane sentinel values: bijective, row/col-decodable."""
    return [(0x51 << 24) | (tag << 16) | (lane_row(l) << 8) | lane_col(l)
            for l in range(LANES)]


def splitmix32(x: int) -> int:
    x = (x + 0x9E3779B9) & M32
    z = x
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & M32
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & M32
    return (z ^ (z >> 16)) & M32


def varied_stimulus(row: int, seed: int = 0) -> list[int]:
    """Genericity-twin stimulus: never lane-id-encoded."""
    return [splitmix32(seed * 1315423911 + row * 37 + l) for l in range(LANES)]


# ---------------------------------------------------------------------------
# FP32 bit helpers (sign-magnitude order, fp arithmetic for goldens)
# ---------------------------------------------------------------------------


def f32_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def f32_round(x: float) -> float:
    """Round a python float to fp32 (one rounding step)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fraction_to_f32(fr) -> float:
    """Exact round-to-nearest-even of a Fraction to fp32 (no
    double-rounding through binary64).  Used for the single-rounded FMA
    contract; python 3.10 has no math.fma."""
    from fractions import Fraction
    if fr == 0:
        return 0.0
    sign = -1.0 if fr < 0 else 1.0
    fr = abs(Fraction(fr))
    num, den = fr.numerator, fr.denominator
    # e = floor(log2(fr)) exactly
    e = num.bit_length() - den.bit_length()
    if (num << max(0, -e)) < (den << max(0, e)):
        e -= 1
    if e < -126:
        e = -126  # subnormal range: fixed scale
    # significand m = fr * 2^(23 - e), rounded half-to-even
    shift = 23 - e
    if shift >= 0:
        num <<= shift
    else:
        den <<= -shift
    q, r = divmod(num, den)
    if 2 * r > den or (2 * r == den and (q & 1)):
        q += 1
    val = sign * float(q) * (2.0 ** e) / (2.0 ** 23)
    return f32_round(val)  # final pack handles overflow-to-inf


def sign_mag_is_smaller(c: int, d: int) -> bool:
    """SFPSWAP.md SignMagIsSmaller: sign-magnitude total order.
    For FP32 bit patterns: -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN.
    """
    # Doc: C ^= (uint32_t)((int32_t)C >> 30) >> 1 (i.e. negative values get
    # their low 31 bits flipped), then two's-complement compare.  As an
    # unsigned monotone key that is: negative -> ~C, positive -> C | 2^31.
    return _smkey(c) < _smkey(d)


# ---------------------------------------------------------------------------
# LaneConfig / lane-enable model (VectorUnit.md, SFPCONFIG.md)
# ---------------------------------------------------------------------------

LC_ENABLE_DEST_INDEX = 1 << 2
LC_EXCHANGE_SRCB_SRCC = 1 << 8
LC_ROW_MASK_SHIFT = 12  # 4 bits, one per subvector row


class LaneState:
    """Per-lane config + predication state relevant to cross-lane ops."""

    def __init__(self):
        self.lane_config = [0] * LANES          # uint18 per lane
        self.lane_flags = [False] * LANES
        self.use_lane_flags = [False] * LANES

    def lane_enabled(self, lane: int) -> bool:
        """VectorUnit.md IsLaneEnabled: ROW_MASK is read from the lane's
        COLUMN slot (LaneConfig[Lane & 7]); flags are per-lane."""
        row_mask = (self.lane_config[lane & 7] >> LC_ROW_MASK_SHIFT) & 0xF
        if (row_mask >> (lane // 8)) & 1:
            return False
        if self.use_lane_flags[lane]:
            return self.lane_flags[lane]
        return True

    def enabled_vec(self) -> list[bool]:
        return [self.lane_enabled(l) for l in range(LANES)]


def all_enabled() -> list[bool]:
    return [True] * LANES


def sfpconfig_write(state: LaneState, vd: int, l0: list[int] | None,
                    imm16: int = 0, mod1: int = 0,
                    lreg_out: dict | None = None) -> None:
    """SFPCONFIG.md functional model for VD in {11..15}.

    - Input comes from the FIRST 8 LANES of LReg[0] (vertical broadcast:
      value used for lane L is l0[L & 7]) unless MOD1_IMM16_IS_VALUE.
    - MOD1_IMM16_IS_LANE_MASK: lane participates iff
      Imm16.Bit[(Lane & 7) * 2] -- a per-COLUMN mask.
    - VD 11..14 (imm form): fixed default constants, NOT Imm16.
    - VD 15: LaneConfig write/OR/AND/XOR; imm form preserves bits >= 16.
    lreg_out: dict to receive {vd: vec} for VD in 11..14.
    """
    IMM_IS_VALUE = 1
    OR_, AND_, XOR_ = 2, 4, 6
    IMM_IS_LANE_MASK = 8
    imm_defaults = {11: 0xBF800000, 12: 0x3B000000,
                    13: 0xBF2CC4C7, 14: 0xBEB08FF9}
    for lane in range(LANES):
        if mod1 & IMM_IS_LANE_MASK:
            if not ((imm16 >> ((lane & 7) * 2)) & 1):
                continue
        if state.use_lane_flags[lane & 7] and not state.lane_flags[lane & 7]:
            continue
        if 11 <= vd <= 14:
            if mod1 & IMM_IS_VALUE:
                value = imm_defaults[vd]
            else:
                value = l0[lane & 7] & M32
            if lreg_out is not None:
                lreg_out.setdefault(vd, [0] * LANES)[lane] = value
        elif vd == 15:
            original = state.lane_config[lane]
            value = (imm16 if (mod1 & IMM_IS_VALUE) else l0[lane & 7]) & 0x3FFFF
            op = mod1 & 6
            if op == 0:
                new = value
            elif op == OR_:
                new = original | value
            elif op == AND_:
                new = original & value
            else:  # XOR
                new = original ^ value
            if mod1 & IMM_IS_VALUE:
                new = (new & 0xFFFF) | (original & ~0xFFFF)
            state.lane_config[lane] = new & 0x3FFFF


# ---------------------------------------------------------------------------
# instruction-level models
# ---------------------------------------------------------------------------


def subvec_shflror1(vc: list[int], enabled=None, dst=None) -> list[int]:
    """SFPSHFT2 Mod1=3 SUBVEC_SHFLROR1 (SFPSHFT2.md):
    VD[lane] = lane&7 ? VC[lane-1] : VC[lane+7]  (rotate right within rows
    of 8).  Disabled lanes keep dst's prior value."""
    out = list(dst) if dst is not None else [0] * LANES
    en = enabled or all_enabled()
    for lane in range(LANES):
        if en[lane]:
            out[lane] = vc[lane - 1] if lane & 7 else vc[lane + 7]
    return out


def subvec_shflshr1(vc: list[int], enabled=None, dst=None,
                    arch: str = "bh") -> list[int]:
    """SFPSHFT2 Mod1=4 SUBVEC_SHFLSHR1.
    BH (SFPSHFT2.md BlackholeA0): VD[lane] = lane&7 ? VC[lane-1] : 0.
    WH: lane 0 of each row is UnpredictableValue (hardware bug; WormholeB0
    SFPSHFT2.md says this mode should not be used on WH).  The WH arm
    returns None sentinels in those lanes so comparisons must skip them.
    """
    out = list(dst) if dst is not None else [0] * LANES
    en = enabled or all_enabled()
    for lane in range(LANES):
        if en[lane]:
            if lane & 7:
                out[lane] = vc[lane - 1]
            else:
                out[lane] = 0 if arch == "bh" else None  # WH: unpredictable
    return out


def shft2_copy4(l0, l1, l2, l3, enabled=None):
    """SFPSHFT2 Mod1=0 COPY4: queue shuffle, no cross-lane movement."""
    en = enabled or all_enabled()
    n0, n1, n2, n3 = list(l0), list(l1), list(l2), list(l3)
    for lane in range(LANES):
        if en[lane]:
            n0[lane], n1[lane], n2[lane], n3[lane] = (
                l1[lane], l2[lane], l3[lane], 0)
    return n0, n1, n2, n3


def shft2_chained_copy4(l0, l1, l2, l3, enabled=None):
    """SFPSHFT2 Mod1=1 SUBVEC_CHAINED_COPY4:
    L3'[lane] = lane < 24 ? oldL0[lane+8] : 0 -- crosses the row boundary."""
    en = enabled or all_enabled()
    v0 = list(l0)
    n0, n1, n2, n3 = list(l0), list(l1), list(l2), list(l3)
    for lane in range(LANES):
        if en[lane]:
            n0[lane] = l1[lane]
            n1[lane] = l2[lane]
            n2[lane] = l3[lane]
            n3[lane] = v0[lane + 8] if lane < 24 else 0
    return n0, n1, n2, n3


def shft2_ror1_and_copy4(l0, l1, l2, l3, vc, enabled=None):
    """SFPSHFT2 Mod1=2 SUBVEC_SHFLROR1_AND_COPY4:
    queue shuffle; L3' = ror1(VC) within rows."""
    en = enabled or all_enabled()
    n0, n1, n2, n3 = list(l0), list(l1), list(l2), list(l3)
    for lane in range(LANES):
        if en[lane]:
            n0[lane] = l1[lane]
            n1[lane] = l2[lane]
            n2[lane] = l3[lane]
            n3[lane] = vc[lane - 1] if lane & 7 else vc[lane + 7]
    return n0, n1, n2, n3


def sfptransp(regs: list[list[int]], enabled=None) -> list[list[int]]:
    """SFPTRANSP (SFPTRANSP.md, WH tree, shared): regs = 8 vectors
    (LReg[0..7]).  Transpose4(0) and Transpose4(4): per column, the 4x4
    (register x subvector-row) matrix is transposed.  Both banks always.
    Disabled lanes are NOT written (per-element predication in the model).
    """
    en = enabled or all_enabled()
    out = [list(r) for r in regs]
    for base in (0, 4):
        for column in range(8):
            for i in range(4):
                for j in range(i):
                    ij = out[base + i][j * 8 + column]
                    ji = out[base + j][i * 8 + column]
                    if en[j * 8 + column]:
                        out[base + i][j * 8 + column] = ji
                    if en[i * 8 + column]:
                        out[base + j][i * 8 + column] = ij
    return out


_VDGETSMIN = {
    1: 0xFFFFFFFF,  # VEC_MIN_MAX: all lanes VD=min, VC=max
    2: 0x0000FFFF,  # SUBVEC_MIN01_MAX23
    3: 0x00FF00FF,  # SUBVEC_MIN02_MAX13
    4: 0xFF0000FF,  # SUBVEC_MIN03_MAX12
    5: 0x000000FF,  # SUBVEC_MIN0_MAX123
    6: 0x0000FF00,  # SUBVEC_MIN1_MAX023
    7: 0x00FF0000,  # SUBVEC_MIN2_MAX013
    8: 0xFF000000,  # SUBVEC_MIN3_MAX012
    9: 0x00000000,  # all lanes VD=max, VC=min (no enum; doc case 9)
}


def sfpswap_should_swap(vc_val: int, vd_val: int, mod1: int, lane: int,
                        exchange: bool = False, tie: str = "doc") -> bool:
    """SFPSWAP decision.

    tie='doc' (SFPSWAP.md): ShouldSwap = SignMagIsSmaller(VC, VD) ||
      (VC == VD && negative), inverted in max lanes -- "max lanes swap
      equal positive values, min lanes swap equal negatives".
    tie='sim' (pinned craq-sim 9f324140, tensix.cpp sfpswap_vd_gets_c):
      min lanes: VC < VD;  max lanes: VC >= VD -- the sign-keyed
      equal-value arm is ABSENT.  Value-invisible for plain min/max,
      VISIBLE via companion movement under ENABLE_DEST_INDEX.
    DOC-VS-SIM DIVERGENCE (lane FB finding, 2026-08-21): silicon
    adjudication pending; goldens must not depend on tie companion
    movement until then.
    """
    if mod1 == 0:
        return True
    vdgetsmin = _VDGETSMIN[mod1]
    if tie == "doc":
        should = (sign_mag_is_smaller(vc_val, vd_val)
                  or ((vc_val == vd_val) and bool(vc_val & 0x80000000)))
        if not ((vdgetsmin >> lane) & 1):
            should = not should
    else:
        if (vdgetsmin >> lane) & 1:
            should = sign_mag_is_smaller(vc_val, vd_val)
        else:
            should = not sign_mag_is_smaller(vc_val, vd_val)  # VC >= VD
    if exchange:
        should = not should
    return should


def sfpswap(vc: list[int], vd: list[int], mod1: int, enabled=None,
            state: LaneState | None = None, tie: str = "doc"):
    """SFPSWAP without ENABLE_DEST_INDEX: returns (new_vc, new_vd).
    Per-lane EXCHANGE_SRCB_SRCC honored from state if given."""
    en = enabled or (state.enabled_vec() if state else all_enabled())
    nvc, nvd = list(vc), list(vd)
    for lane in range(LANES):
        if not en[lane]:
            continue
        exch = bool(state and (state.lane_config[lane] & LC_EXCHANGE_SRCB_SRCC))
        if sfpswap_should_swap(vc[lane], vd[lane], mod1, lane, exch, tie):
            nvc[lane] = vd[lane]
            nvd[lane] = vc[lane]
    return nvc, nvd


def sfpswap_indexed(vc, vd, cc, cd, mod1: int, enabled=None,
                    state: LaneState | None = None, tie: str = "doc"):
    """SFPSWAP.md ENABLE_DEST_INDEX leg: keys in VC/VD (LReg[0..3]),
    companions at value+4 swap on the same decision.
    Returns (new_vc, new_vd, new_cc, new_cd)."""
    en = enabled or (state.enabled_vec() if state else all_enabled())
    nvc, nvd, ncc, ncd = list(vc), list(vd), list(cc), list(cd)
    for lane in range(LANES):
        if not en[lane]:
            continue
        exch = bool(state and (state.lane_config[lane] & LC_EXCHANGE_SRCB_SRCC))
        if sfpswap_should_swap(vc[lane], vd[lane], mod1, lane, exch, tie):
            nvc[lane] = vd[lane]
            nvd[lane] = vc[lane]
            ncc[lane] = cd[lane]
            ncd[lane] = cc[lane]
    return nvc, nvd, ncc, ncd


def sfpconfig_broadcast(l0: list[int]) -> list[int]:
    """SFPCONFIG value form, VD in 11..14: vertical broadcast of the first
    8 lanes -- out[lane] = l0[lane & 7] (SFPCONFIG.md)."""
    return [l0[lane & 7] & M32 for lane in range(LANES)]


# --- partial-register Dst access models (SFPLOAD.md / SFPSTORE.md) ---------


def load_lo16_only(prev: list[int], dst16: list[int]) -> list[int]:
    """MOD0_FMT_LO16_ONLY=14: (prev & 0xffff0000) | Dst16b."""
    return [((p & 0xFFFF0000) | (d & 0xFFFF)) & M32
            for p, d in zip(prev, dst16)]


def load_hi16_only(prev: list[int], dst16: list[int]) -> list[int]:
    """MOD0_FMT_HI16_ONLY=15: (Dst16b << 16) | (prev & 0xffff)."""
    return [(((d & 0xFFFF) << 16) | (p & 0xFFFF)) & M32
            for p, d in zip(prev, dst16)]


def store_lo16(v: list[int]) -> list[int]:
    """SFPSTORE LO16_ONLY=14 / UINT16=6: Dst16b = Datum & 0xffff."""
    return [x & 0xFFFF for x in v]


def store_hi16(v: list[int]) -> list[int]:
    """SFPSTORE HI16_ONLY=15: Dst16b = Datum >> 16."""
    return [(x >> 16) & 0xFFFF for x in v]


# --- Dst 32-bit-layout 16b-view model (pinned craq-sim 9f324140, adjudicated
# on-sim by the lane tracer, 2026-08-21) --------------------------------------
# The Dst banks store the HIGH half of every 32-bit datum in the BF16-swizzled
# layout (dst_encode_bf16 below); INT32/FP32 accesses decode it back.  In the
# harness's fp32-acc config the pinned sim splits the 16-bit view two ways:
#   - RAW 16b modes (UINT16=6 / LO16_ONLY=14 / HI16_ONLY=15) ride the Dst.md
#     Adj16 bank cells (dst_32bit_addr_en=0 path): loads at rows < 8 return
#     dst_encode_bf16(datum32 >> 16) because the Adj algebra makes those
#     cells coincide with the high halves; stores land in their OWN cell and
#     never clobber the same-address 32b datum (doc-consistent).
#   - BF16-FORMAT stores (mod0=2) take the 32-bit write path
#     (sfpu_full_dst32_write = dst_32bit_addr_en || Fp32_enabled): the paired
#     low half is ZEROED, except under ENABLE_DEST_INDEX (LaneConfig bit 2)
#     or the TopK LCONST0 special case, where it is RMW-PRESERVED.
# SFPSTORE.md says a BF16 store writes only its Dst16b cell (always
# preserves); tt-blaze #2475 claims SILICON canonicalizes the paired half.
# THREE-WAY doc/sim/silicon divergence on the BF16-store path -- recorded;
# silicon adjudication pending.


def dst_encode_bf16(x16: int) -> int:
    """craq-sim encode_bf16: (sign, man7 << 8, exp8) from IEEE bf16 bits."""
    e = (x16 >> 7) & 255
    m = x16 & 127
    return (x16 & 0x8000) | (m << 8) | e


def dst_decode_bf16(d16: int) -> int:
    """Inverse of dst_encode_bf16."""
    e = d16 & 255
    m = (d16 >> 8) & 127
    return (d16 & 0x8000) | (e << 7) | m


def sim16_load(datum32: int) -> int:
    """Pinned-sim 16b-view load of a 32-bit Dst datum at the same address."""
    return dst_encode_bf16((datum32 >> 16) & 0xFFFF)


def sim_bf16_store_word(v32: int, low16_prior: int, preserve: bool) -> int:
    """Pinned-sim resulting 32-bit datum after a BF16-format store of the
    fp32 value bits v32 (bf16-exact): high half = truncated top 16 bits;
    low half ZEROED unless preserve (ENABLE_DEST_INDEX / LCONST0 case,
    where it is RMW-preserved)."""
    hi = v32 & 0xFFFF0000
    return (hi | (low16_prior & 0xFFFF)) & M32 if preserve else hi & M32


# ---------------------------------------------------------------------------
# composed consensus primitives (the surface semantics the builder must meet)
# ---------------------------------------------------------------------------


def subvec_rotr(v: list[int], k: int) -> list[int]:
    """rotate right by k within each row of 8 == ror1 composed k times."""
    out = list(v)
    for _ in range(k % ROWW):
        out = subvec_shflror1(out)
    return out


def subvec_broadcast(v: list[int], idx: int) -> list[int]:
    """broadcast lane idx of each row of 8 to the whole row."""
    return [v[lane_row(l) * ROWW + idx] for l in range(LANES)]


def rowvec_broadcast(v: list[int]) -> list[int]:
    """broadcast subvector row 0 to all 4 rows (SFPCONFIG vertical
    broadcast semantics: out[lane] = v[lane & 7])."""
    return sfpconfig_broadcast(v)


def butterfly(v: list[int], k: int) -> list[int]:
    """XOR-pattern exchange within rows of 8 (the __shfl_xor consensus
    primitive): out[lane] = v[lane ^ k], k in {1..7}.  Self-inverse.
    NOTE: no single Tensix instruction; lowering is DERIVED (rotations +
    predicated selects) or refused -- the oracle defines the semantics."""
    assert 0 < k < ROWW
    return [v[(lane_row(l) * ROWW) + (lane_col(l) ^ k)] for l in range(LANES)]


def zip_rows(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    """Interleave subvector rows of two vectors (consensus zip/interleave2,
    row-granular): rows (a0,b0,a1,b1) -> (lo, hi) = rows(a0,b0,a1,b1)."""
    rows = []
    for r in range(NROWS):
        rows.append(a[r * ROWW:(r + 1) * ROWW])
        rows.append(b[r * ROWW:(r + 1) * ROWW])
    lo = [x for row in rows[:NROWS] for x in row]
    hi = [x for row in rows[NROWS:] for x in row]
    return lo, hi


def unzip_rows(lo: list[int], hi: list[int]) -> tuple[list[int], list[int]]:
    """Inverse of zip_rows: de-interleave rows back to (a, b)."""
    rows = [lo[i * ROWW:(i + 1) * ROWW] for i in range(NROWS)] + \
           [hi[i * ROWW:(i + 1) * ROWW] for i in range(NROWS)]
    a = [x for r in range(NROWS) for x in rows[2 * r]]
    b = [x for r in range(NROWS) for x in rows[2 * r + 1]]
    return a, b


# --- reductions -------------------------------------------------------------

_INT_OPS = {
    "add": lambda a, b: (a + b) & M32,
    "xor": lambda a, b: a ^ b,
    "max": lambda a, b: b if sign_mag_is_smaller(a, b) else a,
    "min": lambda a, b: a if sign_mag_is_smaller(a, b) else b,
}


def fp32_add(a_bits: int, b_bits: int) -> int:
    """One SFPADD: fp32 + fp32 with one rounding."""
    return f32_to_bits(f32_round(bits_to_f32(a_bits) + bits_to_f32(b_bits)))


def get_op(op: str):
    if op == "fadd":
        return fp32_add
    return _INT_OPS[op]


def subvec_reduce_tree(v: list[int], op: str) -> list[int]:
    """Segmented (per-row-of-8) all-lanes reduction with the PINNED
    rotate-fold tree: v = op(v, ror(v, 1)); v = op(v, ror(v, 2));
    v = op(v, ror(v, 4)).  The tree order is part of the semantics
    (FP results are bit-defined by this order)."""
    f = get_op(op)
    out = list(v)
    for k in (1, 2, 4):
        rot = subvec_rotr(out, k)
        out = [f(out[l], rot[l]) for l in range(LANES)]
    return out


def subvec_reduce_serial(v: list[int], op: str) -> list[int]:
    """Reference serial (left) fold per row -- the reassociation witness.
    Equal to the tree for association-free ops (int add/xor/min/max)."""
    f = get_op(op)
    out = [0] * LANES
    for r in range(NROWS):
        acc = v[r * ROWW]
        for c in range(1, ROWW):
            acc = f(acc, v[r * ROWW + c])
        for c in range(ROWW):
            out[r * ROWW + c] = acc
    return out


def reduce32_tree(v: list[int], op: str) -> list[int]:
    """Full 32-lane reduction, all lanes hold the result.  PINNED order:
    row-fold tree first (subvec_reduce_tree), then the cross-row combine
    in ascending row order via the transp composition:
    total = op(op(op(row0, row1), row2), row3)."""
    f = get_op(op)
    rowred = subvec_reduce_tree(v, op)
    out = [0] * LANES
    for c in range(ROWW):
        acc = rowred[c]
        for r in range(1, NROWS):
            acc = f(acc, rowred[r * ROWW + c])
        for r in range(NROWS):
            out[r * ROWW + c] = acc
    # every column now holds the same total (row-folds already uniform per
    # row); broadcast is structural, nothing more to do
    return out


# --- scans ------------------------------------------------------------------


def subvec_scan_incl(v: list[int], op: str) -> list[int]:
    """Inclusive prefix within each row of 8, Hillis-Steele shr1 tree:
    for k in (1,2,4): v = op(v, shift_right_in_zero(v, k)).
    Exact for int add/xor/min/max; for fadd the tree order IS the
    semantics."""
    f = get_op(op)
    ident = 0  # shr1 zero-fill == additive/xor identity; min/max need care
    out = list(v)
    for k in (1, 2, 4):
        shifted = [0] * LANES
        for l in range(LANES):
            c = lane_col(l)
            shifted[l] = out[l - k] if c >= k else ident
        nxt = list(out)
        for l in range(LANES):
            if lane_col(l) >= k:
                nxt[l] = f(out[l], shifted[l])
        out = nxt
    return out


def subvec_scan_excl(v: list[int], op: str, identity: int = 0) -> list[int]:
    """Exclusive scan (the API-consensus default): shift of inclusive."""
    incl = subvec_scan_incl(v, op)
    out = [identity] * LANES
    for l in range(LANES):
        if lane_col(l) > 0:
            out[l] = incl[l - 1]
    return out


def rowchain_scan_incl(rows: list[list[int]], op: str) -> list[list[int]]:
    """Register-chain scan (the cumsum/ema shape): rows[i] = op(rows[i],
    rows[i-1]) serially, i = 1..n-1.  Lanewise; the chain axis is the
    REGISTER axis, not the lane axis."""
    f = get_op(op)
    out = [list(r) for r in rows]
    for i in range(1, len(out)):
        out[i] = [f(out[i][l], out[i - 1][l]) for l in range(LANES)]
    return out


# --- serial prefix reference for scans --------------------------------------


def serial_prefix(v: list[int], op: str) -> list[int]:
    f = get_op(op)
    out = [0] * LANES
    for r in range(NROWS):
        acc = None
        for c in range(ROWW):
            x = v[r * ROWW + c]
            acc = x if acc is None else f(acc, x)
            out[r * ROWW + c] = acc
    return out


# ---------------------------------------------------------------------------
# compare-exchange / sort-network semantics
# ---------------------------------------------------------------------------


def sort2(a: list[int], b: list[int], order: str = "asc"):
    """Compare-exchange stage (VQSort Sort2 analogue).
    asc: returns (min, max) per lane == SFPSWAP mod1=1 (VD=min role on the
    first operand).  desc: (max, min) == mod1=9 / EXCHANGE flip.
    Includes the SFPSWAP equal-value quirk -- it is CONTRACT."""
    mod = 1 if order == "asc" else 9
    nvc, nvd = sfpswap(b, a, mod)  # doc roles: VD gets min for mod1=1
    # sfpswap(vc, vd): we map a->VD, b->VC so asc returns (min, max)
    return nvd, nvc


def sort2_kv(k0, k1, p0, p1, order: str = "asc"):
    """Key+payload compare-exchange (ENABLE_DEST_INDEX semantics)."""
    mod = 1 if order == "asc" else 9
    nvc, nvd, ncc, ncd = sfpswap_indexed(k1, k0, p1, p0, mod)
    return nvd, nvc, ncd, ncc


def bitonic_network_stages(n: int):
    """Standard bitonic sorting network for n = power of 2: yields stages,
    each a list of (i, j, dirn) compare-exchanges with i < j; dirn 'asc'
    means element i gets min."""
    assert n & (n - 1) == 0
    stages = []
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            stage = []
            for i in range(n):
                partner = i ^ j
                if partner > i:
                    dirn = "asc" if (i & k) == 0 else "desc"
                    stage.append((i, partner, dirn))
            stage.sort()
            stages.append(stage)
            j //= 2
        k *= 2
    return stages


def bitonic_sort_trace(values: list[int], order: str = "asc"):
    """Sort n values (per independent 'machine' -- caller slices) with the
    bitonic network under SIGN-MAGNITUDE compare-exchange (incl. the
    equal-value quirk).  Returns (sorted_values, [state_after_each_stage]).
    """
    n = len(values)
    state = list(values)
    trace = []
    for stage in bitonic_network_stages(n):
        for (i, j, dirn) in stage:
            want_asc = (dirn == "asc") == (order == "asc")
            a, b = state[i], state[j]
            lo, hi = _ce_pair(a, b, min_first=True)
            state[i], state[j] = (lo, hi) if want_asc else (hi, lo)
        trace.append(list(state))
    return state, trace


def bitonic_sort_kv_trace(keys: list[int], payloads: list[int],
                          order: str = "asc", tie: str = "doc"):
    """Key-value bitonic network trace under sign-magnitude compare-exchange
    with companion movement (ENABLE_DEST_INDEX semantics).  On ties the
    companion movement depends on the UNRESOLVED doc-vs-sim tie divergence
    (see sfpswap_should_swap) -- fixtures use tie-free keys or carry both
    tie-mode variants."""
    n = len(keys)
    ks, ps = list(keys), list(payloads)
    trace = []
    for stage in bitonic_network_stages(n):
        for (i, j, dirn) in stage:
            want_asc = (dirn == "asc") == (order == "asc")
            swap = _ce_should_swap(ks[i], ks[j], min_first=want_asc, tie=tie)
            if swap:
                ks[i], ks[j] = ks[j], ks[i]
                ps[i], ps[j] = ps[j], ps[i]
        trace.append((list(ks), list(ps)))
    return ks, ps, trace


def _ce_should_swap(a: int, b: int, min_first: bool,
                    tie: str = "doc") -> bool:
    """Compare-exchange decision on a pair (a stays first on False):
    with (VC=b, VD=a) and VD-gets-min.  tie per sfpswap_should_swap."""
    if tie == "doc":
        should = sign_mag_is_smaller(b, a) or ((a == b)
                                               and bool(a & 0x80000000))
        return should if min_first else not should
    if min_first:
        return sign_mag_is_smaller(b, a)
    return not sign_mag_is_smaller(b, a)


def _ce_pair(a: int, b: int, min_first: bool, tie: str = "doc"):
    if _ce_should_swap(a, b, min_first, tie):
        return b, a
    return a, b


def topk_select(keys: list[int], k: int, order: str = "desc"):
    """Top-k golden over 32 lanes under the sign-magnitude total order.
    Returns (values, indices) sorted best-first.  Ties broken by lane index
    (stable) -- fixtures must only be compared value-wise unless the
    lowering pins tie behavior."""
    idx = sorted(range(len(keys)),
                 key=lambda i: (_smkey(keys[i]), i),
                 reverse=(order == "desc"))
    take = idx[:k]
    return [keys[i] for i in take], take


def _smkey(c: int) -> int:
    """Monotone unsigned key for the sign-magnitude total order
    (SFPSWAP.md SignMagIsSmaller): negative -> ~C, positive -> C | 2^31."""
    c &= M32
    return (~c & M32) if (c >> 31) else (c | 0x80000000)


# ---------------------------------------------------------------------------
# demand-kernel cross-lane cores (fixture semantics)
# ---------------------------------------------------------------------------


def softmax_k_masked_fold(v_bits: list[int], k: int, op: str = "max"):
    """softmax_k cross-lane core: fold over the first k lanes of each row
    of 8 (k in 1..8), all lanes of the row receive the fold result.

    TWO equivalent formulations, both returned (they must agree -- the
    lane-EX dissolution claim):
      a) mask semantics: lanes >= k are replaced by the fold identity
         before an all-lane row fold (the SFPCONFIG lane-mask reading);
      b) tile-id predicate semantics: participation predicate
         (vConstTileId >> 1) & 7 < k computed per lane, identity
         substituted under the predicate's complement.
    Identity: max -> 0xFF800000 (-inf), add -> +0.
    """
    ident = 0xFF800000 if op in ("max",) else 0
    f = get_op("max" if op == "max" else "fadd")
    # (a) mask formulation
    masked_a = [v_bits[l] if lane_col(l) < k else ident for l in range(LANES)]
    # (b) tile-id predicate formulation
    tid = vconst_tileid()
    masked_b = [v_bits[l] if ((tid[l] >> 1) & 7) < k else ident
                for l in range(LANES)]
    out_a = subvec_reduce_tree(masked_a, "max" if op == "max" else "fadd")
    out_b = subvec_reduce_tree(masked_b, "max" if op == "max" else "fadd")
    return out_a, out_b


def ema_rowchain(x_rows: list[list[int]], alpha_bits: int,
                 y0_bits: list[int]):
    """EMA along the register chain (scan-free reformulation -- consensus
    verdict: no scan primitive; the chain is a serial MAD chain).
    y_i = alpha * x_i + (1 - alpha) * y_{i-1}, computed in fp32.

    TWO candidate arithmetic contracts are returned (the lowering picks
    one; a third value on sim/silicon is a finding):
      'fma'    : y = fp32(alpha * x + fp32(beta * y_prev)) with the outer
                 MAD single-rounded (SFPMAD one-rounding reading);
      'mul_add': every mul and add individually fp32-rounded.
    beta = fp32(1 - alpha).
    """
    a = bits_to_f32(alpha_bits)
    beta = f32_round(1.0 - a)
    outs = {}
    for contract in ("fma", "mul_add"):
        y = [bits_to_f32(b) for b in y0_bits]
        rows_out = []
        for xr in x_rows:
            ny = []
            for l in range(LANES):
                x = bits_to_f32(xr[l])
                t = f32_round(beta * y[l])
                if contract == "fma":
                    from fractions import Fraction
                    exact = Fraction(a) * Fraction(x) + Fraction(t)
                    val = fraction_to_f32(exact)
                else:
                    val = f32_round(f32_round(a * x) + t)
                ny.append(val)
            y = ny
            rows_out.append([f32_to_bits(v) for v in y])
        outs[contract] = rows_out
    return outs


def cumsum_rowchain(x_rows: list[list[int]], dtype: str = "int"):
    """Cumsum decomposition golden: inclusive prefix along the register
    chain (serial adds, order pinned low index -> high).  dtype 'int'
    (exact mod 2^32) or 'fp32' (per-add rounding)."""
    op = "add" if dtype == "int" else "fadd"
    return rowchain_scan_incl(x_rows, op)


# ---------------------------------------------------------------------------
# permutation extraction (identity-battery workhorse)
# ---------------------------------------------------------------------------


def extract_permutation(fn) -> list[int | None]:
    """Trace fn: vec->vec with basis sentinels; returns perm where
    out[l] = in[perm[l]], or None for lanes not sourced from the input
    (zero-fill / unpredictable)."""
    sent = [0x1000 + l for l in range(LANES)]
    out = fn(list(sent))
    perm: list[int | None] = []
    for l in range(LANES):
        v = out[l]
        if v is not None and isinstance(v, int) and 0x1000 <= v < 0x1000 + LANES:
            perm.append(v - 0x1000)
        else:
            perm.append(None)
    return perm


def is_full_permutation(perm) -> bool:
    return (all(p is not None for p in perm)
            and sorted(perm) == list(range(LANES)))
