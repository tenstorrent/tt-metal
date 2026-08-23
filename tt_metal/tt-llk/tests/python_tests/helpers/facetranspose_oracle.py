# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""X6 FPU face-transpose oracle (lane FV, 2026-08-22).

Independent host models of the Matrix-Unit instructions the X6 surface
(sfpi_crosslane.h face_transpose_dst_32b) choreographs, transcribed from
the tt-isa-documentation functional models:

    MOVD2B.md / MOVB2A.md / MOVB2D.md / MOVA2D.md / TRNSPSRCB.md
    (WormholeB0 tree -- the pages carry the Blackhole arms; the
    BlackholeA0 tree has no pages for this family: doc gap, pinned sim
    is the BH oracle).

Everything works on ONE 16x16 face of 32-bit Dst datums (row-major list
of 16 rows x 16 columns) plus a 16x16 SrcB transpose region ([16,32) in
hardware rows -- indexed 0..15 here) and a 16-row SrcA park区 (rows 0..15).
Src datums are 19-bit.

The doc-vs-config ambiguity this oracle EXPOSES rather than hides:
MOVB2D's bit-masking arm keys on SrcBFmt, which on Blackhole without
DISABLE_IMPLIED_SRCB_FMT is ImpliedSrcBFmt of a dummy-validated bank --
NonContractualBehavior.  All srcb_fmt candidates ('tf32', 'fp16',
'other') are computable; the sim test adjudicates empirically and prints
the winner as an arsenal FACT.  The END-TO-END face transpose is
independent of the choice (bits 15..0 are rewritten by the lo16 pass),
which the identity theorem below PROVES by enumeration over the
candidates.
"""

M16 = 0xFFFF
M19 = 0x7FFFF
M32 = 0xFFFFFFFF


# --------------------------------------------------------------------------
# Datum shuffles (MOVD2B.md "Supporting definitions")
# --------------------------------------------------------------------------


def shuffle_bf16(x16):
    # Dst BF16 Sign,Man(7b),Exp(8b) -> Src BF16 Sign,Man(10b),Exp(8b)
    return ((x16 & 0xFF00) << 3) | (x16 & 0xFF)


def shuffle_fp16(x16):
    # Dst FP16 Sign,Man(10b),Exp(5b) -> Src FP16 Sign,Man(10b),0(3b),Exp(5b)
    return ((x16 & 0xFFE0) << 3) | (x16 & 0x1F)


def shuffle_tf32(x19):
    # Dst TF32 Sign,HiMan(7b),Exp(8b),LoMan(3b) -> Src Sign,Man(10b),Exp(8b)
    sign_himan = x19 & 0x7F800
    exp = x19 & 0x007F8
    loman = x19 & 0x00007
    return sign_himan | (loman << 8) | (exp >> 3)


def remove_low_mantissa(v19):
    # Inverse of the 8b-exponent shuffles: Src Sign,Man(10b),Exp(8b) ->
    # 16b Sign,Man(7b),Exp(8b), dropping man bits 2..0 (bits 10..8 of v).
    return ((v19 >> 3) & 0xFF00) | (v19 & 0xFF)


def remove_high_exponent(v19):
    # 5b-exponent path: Src Sign,Man(10b),0(3b),Exp(5b) -> 16b.
    return ((v19 >> 3) & 0xFFE0) | (v19 & 0x1F)


def _flush(v19, flush_denormals):
    # MOVB2A.md / MOVB2D.md / MOVA2D.md: if (FlushDenormals && !(v & 0xff))
    # v = 0.  FlushDenormals = !ALU_ACC_CTRL_Zero_Flag_disabled_src.
    if flush_denormals and (v19 & 0xFF) == 0:
        return 0
    return v19


# --------------------------------------------------------------------------
# Instruction models over face/srcb/srca state
# --------------------------------------------------------------------------


def movd2b_rows_32b(face, srcb, dst_row, src_row, nrows, use_lo16, srca_fmt):
    """MOVD2B.md 32-bit-Dst arm (Fp32_enabled): move NROWS rows of FACE
    starting at DST_ROW into SRCB starting at SRC_ROW (0-based transpose
    region row).  SRCA_FMT in {'bf16', 'tf32'} (the two choreography
    formats; SrcBStyle keys on SrcAFmt per the doc's not-a-typo note)."""
    for k in range(nrows):
        for col in range(16):
            dv = face[dst_row + k][col] & M32
            if use_lo16:
                dv = ((dv << 16) | (dv & M16)) & M32
            if srca_fmt == "bf16":
                sb = shuffle_bf16((dv >> 16) & M16)
            elif srca_fmt == "tf32":
                sb = shuffle_tf32((dv >> 13) & M19)
            else:
                raise ValueError(srca_fmt)
            srcb[src_row + k][col] = sb & M19


def trnspsrcb(srcb):
    """TRNSPSRCB.md: in-place 16x16 transpose of the SrcB region."""
    for i in range(16):
        for j in range(i):
            srcb[i][j], srcb[j][i] = srcb[j][i], srcb[i][j]


def movb2a_rows(srcb, srca, srca_row, srcb_row, nrows, flush_denormals):
    """MOVB2A.md: raw 19-bit copy SrcB->SrcA with the denormal-flush arm."""
    for k in range(nrows):
        for col in range(16):
            srca[srca_row + k][col] = _flush(
                srcb[srcb_row + k][col] & M19, flush_denormals
            )


def movb2d_rows_32b(srcb, face, dst_row, src_row, nrows, srca_fmt, srcb_fmt,
                    flush_denormals):
    """MOVB2D.md, SrcAFmt = TF32 arm of the choreography (writes Dst32b
    bits 31..13, zeroes 12..0; bits 15..13 depend on SRCB_FMT's mask)."""
    assert srca_fmt == "tf32"
    for k in range(nrows):
        for col in range(16):
            sb = _flush(srcb[src_row + k][col] & M19, flush_denormals)
            if srcb_fmt == "fp16":
                sb &= 0x7FF1F
            elif srcb_fmt != "tf32":
                sb &= 0x7F8FF  # drops the relocated LoMan bits 8..10
            val16 = remove_low_mantissa(sb)
            low_mantissa = (sb >> 8) & 7
            face[dst_row + k][col] = ((val16 << 16) | (low_mantissa << 13)) & M32


def mova2d_rows_32b_lo(srca, face, dst_row, src_row, nrows, flush_denormals):
    """MOVA2D.md, SrcAFmt = Float32 (8b exponent) + UseDst32bLo arm:
    Dst32b = (Dst32b & 0xffff0000) | RemoveLowMantissa(SrcAVal)."""
    for k in range(nrows):
        for col in range(16):
            sa = _flush(srca[src_row + k][col] & M19, flush_denormals)
            val16 = remove_low_mantissa(sa)
            face[dst_row + k][col] = (face[dst_row + k][col] & 0xFFFF0000) | val16


# --------------------------------------------------------------------------
# The X6 choreography (sfpi_crosslane.h face_transpose_dst_32b) composed
# from the instruction models above.
# --------------------------------------------------------------------------


def face_transpose_32b(face_in, srcb_fmt="other", zero_flag_disabled=True):
    """Predicted Dst face after the full three-pass choreography.

    face_in: 16x16 list-of-lists of u32.  zero_flag_disabled=True is the
    surface contract (ALU_ACC_CTRL_Zero_Flag_disabled_src=1); False
    models a caller that skipped the cfg block's zero-flag arm."""
    flush = not zero_flag_disabled
    face = [row[:] for row in face_in]
    srcb = [[0] * 16 for _ in range(16)]
    srca = [[0] * 16 for _ in range(16)]

    # Pass 1: lo16 -> SrcB (BF16-shuffled), transpose, park in SrcA.
    for k in range(4):
        movd2b_rows_32b(face, srcb, 4 * k, 4 * k, 4, use_lo16=True,
                        srca_fmt="bf16")
    trnspsrcb(srcb)
    for k in range(4):
        movb2a_rows(srcb, srca, 4 * k, 4 * k, 4, flush)

    # Pass 2: hi16 -> SrcB (TF32-shuffled), transpose, write Dst hi.
    for k in range(4):
        movd2b_rows_32b(face, srcb, 4 * k, 4 * k, 4, use_lo16=False,
                        srca_fmt="tf32")
    trnspsrcb(srcb)
    for k in range(4):
        movb2d_rows_32b(srcb, face, 4 * k, 4 * k, 4, "tf32", srcb_fmt, flush)

    # Pass 3: parked lo16 -> Dst bits 15..0.
    for k in range(2):
        mova2d_rows_32b_lo(srca, face, 8 * k, 8 * k, 8, flush)

    return face


def face_transpose_32b_hi_stage(face_in, srcb_fmt="other",
                                zero_flag_disabled=True):
    """Predicted Dst face after passes 1+2 ONLY (the stage-truncation
    probe): transposed hi16 halves in bits 31..13(16), bits 12..0 zeroed;
    bits 15..13 depend on srcb_fmt."""
    flush = not zero_flag_disabled
    face = [row[:] for row in face_in]
    srcb = [[0] * 16 for _ in range(16)]
    srca = [[0] * 16 for _ in range(16)]
    for k in range(4):
        movd2b_rows_32b(face, srcb, 4 * k, 4 * k, 4, True, "bf16")
    trnspsrcb(srcb)
    for k in range(4):
        movb2a_rows(srcb, srca, 4 * k, 4 * k, 4, flush)
    for k in range(4):
        movd2b_rows_32b(face, srcb, 4 * k, 4 * k, 4, False, "tf32")
    trnspsrcb(srcb)
    for k in range(4):
        movb2d_rows_32b(srcb, face, 4 * k, 4 * k, 4, "tf32", srcb_fmt, flush)
    return face


def dstrow_roundtrip_hi(face_in, rows, srcb_fmt="other",
                        zero_flag_disabled=True):
    """The Dst-row calibration mode: per row r in ROWS, MOVD2B(NORM,
    1 row, SrcAFmt=TF32) then MOVB2D back -- predicted result value for
    each datum v: bits 31..13 survive exactly ('tf32' SrcBFmt) or bits
    31..16 only (other), modulo the flush arm."""
    flush = not zero_flag_disabled
    out = {}
    for r in rows:
        row = []
        for col in range(16):
            v = face_in[r][col] & M32
            sb = shuffle_tf32((v >> 13) & M19)
            sb = _flush(sb, flush)
            if srcb_fmt == "fp16":
                sb &= 0x7FF1F
            elif srcb_fmt != "tf32":
                sb &= 0x7F8FF
            val16 = remove_low_mantissa(sb)
            low_mantissa = (sb >> 8) & 7
            row.append(((val16 << 16) | (low_mantissa << 13)) & M32)
        out[r] = row
    return out


# --------------------------------------------------------------------------
# Host identity theorems (asserted by test_crosslane_facetranspose.py's
# host battery BEFORE any sim run)
# --------------------------------------------------------------------------


def theorem_bitexact_transpose(face):
    """Under the surface contract (zero-flag disabled, SrcBFmt in the
    8b-exponent class), the composition is the EXACT bitwise transpose.

    CONTRACT EDGE (host-proven): the theorem holds for srcb_fmt 'tf32'
    and 'other' (the &0x7F8FF mask only drops the relocated LoMan bits
    8..10, which pass 3 rewrites) but NOT for 'fp16' -- the &0x7FF1F
    mask drops the relocated EXP bits 7..5 and corrupts Dst bits
    31..16.  The choreography does not own SrcBFmt (the hand kernels
    rely on the ambient non-FP16 ALU state too); the surface documents
    this as part of the cfg-block contract."""
    want = [[face[j][i] & M32 for j in range(16)] for i in range(16)]
    for fmt in ("tf32", "other"):
        got = face_transpose_32b(face, srcb_fmt=fmt, zero_flag_disabled=True)
        if got != want:
            return False, fmt, got, want
    return True, None, None, None


def theorem_fp16_srcbfmt_corrupts(face):
    """The negative half of the contract edge: under an FP16-class
    SrcBFmt the composition is NOT the transpose (for generic data)."""
    want = [[face[j][i] & M32 for j in range(16)] for i in range(16)]
    got = face_transpose_32b(face, srcb_fmt="fp16", zero_flag_disabled=True)
    return got != want


def flush_victims(face):
    """Lanes (i, j) of the TRANSPOSED face whose value would be corrupted
    with the zero-flag arm SKIPPED (the contract-necessity twin).  Returns
    {(row, col): corrupted_value} for srcb_fmt='other'."""
    want = [[face[j][i] & M32 for j in range(16)] for i in range(16)]
    got = face_transpose_32b(face, srcb_fmt="other", zero_flag_disabled=False)
    return {
        (i, j): got[i][j]
        for i in range(16)
        for j in range(16)
        if got[i][j] != want[i][j]
    }


# --------------------------------------------------------------------------
# CLOSED-FORM models in SFPU-VALUE space (the datums SFPLOAD/SFPSTORE see).
#
# Layer note that dissolves an apparent divergence: Dst32b CELLS store
# fp32-class datums with a SWIZZLED high half (doc DstDecodeFP32; pinned
# sim encode_fp32/decode_fp32), so the doc's 19-bit-shuffle low-byte
# flush test and the sim's fp32-exponent-field flush test are the SAME
# predicate once expressed on the SFPU-space datum w:
#     hi16 pass:  flush iff (w & 0x7F800000) == 0   (IEEE exponent field)
#     lo16 pass:  flush iff (lo16 & 0xFF)     == 0  (the cell's raw low
#                 half rides the BF16 decode, whose exponent is its low
#                 byte)
#
# X6-F2 (REAL divergence, adjudicated by the hi-stage probe): what the
# hi16 write-back leaves in Dst bits 15..0 differs -- the doc's explicit
# SrcAFmt=TF32 arm writes LowMantissa<<13 and zeroes bits 12..0, while
# the pinned sim PRESERVES the destination low half.  Root cause: the
# pinned sim gates the MOV-family implied-format override on
# DISABLE_IMPLIED_SRCB_FMT_Base (tensix.cpp tensix_srca_fmt_from_srcb)
# where MOVD2B.md specifies DISABLE_IMPLIED_SRCA_FMT_Base -- so with the
# choreography's SRCA disable set, the sim still runs every MOV at the
# bank-implied (bf16-class) format.  END-TO-END INVISIBLE: the bf16-class
# path moves exactly bits 31..16 + preserves 15..0, and pass 3 rewrites
# 15..0 anyway -- the full transpose is bit-exact on BOTH paths (host
# theorem + the passing face probes).  Sim-coverage ledger item for the
# sim owner; the silicon vehicle is insensitive by construction.
# --------------------------------------------------------------------------


def flush_lo16_pred(lo16):
    return (lo16 & 0xFF) == 0


def flush_hi_pred(v):
    return (v & 0x7F800000) == 0


def face_transpose_32b_model(face, zero_flag_disabled=True):
    """Closed-form full-choreography prediction: exact transpose when the
    zero-flag arm is set; per-half flush corruption otherwise."""
    out = [[0] * 16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            w = face[j][i] & M32
            hi = w & 0xFFFF0000
            lo = w & M16
            if not zero_flag_disabled:
                if flush_hi_pred(w):
                    hi = 0
                if flush_lo16_pred(lo):
                    lo = 0
            out[i][j] = hi | lo
    return out


def hi_stage_model(face, dest_face, lo16_mode, srcb_fmt="tf32",
                   zero_flag_disabled=True):
    """Closed-form passes-1+2-only prediction.  lo16_mode:
    'sim-preserve' (Dst bits 15..0 kept -- the pinned sim's bf16-class
    path) or 'doc' (explicit-TF32 arm: LowMantissa<<13, bits 12..0
    zeroed, subject to the srcb_fmt mask)."""
    out = [[0] * 16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            w = face[j][i] & M32
            flushed = (not zero_flag_disabled) and flush_hi_pred(w)
            hi = 0 if flushed else (w & 0xFFFF0000)
            if lo16_mode == "sim-preserve":
                lo = dest_face[i][j] & M16
            else:
                sb = 0 if flushed else shuffle_tf32((w >> 13) & M19)
                if srcb_fmt == "fp16":
                    sb &= 0x7FF1F
                elif srcb_fmt != "tf32":
                    sb &= 0x7F8FF
                lo = (((sb >> 8) & 7) << 13) & M16
                if flushed:
                    lo = 0
            out[i][j] = hi | lo
    return out
