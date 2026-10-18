// SPDX-License-Identifier: Apache-2.0
// TEST ONLY. Native row-broadcast instructions with right-face data loads/stores omitted.
// All four transposed-register multiplies and both transpose operations remain unchanged.
#if defined(TRISC_MATH)
namespace ckernel { namespace sfpu {
template <BinaryOp BINOP>
inline void _resident_left_row_band_(std::uint32_t data_tile_base, std::uint32_t out_tile_base, std::uint32_t face_pair_base, std::uint32_t band_off)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    const std::uint32_t data_slot_base = data_tile_base + face_pair_base + band_off;
    const std::uint32_t out_slot_base  = out_tile_base + face_pair_base + band_off;

    // (1) Load data's 4 col-groups into LREG0..3. Bcast (LREG4..7) was
    //     pre-loaded once by the caller and survives across iterations.
    TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_7, data_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_7, data_slot_base + COL_GROUP_OFFSETS[1]);

    // (2) Transpose: LREG0..3 become the 4 data rows; LREG4..7 become the
    //     transposed bcast rows (of which only LREG4 = bcast row 0 is useful).
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (3) In-register eltwise: LREG_k = LREG_k OP LREG4 for k in 0..3.
    //
    // Pipelining: the 4 binops all read LREG4 and a distinct LREG_k, and each
    // writes a distinct LREG_k. They are pairwise independent, so we can
    // issue them back-to-back with no NOPs between. Only a single trailing
    // NOP is needed before SFPTRANSP reads LREG0..3 (2-cycle SFPADD/SFPMUL
    // latency).
    //
    // IMPORTANT: this block must not write LREG4..7. The persistent bcast
    // values rely on the second SFPTRANSP being a pure inverse of the first
    // for that group. The SUB path uses SFPMAD with LCONST_neg1
    // (data = bcast * -1 + data) to fuse the negation into the FMA instead
    // of the more obvious SFPMOV-NEGATE-into-LREG5 + SFPADD pattern, which
    // would corrupt LREG5 and force a reload of the bcast every band.
    if constexpr (BINOP == BinaryOp::ADD)
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG3, 0);
    }
    else if constexpr (BINOP == BinaryOp::SUB)
    {
        // data - bcast = bcast * (-1) + data, fused into a single SFPMAD per
        // element so we never write LREG4..7 and the bcast survives the
        // second SFPTRANSP.
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
    }
    else if constexpr (BINOP == BinaryOp::MUL)
    {
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    }
    else
    {
        static_assert(BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL, "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
    }
    TTI_SFPNOP; // single drain before SFPTRANSP reads LREG0..3.

    // (4) Transpose back. Since step (3) left LREG4..7 untouched, this also
    //     restores the bcast registers to their pre-transpose face-major
    //     layout, ready for the next band's first SFPTRANSP.
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (5) Store LREG0..3 to the 4 col-groups of the output band.
    TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_7, out_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPSTORE(p_sfpu::LREG1, IM, ADDR_MOD_7, out_slot_base + COL_GROUP_OFFSETS[1]);
}

template <BinaryOp BINOP>
inline void _resident_left_row_tile_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    const std::uint32_t data_base  = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = dst_index_bcast * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base   = dst_index_out * DEST_TILE_SIZE_RAW;

    // Hoisted bcast load: bcast row 0 lives in face 0 row band 0 and is
    // identical for every output band, so we load its 4 col-groups exactly
    // once. The per-band binop is structured to leave LREG4..7 unchanged,
    // so the second SFPTRANSP per band restores them to this face-major
    // layout.
    const std::uint32_t bcast_row0_base = bcast_base + FACE0_BASE;
    TT_SFPLOAD(p_sfpu::LREG4, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[3]);

    // Upper half (face 0/1): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        _resident_left_row_band_<BINOP>(data_base, out_base, FACE0_BASE, band * ROW_BAND_STRIDE);
    }

    // Lower half (face 2/3): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        _resident_left_row_band_<BINOP>(data_base, out_base, FACE2_BASE, band * ROW_BAND_STRIDE);
    }
}

} }
#endif
ALWI void resident_row_delta_sub(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, 2, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a, b, out, VectorMode::RC_custom)));
}
ALWI void resident_row_delta_mul(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 2, DST_ACCUM_MODE), a, b, out, VectorMode::RC_custom)));
}
ALWI void resident_left_mul_bcast_row(uint32_t data, uint32_t row) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _resident_left_row_tile_,
        (ckernel::BinaryOp::MUL), data, row, data, VectorMode::None)));
}
