// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Main body: mirrors piecewise_generic_lut_specialized_N_dual for the gated
// shape (no basis, no range reduction) with parked coefficient fetches.
template <uint32_t POLY_DEG_T, uint32_t NUM_SEG_T, uint32_t LUT_SIZE_T>
inline void piecewise_generic_lut_dst_coeff(const std::array<float, LUT_SIZE_T>& lut) {
    (void)lut;
    dstcoeff::park_all();
    for (int d = 0; d < 32; d += 2) {
        vFloat x_raw1 = dst_reg[d];
        vFloat x_raw2 = dst_reg[d + 1];
        vFloat x1 = prepare_raw_domain_input(x_raw1);
        vFloat x2 = prepare_raw_domain_input(x_raw2);
        vFloat r1, r2;
        dstcoeff::Transport::eval_seg_dual<0>(x1, x2, r1, r2);
        dstcoeff::Transport::cascade_dual<1>(x1, x2, r1, r2);
        // Reload each preserved raw lane only after the dual selected Horner
        // and reconstruction temporaries have retired.  This retains the
        // existing coefficient residency and two-row evaluator schedule.
        vUInt raw_u16_1 = dst_reg[d].mode<DataLayout::U16>();
        vUInt raw_u16_2 = dst_reg[d + 1].mode<DataLayout::U16>();
        finalize_raw_domain_actions(x_raw1, raw_u16_1, r1);
        finalize_raw_domain_actions(x_raw2, raw_u16_2, r2);
        // grad * f'(x) is formed at full SFPU precision and rounded to BF16 once.
        r1 = r1 * dst_reg[32 + d];
        r2 = r2 * dst_reg[33 + d];
        r1 = convert<vFloat16b>(r1, RoundMode::Nearest);
        r2 = convert<vFloat16b>(r2, RoundMode::Nearest);
        dst_reg[d] = r1;
        dst_reg[d + 1] = r2;
    }
}
