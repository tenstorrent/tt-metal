// SPDX-License-Identifier: Apache-2.0
// TEST ONLY: fuse traversal, never arithmetic. Both FP32 opcodes remain distinct.
#if defined(TRISC_MATH)
namespace ckernel::sfpu {
inline void resident_half_product_fold(uint32_t state, uint32_t operand, uint32_t accumulator) {
    // The binary params wrapper starts DST addressing at zero. VectorMode::None
    // delegates the complete traversal here. Raw tile stride64; only faces0/2
    // contain meaningful columns. No incrementing address modifier is used.
    for (uint32_t face = 0; face < 2; ++face) {
        for (uint32_t row = 0; row < 8; ++row) {
            const uint32_t offset = face * 32 + row * 2;
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, state * 64 + offset);
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, operand * 64 + offset);
            // Same native MUL order: state * K (or state * Q), mod1=0.
            // The result retires to a FP32 LREG before the separate ADD reads it.
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            // Independent accumulator load fills the dependent MUL->ADD gap.
            TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, accumulator * 64 + offset);
            // Same native ADD order: accumulated earlier K tiles + product.
            // Explicit volatile instruction emission prevents SFPMAD contraction.
            TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPNOP;  // Conservative two-cycle ADD drain before its result is stored.
            TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, accumulator * 64 + offset);
        }
    }
}
}  // namespace ckernel::sfpu
#endif

ALWI void half_product_fold(uint32_t state, uint32_t operand, uint32_t accumulator) {
    static_assert(DST_SYNC_MODE == DstSync::SyncFull);
    static_assert(DST_ACCUM_MODE);
    static_assert(compute_kernel_lib::DEST_AUTO_LIMIT == 8);
    // Caller preserves mul_binary_tile_init: default SFPU config, ADDR_MOD7=0.
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        resident_half_product_fold,
        state,
        operand,
        accumulator,
        VectorMode::None)));
}
