// SPDX-License-Identifier: Apache-2.0
// TEST ONLY: exact native FP32 column SUM body, pruning independent right-column groups.
// Full 32x32 CB/DST storage is retained. Native initialization remains unchanged.
#if defined(TRISC_MATH)
namespace ckernel {
namespace sfpu {
template <
    PoolType pool_type,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16,
    bool is_signed_int>
inline void half_column_reduce_sum_avg() {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    constexpr std::uint32_t UPPER_FACE_ADDRS[NUM_FACES] = {0, 0, 16, 16};    // Face 0, 0, 1, 1
    constexpr std::uint32_t LOWER_FACE_ADDRS[NUM_FACES] = {32, 32, 48, 48};  // Face 2, 2, 3, 3
    constexpr std::uint32_t COLUMN_OFFSETS[NUM_FACES] = {0, 2, 0, 2};        // even, odd, even, odd

    // Optimized column reduction: Reduce → Add → Transpose → HalfReduce
    // Instead of the naive Transpose → Reduce → Transpose → Reduce → Add approach, we first reduce
    // across registers, then add upper+lower faces (all 4 positions carry meaningful partial sums),
    // then transpose, then do a final half-reduce on LREG0-3 only. This eliminates one transpose
    // and halves the second reduction pass, saving 4 instructions per iteration.
    for (std::uint32_t i = 0; i < 2; i++) {
        const std::uint32_t upper_face_addr = UPPER_FACE_ADDRS[i];
        const std::uint32_t lower_face_addr = LOWER_FACE_ADDRS[i];
        const std::uint32_t column_offset = COLUMN_OFFSETS[i];

        // Step 1: Tree-reduce across registers (LREG0-3→LREG0, LREG4-7→LREG4) without transpose.
        // After this, each of the 4 positions in LREG0 holds the sum of rows at that position
        // across all 4 loaded LREGs (e.g., LREG0[i] = sum of row[i], row[i+4], row[i+8], row[i+12]).
        load_face_data<INSTRUCTION_MODE, clear_high_bits>(upper_face_addr, lower_face_addr, column_offset);
        lltt::replay(0, 6);

        // Step 2: Cross-face addition. Unlike the old approach where only position 0 of the
        // cross-face sum was meaningful, here ALL 4 positions carry useful partial sums.
        if constexpr (is_integer_mode) {
            TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);  // LREG0 = upper + lower (int)
        } else {
            TTI_SFPADD(
                p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // LREG0 = upper + lower (float)
        }

        // Result of column reduction now stored in LREG0 as 4 partial sums
        // Step 3: Transpose to rearrange the 4 partial sums for final reduction
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4: Final tree-reduce across LREG0-3 only (LREG4-7 no longer needed).
        // This sums the 4 partial sums into LREG0[0] = total column sum.
        lltt::replay(6, 3);

        // Perform averaging if requested (different for int vs float)
        if constexpr (pool_type == PoolType::AVG) {
            if constexpr (is_integer_mode) {
                perform_int_average<INSTRUCTION_MODE, is_signed_int>();
            } else {
                perform_float_average();
            }
        }
        // Store the final column sum/average to the first row.
        // Mode 9 (SFPSTORE_MOD0_FMT_LO16) is only needed when the packer-visible OUTPUT is UInt16 in a
        // 32-bit dest: there the reduced value sits in the low 16 bits but the packer reads the high 16,
        // so we move low->high. When the output is a full 32-bit format (e.g. UInt32) the packer reads
        // the whole dest word, so we use the plain INSTRUCTION_MODE store even for UInt16 input.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        // Runtime-address store (TT_ not TTI_): the face address comes from the loop index and is not
        // guaranteed to fold to a compile-time constant.
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, upper_face_addr + column_offset);
    }
}

inline void half_column_reduce_sum() {
    half_column_reduce_sum_avg<
        PoolType::SUM,
        GetSfpLoadStoreInstrMod<DataFormat::Float32, true>(),
        false,
        false,
        false>();
}
}  // namespace sfpu
}  // namespace ckernel
#endif

ALWI void half_mul(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::C)));
}
ALWI void half_add(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::C)));
}
ALWI void half_sub(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::C)));
}

template <uint32_t Out>
void half_reduce_product() {
    cb_wait_front(8, 4);
    reconfig_data_format(8, 6);
    pack_reconfig_data_format(Out);
    copy_init(8);
    cb_wait_front(6, 1);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL, PackMode::Default>(Out)));
    tile_regs_acquire();
    add_binary_tile_init();
    copy_tile(8, 0, 0);
    // Preserve native Accurate reducer's K-tile fold order and work-DST index.
    constexpr uint32_t work_dst = compute_kernel_lib::DEST_AUTO_LIMIT - 1;
    for (uint32_t ht = 1; ht < 4; ++ht) {
        copy_tile(8, ht, work_dst);
        half_add(0, work_dst, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, half_column_reduce_sum, 0, VectorMode::RC_custom)));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(Out, 1);
    pack_tile(0, Out);
    cb_push_back(Out, 1);
    tile_regs_release();
    PACK((llk_pack_reduce_mask_clear()));
    cb_pop_front(8, 4);
}
