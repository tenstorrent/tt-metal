// SPDX-License-Identifier: Apache-2.0
// TEST ONLY. Prepended helpers are checksum-pinned from the validated half-column probe.
// Full-sync FP32 DST0..3 retain state, DST4 accumulator/delta, DST5/6 scratch.
// Every multiply/add is separate and retains the baseline operand and K-tile order.
ALWI void reduce_dst4() {
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, half_column_reduce_sum, 4, VectorMode::RC_custom)));
}

void kernel_main() {
    static_assert(DST_SYNC_MODE == DstSync::SyncFull);
    static_assert(DST_ACCUM_MODE);
    static_assert(compute_kernel_lib::DEST_AUTO_LIMIT == 8);
    compute_kernel_hw_startup(0, 4, 7);
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    const uint32_t length = get_arg_val<uint32_t>(1);
    for (uint32_t job = 0; job < jobs; ++job) {
        cb_wait_front(0, 4);
        tile_regs_acquire();
        for (uint32_t i = 0; i < 4; ++i) load(0, i, i);
        cb_pop_front(0, 4);
        for (uint32_t token = 0; token < length; ++token) {
            if (token != 0) tile_regs_acquire();
            cb_wait_front(1, 4);
            cb_wait_front(2, 4);
            cb_wait_front(3, 1);
            cb_wait_front(4, 1);
            cb_wait_front(5, 1);
            // Decay: state * decay, unchanged operand order, FP32 rounding after MUL.
            load(4, 0, 4);
            for (uint32_t i = 0; i < 4; ++i) {
                mul_binary_tile_init();
                half_mul(i, 4, i);
            }
            // Prediction: ((state0*K0 + state1*K1) + state2*K2) + state3*K3.
            for (uint32_t i = 0; i < 4; ++i) {
                load(2, i, 6);
                mul_binary_tile_init();
                half_mul(i, 6, i == 0 ? 4 : 5);
                if (i != 0) {
                    add_binary_tile_init();
                    half_add(4, 5, 4);
                }
            }
            reduce_dst4();
            // Delta: (V - prediction) * beta. No fused multiply-add.
            load(3, 0, 5);
            load(5, 0, 6);
            sub_binary_tile_init();
            half_sub(5, 4, 4);
            mul_binary_tile_init();
            half_mul(4, 6, 4);
            // Update: (K * delta_row) + decayed_state, preserving add operand order.
            // The validated SFPU row broadcast reads only row0 of DST4; no CB11 roundtrip.
            for (uint32_t i = 0; i < 4; ++i) {
                load(2, i, 6);
                sfpu_mul_bcast_row_init();
                sfpu_mul_bcast_row(6, 4);
                add_binary_tile_init();
                half_add(6, i, i);
            }
            // Output uses the updated state and the identical four-K-tile fold/tree.
            for (uint32_t i = 0; i < 4; ++i) {
                load(1, i, 6);
                mul_binary_tile_init();
                half_mul(i, 6, i == 0 ? 4 : 5);
                if (i != 0) {
                    add_binary_tile_init();
                    half_add(4, 5, 4);
                }
            }
            reduce_dst4();
            cb_reserve_back(13, 1);
            if (token + 1 == length) cb_reserve_back(12, 4);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(13);
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL, PackMode::Default>(13)));
            pack_tile(4, 13);
            PACK((llk_pack_reduce_mask_clear()));
            if (token + 1 == length) {
                pack_reconfig_data_format(12);
                for (uint32_t i = 0; i < 4; ++i) pack_tile(i, 12, i);
            }
            // Native no-clear release waits for PACK before releasing MATH; full-sync
            // never flips DST banks. Standard tile_regs_release would zero all state.
            PACK((_llk_pack_dest_section_done_<DstSync::SyncFull, true, false>()));
            cb_push_back(13, 1);
            if (token + 1 == length) cb_push_back(12, 4);
            cb_pop_front(1, 4);
            cb_pop_front(2, 4);
            cb_pop_front(3, 1);
            cb_pop_front(4, 1);
            cb_pop_front(5, 1);
        }
    }
    cb_pop_front(6, 1);
}
