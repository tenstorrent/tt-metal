// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_rational_helpers.inc"

// Integration ownership only; the selected configuration and helpers remain.
#ifndef TTPOLY_LLK_COMPILATION
void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t num_degree = NUM_DEGREE;
    constexpr uint32_t den_degree = DEN_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t lut_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_degree = get_compile_time_arg_val(1);
    constexpr uint32_t den_degree = get_compile_time_arg_val(2);
    constexpr uint32_t num_segments = get_compile_time_arg_val(3);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, lut_size>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

#ifdef FUSE_GRAD_MUL
    // Second input stream: the incoming gradient for the fused backward
    // multiply. Declared outside the EMBEDDED_LUT branch split because BOTH
    // modes need it (same convention as piecewise_generic.cpp).
    constexpr auto cb_grad = tt::CBIndex::c_1;
#endif

#ifdef TRISC_MATH
    // Clear any ADDR_MOD_SET_Base left set by a previous PROCESS before
    // init_sfpu programs the SFPU addr_mods. No-op on Blackhole. MATH-only:
    // the declaration is unreachable on the unpack/pack TUs.
    TT_REPLAY_WH_ADDRMOD_SCRUB();
#endif
    init_sfpu(cb_in, cb_out);

#include "shared_rational_init.inc"

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
#ifdef FUSE_GRAD_MUL
        // Park `grad` in DST tile 1 so the epilogue can form grad * f'(x).
        // ORDER MATTERS: copy grad FIRST so the input copy leaves the sfpi
        // dst_reg[] base on tile 0 where the evaluator expects it — see the
        // measured failure mode in piecewise_generic.cpp's twin block.
        cb_wait_front(cb_grad, 1);
        copy_tile(cb_grad, 0, 1);
#endif
#if defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_CLASS_REP)
        // One copy owns both the selected evaluator's input and the quotient's
        // raw shadow.  The replay loads tile 1 at offset 64 and writes tile 0;
        // rebasing the D-RWC after copy is one configuration issue and avoids
        // the former second full-tile datacopy.
        copy_tile(cb_in, 0, 1);
#ifdef TRISC_MATH
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::get_dest_buffer_base());
#endif
#else
        copy_tile(cb_in, 0, 0);
#endif

#include "shared_rational_tile.inc"

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
#ifdef FUSE_GRAD_MUL
        cb_pop_front(cb_grad, 1);
#endif
        tile_regs_release();
    }
}
#endif  // TTPOLY_LLK_COMPILATION
