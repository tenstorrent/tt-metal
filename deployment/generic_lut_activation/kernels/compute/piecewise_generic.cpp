// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_generic_helpers.inc"

// Integration ownership only: LLK callers provide their own entry/lifecycle.
// The selected configuration and shared evaluator helpers remain identical.
#ifndef TTPOLY_LLK_COMPILATION
void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t poly_degree = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t lut_size = get_compile_time_arg_val(0);
    constexpr uint32_t poly_degree = get_compile_time_arg_val(1);
    constexpr uint32_t num_segments = get_compile_time_arg_val(2);

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
    // modes need it. c_1 is otherwise unused (only c_0, c_16 and, in generic
    // LUT mode, c_25 are taken).
    constexpr auto cb_grad = tt::CBIndex::c_1;
#endif

#ifdef TRISC_MATH
    // Clear any ADDR_MOD_SET_Base left set by a previous PROCESS before
    // init_sfpu programs the SFPU addr_mods. No-op on Blackhole. MATH-only:
    // the declaration is unreachable on the unpack/pack TUs.
    TT_REPLAY_WH_ADDRMOD_SCRUB();
#endif
    init_sfpu(cb_in, cb_out);

#include "shared_generic_init.inc"

#ifdef PACK_RELU_MODE
    // Lever 1 (packer-side ReLU). The packer applies ReLU to datums leaving DST
    // just after early format conversion, at ZERO per-tile cost -- this is one
    // config-register RMW, hoisted out of the loop. For the covered ops the
    // packer computes the exact function, so the SFPU eval body below collapses
    // to the identity path and no polynomial is evaluated at all.
    //
    // STACC_RELU is device config state that PERSISTS past this program, so it
    // is reset to NO_RELU after the loop. Leaving it armed would silently clamp
    // negatives in whatever op ran next on this core.
#if PACK_RELU_MODE == 1
    pack_relu_config(ckernel::ReluConfig::zero());
#elif PACK_RELU_MODE == 2
    pack_relu_config(ckernel::ReluConfig::min_threshold(PACK_RELU_THRESHOLD));
#elif PACK_RELU_MODE == 3
    pack_relu_config(ckernel::ReluConfig::max_threshold(PACK_RELU_THRESHOLD));
#else
#error "PACK_RELU_MODE must be 1 (zero), 2 (min_threshold) or 3 (max_threshold)"
#endif
#endif  // PACK_RELU_MODE

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
#if defined(TT_TARGET_SELECTED_BF16_SCRATCH_BOUNDARY)
        // The preceding iteration ends with c24 as the unpack source.  Source
        // selection is unpacker state, not an argument carried by copy_tile;
        // restore c0 before consuming the next raw tile.
        copy_init(cb_in);
#endif
#ifdef FUSE_GRAD_MUL
        // Park `grad` in DST tile 1 so the epilogue can form grad * f'(x).
        // cb_grad shares data_format and tile_size with cb_in by construction,
        // so the unpacker config from init_sfpu(cb_in, ...) applies unchanged.
        //
        // ORDER MATTERS: sfpi's dst_reg[] is relative to the DST base that the
        // LAST copy_tile established. Copying grad AFTER the input would leave
        // the base on tile 1, so the evaluator would transform grad instead of
        // x and pack_tile(0, ...) would emit the untouched input. Measured:
        // that ordering makes every output exactly x. Copy grad FIRST, so the
        // input copy leaves the base on tile 0 where the evaluator expects it.
        cb_wait_front(cb_grad, 1);
        copy_tile(cb_grad, 0, 1);
#endif
        copy_tile(cb_in, 0, 0);

#include "shared_generic_tile.inc"

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
#if defined(TT_TARGET_SELECTED_BF16_SCRATCH_BOUNDARY)
        // copy_tile is asynchronous with respect to the unpacker.  Follow the
        // in-tree temporary-CB pattern: retain both pages through the final
        // pack, then pop them immediately before the ordinary input pop and
        // destination release.  This is the first point at which all three
        // TRISC consumers have crossed the phase boundary.
        cb_pop_front(tt::CBIndex::c_24, 2);
#endif
        cb_pop_front(cb_in, 1);
#ifdef FUSE_GRAD_MUL
        cb_pop_front(cb_grad, 1);
#endif
        tile_regs_release();
    }
#ifdef PACK_RELU_MODE
    // Disarm: STACC_RELU outlives this program (see above).
    pack_relu_config(ckernel::ReluConfig::none());
#endif
}
#endif  // TTPOLY_LLK_COMPILATION
