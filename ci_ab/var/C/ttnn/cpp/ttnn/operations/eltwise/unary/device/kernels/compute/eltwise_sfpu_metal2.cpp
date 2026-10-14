// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of eltwise_sfpu.cpp, which lives beside it. Ops ported to Metal 2.0
// bind this file; the original serves the consumers still on the legacy API. Until the last of them
// migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::in, dfb::out) and the named argument set (num_tiles) are this fork's
// interface: every later consumer inherits them, so they are taken from the kernel's own vocabulary
// rather than any one op's locals, and are not renamed once a consumer exists. SFPU_OP_CHAIN_0, when
// defined, is expanded once per tile between the copy into Dest and the pack.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#if defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_TILE)

// The chain forms the program factory emits into SFPU_OP_CHAIN_0_TILE: on a later tile an op re-programs only the state
// another op of the chain wrote.
namespace ckernel {
#ifdef SFPU_OP_EXP_INCLUDE
template <bool approx, bool constants, bool upper_macros>
ALWI void exp_tile_chain_reinit() {
    MATH((sfpu::exp_init<approx, 0x3F800000, true, DST_ACCUM_MODE, false, constants, upper_macros>()));
}
#endif
#ifdef SFPU_OP_RECIP_INCLUDE
ALWI void recip_tile_chain_init() {
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::recip_init, (APPROX, DST_ACCUM_MODE, true, false)));
}
template <bool own_state>
ALWI void recip_tile_chain_reinit() {
    MATH((sfpu::recip_init<APPROX, DST_ACCUM_MODE, false, false, own_state>()));
}
ALWI void recip_tile_chain_rerecord() { MATH((sfpu::_record_reciprocal_fast_24b_5c_())); }
#endif
#ifdef SFPU_OP_RSQRT_INCLUDE
ALWI void rsqrt_tile_chain_reinit() { MATH((sfpu::rsqrt_init<APPROX>())); }
#endif
#ifdef SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
template <bool prgm_rounding>
ALWI void square_tile_chain_init() {
    MATH(llk_math_sfpu_init_once());
    MATH((sfpu::_square_init_<ADDR_MOD_4, prgm_rounding>()));
}
template <bool prgm_rounding>
ALWI void square_tile_chain(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_square,
        (APPROX, DST_ACCUM_MODE, 32, ADDR_MOD_4, prgm_rounding),
        idst,
        VectorMode::None));
}
#endif
}  // namespace ckernel

// Expanded in kernel_main's loops: i == 0 on the first tile or block.
#define SFPU_OP_CHAIN_FIRST_TILE_ONLY(init) \
    if (i == 0) {                           \
        init                                \
    }
#define SFPU_OP_CHAIN_FIRST_OR_LATER_TILE(first, later) \
    if (i == 0) {                                       \
        first                                           \
    } else {                                            \
        later                                           \
    }
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
#if defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_BLOCK)
    // Op-major blocks: every op runs over the block's tiles before the next op, so an init or re-program the chain
    // repeats is paid once per block; each tile is copied just before the first op runs on it.
#define SFPU_OP_CHAIN_FOR_EACH_TILE(func) \
    for (uint32_t t = 0; t < n; ++t) {    \
        if (!copied) {                    \
            dfb_in.wait_front(t + 1);     \
            copy_tile(dfb::in, t, t);     \
        }                                 \
        func                              \
    }                                     \
    copied = true;
    for (uint32_t i = 0; i < num_tiles; i += SFPU_OP_CHAIN_0_BLOCK) {
        const uint32_t n = num_tiles - i < SFPU_OP_CHAIN_0_BLOCK ? num_tiles - i : SFPU_OP_CHAIN_0_BLOCK;
        tile_regs_acquire();

        dfb_out.reserve_back(n);

        bool copied = false;
        SFPU_OP_CHAIN_0_BLOCK_OPS

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t t = 0; t < n; ++t) {
            pack_tile(t, dfb::out);
        }

        dfb_in.pop_front(n);
        dfb_out.push_back(n);

        tile_regs_release();
    }
#else
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        copy_tile(dfb::in, 0, 0);

#if defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_FUNC_0) && !defined(SFPU_OP_CHAIN_0_FUNC_1)
        // A single op's init state outlives its calls, so only the first tile runs the init.
        if (i == 0) {
            SFPU_OP_CHAIN_0_INIT_0
        }
        if (i != 0) {
            SFPU_OP_CHAIN_0_FUNC_0
        }
#elif defined(ARCH_BLACKHOLE) && defined(SFPU_OP_CHAIN_0_TILE)
        // The host wraps the inits whose state nothing after them writes, and the inits a later tile repeats.
        SFPU_OP_CHAIN_0_TILE
#elif defined(SFPU_OP_CHAIN_0)
        SFPU_OP_CHAIN_0
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb::out);

        dfb_in.pop_front(1);
        dfb_out.push_back(1);

        tile_regs_release();
    }
#endif
}
