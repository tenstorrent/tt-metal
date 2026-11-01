// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Ternary decode matmul with the per-128-group scales applied inside the kernel (A3, step b).
//
//   y[n] = sum_g s[g, n] * sum_{k in g} x[k] W[k, n],   g = k / 128
//
// x arrives block-masked as 8-row tiny tiles: row g % 8 of K-tile k carries x where
// g = k / 4, every other row is zero. A super-group G is 8 groups = 32 K-tiles, so one
// custom_mm over 32 tiles leaves P[r, :] = the partial of group 8G + r in DST rows 0-7,
// exact in fp32 (x bf16, W in {-1,0,1}).
//
// Per output tile n (DST full sync, fp32, one acquire per n):
//   slot 0 = P, slot 1 = ACC, slots 2.. = the scales of all groups for this n, unpacked
//   to DST once (fp32, 4 super-groups per 32x32 tile: rows (G%4)*8 .. +7 of tile G/4).
//   for G: P = custom_mm(32 tiles); SFPU ACC[0:8] += P[0:8] * S_G; P = 0
//   SFPU column sum of ACC -> row 0; pack it.
// The unpacker switches between custom_mm and copy twice per n, never per group.
//
// NCRISC streams the codes (deepseek reader, unchanged). BRISC streams the scale tiles.

#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "../../../../../deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp"
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"
#endif
#endif

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("tm_is_active_core") == 1;
};

constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_s = 2;
constexpr uint32_t cb_out = 4;
constexpr uint32_t num_g = get_named_compile_time_arg_val("tm_num_subblocks_k");  // super-groups
constexpr uint32_t s_tiles = (num_g + 3) / 4;
constexpr uint32_t pcn = get_named_compile_time_arg_val("tm_per_core_n");

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
// DST addressing for sfpi: 32 units per 64-row tile slot, one unit = 2 rows; face f
// starts at unit 8 f. Rows 0-7 of a tiny 8x32 tile are units 0-3 (face 0) and 8-11 (face 1).
constexpr uint32_t U = 32;
constexpr uint32_t P = 0 * U, A = 1 * U;

template <uint32_t G>
inline void scale_acc() {
    constexpr uint32_t b = (G % 4) * 8;
    constexpr uint32_t s0 = (2 + G / 4) * U + (b < 16 ? b / 2 : 16 + (b - 16) / 2);
#pragma GCC unroll 4
    for (uint32_t i = 0; i < 4; ++i) {
        sfpi::dst_reg[A + i] = sfpi::dst_reg[A + i] + sfpi::dst_reg[P + i] * sfpi::dst_reg[s0 + i];
        sfpi::dst_reg[P + i] = sfpi::vFloat(0.0f);
        sfpi::dst_reg[A + 8 + i] = sfpi::dst_reg[A + 8 + i] + sfpi::dst_reg[P + 8 + i] * sfpi::dst_reg[s0 + 8 + i];
        sfpi::dst_reg[P + 8 + i] = sfpi::vFloat(0.0f);
    }
}

inline void zero_p_acc() {
#pragma GCC unroll 8
    for (uint32_t i = 0; i < 2 * U; ++i) {
        sfpi::dst_reg[i] = sfpi::vFloat(0.0f);
    }
}

template <typename F>
inline void sfpu_section(F f) {
    _llk_math_eltwise_sfpu_start_(0);  // dst base = slot 0 of this section, wait for the FPU
    f();
    _llk_math_eltwise_sfpu_done_();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}
#endif

#if defined(COMPILE_FOR_TRISC)
template <uint32_t G>
struct Groups {
    static inline void run() {
        Groups<G - 1>::run();
        cb_wait_front(cb_in1, 32);
        custom_mm_block<CustomMmDestLayout::Default, false>(cb_in0, cb_in1, (G - 1) * 32, 0, 0, 32);
        cb_pop_front(cb_in1, 32);
        MATH((sfpu_section([] { scale_acc<G - 1>(); })));
    }
};
template <>
struct Groups<0> {
    static inline void run() {}
};
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
        cb_in1,
        cb_out,
        get_named_compile_time_arg_val("tm_in1_addr"),
        get_named_compile_time_arg_val("tm_in1_page_size"),
        get_named_compile_time_arg_val("tm_in1_num_pages"),
        32,
        pcn,
        get_named_compile_time_arg_val("tm_in1_block_bytes"),
        pcn,
        num_g,
        get_named_compile_time_arg_val("tm_bank_id"),
        get_named_compile_time_arg_val("tm_vc")>;
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(cb_in0, get_named_compile_time_arg_val("tm_num_tiles_k"));
    }
    deepseek_b1_ops::DRAMStreamingMatmul::Op<CTArgs, Core::is_active_core, false> mm;
    mm();
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_active_core) {
        constexpr uint32_t s_tile_bytes = 4096;  // fp32 32x32
        constexpr uint32_t bank = get_named_compile_time_arg_val("tm_bank_id");
        uint32_t addr = get_named_compile_time_arg_val("tm_s_addr");
        for (uint32_t n = 0; n < pcn; ++n) {
            cb_reserve_back(cb_s, s_tiles);
            uint32_t l1 = get_write_ptr(cb_s);
            for (uint32_t j = 0; j < s_tiles; ++j) {
                noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr), l1, s_tile_bytes);
                addr += s_tile_bytes;
                l1 += s_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_s, s_tiles);
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    if constexpr (Core::is_active_core) {
        custom_mm_block_init<false, CustomMmDestLayout::Default, true>(cb_in0, cb_in1, cb_out);
        cb_wait_front(cb_in0, get_named_compile_time_arg_val("tm_num_tiles_k"));
        for (uint32_t n = 0; n < pcn; ++n) {
            tile_regs_acquire();
            // scales for this n into slots 2.. (unpack straight to DST, fp32)
            cb_wait_front(cb_s, s_tiles);
            reconfig_data_format_srca(cb_in1, cb_s);
            copy_init(cb_s);
            for (uint32_t j = 0; j < s_tiles; ++j) {
                copy_tile(cb_s, j, 2 + j);
            }
            cb_pop_front(cb_s, s_tiles);
            reconfig_full_operand<SrcOrder::Reverse>(cb_in0, cb_in1);
            custom_mm_block_init_short<false, CustomMmDestLayout::Default>(cb_in0, cb_in1, cb_out);
            MATH((sfpu_section([] { zero_p_acc(); })));
            Groups<num_g>::run();
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(1);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_block_contiguous_init(cb_out);
            pack_block_contiguous(1, cb_out, 1);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_in0, get_named_compile_time_arg_val("tm_num_tiles_k"));
    }
#endif
}
