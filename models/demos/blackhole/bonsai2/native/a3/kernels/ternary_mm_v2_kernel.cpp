// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Ternary decode matmul with in-kernel group scales, model version (A3 v2).
//
// Same math as ternary_mm_scaled_kernel.cpp (see there): block-masked 8-row x, one
// custom_mm per super-group of 8 groups, SFPU ACC += P * S_G, sfpu_reduce at the end.
// Differences:
//   - weight and scale addresses are runtime args, so one compile serves every layer;
//   - x comes from the persistent masked buffer the prep kernel fills (cb 0, sharded);
//   - GU mode (tm_gu=1): the column stream holds (gate, up) pairs and the epilogue
//     writes silu(gate) * up, so gate/up/silu/mul is one op;
//   - BRISC writes the output to an interleaved tensor: 64 B 1x32 pages (tm_out_tile32=0,
//     the next prep reads them) or 2 KB 32x32 pages with row 0 live and zeros elsewhere
//     (tm_out_tile32=1, what the model's residual add expects).
//
// Runtime args, per core:
//   NCRISC: w_addr, bank, vc     BRISC: s_addr, out_addr, out_tile0 (first output tile of this core),
//           [3] out_tiles, [4] bank, [5] out2_addr, [6] n_split (second output from this tile on),
//                                       out_tiles (real output tiles; the rest are padding columns), bank
// Bank and vc are runtime, not per-core compile-time: per-core CT args split the program into
// one kernel binary per (bank, vc) group, and the extra dispatch cost ate the GDN out-proj gain.

#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/pack_block.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"
#endif
#endif

constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_in1 = 1;
constexpr uint32_t cb_s = 2;
constexpr uint32_t cb_out = 4;
constexpr uint32_t num_g = get_named_compile_time_arg_val("tm_num_g");  // super-groups (8 groups each)
constexpr uint32_t s_tiles = (num_g + 3) / 4;
// Scale tiles per column: s_full full tiles (4 super-groups each) plus, when num_g % 4 = rem > 0, a
// tail. tail_t consecutive columns share one tail tile in DRAM (column ti's rem super-groups at rows
// (ti * rem + r) * 8); BRISC reads it once per group and copies it into each column's slot.
constexpr uint32_t s_full = num_g / 4;
constexpr uint32_t rem = num_g % 4;
constexpr uint32_t tail_t = get_named_compile_time_arg_val("tm_tail_t");
// tm_s16: scale tiles are fp16 (2 KB, the GGUF's own type; the CB is declared bf16 as a carrier). TRISC
// reprograms the unpacker's source format to Float16 for the scale copy; DST holds them as fp32 (exact).
constexpr bool s16 = get_named_compile_time_arg_val("tm_s16") == 1;
constexpr uint32_t s_tb = s16 ? 2048 : 4096;
constexpr uint32_t pcn = get_named_compile_time_arg_val("tm_per_core_n");  // output tiles per core
constexpr bool gu = get_named_compile_time_arg_val("tm_gu") == 1;
constexpr uint32_t cols = gu ? 2 : 1;                                   // weight columns per output tile
constexpr uint32_t kt = num_g * 32;
constexpr uint32_t cols_per_core = pcn * cols;
// A2b: code tiles are 256 B (no exponent section); the unpacker forces every exponent to 127 (+-1.0).
constexpr bool shared_exp = get_named_compile_time_arg_val("tm_shared_exp") == 1;

#if defined(COMPILE_FOR_TRISC)
inline void force_shared_exp() {
    if constexpr (shared_exp) {
        UNPACK((cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(127)));
        UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1)));
    }
}
#endif

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
// sfpi units per 64-row DST tile slot; face f starts at unit 8 f. Unit 2q is rows 4q..4q+3 of a
// face's even columns, unit 2q+1 the same rows' odd columns, so units 4p..4p+3 = rows 8p..8p+7.
constexpr uint32_t U = 32;

template <uint32_t G, uint32_t ACC, uint32_t SBASE, uint32_t TI = 0>
inline void scale_acc() {
    // full tiles: super-group G at rows (G%4)*8 of tile G/4; tail: rows (TI*rem + G-4*s_full)*8 of tile s_full
    constexpr bool tail = G >= 4 * s_full;
    constexpr uint32_t b = tail ? (TI * rem + G - 4 * s_full) * 8 : (G % 4) * 8;
    constexpr uint32_t s0 = (SBASE + (tail ? s_full : G / 4)) * U + (b < 16 ? b / 2 : 16 + (b - 16) / 2);
#pragma GCC unroll 4
    for (uint32_t i = 0; i < 4; ++i) {
        sfpi::dst_reg[ACC * U + i] = sfpi::dst_reg[ACC * U + i] + sfpi::dst_reg[i] * sfpi::dst_reg[s0 + i];
        sfpi::dst_reg[i] = sfpi::vFloat(0.0f);
        sfpi::dst_reg[ACC * U + 8 + i] =
            sfpi::dst_reg[ACC * U + 8 + i] + sfpi::dst_reg[8 + i] * sfpi::dst_reg[s0 + 8 + i];
        sfpi::dst_reg[8 + i] = sfpi::vFloat(0.0f);
    }
}

inline void zero_slots(uint32_t nslots) {
    for (uint32_t i = 0; i < nslots * U; ++i) {
        sfpi::dst_reg[i] = sfpi::vFloat(0.0f);
    }
}

// slot 1 row 0 *= slot 2 row 0. A unit is 4 rows of the even (unit 2q) or odd (unit 2q+1)
// columns of a face, so row 0 of faces 0/1 needs units 0, 1, 8, 9.
inline void mul_row0() {
#pragma GCC unroll 4
    for (uint32_t i : {0u, 1u, 8u, 9u}) {
        sfpi::dst_reg[U + i] = sfpi::dst_reg[U + i] * sfpi::dst_reg[2 * U + i];
    }
}

template <typename F>
inline void sfpu_section(F f) {
    _llk_math_eltwise_sfpu_start_(0);
    f();
    _llk_math_eltwise_sfpu_done_();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}
#endif

#if defined(COMPILE_FOR_TRISC)
// super-groups [G0, G0 + N) of one column
template <uint32_t G0, uint32_t N, uint32_t ACC, uint32_t SBASE, uint32_t TI>
struct Groups {
    static inline void run() {
        Groups<G0, N - 1, ACC, SBASE, TI>::run();
        cb_wait_front(cb_in1, 32);
        custom_mm_block<CustomMmDestLayout::Default, false>(cb_in0, cb_in1, (G0 + N - 1) * 32, 0, 0, 32);
        cb_pop_front(cb_in1, 32);
        MATH((sfpu_section([] { scale_acc<G0 + N - 1, ACC, SBASE, TI>(); })));
    }
};
template <uint32_t G0, uint32_t ACC, uint32_t SBASE, uint32_t TI>
struct Groups<G0, 0, ACC, SBASE, TI> {
    static inline void run() {}
};

// one column: the full-tile super-groups, then the tail ones at this column's slot ti of the shared tail
template <uint32_t ACC, uint32_t SBASE>
inline void column(uint32_t ti) {
    Groups<0, 4 * s_full, ACC, SBASE, 0>::run();
    if constexpr (rem > 0) {
        switch (ti) {
            case 0: Groups<4 * s_full, rem, ACC, SBASE, 0>::run(); break;
            case 1: if constexpr (tail_t > 1) { Groups<4 * s_full, rem, ACC, SBASE, 1>::run(); } break;
            case 2: if constexpr (tail_t > 2) { Groups<4 * s_full, rem, ACC, SBASE, 2>::run(); } break;
            default: if constexpr (tail_t > 3) { Groups<4 * s_full, rem, ACC, SBASE, 3>::run(); } break;
        }
    }
}
#endif

void kernel_main() {
    constexpr bool is_active = get_named_compile_time_arg_val("tm_is_active_core") == 1;
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (is_active) {
        // codes: this core's column sticks, contiguous in its DRAM bank (deepseek reader, address at runtime)
        constexpr uint32_t page = get_named_compile_time_arg_val("tm_in1_page_size");
        constexpr uint32_t pages = get_named_compile_time_arg_val("tm_in1_num_pages");
        constexpr uint32_t block = get_named_compile_time_arg_val("tm_in1_block_bytes");  // 32 tiles
        constexpr uint32_t nbuf = 3;
        unified_kernels::setup_sharded_buffer(cb_in0, kt);
        const uint32_t w_addr = get_arg_val<uint32_t>(0);
        const uint32_t bank = get_arg_val<uint32_t>(1);
        const uint32_t vc = get_arg_val<uint32_t>(2);
        const uint64_t base = get_noc_addr_from_bank_id<true>(bank, w_addr);
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        noc_async_read_one_packet_set_state<true>(base, page, vc);
        cb_reserve_back(cb_in1, 32 * 2);
        uint32_t l1 = get_write_ptr(cb_in1);
        auto& iface = get_local_cb_interface(cb_in1);
        const uint32_t cb_base = iface.fifo_limit - iface.fifo_size;
        const uint32_t cb_end = cb_base + nbuf * block;
        uint32_t src = 0, free_blocks = nbuf, trid = 1, trid_wait = 1;
        for (uint32_t it = 0; it < num_g * cols_per_core; ++it) {
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(base, src, l1, trid);
                src += page;
                l1 += page;
            }
            if (free_blocks == nbuf - 1) {
                noc_async_read_barrier_with_trid(trid_wait);
                cb_push_back(cb_in1, 32);
                trid_wait = trid_wait == nbuf ? 1 : trid_wait + 1;
                cb_reserve_back(cb_in1, 32 * 2);
            } else {
                free_blocks -= 1;
            }
            trid = trid == nbuf ? 1 : trid + 1;
            if (l1 >= cb_end) {
                l1 = cb_base;
            }
        }
        noc_async_read_barrier_with_trid(trid_wait);
        cb_push_back(cb_in1, 32);
    }
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (is_active) {
        constexpr bool out_tile32 = get_named_compile_time_arg_val("tm_out_tile32") == 1;
        const uint32_t bank = get_arg_val<uint32_t>(4);
        constexpr uint32_t out_page = out_tile32 ? 2048 : 64;
        uint32_t s_addr = get_arg_val<uint32_t>(0);
        constexpr bool out_is_dram = get_named_compile_time_arg_val("tm_out_is_dram") == 1;
        const InterleavedAddrGen<out_is_dram> og{.bank_base_address = get_arg_val<uint32_t>(1), .page_size = out_page};
        const uint32_t out_tile0 = get_arg_val<uint32_t>(2);
        const uint32_t out_tiles = get_arg_val<uint32_t>(3);  // tiles past this are padding columns
        // optional second output: tiles from n_split on go to out2 (page t - n_split), e.g. GDN [qkv | z]
        const InterleavedAddrGen<out_is_dram> og2{.bank_base_address = get_arg_val<uint32_t>(5), .page_size = out_page};
        const uint32_t n_split = get_arg_val<uint32_t>(6);
        // out_tile32: a zeroed 2 KB staging tile; each output overwrites row 0 (face 0/1 row 0) only
        constexpr uint32_t cb_stage = 5;
        uint32_t stage = 0;
        if constexpr (out_tile32) {
            cb_reserve_back(cb_stage, 1);
            stage = get_write_ptr(cb_stage);
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
            for (uint32_t i = 0; i < 512; ++i) {
                z[i] = 0;
            }
        }
        auto write_out = [&](uint32_t n) {
            cb_wait_front(cb_out, 1);
            if (out_tile0 + n >= out_tiles) {
                cb_pop_front(cb_out, 1);
                return;
            }
            const uint32_t src = get_read_ptr(cb_out);
            const uint32_t t = out_tile0 + n;
            const uint64_t dst = t < n_split ? get_noc_addr(t, og) : get_noc_addr(t - n_split, og2);
            if constexpr (out_tile32) {
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                for (uint32_t i = 0; i < 8; ++i) {
                    d[i] = s[i];          // face 0 row 0
                    d[128 + i] = s[8 + i];  // face 1 row 0 (+512 B)
                }
                noc_async_write(stage, dst, 2048);
            } else {
                noc_async_write(src, dst, 64);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        };
        // per group of tail_t columns in DRAM: [tail][main c0][main c1]...; the tail goes to scratch once
        constexpr uint32_t cb_tail = 6;
        uint32_t tail_l1 = 0;
        if constexpr (rem > 0) {
            cb_reserve_back(cb_tail, 1);
            tail_l1 = get_write_ptr(cb_tail);
        }
        for (uint32_t n = 0; n < pcn; ++n) {
            cb_reserve_back(cb_s, cols * s_tiles);
            uint32_t l1 = get_write_ptr(cb_s);
            for (uint32_t j = 0; j < cols; ++j) {
                if constexpr (rem > 0) {
                    if ((n * cols + j) % tail_t == 0) {
                        noc_async_read(get_noc_addr_from_bank_id<true>(bank, s_addr), tail_l1, s_tb);
                        s_addr += s_tb;
                        noc_async_read_barrier();
                    }
                }
                if constexpr (s_full > 0) {
                    noc_async_read(get_noc_addr_from_bank_id<true>(bank, s_addr), l1, s_full * s_tb);
                    s_addr += s_full * s_tb;
                }
                if constexpr (rem > 0) {
                    noc_async_read(get_noc_addr(tail_l1), l1 + s_full * s_tb, s_tb);  // local copy
                }
                l1 += s_tiles * s_tb;
            }
            noc_async_read_barrier();
            cb_push_back(cb_s, cols * s_tiles);
            if (n > 0) {
                write_out(n - 1);
            }
        }
        write_out(pcn - 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    if constexpr (is_active) {
        custom_mm_block_init<false, CustomMmDestLayout::Default, true>(cb_in0, cb_in1, cb_out);
        force_shared_exp();
        cb_wait_front(cb_in0, kt);
        for (uint32_t n = 0; n < pcn; ++n) {
            tile_regs_acquire();
            cb_wait_front(cb_s, cols * s_tiles);
            reconfig_data_format_srca(cb_in1, cb_s);
            copy_init(cb_s);
            if constexpr (s16) {  // the CB says bf16; the bytes are fp16
                UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
                    (uint32_t)DataFormat::Float16, (uint32_t)DataFormat::Float16, 2048)));
                MATH((_llk_math_reconfig_data_format_srca_<DST_ACCUM_MODE>((uint32_t)DataFormat::Float16)));
            }
            for (uint32_t j = 0; j < cols * s_tiles; ++j) {
                copy_tile(cb_s, j, 3 + j);  // gate scales at 3.., up scales at 3 + s_tiles..
            }
            cb_pop_front(cb_s, cols * s_tiles);
            reconfig_full_operand<SrcOrder::Reverse>(cb_in0, cb_in1);
            custom_mm_block_init_short<false, CustomMmDestLayout::Default>(cb_in0, cb_in1, cb_out);
            force_shared_exp();
            MATH((sfpu_section([] { zero_slots(3); })));
            column<1, 3>((n * cols) % tail_t);
            if constexpr (gu) {
                column<2, 3 + s_tiles>((n * cols + 1) % tail_t);
            }
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(1);
            if constexpr (gu) {
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(2);
                silu_tile_init();
                silu_tile(1);
                MATH((sfpu_section([] { mul_row0(); })));
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_block_contiguous_init(cb_out);
            pack_block_contiguous(1, cb_out, 1);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_in0, kt);
    }
#endif
}
