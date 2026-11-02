// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused decode RMSNorm + prep: y = x * rstd * g (g = 1 + w), then x' = H(D y) per 1024-block,
// scattered into the A3 workers' masked buffer exactly as prep_hadamard_kernel.cpp does.
// Replaces to_memory_config + sharded rms_norm + sharded_to_interleaved + prep at the 5120 sites.
//
// One unit (b, r) per core (prep's units). rstd needs the whole row, so the cores exchange their
// block's partial sum of squares through a master core (the first unit's core):
//   TRISC  per-column partial sums of x_b^2 -> cb_p (1x32 fp32)
//   BRISC  sums the 32 values, writes partial[b] into the master's slot array, increments the
//          master's sem 0; the master waits for every core, computes rstd, writes it to every core
//          and increments their sem 1; every BRISC then fills cb_r and does the scatter.
//   TRISC  y = x * g * rstd in fp32 (SFPU), packed bf16 -> cb_y, then x_b' = y_b @ M_b (custom_mm).
//
// Runtime args, per core:
//   TRISC:  rb_addr
//   NCRISC: x_addr, m_addr, g_addr, b, r
//   BRISC:  x8_addr, b, r, rb_addr, master_xy, is_master, num_cores, num_blocks, inv_k (fp32 bits),
//           eps (fp32 bits), num_workers, worker xy..., [master only] core xy... for the rstd broadcast
// Compile time: x_is_dram, m_page, units_per_bank.

#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/pack_block.h"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_common.h"
#endif
#endif

constexpr uint32_t cb_x = 0, cb_m = 1, cb_o = 2, cb_stage = 3, cb_g = 7, cb_p = 8, cb_r = 9, cb_y = 10, cb_scr = 11;
// cb_sync (4 KB, same address on every core of the program): partial[b] at +16 b, rstd at +1024,
// NoC staging words at +2048 / +2064 (16 B strides: NoC writes keep the 16 B phase)
constexpr uint32_t cb_sync = 12;
constexpr uint32_t KT = 32;  // K-tiles per 1024-block
constexpr uint32_t GRP = 4;  // tiles per SFPU group in the y pass (DST slots 0-3 x, 4-7 g)

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
constexpr uint32_t U = 32;
// row 0 of a 1x32 tile in DST: units 0/1 (face 0 even/odd columns), 8/9 (face 1)
inline void sq_acc() {  // slot 0 += slot 1 ^ 2
#pragma GCC unroll 4
    for (uint32_t i : {0u, 1u, 8u, 9u}) {
        sfpi::vFloat v = sfpi::dst_reg[U + i];
        sfpi::dst_reg[i] = sfpi::dst_reg[i] + v * v;
    }
}
inline void zero2() {
#pragma GCC unroll 64
    for (uint32_t i = 0; i < 2 * U; ++i) {
        sfpi::dst_reg[i] = sfpi::vFloat(0.0f);
    }
}
template <uint32_t J>
inline void scale_one(sfpi::vFloat r) {  // slot J = slot J * slot (J + 4) * r
#pragma GCC unroll 4
    for (uint32_t i : {0u, 1u, 8u, 9u}) {
        sfpi::dst_reg[J * U + i] = sfpi::dst_reg[J * U + i] * sfpi::dst_reg[(J + 4) * U + i] * r;
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

#if defined(COMPILE_FOR_BRISC)
inline float as_f(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } c{u};
    return c.f;
}
inline uint32_t as_u(float f) {
    union {
        float f;
        uint32_t u;
    } c{f};
    return c.u;
}
inline float rsqrt_newton(float v) {  // soft-float rsqrt to full fp32 precision
    uint32_t i = 0x5f3759df - (as_u(v) >> 1);
    float y = as_f(i);
    for (int k = 0; k < 4; ++k) {
        y = y * (1.5f - 0.5f * v * y * y);
    }
    return y;
}
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr bool x_is_dram = get_named_compile_time_arg_val("rp_x_is_dram") == 1;
    constexpr uint32_t m_page = get_named_compile_time_arg_val("rp_m_page");
    constexpr uint32_t upb = get_named_compile_time_arg_val("rp_units_per_bank");
    const uint32_t x_addr = get_arg_val<uint32_t>(0), m_addr = get_arg_val<uint32_t>(1), g_addr = get_arg_val<uint32_t>(2);
    const uint32_t b = get_arg_val<uint32_t>(3), r = get_arg_val<uint32_t>(4);
    const InterleavedAddrGen<x_is_dram> xg{.bank_base_address = x_addr, .page_size = 2048};
    const InterleavedAddrGen<true> gg{.bank_base_address = g_addr, .page_size = 128};
    // x row 0 of each K-tile: 64 B-aligned reads (rows 0-1 of face 0 at +0, of face 1 at +512) into scratch
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    cb_reserve_back(cb_x, KT);
    cb_reserve_back(cb_g, KT);
    for (uint32_t i = 0; i < KT; ++i) {
        const uint32_t page = b * KT + i;
        noc_async_read(get_noc_addr(page, xg, 0), scr + i * 128, 64);
        noc_async_read(get_noc_addr(page, xg, 512), scr + i * 128 + 64, 64);
        noc_async_read(get_noc_addr(page, gg, 0), get_write_ptr(cb_g) + i * 128, 128);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_x));
    for (uint32_t i = 0; i < KT; ++i) {
        for (uint32_t w = 0; w < 8; ++w) {
            d[i * 16 + w] = s[i * 32 + w];           // face 0 row 0
            d[i * 16 + 8 + w] = s[i * 32 + 16 + w];  // face 1 row 0
        }
    }
    cb_push_back(cb_x, KT);
    cb_push_back(cb_g, KT);
    const uint32_t unit = 8 * b + r;
    uint32_t src = m_addr + (unit % upb) * 4 * KT * m_page;
    const uint32_t bank = unit / upb;
    for (uint32_t jj = 0; jj < 4; ++jj) {
        cb_reserve_back(cb_m, KT);
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), get_write_ptr(cb_m), KT * m_page);
        noc_async_read_barrier();
        src += KT * m_page;
        cb_push_back(cb_m, KT);
    }
#elif defined(COMPILE_FOR_BRISC)
    const uint32_t x8_addr = get_arg_val<uint32_t>(0);
    const uint32_t b = get_arg_val<uint32_t>(1), r = get_arg_val<uint32_t>(2);
    cb_reserve_back(cb_sync, 1);
    const uint32_t slots = get_write_ptr(cb_sync);
    const uint32_t mxy = get_arg_val<uint32_t>(4);
    const bool is_master = get_arg_val<uint32_t>(5) == 1;
    const uint32_t ncores = get_arg_val<uint32_t>(6), nblocks = get_arg_val<uint32_t>(7);
    const float inv_k = as_f(get_arg_val<uint32_t>(8)), eps = as_f(get_arg_val<uint32_t>(9));
    const uint32_t nw = get_arg_val<uint32_t>(10);
    const uint32_t sem_part = get_semaphore(0), sem_rstd = get_semaphore(1);
    // 1. this block's partial sum of squares -> master's slot b
    cb_wait_front(cb_p, 1);
    volatile tt_l1_ptr float* p = reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(cb_p));
    float part = 0.0f;
    for (uint32_t c = 0; c < 32; ++c) {
        part += p[c];
    }
    cb_pop_front(cb_p, 1);
    volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
    sl[2048 / 4] = as_u(part);  // staging word for the NoC write
    noc_async_write(slots + 2048, get_noc_addr(mxy >> 16, mxy & 0xffff, slots + 16 * b), 4);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(mxy >> 16, mxy & 0xffff, sem_part), 1);
    // 2. master: all partials in -> rstd to every core
    if (is_master) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_part), ncores);
        float sum = 0.0f;
        for (uint32_t i = 0; i < nblocks; ++i) {
            sum += as_f(sl[4 * i]);
        }
        sl[2064 / 4] = as_u(rsqrt_newton(sum * inv_k + eps));
        const uint32_t list = 11 + nw;
        for (uint32_t c = 0; c < ncores; ++c) {
            const uint32_t xy = get_arg_val<uint32_t>(list + c);
            noc_async_write(slots + 2064, get_noc_addr(xy >> 16, xy & 0xffff, slots + 1024), 4);
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < ncores; ++c) {
            const uint32_t xy = get_arg_val<uint32_t>(list + c);
            noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xffff, sem_rstd), 1);
        }
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_rstd), 1);
    // 3. rstd as a 1x32 fp32 row for TRISC
    cb_reserve_back(cb_r, 1);
    volatile tt_l1_ptr uint32_t* rr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_r));
    const uint32_t rbits = sl[1024 / 4];
    for (uint32_t c = 0; c < 32; ++c) {
        rr[c] = rbits;
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3)) = rbits;  // rb shard: what MATH reads
    cb_push_back(cb_r, 1);  // the signal: TRISC unpack waits on it, MATH follows the unpacker
    // 4. scatter, as prep_hadamard_kernel.cpp
    cb_reserve_back(cb_stage, 1);
    const uint32_t stage = get_write_ptr(cb_stage);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    for (uint32_t i = 0; i < 128; ++i) {
        st[i] = 0;
    }
    for (uint32_t j = 4 * r; j < 4 * r + 4; ++j) {
        const uint32_t k = b * KT + j;
        cb_wait_front(cb_o, 1);
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_o));
        for (uint32_t i = 0; i < 8; ++i) {
            st[r * 8 + i] = row[i];
            st[64 + r * 8 + i] = row[8 + i];
        }
        cb_pop_front(cb_o, 1);
        for (uint32_t w = 0; w < nw; ++w) {
            const uint32_t xy = get_arg_val<uint32_t>(11 + w);
            noc_async_write(stage, get_noc_addr(xy >> 16, xy & 0xffff, x8_addr + k * 512), 512);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < 8; ++i) {
            st[r * 8 + i] = 0;
            st[64 + r * 8 + i] = 0;
        }
    }
#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    // 1. per-column partial sums of x^2 over this block's 32 K-tiles (slot 0), x tiles land in slot 1
    copy_init(cb_x);
    cb_wait_front(cb_x, KT);
    tile_regs_acquire();
    MATH((sfpu_section([] { zero2(); })));
    for (uint32_t t = 0; t < KT; ++t) {
        copy_tile(cb_x, t, 1);
        MATH((sfpu_section([] { sq_acc(); })));
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_p, 1);
    pack_reconfig_data_format(cb_p);
    pack_block_contiguous_init(cb_p);
    pack_block_contiguous(0, cb_p, 1);
    cb_push_back(cb_p, 1);
    tile_regs_release();
    // 2. y = x * g * rstd, GRP tiles at a time (x in slots 0-3, g in 4-7), packed bf16 -> cb_y
    // rstd: read by the MATH thread inside the SFPU step, which runs after copy_tile has waited on the
    // unpacker, which waited on cb_r (MATH itself never waits on a CB, and has no CB interface).
    const uint32_t r_addr = get_arg_val<uint32_t>(0);  // rb shard (same L1 address on every prep core)
    cb_wait_front(cb_r, 1);
    cb_wait_front(cb_g, KT);
    cb_reserve_back(cb_y, KT);
    pack_reconfig_data_format(cb_y);
    pack_block_contiguous_init(cb_y);
    for (uint32_t t0 = 0; t0 < KT; t0 += GRP) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        for (uint32_t j = 0; j < GRP; ++j) {
            copy_tile(cb_x, t0 + j, j);
        }
        reconfig_data_format_srca(cb_g);
        copy_init(cb_g);
        for (uint32_t j = 0; j < GRP; ++j) {
            copy_tile(cb_g, t0 + j, 4 + j);
        }
        MATH((sfpu_section([r_addr] {
            const sfpi::vFloat rv = *reinterpret_cast<volatile float*>(r_addr);
            scale_one<0>(rv);
            scale_one<1>(rv);
            scale_one<2>(rv);
            scale_one<3>(rv);
        })));
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, cb_y, GRP);
        tile_regs_release();
    }
    cb_push_back(cb_y, KT);
    cb_pop_front(cb_x, KT);
    cb_pop_front(cb_g, KT);
    cb_pop_front(cb_r, 1);
    // 3. x_b' = y_b @ M_b for this unit's 4 output tiles
    reconfig_full_operand<SrcOrder::Reverse>(cb_y, cb_m);
    custom_mm_block_init<false, CustomMmDestLayout::Default, true>(cb_y, cb_m, cb_o);
    pack_block_contiguous_init(cb_o);
    cb_wait_front(cb_y, KT);
    for (uint32_t j = 0; j < 4; ++j) {
        cb_wait_front(cb_m, KT);
        tile_regs_acquire();
        custom_mm_block<CustomMmDestLayout::Default, false>(cb_y, cb_m, 0, 0, 0, KT);
        tile_regs_commit();
        cb_pop_front(cb_m, KT);
        tile_regs_wait();
        cb_reserve_back(cb_o, 1);
        pack_block_contiguous(0, cb_o, 1);
        cb_push_back(cb_o, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_y, KT);
#endif
}
