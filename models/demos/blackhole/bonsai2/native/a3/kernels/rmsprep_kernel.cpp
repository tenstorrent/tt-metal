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
//   TRISC  y = x * g * rstd in fp32 (SFPU), then x_b' = y_b @ M_b (custom_mm).
// The block is handled as ONE 32x32 tile (row t = K-tile t's 32 values): NCRISC builds the x tile,
// g comes as one fp32 tile per block, TRISC does whole-tile SFPU steps and packs one bf16 tile, and
// BRISC re-lays its 32 rows into the 1x32 tiles custom_mm takes (32 tiny-tile copies cost 25 us).
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
#include "api/compute/reduce.h"
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
constexpr uint32_t cb_y32 = 13;  // y as one bf16 32x32 tile (cb_x holds x the same way, cb_g holds g)
constexpr uint32_t cb_yo = 14;   // emit: the normalized y = y' * rstd as one bf16 32x32 tile
constexpr uint32_t cb_ystage = 15;  // emit: a zeroed 2 KB tile staging one row-0 output page
// rp_emit: also write the normalized row (for consumers that need it besides the ternary site, e.g. the
// GDN a|b matmul) into y_out [.., 32, K] 32x32 tiles, row 0 live, zeros elsewhere; unit r writes the
// block's K-tiles 4r..4r+3.
constexpr bool emit = get_named_compile_time_arg_val("rp_emit") == 1;
// cb_sync (4 KB, same address on every core of the program): partial[b] at +16 b, rstd at +1024,
// NoC staging words at +2048 / +2064 (16 B strides: NoC writes keep the 16 B phase)
constexpr uint32_t cb_sync = 12;
constexpr uint32_t KT = 32;  // K-tiles per 1024-block

#if defined(COMPILE_FOR_TRISC) && defined(TRISC_MATH)
constexpr uint32_t U = 32;
inline void square_slot0() {
#pragma GCC unroll 32
    for (uint32_t i = 0; i < U; ++i) {
        sfpi::vFloat v = sfpi::dst_reg[i];
        sfpi::dst_reg[i] = v * v;
    }
}
inline void xg_and_sq() {  // slot 2 = slot 0 ^ 2 (for the partial), slot 0 = slot 0 * slot 1 (y' = x * g)
#pragma GCC unroll 32
    for (uint32_t i = 0; i < U; ++i) {
        sfpi::vFloat v = sfpi::dst_reg[i];
        sfpi::dst_reg[2 * U + i] = v * v;
        sfpi::dst_reg[i] = v * sfpi::dst_reg[U + i];
    }
}
inline void scale_slot5(sfpi::vFloat r) {  // slot 5 (y') *= rstd, whole tile
#pragma GCC unroll 32
    for (uint32_t i = 0; i < U; ++i) {
        sfpi::dst_reg[5 * U + i] = sfpi::dst_reg[5 * U + i] * r;
    }
}
inline void scale_rows() {  // row 0 of slots 0-3 *= rstd (row 0 of slot 4)
#pragma GCC unroll 4
    for (uint32_t i : {0u, 1u, 8u, 9u}) {
        sfpi::vFloat r = sfpi::dst_reg[4 * U + i];
#pragma GCC unroll 4
        for (uint32_t j = 0; j < 4; ++j) {
            sfpi::dst_reg[j * U + i] = sfpi::dst_reg[j * U + i] * r;
        }
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
    const InterleavedAddrGen<true> gg{.bank_base_address = g_addr, .page_size = 4096};
    // x row 0 of each K-tile: 64 B-aligned reads (rows 0-1 of face 0 at +0, of face 1 at +512) into scratch
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    cb_reserve_back(cb_x, 1);
    cb_reserve_back(cb_g, 1);
    noc_async_read(get_noc_addr(b, gg, 0), get_write_ptr(cb_g), 4096);  // g for block b: one fp32 tile
    // tile row i = K-tile i's row 0: cols 0-15 at face (i/16)*2 row i%16, cols 16-31 in the next face.
    // NoC copies, not RISC word loops (512 volatile L1 words per pass cost several us).
    const uint32_t xt = get_write_ptr(cb_x);
    if constexpr (!x_is_dram) {
        for (uint32_t i = 0; i < KT; ++i) {
            const uint32_t page = b * KT + i, f0 = xt + (i / 16) * 1024 + (i % 16) * 32;
            noc_async_read(get_noc_addr(page, xg, 0), f0, 32);
            noc_async_read(get_noc_addr(page, xg, 512), f0 + 512, 32);
        }
        noc_async_read_barrier();
    } else {  // DRAM: 64 B-aligned reads into scratch first (a 32 B read into another 64 B phase comes back wrong)
        for (uint32_t i = 0; i < KT; ++i) {
            const uint32_t page = b * KT + i;
            noc_async_read(get_noc_addr(page, xg, 0), scr + i * 128, 64);
            noc_async_read(get_noc_addr(page, xg, 512), scr + i * 128 + 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < KT; ++i) {
            const uint32_t f0 = xt + (i / 16) * 1024 + (i % 16) * 32;
            noc_async_read(get_noc_addr(scr + i * 128), f0, 32);
            noc_async_read(get_noc_addr(scr + i * 128 + 64), f0 + 512, 32);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_x, 1);
    cb_push_back(cb_g, 1);
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
    const float part = p[0];  // TRISC reduced the block to one value
    cb_pop_front(cb_p, 1);
    volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
    sl[2048 / 4] = as_u(part);  // staging word for the NoC write
    // y (one 32x32 bf16 tile) -> 32 1x32 tiles for custom_mm: tile t = [row t cols 0-15][row t cols 16-31]
    cb_wait_front(cb_y32, 1);
    cb_reserve_back(cb_y, KT);
    {
        const uint32_t ys = get_read_ptr(cb_y32), yd = get_write_ptr(cb_y);
        for (uint32_t t = 0; t < KT; ++t) {
            const uint32_t f0 = ys + (t / 16) * 1024 + (t % 16) * 32;
            noc_async_read(get_noc_addr(f0), yd + t * 64, 32);
            noc_async_read(get_noc_addr(f0 + 512), yd + t * 64 + 32, 32);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_y, KT);
    if constexpr (!emit) {
        cb_pop_front(cb_y32, 1);  // when emitting, TRISC copies it again and pops it
    }
    constexpr uint32_t dbg = get_named_compile_time_arg_val("rp_debug");  // 1: no rstd exchange (rstd = 1/sqrt(partial*nb/K))
    if constexpr (dbg & 1) {
        sl[1024 / 4] = as_u(rsqrt_newton(part * inv_k * nblocks + eps));
    } else {
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
    }
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
    if constexpr (emit) {
        // y rows of this unit's K-tiles -> y_out pages as full 32x32 tiles: row 0 = y, zeros elsewhere
        const InterleavedAddrGen<false> yg{.bank_base_address = get_arg_val<uint32_t>(11 + nw + (is_master ? ncores : 0)),
                                           .page_size = 2048};
        cb_reserve_back(cb_ystage, 1);
        const uint32_t ys = get_write_ptr(cb_ystage);
        volatile tt_l1_ptr uint32_t* yz = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ys);
        for (uint32_t i = 0; i < 512; ++i) {
            yz[i] = 0;
        }
        cb_wait_front(cb_yo, 1);
        const uint32_t yt = get_read_ptr(cb_yo);
        for (uint32_t t = 4 * r; t < 4 * r + 4; ++t) {
            const uint32_t f0 = yt + (t / 16) * 1024 + (t % 16) * 32;  // row t of the 32x32 y tile
            noc_async_read(get_noc_addr(f0), ys, 32);
            noc_async_read(get_noc_addr(f0 + 512), ys + 512, 32);
            noc_async_read_barrier();
            noc_async_write(ys, get_noc_addr(b * KT + t, yg), 2048);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_yo, 1);
    }
#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    // 1. one acquire: x -> slot 0, g -> slot 1; slot 2 = x^2 column-reduced (the partial, -> cb_p),
    //    slot 0 = y' = x * g (-> cb_y32 as one bf16 tile; BRISC re-lays it for custom_mm). rstd is
    //    applied to the output rows at the end: H(D (x g rstd)) = rstd H(D (x g)), so the exchange
    //    overlaps the matmul instead of sitting in front of it.
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_g, 1);
    tile_regs_acquire();
    copy_init(cb_x);
    copy_tile(cb_x, 0, 0);
    reconfig_data_format_srca(cb_x, cb_g);
    copy_init(cb_g);
    copy_tile(cb_g, 0, 1);
    constexpr uint32_t tdbg = get_named_compile_time_arg_val("rp_tdebug");  // 2: no reduce, 4: no x*g / x^2
    if constexpr (!(tdbg & 4)) {
        MATH((sfpu_section([] { xg_and_sq(); })));
    }
    if constexpr (!(tdbg & 2)) {
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(2);
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2);  // scalar at [0, 0]
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_p, 1);
    pack_reconfig_data_format(cb_p);
    pack_block_contiguous_init(cb_p);
    pack_block_contiguous(2, cb_p, 1);
    cb_push_back(cb_p, 1);
    cb_reserve_back(cb_y32, 1);
    pack_reconfig_data_format(cb_y32);
    pack_block_contiguous_init(cb_y32);
    pack_block_contiguous(0, cb_y32, 1);
    cb_push_back(cb_y32, 1);
    tile_regs_release();
    cb_pop_front(cb_x, 1);
    cb_pop_front(cb_g, 1);
    // 2. x_b' = y'_b @ M_b for this unit's 4 output tiles, kept in DST slots 0-3 (full sync); then
    //    rstd (cb_r, a 1x32 fp32 row) is copied into slot 4 -- the copy makes MATH wait for it -- and
    //    one SFPU step scales the 4 output rows; one pack writes all 4.
    reconfig_full_operand<SrcOrder::Reverse>(cb_y, cb_m);
    custom_mm_block_init<false, CustomMmDestLayout::Default, true>(cb_y, cb_m, cb_o);
    cb_wait_front(cb_y, KT);
    tile_regs_acquire();
    for (uint32_t j = 0; j < 4; ++j) {
        cb_wait_front(cb_m, KT);
        custom_mm_block<CustomMmDestLayout::Default, false>(cb_y, cb_m, 0, 0, j, KT);
        cb_pop_front(cb_m, KT);
    }
    cb_wait_front(cb_r, 1);
    reconfig_data_format_srca(cb_m, cb_r);
    copy_init(cb_r);
    copy_tile(cb_r, 0, 4);
    MATH((sfpu_section([] { scale_rows(); })));
    if constexpr (emit) {
        const uint32_t r_addr = get_arg_val<uint32_t>(0);  // rb shard: rstd as a scalar for a whole-tile scale
        cb_wait_front(cb_y32, 1);
        reconfig_data_format_srca(cb_r, cb_y32);
        copy_init(cb_y32);
        copy_tile(cb_y32, 0, 5);
        MATH((sfpu_section([r_addr] { scale_slot5(sfpi::vFloat(*reinterpret_cast<volatile float*>(r_addr))); })));
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_o, 4);
    pack_reconfig_data_format(cb_o);
    pack_block_contiguous_init(cb_o);
    pack_block_contiguous(0, cb_o, 4);
    cb_push_back(cb_o, 4);
    if constexpr (emit) {
        cb_reserve_back(cb_yo, 1);
        pack_reconfig_data_format(cb_yo);
        pack_block_contiguous_init(cb_yo);
        pack_block_contiguous(5, cb_yo, 1);
        cb_push_back(cb_yo, 1);
        cb_pop_front(cb_y32, 1);
    }
    tile_regs_release();
    cb_pop_front(cb_y, KT);
    cb_pop_front(cb_r, 1);
#endif
}
