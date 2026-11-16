// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 q / k / v head norm + RoPE, reader (NCRISC) (fused_decode/heads.py: HeadsB1). One head per core.
//
// The head's D values are packed into one tile: value d = 32 r + c at row r, column c (R = D / 32 <= 16 rows, faces
// 0 / 1; the other rows zero). Builds, from row 0 of the fused QKV projection output ([1, 1, 1, N] BF16 TILE, L1):
//   cb_x:   the head packed;
//   cb_xr:  (q / k) the head with rows rotated by R / 2 (row r = rows (r + R / 2) % R): NeoX rotate_half without its
//           sign, which rides in the rotated gamma;
//   cb_cs:  (q / k) cos and sin of the decode position packed (row 0 of the position-gathered [1, 1, 32, D] rows,
//           64 B aligned reads: the rows may live in DRAM);
//   cb_g:   (q / k) the packed gamma and the packed, rotated, sign-folded gamma (host-built, one DRAM page each);
//   cb_scaler: the reduce scaler (1.0 in row 0 of every face).
//
// from_table: cos / sin come straight from the row-major [context, D] RoPE tables: the decode position (element 0 of
// the [1, 32] UINT32 position tensor) selects one row (page) of each; packed row r = bytes [64 r, 64 r + 64).
//
// runtime args: [x_addr, first_tile, cos_addr, sin_addr, g_addr, g_page, pos_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);  // the head's first column tile in the QKV output
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_page = get_arg_val<uint32_t>(5);  // the tensor's (gamma, rotated gamma) pair: pages 2 g_page ..
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(2);
    constexpr uint32_t cb_g = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(5);
    constexpr uint32_t R = get_compile_time_arg_val(6);
    constexpr uint32_t rope = get_compile_time_arg_val(7);  // q / k
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t from_table = get_compile_time_arg_val(g_args.next_compile_time_args_offset());
    constexpr auto p_args = TensorAccessorArgs<g_args.next_compile_time_args_offset() + 1>();
    constexpr uint32_t page = 2048, face = 512;
    static_assert(R <= 16 && R % 2 == 0, "head_dim <= 512, a multiple of 64");

    const auto s_x = TensorAccessor(x_args, x_addr, page);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    auto zero = [&](uint32_t dst, uint32_t bytes) {
        for (uint32_t z = 0; z < bytes; z += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, dst + z, MEM_ZEROS_SIZE < bytes - z ? MEM_ZEROS_SIZE : bytes - z);
        }
    };
    // packed row r <- row 0 of QKV tile first + src_r
    auto gather = [&](uint32_t dst, bool rotate) {
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t src = first + (rotate ? (r + R / 2) % R : r);
            noc_async_read(s_x.get_noc_addr(src, 0), dst + 32 * r, 32);
            noc_async_read(s_x.get_noc_addr(src, face), dst + face + 32 * r, 32);
        }
    };

    // All reads in flight at once, one barrier. Only x needs zeros past its R rows (the norm sums the whole tile),
    // and those rows are disjoint from the gathered ones; every other operand's rows >= R only reach output rows the
    // writer never sends.
    cb_reserve_back(cb_x, 1);
    cb_reserve_back(cb_scaler, 1);
    const uint32_t xt = get_write_ptr(cb_x);
    zero(xt + 32 * R, face - 32 * R);
    zero(xt + face + 32 * R, face - 32 * R);
    zero(xt + 2 * face, 2 * face);
    zero(get_write_ptr(cb_scaler), page);
    gather(xt, false);
    if constexpr (rope) {
        cb_reserve_back(cb_xr, 1);
        cb_reserve_back(cb_cs, 2);
        cb_reserve_back(cb_g, 2);
        const auto s_g = TensorAccessor(g_args, g_addr, page);
        noc_async_read(s_g.get_noc_addr(2 * g_page), get_write_ptr(cb_g), page);
        noc_async_read(s_g.get_noc_addr(2 * g_page + 1), get_write_ptr(cb_g) + page, page);
        const uint32_t stage = get_write_ptr(cb_stage);
        if constexpr (from_table) {
            // position, then one row of each table (64 B aligned stage offsets)
            noc_async_read(TensorAccessor(p_args, get_arg_val<uint32_t>(6), 128).get_noc_addr(0), stage, 64);
            noc_async_read_barrier();
            const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0];
            noc_async_read(TensorAccessor(c_args, cos_addr, 64 * R).get_noc_addr(pos), stage + 64, 64 * R);
            noc_async_read(TensorAccessor(s_args, sin_addr, 64 * R).get_noc_addr(pos), stage + 64 + 64 * R, 64 * R);
        } else {
            // cos / sin rows: rows 0-1 of faces 0 / 1 (64 B, aligned) of each of the R source tiles into the stage
            const auto s_c = TensorAccessor(c_args, cos_addr, page);
            const auto s_s = TensorAccessor(s_args, sin_addr, page);
            for (uint32_t r = 0; r < R; ++r) {
                noc_async_read(s_c.get_noc_addr(r, 0), stage + 256 * r, 64);
                noc_async_read(s_c.get_noc_addr(r, face), stage + 256 * r + 64, 64);
                noc_async_read(s_s.get_noc_addr(r, 0), stage + 256 * r + 128, 64);
                noc_async_read(s_s.get_noc_addr(r, face), stage + 256 * r + 192, 64);
            }
        }
        gather(get_write_ptr(cb_xr), true);
        noc_async_read_barrier();
        const uint32_t cs = get_write_ptr(cb_cs);
        if constexpr (from_table) {
            for (uint32_t r = 0; r < R; ++r) {
                volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64 + 64 * r);
                volatile tt_l1_ptr uint32_t* sn =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64 + 64 * R + 64 * r);
                volatile tt_l1_ptr uint32_t* c0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + 32 * r);
                volatile tt_l1_ptr uint32_t* c1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + face + 32 * r);
                volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + page + 32 * r);
                volatile tt_l1_ptr uint32_t* s1 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + page + face + 32 * r);
                for (uint32_t w = 0; w < 8; ++w) {
                    c0[w] = c[w];
                    c1[w] = c[8 + w];
                    s0[w] = sn[w];
                    s1[w] = sn[8 + w];
                }
            }
        }
        for (uint32_t r = 0; r < (from_table ? 0 : R); ++r) {
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 256 * r);
            volatile tt_l1_ptr uint32_t* c0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + 32 * r);
            volatile tt_l1_ptr uint32_t* c1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + face + 32 * r);
            volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + page + 32 * r);
            volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + page + face + 32 * r);
            for (uint32_t w = 0; w < 8; ++w) {  // row 0 of each face = the first 32 B of each 64 B block
                c0[w] = src[w];
                c1[w] = src[16 + w];
                s0[w] = src[32 + w];
                s1[w] = src[48 + w];
            }
        }
    } else {
        noc_async_read_barrier();
    }
    {
        // reduce scaler (zeroed above): 1.0 in row 0 of every face
        volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t i = 0; i < 8; ++i) {
                sc[f * 128 + i] = 0x3F803F80;
            }
        }
    }
    cb_push_back(cb_scaler, 1);
    cb_push_back(cb_x, 1);
    if constexpr (rope) {
        cb_push_back(cb_xr, 1);
        cb_push_back(cb_cs, 2);
        cb_push_back(cb_g, 2);
    }
}
