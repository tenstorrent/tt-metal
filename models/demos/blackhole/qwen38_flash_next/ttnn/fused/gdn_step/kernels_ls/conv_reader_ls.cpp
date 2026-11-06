// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Conv core of the lane-split GDN step (QWEN38_GDN_LANE_SPLIT, rows 8 / 16 / 32): conv_reader.cpp with row-limited
// reads (design F1 of plan_nov05/g1_gdnlane/PLAN.md section 3).  The projection tile P[T] and the three older ring
// slots' tile T are read for the live rows 0 .. ROWS-1 only; the dead rows of those four CB pages are filled with zero
// from the core's zero region (the live and the zero reads cover disjoint byte ranges: one batch, one barrier), so a
// stale or non-finite pad row never reaches the conv, the l2 norms or the state.  The taps are read whole (the conv
// uses their row 0 only).  The newest ring slot's tile T is written for the live rows only, like the default ROWREAD
// path. bf16 TILE page layout: four 16 x 16 faces of 512 B (0 = rows 0-15 left, 1 = rows 0-15 right, 2 / 3 = rows
// 16-31), 32 B per face row.  Compile-time args: 0 ROWS; then TensorAccessorArgs of projected, slot0, slot1, slot2,
// tap0..tap3, newest.  Runtime args: 0 projected, 1-3 slots, 4-7 taps, 8 newest addresses, 9 T.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_P = 0, CB_S = 1, CB_T = 2;
constexpr uint32_t BF16_TILE = 2048, FACE_BYTES = 512, ROW_BYTES = 32, FACE_ROWS = 16;
constexpr uint32_t ROWS = get_compile_time_arg_val(0);
constexpr uint32_t TOP_ROWS = ROWS < FACE_ROWS ? ROWS : FACE_ROWS;         // live rows of faces 0 and 1
constexpr uint32_t BOTTOM_ROWS = ROWS > FACE_ROWS ? ROWS - FACE_ROWS : 0;  // live rows of faces 2 and 3
constexpr uint32_t TILE_ROWS = 32;

constexpr uint32_t live_rows_of_face(uint32_t face) { return face < 2 ? TOP_ROWS : BOTTOM_ROWS; }

// the live byte ranges of one bf16 page, from the page to the same offsets of the L1 page
template <typename Acc>
FORCE_INLINE void read_live(const Acc& acc, uint32_t page, uint32_t l1_page) {
    if constexpr (ROWS == TILE_ROWS) {
        noc_async_read(acc.get_noc_addr(page, 0), l1_page, BF16_TILE);
    } else {
        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t n = live_rows_of_face(face) * ROW_BYTES;
            if (n != 0) {
                noc_async_read(acc.get_noc_addr(page, face * FACE_BYTES), l1_page + face * FACE_BYTES, n);
            }
        }
    }
}

template <typename Acc>
FORCE_INLINE void write_live(const Acc& acc, uint32_t page, uint32_t l1_page) {
    if constexpr (ROWS == TILE_ROWS) {
        noc_async_write(l1_page, acc.get_noc_addr(page, 0), BF16_TILE);
    } else {
        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t n = live_rows_of_face(face) * ROW_BYTES;
            if (n != 0) {
                noc_async_write(l1_page + face * FACE_BYTES, acc.get_noc_addr(page, face * FACE_BYTES), n);
            }
        }
    }
}

// zero the dead rows of one L1 bf16 page from MEM_ZEROS (at most 512 B per read, one read per face); the caller waits
// on a read barrier
FORCE_INLINE void zero_dead(uint32_t l1_page) {
    if constexpr (ROWS != TILE_ROWS) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t live = live_rows_of_face(face) * ROW_BYTES;
            if (live != FACE_BYTES) {
                noc_async_read(zeros, l1_page + face * FACE_BYTES + live, FACE_BYTES - live);
            }
        }
    }
}
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t s2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t newest_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tile = get_arg_val<uint32_t>(arg++);

    constexpr auto p_args = TensorAccessorArgs<1>();
    constexpr auto s0_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    constexpr auto t0_args = TensorAccessorArgs<s2_args.next_compile_time_args_offset()>();
    constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
    constexpr auto t2_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
    constexpr auto t3_args = TensorAccessorArgs<t2_args.next_compile_time_args_offset()>();
    constexpr auto newest_args = TensorAccessorArgs<t3_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(p_args, p_addr);
    const auto s0 = TensorAccessor(s0_args, s0_addr);
    const auto s1 = TensorAccessor(s1_args, s1_addr);
    const auto s2 = TensorAccessor(s2_args, s2_addr);
    const auto t0 = TensorAccessor(t0_args, t0_addr);
    const auto t1 = TensorAccessor(t1_args, t1_addr);
    const auto t2 = TensorAccessor(t2_args, t2_addr);
    const auto t3 = TensorAccessor(t3_args, t3_addr);
    const auto newest = TensorAccessor(newest_args, newest_addr);

    FUSED_ZONE("fz_gls_cr");
    cb_reserve_back(CB_P, 1);
    cb_reserve_back(CB_S, 3);
    cb_reserve_back(CB_T, 4);
    const uint32_t p_l1 = get_write_ptr(CB_P);
    const uint32_t s_l1 = get_write_ptr(CB_S);
    const uint32_t t_l1 = get_write_ptr(CB_T);
    read_live(p, tile, p_l1);
    read_live(s0, tile, s_l1);
    read_live(s1, tile, s_l1 + BF16_TILE);
    read_live(s2, tile, s_l1 + 2 * BF16_TILE);
    zero_dead(p_l1);
    zero_dead(s_l1);
    zero_dead(s_l1 + BF16_TILE);
    zero_dead(s_l1 + 2 * BF16_TILE);
    noc_async_read_page(tile, t0, t_l1);
    noc_async_read_page(tile, t1, t_l1 + BF16_TILE);
    noc_async_read_page(tile, t2, t_l1 + 2 * BF16_TILE);
    noc_async_read_page(tile, t3, t_l1 + 3 * BF16_TILE);
    noc_async_read_barrier();
    cb_push_back(CB_P, 1);
    cb_push_back(CB_S, 3);
    cb_push_back(CB_T, 4);
    write_live(
        newest,
        tile,
        p_l1);  // the newest ring slot's tile T, live rows only (the dead rows of the slot stay as they were)
    noc_async_write_barrier();
}
