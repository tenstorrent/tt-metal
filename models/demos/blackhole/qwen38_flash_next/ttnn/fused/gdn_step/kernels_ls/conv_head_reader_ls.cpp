// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader of a conv core that also takes head items (QWEN38_GDN_LS_CONVHEAD=1, option B of plan_nov05/g1_gdnlane): the
// conv phase of conv_reader_ls.cpp (its code, below, in namespace convphase), then the head reader of reader_ls.cpp
// (included as it is, its kernel_main renamed).  Runtime args: 0-9 the head reader's (reader_ls.cpp), 10-19 the conv
// reader's (conv_reader_ls.cpp: addresses of projected, 3 slots, 4 taps, newest, then T).  Compile-time args: the head
// reader's (reader_ls.cpp: 0-7, then the TensorAccessorArgs of projected, dtna, norm, state, scalers, masks), then the
// conv reader's accessors (projected, slot0-2, tap0-3, newest; ROWS is arg 0 for both).  The conv phase reads only DRAM
// and writes its CB pages and the DRAM ring slot; it never waits for another core.  CB 0 / 1 / 2 are the conv inputs
// (the head kernels do not use them).
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

#define kernel_main ls_head_reader_main
#include "reader_ls.cpp"
#undef kernel_main

namespace convphase {
constexpr auto h_p = TensorAccessorArgs<8>();
constexpr auto h_dtna = TensorAccessorArgs<h_p.next_compile_time_args_offset()>();
constexpr auto h_w = TensorAccessorArgs<h_dtna.next_compile_time_args_offset()>();
constexpr auto h_state = TensorAccessorArgs<h_w.next_compile_time_args_offset()>();
constexpr auto h_scalers = TensorAccessorArgs<h_state.next_compile_time_args_offset()>();
constexpr auto h_masks = TensorAccessorArgs<h_scalers.next_compile_time_args_offset()>();
constexpr uint32_t CONV_CTA = h_masks.next_compile_time_args_offset();
constexpr uint32_t CONV_ARG0 = 10;
}  // namespace convphase

namespace convphase {
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
}  // namespace convphase

namespace convphase {
// conv phase of a conv core that takes head items afterwards; its runtime args start at ARG0
FORCE_INLINE void run(uint32_t arg) {
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

    constexpr auto p_args = TensorAccessorArgs<CONV_CTA>();
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
}  // namespace convphase

void kernel_main() {
    convphase::run(convphase::CONV_ARG0);  // the whole conv phase (reads, push, newest-slot write, barriers) first
    ls_head_reader_main();
}
