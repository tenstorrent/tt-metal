// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DeepSeek-V4.1-Flash paged PREFILL, SINGLE-TRACE variant (DSV41_PF_SINGLE=1, tt/prefill_single.py): translate the
// indexer's selected ids (slots of the position-indexed key slab with the current chunk in front) into rows of the
// paged KV pool. Same as prefill_id_translate.cpp except for the slot -> entry map:
//   slots [0, CCUR)       the CURRENT chunk's entries: slot s = entry cs + s, cs = off[0] (= s0 / ratio)
//   slots [CCUR, ...)     the history: slot s = entry s - CCUR
// (the FIFO variant: entry = slot - off).
//
//   ids   uint32 ROW_MAJOR [U*C, NID]   FIFO slots of the selected compressed entries (0xFFFFFFFF = none)
//   pt    int32  ROW_MAJOR [U, MAXP]    page table of the user held by each prefill slot (-1 = no page)
//   off   uint32 ROW_MAJOR [1, 32]      element 0: cs (first entry of the current chunk)
//   out   uint32 ROW_MAJOR [U*C, NID]   pool rows, 0xFFFFFFFF where the id is a sentinel / not a visible entry / its
//   page is unallocated
//
// Entry j of a source with ratio RATIO lives at pool row pt[u][(j*RATIO)/PAGE_TOK] * PAGE_ROWS + SRC_OFF + ((j*RATIO) %
// PAGE_TOK) / RATIO (the same mapping as paged_kv_step.cpp / PageLayout.phys_rows). One core handles ROWS_PER_CORE
// consecutive rows of one user.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t RPC = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t RATIO = get_compile_time_arg_val(2);
    constexpr uint32_t SRC_OFF = get_compile_time_arg_val(3);
    constexpr uint32_t MAXP = get_compile_time_arg_val(4);
    constexpr uint32_t NID = get_compile_time_arg_val(5);
    constexpr uint32_t PAGE_TOK = get_compile_time_arg_val(6);
    constexpr uint32_t PAGE_ROWS = get_compile_time_arg_val(7);
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t CCUR = get_compile_time_arg_val(9);
    constexpr auto ids_args = TensorAccessorArgs<10>();
    constexpr auto pt_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto off_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<off_args.next_compile_time_args_offset()>();

    constexpr uint32_t ROW_B = NID * 4;
    constexpr uint32_t PT_B = ((MAXP * 4 + 63) / 64) * 64;
    constexpr uint32_t B_PT = 0, B_OFF = B_PT + PT_B, B_IN = B_OFF + 64, B_OUT = B_IN + ROW_B;

    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t a_ids = get_common_arg_val<uint32_t>(0), a_pt = get_common_arg_val<uint32_t>(1),
                   a_off = get_common_arg_val<uint32_t>(2), a_out = get_common_arg_val<uint32_t>(3);

    Noc noc;
    const auto ids_acc = TensorAccessor(ids_args, a_ids, ROW_B);
    const auto pt_acc = TensorAccessor(pt_args, a_pt, MAXP * 4);
    const auto off_acc = TensorAccessor(off_args, a_off, 128);
    const auto out_acc = TensorAccessor(out_args, a_out, ROW_B);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr int32_t* ptb = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + B_PT);
    volatile tt_l1_ptr uint32_t* offb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_OFF);
    volatile tt_l1_ptr uint32_t* inb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_IN);
    volatile tt_l1_ptr uint32_t* outb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + B_OUT);

    const uint32_t row0 = core * RPC;
    const uint32_t user = row0 / C;
    noc.async_read(pt_acc, cb, MAXP * 4, {.page_id = user, .offset_bytes = 0}, {.offset_bytes = B_PT});
    noc.async_read(off_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = B_OFF});
    noc.async_read_barrier();
    const uint32_t off = offb[0];

    for (uint32_t r = 0; r < RPC; ++r) {
        noc.async_read(ids_acc, cb, ROW_B, {.page_id = row0 + r, .offset_bytes = 0}, {.offset_bytes = B_IN});
        noc.async_read_barrier();
        for (uint32_t i = 0; i < NID; ++i) {
            const uint32_t s = inb[i];
            uint32_t res = 0xFFFFFFFFu;
            if (s != 0xFFFFFFFFu) {
                const uint32_t j = s < CCUR ? off + s : s - CCUR;
                const uint32_t tok = j * RATIO;
                const uint32_t pg = tok / PAGE_TOK;
                if (pg < MAXP) {
                    const int32_t page = ptb[pg];
                    if (page >= 0) {
                        res = (uint32_t)page * PAGE_ROWS + SRC_OFF + (tok % PAGE_TOK) / RATIO;
                    }
                }
            }
            outb[i] = res;
        }
        noc.async_write(cb, out_acc, ROW_B, {.offset_bytes = B_OUT}, {.page_id = row0 + r, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
