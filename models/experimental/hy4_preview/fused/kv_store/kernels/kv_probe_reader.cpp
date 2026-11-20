// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Probe of kv_frames::locate on the device (data movement, one core): for every request (decode row, layer,
// stripe row) read one item from a cache tensor at the address the frame table gives and write it to page i of
// the output. An item is a packed latent row (ROW_STRIDE = ITEM_BYTES = 704) or the 32-row index-key block that
// holds the row (ROW_STRIDE 0, ITEM_BYTES 4352, written as 4 pages of 1088 bytes).
//
// The tables live in an L1 tensor behind cb_table: [kinds][decode rows][TABLE_FRAMES] uint32 frame bases; this
// kernel uses the tables at word offset `table_words`.
//
// Requests: uint32 pages of REQ_PER_PAGE entries of 4 words [decode row, layer, stripe row, 0].
//
// mode bit 0 (timing): no DRAM access for the items; the addresses are folded into a checksum written to output
//                      page 0, words [0] checksum, [1] count. Every request page is evaluated `repeat` times.
// mode bit 1 (timing): today's contiguous arithmetic instead of the table (sparse_mla/kernels/sm_common.hpp
//                      kv_row_bank / kv_row_offset at page (decode row * slot_stride + layer) * rows_per_slot +
//                      row): the same addresses as the identity table of a slot-major cache.
// mode bit 2 (timing): requests are read, nothing is computed (the floor of the two above).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "kv_frames.hpp"
#include "sm_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_table = get_named_compile_time_arg_val("cb_table");
    constexpr uint32_t cb_req = get_named_compile_time_arg_val("cb_req");
    constexpr uint32_t cb_data = get_named_compile_time_arg_val("cb_data");
    constexpr uint32_t ITEM_BYTES = get_named_compile_time_arg_val("item_bytes");
    constexpr uint32_t ROW_STRIDE = get_named_compile_time_arg_val("row_stride");
    constexpr uint32_t OUT_PAGE = get_named_compile_time_arg_val("out_page_bytes");
    constexpr uint32_t ITEM_PAGES = get_named_compile_time_arg_val("item_pages");
    constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch");
    constexpr uint32_t REQ_PER_PAGE = get_named_compile_time_arg_val("requests_per_page");
    constexpr uint32_t SLOT_BYTES = get_named_compile_time_arg_val("slot_bytes");  // per layer step
    constexpr uint32_t TABLE_WORDS = get_named_compile_time_arg_val("table_words");
    constexpr uint32_t MODE = get_named_compile_time_arg_val("mode");
    constexpr uint32_t REPEAT = get_named_compile_time_arg_val("repeat");
    constexpr uint32_t ROWS_PER_SLOT = get_named_compile_time_arg_val("rows_per_slot");
    constexpr uint32_t SLOT_STRIDE = get_named_compile_time_arg_val("slot_stride");
    constexpr bool no_reads = (MODE & 1) != 0;
    constexpr bool contiguous = (MODE & 2) != 0;
    constexpr bool floor_only = (MODE & 4) != 0;
    static_assert(ITEM_PAGES * OUT_PAGE >= ITEM_BYTES && ITEM_BYTES <= NOC_MAX_BURST_SIZE);

    constexpr auto req_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<req_args.next_compile_time_args_offset()>();

    const uint32_t pool_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t req_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t count = get_common_arg_val<uint32_t>(3);

    const auto req = TensorAccessor(req_args, req_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    volatile tt_l1_ptr uint32_t* tables =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_table)) + TABLE_WORDS;
    cb_reserve_back(cb_req, 1);
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_req));
    cb_reserve_back(cb_data, BATCH);
    const uint32_t data = get_write_ptr(cb_data);

    auto where = [&](uint32_t i) -> kv_frames::Loc {
        const uint32_t u = r[4 * i], layer = r[4 * i + 1], row = r[4 * i + 2];
        if constexpr (contiguous) {
            const uint32_t page = (u * SLOT_STRIDE + layer) * ROWS_PER_SLOT + row;
            return kv_frames::Loc{sm::kv_row_bank<8, 32>(page), sm::kv_row_offset<8, 32>(page, ROW_STRIDE)};
        } else {
            return kv_frames::locate(tables + u * kv_frames::TABLE_FRAMES, layer * SLOT_BYTES, row, ROW_STRIDE);
        }
    };

    uint32_t checksum = 0, evaluated = 0;
    for (uint32_t first = 0; first < count; first += REQ_PER_PAGE) {
        const uint32_t here = (count - first) < REQ_PER_PAGE ? (count - first) : REQ_PER_PAGE;
        noc_async_read(
            req.get_noc_addr(first / REQ_PER_PAGE), reinterpret_cast<uint32_t>(r), here * 4 * sizeof(uint32_t));
        noc_async_read_barrier();
        if constexpr (floor_only) {
            continue;
        }
        if constexpr (no_reads) {
            for (uint32_t rep = 0; rep < REPEAT; ++rep) {
                for (uint32_t i = 0; i < here; ++i) {
                    const kv_frames::Loc at = where(i);
                    checksum = ((checksum << 1) | (checksum >> 31)) + at.offset + at.bank;
                    ++evaluated;
                }
            }
            continue;
        }
        for (uint32_t b = 0; b < here; b += BATCH) {
            const uint32_t n = (here - b) < BATCH ? (here - b) : BATCH;
            for (uint32_t i = 0; i < n; ++i) {
                const kv_frames::Loc at = where(b + i);
                noc_async_read_one_packet(
                    get_noc_addr_from_bank_id<true>(at.bank, pool_addr + at.offset), data + i * ITEM_BYTES, ITEM_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t p = 0; p < ITEM_PAGES; ++p) {
                    noc_async_write(
                        data + i * ITEM_BYTES + p * OUT_PAGE,
                        out.get_noc_addr((first + b + i) * ITEM_PAGES + p),
                        OUT_PAGE);
                }
            }
            noc_async_write_barrier();
        }
    }
    if constexpr (no_reads) {
        volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data);
        words[0] = checksum;
        words[1] = evaluated;
        noc_async_write(data, out.get_noc_addr(0), 64);
        noc_async_write_barrier();
    }
}
