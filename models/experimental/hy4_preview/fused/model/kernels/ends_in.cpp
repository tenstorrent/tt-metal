// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// IN of the fused decode step (fused/model/ends_fused.py EndsIn; PLAN.md section 46): the 8 token ids of the step ->
// their embedding rows as the iHC site's F tensor, the site's streams zeroed and its coefficient page set to
// "post = 1", so that the leading post-mix of layer 0's P1 (streams += post x F) leaves the 4 streams equal to the
// embedding row (reference/ihc.py ihc_expand), bit for bit.
//
// One instance on each of the K site cores of the root chip. Core k:
//   1. reads X_PAGES zero pages of the constants tensor into its streams pages and the constants' last page into its
//      coefficient page (the site's tensors are width sharded: the same L1 address on every site core);
//   2. reads the 8 fed-back ids and the 8 forced ids (two 64-byte DRAM pages); a forced id with bit 31 set wins;
//   3. reads SLICE bytes of token row r = k mod 8 from the embedding table (one DRAM page per token, BF16, the
//      checkpoint's column order): the hidden columns owned by the HALF of the site cores this core is not in;
//   4. writes them as pieces of 32 bytes (16 values: one face-row half of an [8, 32] row tile) into the F shards of
//      those cores: the piece of tile t, half h goes to byte t * 512 + (8 h + r) * 32 of the owner's shard
//      (ihc_site/layout.py rows_to_pages). Rows of different tokens interleave at 32 bytes there, so a piece is the
//      largest unit that moves whole. Source and destination offsets are multiples of 32.
// Data movement only: NoC reads from DRAM into this core's L1, NoC writes into other site cores' L1. No circular
// buffer wait, no semaphore, no fabric, no loop without a bound: it cannot stall on a peer.
//
// Compile-time args: the table's, the id tensors' and the constants' TensorAccessorArgs, then
//   scratch CB, X_PAGES (streams pages per core), SLICE (bytes of the table row this core moves), PEERS (cores it
//   writes to), TILES (32-column tiles per F shard), ROW_BYTES (a table page), VOCAB (table pages: an id at or
//   above it reads token 0's row instead of an address outside the table).
// Runtime args (per core):
//   0 token row r   1 byte offset of the slice in the token's table row
//   2 table address 3 ids address 4 forced ids address 5 constants address
//   6 streams address 7 coefficient page address 8 F address
//   9 .. 9 + PEERS - 1: the peers' NoC x | y << 16, in the order of their hidden columns

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto table_args = TensorAccessorArgs<0>();
    constexpr auto ids_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto const_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = const_args.next_compile_time_args_offset();
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(base + 0);
    constexpr uint32_t x_pages = get_compile_time_arg_val(base + 1);
    constexpr uint32_t slice = get_compile_time_arg_val(base + 2);
    constexpr uint32_t peers = get_compile_time_arg_val(base + 3);
    constexpr uint32_t tiles = get_compile_time_arg_val(base + 4);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(base + 5);
    constexpr uint32_t vocab = get_compile_time_arg_val(base + 6);
    constexpr uint32_t PAGE = 4096;  // an fp32 page of the site
    constexpr uint32_t PIECE = 32;   // 16 BF16 values
    constexpr uint32_t IDS = 64;     // a page of 16 uint32
    static_assert(slice == peers * tiles * 2 * PIECE, "the slice is the peers' F columns");
    static_assert(slice % 64 == 0 && row_bytes % 64 == 0, "DRAM reads are whole 64-byte words");

    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t slice_offset = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t ids_addr = get_arg_val<uint32_t>(3);
    const uint32_t force_addr = get_arg_val<uint32_t>(4);
    const uint32_t const_addr = get_arg_val<uint32_t>(5);
    const uint32_t streams_addr = get_arg_val<uint32_t>(6);
    const uint32_t coef_addr = get_arg_val<uint32_t>(7);
    const uint32_t f_addr = get_arg_val<uint32_t>(8);

    // scratch: [ids 64 | forced ids 64 | the slice], on a 64-byte boundary (every DRAM read below has source,
    // destination and length on whole 64-byte words; the buffer is 64 bytes longer than its contents)
    const uint32_t scratch = (get_write_ptr(cb_scratch) + 63u) & ~63u;
    const uint32_t slice_l1 = scratch + 2 * IDS;

    const auto consts = TensorAccessor(const_args, const_addr, PAGE);
    for (uint32_t p = 0; p < x_pages; ++p) {
        noc_async_read(consts.get_noc_addr(p), streams_addr + p * PAGE, PAGE);
    }
    noc_async_read(consts.get_noc_addr(x_pages), coef_addr, PAGE);
    const auto ids = TensorAccessor(ids_args, ids_addr, IDS);
    const auto forced = TensorAccessor(ids_args, force_addr, IDS);
    noc_async_read(ids.get_noc_addr(0), scratch, IDS);
    noc_async_read(forced.get_noc_addr(0), scratch + IDS, IDS);
    noc_async_read_barrier();
    invalidate_l1_cache();

    const volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t fed = words[row];
    const uint32_t force = words[16 + row];
    uint32_t token = (force >> 31) ? (force & 0x7FFFFFFFu) : fed;
    if (token >= vocab) {
        token = 0;
    }

    const auto table = TensorAccessor(table_args, table_addr, row_bytes);
    noc_async_read(table.get_noc_addr(token, slice_offset), slice_l1, slice);
    noc_async_read_barrier();

    uint32_t src = slice_l1;
    for (uint32_t d = 0; d < peers; ++d) {
        const uint32_t xy = get_arg_val<uint32_t>(9 + d);
        const uint32_t px = xy & 0xFFFF;
        const uint32_t py = xy >> 16;
        for (uint32_t t = 0; t < tiles; ++t) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t dst = f_addr + t * 512 + (8 * h + row) * PIECE;
                noc_async_write_one_packet(src, get_noc_addr(px, py, dst), PIECE);
                src += PIECE;
            }
        }
    }
    noc_async_write_barrier();
}
