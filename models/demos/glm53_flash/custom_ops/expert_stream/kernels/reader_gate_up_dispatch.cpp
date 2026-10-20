// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert gate|up weight streamer (NCRISC, NOC0) of the per-expert dispatch op
// (custom_ops/expert_stream/op.py, ExpertGateUpDispatch). Same weight layout and column pipeline as
// reader_gate_up.cpp; the experts of this core come from the LPT assignment over the per-expert token counts
// (dispatch_common.hpp). With one column pair per bank an expert's two columns are streamed once however many
// 32-row passes it takes (the compute keeps them resident); with more pairs they are streamed once per pass.
//
// runtime args: [w_addr, counts_addr, bank_id, vc, group]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatch_common.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t group = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_c = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t groups = get_compile_time_arg_val(4);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t ppb = get_compile_time_arg_val(8);  // column pairs per bank
    constexpr auto counts_args = TensorAccessorArgs<9>();

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t col_bytes = kt * tile_bytes;
    constexpr uint32_t pages_per_col = col_bytes / page_bytes;
    static_assert(pages_per_col * page_bytes == col_bytes, "page size must divide the column");

    cb_reserve_back(cb_c, E_TILES);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    const auto s_c = TensorAccessor(counts_args, counts_addr, 2048);
    for (uint32_t i = 0; i < E_TILES; ++i) {
        noc_async_read(s_c.get_noc_addr(i), c_l1 + i * 2048, 2048);
    }
    noc_async_read_barrier();
    uint16_t n_tok[E], passes[E], base[E], mine[E];
    read_counts<E>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), n_tok);
    pass_bases<E>(n_tok, passes, base);
    const uint32_t n_mine = assign_group<E, 32>(passes, groups, group, mine);

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    // (gate, up) column pairs are the streaming unit (contiguous in DRAM): two pair slots, one pair in flight
    // ahead of the compute, which indexes gate tiles 0..kt-1 and up tiles kt..2kt-1 of the front pair (a pair
    // never straddles the CB end, unlike single-column slots).
    constexpr uint32_t pair_bytes = 2 * col_bytes;
    constexpr uint32_t pages_per_pair = pair_bytes / page_bytes;
    constexpr uint32_t num_buffers = 2;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, 2 * kt * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * pair_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t pushed = 0;
    uint32_t issued = 0;

    for (uint32_t i = 0; i < n_mine; ++i) {
        // one pair per bank: the pair stays resident in the compute over the expert's passes (streamed once);
        // more pairs: the ppb pairs are streamed again for every pass
        const uint32_t repeats = ppb == 1 ? 1 : passes[mine[i]];
        for (uint32_t rep = 0; rep < repeats; ++rep) {
            uint32_t src = static_cast<uint32_t>(mine[i]) * expert_stride;
            for (uint32_t q = 0; q < ppb; ++q) {
                noc_async_read_set_trid(trid);
                for (uint32_t p = 0; p < pages_per_pair; ++p) {
                    noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                    src += page_bytes;
                    l1_write += page_bytes;
                }
                ++issued;
                if (free_blocks == num_buffers - extra_in_flight) {
                    noc_async_read_barrier_with_trid(trid_to_wait);
                    cb_push_back(cb_w, 2 * kt);
                    ++pushed;
                    trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                    cb_reserve_back(cb_w, 2 * kt * (extra_in_flight + 1));
                } else {
                    free_blocks -= 1;
                }
                trid = trid == num_buffers ? 1 : trid + 1;
                if (l1_write >= cb_end) {
                    l1_write = cb_base;
                }
            }
        }
    }
    while (pushed < issued) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, 2 * kt);
        ++pushed;
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
}
