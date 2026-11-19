// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down-projection weight streamer (NCRISC, NOC0) of the per-expert dispatch op in its EXPERT-PARALLEL split
// (custom_ops/expert_stream/op.py, ExpertDownDispatch(down_split="experts"); decode roofline lever E3). The work
// unit is (expert e, column group q): the bank's `unit_cols` output columns [q unit_cols, (q + 1) unit_cols) of
// W2[e] (unit_cols x seg_tiles tiles, one contiguous block of the stacked layout: 8 x 8 = 64 tiles = 70 KB at the
// 8x4 shapes, the block op's streaming unit), LPT-assigned over the bank's `readers` cores by pass count
// (dispatch_common.hpp for_each_unit, replayed by every kernel: no unit list in memory). Each unit's block is
// streamed once and kept by the compute over the expert's passes, two or three unit slots with one in flight.
// The active set is the writer's: the per-expert counts go through the same pass budget (clamp_passes), so an
// expert the writer truncated away is not streamed either.
//
// runtime args: [w_addr, counts_addr, bank_id, vc, reader (this core's index within the bank)]
// REGIONS_SRC (regions_src.hpp): the regions lists' per-expert down tensors (common runtime args down_addr[0..E));
// the unit's columns of K tile i are one packet of unit_cols tiles at (i * C + q * unit_cols) * tile_bytes,
// seg_tiles packets per unit, k-major in the CB (compute_down_dispatch_experts.cpp REGIONS_SRC).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatch_common.hpp"
#include "regions_src.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t reader = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_c = get_compile_time_arg_val(1);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t unit_cols = get_compile_time_arg_val(4);
    constexpr uint32_t bank_expert_stride = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t T = get_compile_time_arg_val(8);
    constexpr uint32_t readers = get_compile_time_arg_val(9);       // cores per bank = the LPT groups
    constexpr uint32_t UPE = get_compile_time_arg_val(10);          // units per expert = C / unit_cols
    constexpr uint32_t num_buffers = get_compile_time_arg_val(11);  // unit block slots (2 or 3)
    constexpr auto counts_args = TensorAccessorArgs<12>();

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t unit_tiles = unit_cols * seg_tiles;
    constexpr uint32_t unit_bytes = unit_cols * seg_bytes;
    constexpr uint32_t pages_per_unit = unit_bytes / page_bytes;
    static_assert(pages_per_unit * page_bytes == unit_bytes, "the packet must divide the unit block");
    static_assert(page_bytes <= NOC_MAX_BURST_SIZE, "one packet must fit one NOC burst");
    static_assert(readers <= 32, "for_each_unit's group bound");

    cb_reserve_back(cb_c, E_TILES);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    const auto s_c = TensorAccessor(counts_args, counts_addr, 2048);
    for (uint32_t i = 0; i < E_TILES; ++i) {
        noc_async_read(s_c.get_noc_addr(i), c_l1 + i * 2048, 2048);
    }
    uint16_t n_tok[E];
    noc_async_read_barrier();
    read_counts<E>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), n_tok);
    uint16_t passes[E], base[E];
    pass_bases<E>(n_tok, passes, base);
    bool clamped = false;  // the pass budget, the same truncation in every kernel of both ops (dispatch_common.hpp)
    clamp_passes<E>(n_tok, passes, base, dispatch_max_passes(T, E), clamped);

#ifdef REGIONS_SRC
    constexpr uint32_t C = bank_expert_stride / seg_bytes;  // output columns per bank
    constexpr uint32_t src_step = C * tile_bytes;           // the next K tile of the unit's columns
    static_assert(page_bytes == unit_cols * tile_bytes, "regions-sourced unit packets are one K tile of the unit");
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
    (void)w_addr;  // the per-expert addresses come from the common runtime args
#else
    constexpr uint32_t src_step = page_bytes;
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
#endif
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    static_assert(num_buffers == 2 || num_buffers == 3, "two or three unit block slots");
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, unit_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * unit_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t pushed = 0;
    uint32_t issued = 0;

    // this core's units in LPT order (the compute consumes them in the same order, the writer feeds them so)
    for_each_unit<E, UPE, 32>(passes, readers, reader, [&](uint32_t e, uint32_t q) {
        noc_async_read_set_trid(trid);
#ifdef REGIONS_SRC
        uint32_t src = get_common_arg_val<uint32_t>(e) + q * unit_cols * tile_bytes;
#else
        uint32_t src = e * bank_expert_stride + q * unit_bytes;
#endif
        for (uint32_t c = 0; c < pages_per_unit; ++c) {
            noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
            src += src_step;
            l1_write += page_bytes;
        }
        ++issued;
        if (free_blocks == num_buffers - extra_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, unit_tiles);
            ++pushed;
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
            cb_reserve_back(cb_w, unit_tiles * (extra_in_flight + 1));
        } else {
            free_blocks -= 1;
        }
        trid = trid == num_buffers ? 1 : trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    });
    while (pushed < issued) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, unit_tiles);
        ++pushed;
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
    noc_async_full_barrier();  // exit contract: every NOC read / write / atomic acknowledged before the next kernel
}
