// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Weight streamer of one projection unit of the Hy4 decode projection programs (NCRISC, NOC0)
// (fused/projections/program.py). Derived from fused/moe/kernels/stream_reader.cpp (itself from
// models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_union_reader.cpp), reduced to what a projection needs:
// the unit's weights are one fixed run of bytes in its core's DRAM bank, so the stream waits for nobody. It starts
// with the kernel, before the hub has released anything, and is bounded only by the ring it fills.
//
// The run is n_blocks blocks of block_bytes (n_pk equal NOC packets per block) read into a ring of ring_blocks
// blocks, extra_in_flight + 1 blocks outstanding, one transaction id per block, on this core's response virtual
// channel.
//
// A unit that is never released does not take its weights: the wait for room in the ring is bounded
// (8 x HY4_SPIN_LIMIT polls), after which the stream ends once its outstanding reads have landed.
//
// runtime args: [w_addr, bank_id, vc, core_offset_bytes]
// compile-time args: [cb_w, n_blocks, block_tiles, block_bytes, n_pk, ring_blocks, extra_in_flight]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "proj_common.hpp"

namespace {
// Room for `tiles` more pages at the back of cb, polled so that a core nobody releases still ends. The bound is
// several times the data-movement kernel's own (its wait for the release, then for its group's tiles): the stream
// must outlast a unit that starts late.
FORCE_INLINE bool room(uint32_t cb, uint32_t tiles) {
    for (uint32_t round = 0; round < 8; ++round) {
        for (uint32_t spin = 0; spin < HY4_SPIN_LIMIT; ++spin) {
            if (cb_pages_reservable_at_back(cb, tiles)) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace

void kernel_main() {
#ifdef HY4_EMPTY  // measurement only: the launch floor of the program
    return;
#endif
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t core_offset = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t n_pk = get_compile_time_arg_val(4);
    constexpr uint32_t ring_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t extra_in_flight = get_compile_time_arg_val(6);

    constexpr uint32_t pkt = block_bytes / n_pk;
    static_assert(pkt * n_pk == block_bytes && pkt <= NOC_MAX_BURST_SIZE, "a block is n_pk equal packets");
    static_assert(ring_blocks >= extra_in_flight + 2 && ring_blocks <= 15, "ring size (one transaction id per block)");

    const uint32_t cb_base = get_write_ptr(cb_w);  // the ring's first block (nothing reserved yet)
    const uint32_t cb_end = cb_base + ring_blocks * block_bytes;
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, pkt, vc);

    uint32_t src = core_offset;
    uint32_t l1_write = cb_base;
    uint32_t trid = 1;          // of the next block to issue
    uint32_t trid_to_wait = 1;  // of the oldest block in flight
    uint32_t in_flight = 0;
    bool stopped = false;
    for (uint32_t blk = 0; blk < n_blocks && !stopped; ++blk) {
        // room for this block behind the ones in flight (they are pushed in order)
        if (!room(cb_w, block_tiles * (in_flight + 1))) {
            stopped = true;
            break;
        }
#ifndef HY4_NO_READ
        noc_async_read_set_trid(trid);
        for (uint32_t p = 0; p < n_pk; ++p) {
            noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
            src += pkt;
            l1_write += pkt;
        }
#else  // measurement only: hand compute whatever the ring holds
        l1_write += block_bytes;
#endif
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
        trid = trid == ring_blocks ? 1 : trid + 1;
        ++in_flight;
        if (in_flight == extra_in_flight + 1) {
#ifndef HY4_NO_READ
            noc_async_read_barrier_with_trid(trid_to_wait);
#endif
            cb_push_back(cb_w, block_tiles);
            trid_to_wait = trid_to_wait == ring_blocks ? 1 : trid_to_wait + 1;
            --in_flight;
        }
    }
    // blocks issued but not yet pushed
    while (in_flight > 0) {
#ifndef HY4_NO_READ
        noc_async_read_barrier_with_trid(trid_to_wait);
#endif
        cb_push_back(cb_w, block_tiles);
        trid_to_wait = trid_to_wait == ring_blocks ? 1 : trid_to_wait + 1;
        --in_flight;
    }
}
