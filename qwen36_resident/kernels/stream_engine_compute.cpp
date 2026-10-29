// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Weight-streaming engine compute (TRISC): for every matmul of the schedule, computes this core's
// output columns out[1, n] = in0[1, K] @ W[K, n] with the M=1 custom matmul (1x32 activation tiles),
// accumulating over the streamed K blocks. Each matmul after the first waits for the writer's gate
// token (the activation-ready signal of the modeled chain).
//
// Runtime args: per entry e: n_tiles.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "stream_engine_common.hpp"

using namespace stream_engine;

namespace {

RingCursor cursor;
uint32_t ring_base;           // cb_addr_shift units
uint32_t consumed_addr_word;  // Tensix word address of the consumed-units register

// Next block of entry E: point the entry's CB at its ring slot, wait for it, run fn, then release it.
// The consumed count is stored by the Tensix core after the unpacker finishes (same as cb_pop_front).
template <uint32_t E, typename F>
FORCE_INLINE void with_block(F fn) {
    using En = Entry<E>;
    const uint32_t at = cursor.place<E>();
    UNPACK((get_local_cb_interface(En::cb).fifo_rd_ptr = ring_base + (at >> cb_addr_shift)));
    cb_wait_front(En::cb, En::sb);
    fn();
    cb_pop_front(En::cb, En::sb);
    UNPACK((TT_SETDMAREG(0, cursor.units & 0xffff, 0, LO_16(4))));
    UNPACK((TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK)));
    UNPACK((TT_STOREREG(4, consumed_addr_word)));
}

template <uint32_t E>
FORCE_INLINE void run_entry(bool gated) {
    using En = Entry<E>;
    const uint32_t n_tiles = get_arg_val<uint32_t>(E);
    if (gated) {
        cb_wait_front(cb_gate, 1);
        cb_pop_front(cb_gate, 1);
    }
    reconfig_data_format<SrcOrder::Reverse>(cb_in0, En::cb);
    custom_mm_block_init_short<false, true, false>(cb_in0, En::cb, cb_out);
    for (uint32_t n = 0; n < n_tiles; n++) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb + 1 < En::nkb; kb++) {
            with_block<E>([&] { custom_mm_block<false>(cb_in0, En::cb, kb * En::sb, 0, 0, En::sb); });
        }
        with_block<E>([&] { custom_mm_block<true>(cb_in0, En::cb, (En::nkb - 1) * En::sb, 0, 0, En::sb); });
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_block_contiguous(0, cb_out, 1);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}

}  // namespace

void kernel_main() {
#ifdef TRISC_UNPACK
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        // A stream register (low address space) - TT_STOREREG only reaches 18-bit word addresses.
        consumed_addr_word = static_cast<uint32_t>(
            (reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(cb_consumed)) >> 2) & 0x3ffff);
    }
#endif
    custom_mm_block_init<false, true, false>(cb_in0, Entry<0>::cb, cb_out);
    pack_block_contiguous_init(cb_out);
    cb_wait_front(cb_in0, Entry<0>::Kt);
    bool gated = false;
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t l = 0; l < layers; l++) {
            run_entry<0>(gated);
            gated = true;
            run_entry<1>(true);
            run_entry<2>(true);
            run_entry<3>(true);
        }
    }
    custom_mm_block_uninit<false>();
}
