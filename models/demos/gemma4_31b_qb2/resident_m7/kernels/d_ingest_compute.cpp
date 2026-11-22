// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter ingest, streamer compute (TRISC; M6b C1), see d_ingest_common.hpp:
//   partial = taps @ W_fc (this core's hidden columns) -> cb_pout
//   per drafter layer: k | v = ctx @ W_kv (this core's 2 column tiles) -> cb_kv
// matmul_block on 32 x 32 activation tiles straight from the weight ring, as d_streamer_compute.cpp.
//
// Runtime args: 0 nd (fc column tiles), 1 nkv (k | v column tiles)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/experimental/pack_block.h"
#include "d_ingest_common.hpp"

using namespace ding;

namespace {

RingCursor cursor;
uint32_t ring_base;  // cb_addr_shift units
uint32_t consumed_addr_word;

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

// out[n] = in0[32, Kt] @ W[Kt, n] for n_tiles columns streamed through ring entry E
template <uint32_t E>
FORCE_INLINE void matmul(uint32_t in0_cb, uint32_t out_cb, uint32_t n_tiles, uint32_t block_tiles) {
    using En = Entry<E>;
    cb_wait_front(in0_cb, En::Kt);
    reconfig_full_operand<SrcOrder::Reverse>(in0_cb, En::cb);
    pack_reconfig_data_format<true>(out_cb);
    matmul_block_init(in0_cb, En::cb, 0, 1, 1, En::sb);
    pack_block_contiguous_init(out_cb);
    cb_reserve_back(out_cb, block_tiles);
    for (uint32_t n = 0; n < n_tiles; n++) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < En::nkb; kb++) {
            with_block<E>([&] {
                for (uint32_t k = 0; k < En::sb; k++) {
                    matmul_block(in0_cb, En::cb, kb * En::sb + k, k, 0, 0, 1, 1, En::sb);
                }
            });
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, out_cb, 1);
        tile_regs_release();
    }
    cb_push_back(out_cb, block_tiles);
}

}  // namespace

void kernel_main() {
    const uint32_t nd = get_arg_val<uint32_t>(0);
    const uint32_t nkv = get_arg_val<uint32_t>(1);
#ifdef TRISC_UNPACK
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed_addr_word = static_cast<uint32_t>(
            (reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(cb_consumed)) >> 2) & 0x3ffff);
    }
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in, cb_w0, cb_pout);
    matmul<kFc>(cb_in, cb_pout, nd, nd_max);
    cb_pop_front(cb_in, Kin);
    for (uint32_t l = 0; l < layers; l++) {
        matmul<kKv>(cb_h, cb_kv, nkv, kKvTiles);
    }
    cb_pop_front(cb_h, Ht);
}
