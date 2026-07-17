// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tensix consumer extent probe: snapshot getters on UNPACK. When rotate_tc is set,
// advances tc_idx with wait_front + copy_tile + pop_front between snapshots (minimal
// credits). When the producer rotates (push_back between TC snapshots), consumers
// paired with producer TC0 must drain one credit each or finish() hangs in AAW.

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_tc_snapshots = get_arg(args::num_tc_snapshots);
    constexpr uint32_t rotate_tc = get_arg(args::rotate_tc);
    constexpr uint32_t drain_producer_rotate_credits = get_arg(args::drain_producer_rotate_credits);
    constexpr uint32_t drain_last_tc_credit = get_arg(args::drain_last_tc_credit);
    constexpr uint32_t num_producers = get_arg(args::num_producers);

    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);
    const uint32_t consumer_idx = get_my_thread_id();

    DataflowBuffer dfb(dfb::in);

    if constexpr (rotate_tc || drain_producer_rotate_credits || drain_last_tc_credit) {
        unary_op_init_common(dfb.get_id(), dfb.get_id());
    }

    for (uint32_t tc = 0; tc < num_tc_snapshots; ++tc) {
#ifdef TRISC_UNPACK
        constexpr uint32_t extent_record_bytes = 8 * sizeof(uint32_t);
#ifdef ARCH_QUASAR
        const uint32_t result_l1_ptr_addr =
            result_l1_addr + MEM_L1_UNCACHED_BASE + tc * extent_record_bytes;
#else
        const uint32_t result_l1_ptr_addr = result_l1_addr + tc * extent_record_bytes;
#endif
        volatile tt_l1_ptr uint32_t* const out =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_ptr_addr);
        out[0] = dfb.get_entry_size();
        out[1] = dfb.get_stride_size();
        out[2] = dfb.get_total_num_entries();
        out[3] = dfb.get_total_size_bytes();
        out[4] = dfb.get_local_num_entries();
        out[5] = dfb.get_local_size_bytes();
        out[6] = dfb.get_ring_span_bytes();
        out[7] = dfb.get_ring_span_num_entries();
#endif

        if constexpr (rotate_tc) {
            if (tc + 1 < num_tc_snapshots) {
                tile_regs_acquire();
                dfb.wait_front(1);
                copy_tile(dfb.get_id(), 0, 0);
                dfb.pop_front(1);
                tile_regs_release();
            }
        }
    }

    // After rotating through all TCs, pop once more to drain credits posted to the last TC
    // (e.g. 2Sx4A: producer[1] posts to consumer TC1 while rotate only pops between TC0→TC1).
    if constexpr (drain_last_tc_credit) {
        tile_regs_acquire();
        dfb.wait_front(1);
        copy_tile(dfb.get_id(), 0, 0);
        dfb.pop_front(1);
        tile_regs_release();
    }

    if constexpr (drain_producer_rotate_credits) {
        if (consumer_idx < num_producers) {
            tile_regs_acquire();
            dfb.wait_front(1);
            copy_tile(dfb.get_id(), 0, 0);
            dfb.pop_front(1);
            tile_regs_release();
        }
    }

    dfb.finish();
}
