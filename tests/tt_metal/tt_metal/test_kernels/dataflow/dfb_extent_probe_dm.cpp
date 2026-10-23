// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DM DFB extent probe: snapshot static layout getters to L1, optionally rotate tc_idx
// via reserve_back/push_back (no DRAM / NoC). Multi-threaded: one record per thread at
// tc_idx == 0 unless rotate_tc is set (single-thread only for rotation).

#include "api/dataflow/dataflow_buffer.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

namespace {

inline uint32_t l1_ptr_addr(uint32_t byte_addr) {
#ifdef ARCH_QUASAR
    return byte_addr + MEM_L1_UNCACHED_BASE;
#else
    return byte_addr;
#endif
}

inline void snapshot_extent(DataflowBuffer& dfb, uint32_t result_l1_addr, uint32_t record_index) {
    constexpr uint32_t extent_record_bytes = 8 * sizeof(uint32_t);
    const uint32_t out_addr = l1_ptr_addr(result_l1_addr + record_index * extent_record_bytes);
    volatile tt_l1_ptr uint32_t* const out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
    out[0] = dfb.get_entry_size();
    out[1] = dfb.get_stride_size();
    out[2] = dfb.get_total_num_entries();
    out[3] = dfb.get_total_size_bytes();
    out[4] = dfb.get_local_num_entries();
    out[5] = dfb.get_local_size_bytes();
    out[6] = dfb.get_ring_span_bytes();
    out[7] = dfb.get_ring_span_num_entries();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_tc_snapshots = get_arg(args::num_tc_snapshots);
    constexpr uint32_t rotate_tc = get_arg(args::rotate_tc);
    constexpr uint32_t credits_to_post = get_arg(args::credits_to_post);

    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);
    const uint32_t thread_id = get_my_thread_id();

    DataflowBuffer dfb(dfb::out);

    const uint32_t base_record = thread_id * num_tc_snapshots;
    for (uint32_t tc = 0; tc < num_tc_snapshots; ++tc) {
        snapshot_extent(dfb, result_l1_addr, base_record + tc);
        if constexpr (rotate_tc) {
            if (tc + 1 < num_tc_snapshots) {
                dfb.reserve_back(1);
                dfb.push_back(1);
            }
        }
    }

    if constexpr (credits_to_post > 0) {
        for (uint32_t i = 0; i < credits_to_post; ++i) {
            dfb.reserve_back(1);
            dfb.push_back(1);
        }
    }

    dfb.finish();
}
