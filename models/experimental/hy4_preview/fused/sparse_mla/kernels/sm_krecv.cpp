// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sparse-MLA K receiver (data movement, on every attention core): the other end of sm_mcast.cpp. Per chunk of the
// S-block: reserve the chunk in cb_k (which waits for the attention kernel to free it), tell every converter of
// the S-block, wait until all of them have counted the chunk up, publish the chunk to the attention kernel.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "sm_dm.hpp"

void kernel_main() {
#ifdef SM_NOP
    return;
#endif
    constexpr uint32_t cb_k = get_named_compile_time_arg_val("cb_k");
    constexpr uint32_t cb_hdr = get_named_compile_time_arg_val("cb_hdr_k");
    constexpr uint32_t USERS = get_named_compile_time_arg_val("users");
    constexpr uint32_t GROUPS = get_named_compile_time_arg_val("groups");
    constexpr uint32_t NCONV = get_named_compile_time_arg_val("converters");
    constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk_rows");
    constexpr uint32_t DHt = get_named_compile_time_arg_val("dht");
    constexpr uint32_t sem_ready = get_named_compile_time_arg_val("sem_ready");
    constexpr uint32_t sem_data = get_named_compile_time_arg_val("sem_data");
    constexpr uint32_t sem_err = get_named_compile_time_arg_val("sem_err");
    constexpr uint32_t TBL = get_named_compile_time_arg_val("tbl_krecv");
    constexpr uint32_t SLABS = CHUNK / sm::TILE_ROWS;

    constexpr auto sel_args = TensorAccessorArgs<0>();

    const sm::Role role =
        sm::role_from_position<get_named_compile_time_arg_val("x0_left"), get_named_compile_time_arg_val("x0_right")>();
    const sm::Coords at = sm::block_coords(TBL, role.block);
    const uint32_t sel_addr = get_common_arg_val<uint32_t>(0);

    const auto sel = TensorAccessor(sel_args, sel_addr);
    uint32_t counts[sm::MAX_USERS];
    sm::read_counts<USERS>(sel, cb_hdr, counts);

    volatile tt_l1_ptr uint32_t* sent = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_data);
    volatile tt_l1_ptr uint32_t* errors = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_err);

    uint32_t chunks = 0;
    sm::for_each_chunk<USERS, GROUPS, CHUNK>(counts, role.block, [&](uint32_t, uint32_t, uint32_t) {
        cb_reserve_back(cb_k, SLABS * DHt);
        for (uint32_t v = 0; v < NCONV; ++v) {
            noc_semaphore_inc(get_noc_addr(at.reader_x, at.reader_y + v, sem_ready), 1);
        }
        ++chunks;
        if (!sm::wait_min_bounded(sent, chunks * NCONV)) {
            *errors = *errors | 4u;  // read by the I/O kernel of this core into the status page
        }
        cb_push_back(cb_k, SLABS * DHt);
    });
    noc_async_atomic_barrier();
    noc_semaphore_set(sent, 0);  // every increment has been waited for: zero again for the next launch
}
