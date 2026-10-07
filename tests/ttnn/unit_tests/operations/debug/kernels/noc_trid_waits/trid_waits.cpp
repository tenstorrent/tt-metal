// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Issues one transaction with a transaction ID, waits for it with a per-trid wait right away, and checks
// that the transaction had really finished. The per-trid counters are incremented when NOC_CMD_CTRL is
// written; if the wait's first counter read overtakes that store it sees 0 and returns too early.
//
// MODE 0: noc_async_read_barrier_with_trid   - the read data must have landed.
// MODE 1: noc_async_write_barrier_with_trid  - NIU_MST_REQS_OUTSTANDING_ID(trid) must be 0.
// MODE 2: noc_async_write_flushed_with_trid  - NIU_MST_WRITE_REQS_OUTGOING_ID(trid) must be 0.
// IMPL 0: the API; IMPL 1: the counter poll the API used before ordering it after NOC_CMD_CTRL.
//
// Scratch layout (same L1 address on both cores): [0, 16 KB) local buffer, [16, 32 KB) remote buffer
// (host-filled with word index i at word i), result words at +32 KB.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t LOCAL_OFF = 0;
constexpr uint32_t REMOTE_OFF = 16 * 1024;
constexpr uint32_t RESULT_OFF = 32 * 1024;
constexpr uint32_t TRID = 1;

FORCE_INLINE volatile tt_l1_ptr uint32_t* l1_word(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

void kernel_main() {
    constexpr uint32_t ITERS = get_compile_time_arg_val(0);
    constexpr uint32_t MODE = get_compile_time_arg_val(1);
    constexpr uint32_t IMPL = get_compile_time_arg_val(2);
    constexpr uint32_t BYTES = get_compile_time_arg_val(3);
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t remote_x = get_arg_val<uint32_t>(1);
    const uint32_t remote_y = get_arg_val<uint32_t>(2);

    const uint32_t local = base + LOCAL_OFF;
    const uint32_t remote = base + REMOTE_OFF;
    volatile tt_l1_ptr uint32_t* result = l1_word(base + RESULT_OFF);
    volatile tt_l1_ptr uint32_t* local_words = l1_word(local);
    constexpr uint32_t LAST = BYTES / 4 - 1;
    // The remote buffer holds word index i at word i (REMOTE_OFF / 4 + i in the scratch tensor).
    constexpr uint32_t FIRST_VALUE = REMOTE_OFF / 4;
    constexpr uint32_t LAST_VALUE = REMOTE_OFF / 4 + LAST;

    uint32_t early = 0;
    if constexpr (MODE == 0) {
        noc_async_read_one_packet_set_state(get_noc_addr(remote_x, remote_y, remote), BYTES);
        noc_async_read_set_trid(TRID);
    }
    for (uint32_t i = 0; i < ITERS; ++i) {
        if constexpr (MODE == 0) {
            local_words[0] = 0;
            local_words[LAST] = 0;
            noc_async_read_one_packet_with_state_with_trid(remote, 0, local, TRID);
            if constexpr (IMPL == 0) {
                noc_async_read_barrier_with_trid(TRID);
            } else {
                while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, TRID)) {
                }
                invalidate_l1_cache();
            }
            if (local_words[0] != FIRST_VALUE || local_words[LAST] != LAST_VALUE) {
                early++;
                // Let the read land before clearing the buffer for the next iteration.
                noc_async_read_barrier();
            }
        } else {
            noc_async_write_one_packet_with_trid(local, get_noc_addr(remote_x, remote_y, local), BYTES, TRID);
            if constexpr (MODE == 1) {
                if constexpr (IMPL == 0) {
                    noc_async_write_barrier_with_trid(TRID);
                } else {
                    while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, TRID)) {
                    }
                }
                ncrisc_noc_order_after_cmd_ctrl_write(noc_index, write_cmd_buf);
                if (NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(TRID)) != 0) {
                    early++;
                }
            } else {
                if constexpr (IMPL == 0) {
                    noc_async_write_flushed_with_trid(TRID);
                } else {
                    while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, TRID)) {
                    }
                }
                ncrisc_noc_order_after_cmd_ctrl_write(noc_index, write_cmd_buf);
                if (NOC_STATUS_READ_REG(noc_index, NIU_MST_WRITE_REQS_OUTGOING_ID(TRID)) != 0) {
                    early++;
                }
            }
            noc_async_write_barrier();
        }
    }
    result[0] = ITERS;
    result[1] = early;
}
