// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace {
constexpr bool WAIT_FOR_COMPLETION = get_compile_time_arg_val(0) != 0;
constexpr bool STATEFUL = get_compile_time_arg_val(1) != 0;
constexpr uint32_t SIZE = get_compile_time_arg_val(2);
constexpr uint32_t ITERATIONS = get_compile_time_arg_val(3);
constexpr uint32_t CMD_BUF = get_compile_time_arg_val(4) != 0 ? write_reg_cmd_buf : write_cmd_buf;
constexpr uint32_t TRID = 5;
constexpr uint32_t PAGE_BYTES = 2048;
constexpr uint32_t POLL_CAP = 4000000;
constexpr uint32_t MARKER = 0x11111111;

FORCE_INLINE uint32_t sent() { return NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT); }
FORCE_INLINE uint32_t acked() { return NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED); }

FORCE_INLINE void finish_store(volatile tt_l1_ptr uint32_t* word) {
    uint32_t sink;
    asm volatile("lw %0, 0(%1)\n\tand x0, x0, %0" : "=&r"(sink) : "r"(word) : "memory");
}
}  // namespace

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<5>();
    const auto buffer = TensorAccessor(args, get_arg_val<uint32_t>(0), PAGE_BYTES);
    const uint64_t dst = buffer.get_noc_addr(0);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t src = scratch + PAGE_BYTES;
    const uint32_t readback = scratch + 2 * PAGE_BYTES;
    volatile tt_l1_ptr uint32_t* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(readback);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    uint32_t iterations = 0, early = 0, wrong = 0, timeouts = 0;

    if constexpr (STATEFUL) {
        noc_async_write_one_packet_with_trid_set_state(dst, CMD_BUF);
    }
    for (uint32_t i = 0; i < ITERATIONS && timeouts == 0; ++i) {
        const uint32_t pattern = 0x5A5A0000u | i;
        for (uint32_t w = 0; w < SIZE / sizeof(uint32_t); ++w) {
            source[w] = pattern;
        }
        finish_store(source + SIZE / sizeof(uint32_t) - 1);
        const uint32_t before_sent = sent();
        const uint32_t before_acked = acked();

        if constexpr (STATEFUL) {
            noc_async_write_one_packet_with_trid_with_state(src, static_cast<uint32_t>(dst), SIZE, TRID, CMD_BUF);
        } else {
            noc_async_write_one_packet_with_trid(src, dst, SIZE, TRID, CMD_BUF);
        }
        if constexpr (WAIT_FOR_COMPLETION) {
            noc_async_write_barrier_with_trid(TRID);
        } else {
            noc_async_write_flushed_with_trid(TRID);
        }
        // A completion barrier must observe the ack; a flush must observe the departing request.
        if constexpr (WAIT_FOR_COMPLETION) {
            early += (acked() == before_acked);
        } else {
            early += (sent() == before_sent);
        }
        // Either API permits immediate reuse of the source buffer.
        for (uint32_t w = 0; w < SIZE / sizeof(uint32_t); ++w) {
            source[w] = MARKER;
        }
        finish_store(source + SIZE / sizeof(uint32_t) - 1);
        iterations++;

        // Bound the observation wait and finish this transfer before the next iteration.
        uint32_t polls = 0;
        while (sent() == before_sent || acked() == before_acked) {
            if (++polls > POLL_CAP) {
                timeouts++;
                break;
            }
        }
        if (timeouts != 0) {
            break;
        }
        noc_async_read(dst, readback, SIZE);
        noc_async_read_barrier();
        for (uint32_t w = 0; w < SIZE / sizeof(uint32_t); ++w) {
            wrong += (received[w] != pattern);
        }
    }
    ncrisc_noc_set_transaction_id(noc_index, CMD_BUF, 0);
    for (uint32_t w = 0; w < 16; ++w) {
        result[w] = 0;
    }
    result[0] = 0xC0DE7B1D;
    result[1] = iterations;
    result[2] = early;
    result[3] = wrong;
    result[4] = timeouts;
    result[5] = noc_index;
    result[6] = 0xD0E5;
    finish_store(result + 15);
    noc_async_write(scratch, buffer.get_noc_addr(1), 64);
    noc_async_write_barrier();
}
