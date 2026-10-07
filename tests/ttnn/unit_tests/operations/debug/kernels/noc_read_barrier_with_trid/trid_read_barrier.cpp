// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A transaction-id read followed directly by its barrier, the way a kernel writes it:
//     noc_async_read_one_packet_with_state_with_trid(...);
//     noc_async_read_barrier_with_trid(trid);
// Each iteration first stores a marker into the destination word. If the barrier returns before the read has
// landed, the destination still holds the marker ("stale") and the read-response counter has not moved ("early").
// After the check the iteration waits (capped) for the read to complete, so iterations never overlap.
//
// MODE 0 uses noc_async_read_barrier_with_trid. MODE 1 is a diagnostic control that polls the outstanding counter
// directly after the issue, as the barrier did before it read NOC_CMD_CTRL back; where that race fires, it shows the
// detector works.
// FORM selects the read call: 0 default, 1 skip_ptr_update, 2 skip_cmdbuf_chk.
// The observation waits this kernel adds are capped; the API calls under test and the MODE 1 poll are not.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace {
constexpr uint32_t MODE = get_compile_time_arg_val(0);
constexpr uint32_t FORM = get_compile_time_arg_val(1);
constexpr uint32_t ITERATIONS = get_compile_time_arg_val(2);
constexpr uint32_t TRID = 5;
constexpr uint32_t PATTERN = 0x5A5A5A5A;  // contents of the source DRAM page
constexpr uint32_t POLL_CAP = 4000000;
constexpr uint32_t PAGE_BYTES = 2048;

// Result words (also read by the host).
constexpr uint32_t R_MAGIC = 0, R_ITERATIONS = 1, R_EARLY = 2, R_STALE = 3, R_TIMEOUTS = 4, R_WRONG = 5, R_NOC = 6,
                   R_DONE = 7, R_WORDS = 8;
constexpr uint32_t MAGIC = 0xC0DE7B1D, DONE = 0xD0E5;

FORCE_INLINE void issue(uint32_t src_lo, uint32_t dst) {
    if constexpr (FORM == 0) {
        noc_async_read_one_packet_with_state_with_trid(src_lo, 0, dst, TRID);
    } else if constexpr (FORM == 1) {
        noc_async_read_one_packet_with_state_with_trid<true, false>(src_lo, 0, dst, TRID);
    } else {
        noc_async_read_one_packet_with_state_with_trid<false, true>(src_lo, 0, dst, TRID);
    }
}

// The skip_ptr_update form leaves the read uncounted in software; count it so later read barriers stay exact.
FORCE_INLINE void count_uncounted_read() {
    if constexpr (FORM == 1) {
        if constexpr (noc_mode == DM_DEDICATED_NOC) {
            noc_reads_num_issued[noc_index] += 1;
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc_index, 1);
        }
    }
}

FORCE_INLINE uint32_t read_responses() { return NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED); }
}  // namespace

void kernel_main() {
    const uint32_t buffer_addr = get_arg_val<uint32_t>(0);
    constexpr auto args = TensorAccessorArgs<3>();
    const auto buffer = TensorAccessor(args, buffer_addr, PAGE_BYTES);

    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t dst = scratch + 1024;
    volatile tt_l1_ptr uint32_t* dst_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    const uint64_t src = buffer.get_noc_addr(0);  // page 0 holds PATTERN

    uint32_t iterations = 0, early = 0, stale = 0, timeouts = 0, wrong = 0;
    noc_async_read_set_trid(TRID);
    noc_async_read_one_packet_set_state(src, 4);
    for (uint32_t i = 0; i < ITERATIONS && timeouts == 0; ++i) {
        // the command buffer is idle before each issue (the skip_cmdbuf_chk form requires it)
        uint32_t ready_polls = 0;
        while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
            if (++ready_polls > POLL_CAP) {
                timeouts++;
                break;
            }
        }
        if (timeouts != 0) {
            break;
        }
        const uint32_t marker = 0x11110000u | (i & 0xFFFFu);
        *dst_word = marker;
        {
            // complete the marker store before the read is issued
            uint32_t sink;
            asm volatile("lw %0, 0(%1)\n\tand x0, x0, %0" : "=&r"(sink) : "r"(dst_word) : "memory");
        }
        const uint32_t before = read_responses();
        issue((uint32_t)src, dst);
        if constexpr (MODE == 0) {
            noc_async_read_barrier_with_trid(TRID);
        } else {
            while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, TRID)) {
            }
            invalidate_l1_cache();
        }
        const uint32_t after = read_responses();
        const uint32_t value = *dst_word;
        count_uncounted_read();
        iterations++;
        early += (after == before);
        stale += (value == marker);

        // Let the read finish before the next iteration; it must then have delivered the source word.
        uint32_t polls = 0;
        while ((uint32_t)(read_responses() - before) < 1 ||
               NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(TRID)) != 0) {
            if (++polls > POLL_CAP) {
                timeouts++;
                break;
            }
        }
        if (timeouts == 0) {
            uint32_t landed = 0;
            for (uint32_t p = 0; p < 64 && landed != PATTERN; ++p) {
                invalidate_l1_cache();
                landed = *dst_word;
            }
            wrong += (landed != PATTERN);
        }
    }
    noc_async_read_set_trid(0);

    for (uint32_t w = 0; w < 16; ++w) {
        result[w] = 0;
    }
    result[R_MAGIC] = MAGIC;
    result[R_ITERATIONS] = iterations;
    result[R_EARLY] = early;
    result[R_STALE] = stale;
    result[R_TIMEOUTS] = timeouts;
    result[R_WRONG] = wrong;
    result[R_NOC] = noc_index;
    result[R_DONE] = DONE;
    {
        uint32_t sink;
        asm volatile("lw %0, 28(%1)\n\tand x0, x0, %0" : "=&r"(sink) : "r"(result) : "memory");
    }
    noc_async_write(scratch, buffer.get_noc_addr(1), R_WORDS * sizeof(uint32_t) * 2);
    noc_async_write_barrier();
}
