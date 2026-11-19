// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Diagnostic only: native helper or identical native zero APIs with a bounded final wait.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/zero_init_common.hpp"

void kernel_main() {
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(0);
    constexpr uint32_t poll_limit = get_compile_time_arg_val(1);
    constexpr bool native_helper = get_compile_time_arg_val(2);
    const auto output = TensorAccessor(TensorAccessorArgs<3>(), get_arg_val<uint32_t>(0));
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t end = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    d[32] = 0x41434b42;  // ACKB
    d[33] = noc_index;
    d[34] = noc_nonposted_writes_num_issued[noc_index];
    d[35] = noc_nonposted_writes_acked[noc_index];
    d[36] = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
    d[37] = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
    d[44] = start;
    d[45] = end;
    d[46] = get_arg_val<uint32_t>(0);
    d[47] = get_write_ptr(scratch_cb);
    if constexpr (native_helper) {
        zero_pages(scratch_cb, start, end, 4096, output);
        d[43] = 1;
    } else {
        // Exactly zero_pages' native scratch and page-write operations. Only the final
        // write-ACK wait is bounded and exposes FAILURE instead of hanging the host.
        Noc noc;
        CircularBuffer scratch(scratch_cb);
        noc.async_write_zeros(scratch, NOC_MAX_BURST_SIZE);
        noc.write_zeros_l1_barrier();
        for (uint32_t page = start; page < end; ++page) {
            noc.async_write_zeros(output, 4096, {.page_id = page}, scratch);
        }
        const uint32_t expected = noc_nonposted_writes_acked[noc_index];
        uint32_t actual = 0;
        uint32_t poll = 0;
        for (; poll < poll_limit; ++poll) {
            actual = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
            if (actual == expected || static_cast<int32_t>(actual - expected) > 0) {
                break;
            }
        }
        d[38] = noc_nonposted_writes_num_issued[noc_index];
        d[39] = expected;
        d[40] = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
        d[41] = actual;
        d[42] = poll;
        // 2=ACK overshoot; 3=bounded timeout/undershoot. Neither is a valid output.
        d[43] = actual == expected ? 1 : (static_cast<int32_t>(actual - expected) > 0 ? 2 : 3);
    }
    asm volatile("fence" ::: "memory");
}
