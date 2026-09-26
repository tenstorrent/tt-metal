// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCIe peer-to-peer responder. Runs on chip B. Spins on a local L1 flag that
// chip A writes over PCIe, and echoes each sequence number back into chip A's
// L1 through B's own outbound iATU region.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t local_flag = get_arg_val<uint32_t>(i++);
    const uint32_t ack_dst_lo = get_arg_val<uint32_t>(i++);
    const uint32_t ack_dst_hi = get_arg_val<uint32_t>(i++);
    const uint32_t local_ack_src = get_arg_val<uint32_t>(i++);
    const uint32_t results = get_arg_val<uint32_t>(i++);
    const uint32_t iters = get_arg_val<uint32_t>(i++);
    const uint32_t timeout_lo = get_arg_val<uint32_t>(i++);
    const uint32_t timeout_hi = get_arg_val<uint32_t>(i++);
    const uint32_t flag_bytes = get_arg_val<uint32_t>(i++);

    const uint64_t ack_dst = (static_cast<uint64_t>(ack_dst_hi) << 32) | ack_dst_lo;
    const uint64_t timeout = (static_cast<uint64_t>(timeout_hi) << 32) | timeout_lo;

    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_flag);
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_ack_src);
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results);

    res[0] = 0;
    res[1] = 0;
    // Announce readiness to the initiator: ack[1] carries the magic on every ack write.
    ack[0] = 0;
    ack[1] = 0x52454459;
    noc_async_write(local_ack_src, ack_dst, flag_bytes);
    for (uint32_t k = 1; k <= iters; ++k) {
        const uint64_t t0 = get_timestamp();
        while (flag[0] != k) {
            if (get_timestamp() - t0 > timeout) {
                res[0] = 2;
                res[1] = k - 1;
                noc_async_write_barrier();
                return;
            }
        }
        ack[0] = k;
        noc_async_write(local_ack_src, ack_dst, flag_bytes);
    }
    noc_async_write_barrier();
    res[0] = 1;
    res[1] = iters;
}
