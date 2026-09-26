// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCIe peer-to-peer initiator. Runs on chip A and writes straight from its own
// L1, through the PCIe NoC endpoint and an outbound iATU region, into a TLB
// window on chip B that points at a core's L1 (or a DRAM bank) on B.
//
// Modes:
//   0 SMOKE     one 4 KB write into B, host verifies the pattern landed
//   1 PINGPONG  write seq to B's flag, spin on local flag for the echo, log RTT
//   2 BW        stream total_bytes in chunks into B's data window, then flag,
//               wait for B's ack, log issue cycles and end-to-end cycles

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

static constexpr uint32_t RESULT_STATUS_OK = 1;
static constexpr uint32_t RESULT_STATUS_TIMEOUT = 2;
static constexpr uint32_t RESULT_STATUS_NO_RESPONDER = 3;
static constexpr uint32_t READY_MAGIC = 0x52454459;  // "REDY", written by the responder into flag[1]

void kernel_main() {
    uint32_t i = 0;
    const uint32_t mode = get_arg_val<uint32_t>(i++);
    const uint32_t data_dst_lo = get_arg_val<uint32_t>(i++);  // B data window  (64-bit NoC addr)
    const uint32_t data_dst_hi = get_arg_val<uint32_t>(i++);
    const uint32_t flag_dst_lo = get_arg_val<uint32_t>(i++);  // B flag        (64-bit NoC addr)
    const uint32_t flag_dst_hi = get_arg_val<uint32_t>(i++);
    const uint32_t local_flag = get_arg_val<uint32_t>(i++);      // L1: echoed seq lands here
    const uint32_t local_src = get_arg_val<uint32_t>(i++);       // L1: source buffer (>= chunk bytes)
    const uint32_t local_flag_src = get_arg_val<uint32_t>(i++);  // L1: 64 B block used to carry seq
    const uint32_t results = get_arg_val<uint32_t>(i++);         // L1: results array
    const uint32_t iters = get_arg_val<uint32_t>(i++);
    const uint32_t chunk = get_arg_val<uint32_t>(i++);
    const uint32_t total_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t window_span = get_arg_val<uint32_t>(i++);  // bytes of B window to cycle through
    const uint32_t timeout_lo = get_arg_val<uint32_t>(i++);
    const uint32_t timeout_hi = get_arg_val<uint32_t>(i++);
    const uint32_t flag_bytes = get_arg_val<uint32_t>(i++);  // 16 or 64

    const uint64_t data_dst = (static_cast<uint64_t>(data_dst_hi) << 32) | data_dst_lo;
    const uint64_t flag_dst = (static_cast<uint64_t>(flag_dst_hi) << 32) | flag_dst_lo;
    const uint64_t timeout = (static_cast<uint64_t>(timeout_hi) << 32) | timeout_lo;

    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_flag);
    volatile tt_l1_ptr uint32_t* seq_blk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_flag_src);
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results);

    // results[0] = status, results[1] = iterations completed, results[2..3] = 64-bit cycles (BW),
    // results[4..5] = 64-bit end-to-end cycles (BW), results[8 + k] = per-iteration RTT (PINGPONG)
    res[0] = 0;
    res[1] = 0;

    if (mode == 0) {
        noc_async_write(local_src, data_dst, 4096);
        noc_async_write_barrier();
        res[0] = RESULT_STATUS_OK;
        res[1] = 1;
        return;
    }

    // Persistent semantics: wait until the responder on B is up and spinning before starting any clock.
    {
        const uint64_t t0 = get_timestamp();
        while (flag[1] != READY_MAGIC) {
            if (get_timestamp() - t0 > timeout) {
                res[0] = RESULT_STATUS_NO_RESPONDER;
                return;
            }
        }
    }

    if (mode == 1) {
        uint32_t done = 0;
        for (uint32_t k = 1; k <= iters; ++k) {
            seq_blk[0] = k;
            const uint64_t t0 = get_timestamp();
            noc_async_write(local_flag_src, flag_dst, flag_bytes);
            while (flag[0] != k) {
                if (get_timestamp() - t0 > timeout) {
                    res[0] = RESULT_STATUS_TIMEOUT;
                    res[1] = done;
                    noc_async_write_barrier();
                    return;
                }
            }
            const uint64_t t1 = get_timestamp();
            res[8 + (k - 1)] = static_cast<uint32_t>(t1 - t0);
            ++done;
        }
        noc_async_write_barrier();
        res[0] = RESULT_STATUS_OK;
        res[1] = done;
        return;
    }

    // mode 2: bandwidth
    {
        uint32_t off = 0;
        const uint64_t t0 = get_timestamp();
        for (uint32_t sent = 0; sent < total_bytes; sent += chunk) {
            noc_async_write(local_src, data_dst + off, chunk);
            off += chunk;
            if (off + chunk > window_span) {
                off = 0;
            }
        }
        noc_async_write_barrier();
        const uint64_t t1 = get_timestamp();
        // Completion flag: B's responder echoes it back once it has observed it.
        seq_blk[0] = 1;
        noc_async_write(local_flag_src, flag_dst, flag_bytes);
        while (flag[0] != 1) {
            if (get_timestamp() - t0 > timeout) {
                res[0] = RESULT_STATUS_TIMEOUT;
                noc_async_write_barrier();
                return;
            }
        }
        const uint64_t t2 = get_timestamp();
        noc_async_write_barrier();
        res[2] = static_cast<uint32_t>(t1 - t0);
        res[3] = static_cast<uint32_t>((t1 - t0) >> 32);
        res[4] = static_cast<uint32_t>(t2 - t0);
        res[5] = static_cast<uint32_t>((t2 - t0) >> 32);
        res[0] = RESULT_STATUS_OK;
        res[1] = total_bytes / chunk;
    }
}
