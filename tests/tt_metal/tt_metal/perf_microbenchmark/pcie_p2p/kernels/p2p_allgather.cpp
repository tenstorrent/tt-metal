// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PCIe peer-to-peer all-gather step. One worker per chip. Each iteration k:
//   * write my payload (size bytes) into slot[me] of every peer, straight over PCIe
//   * write flag k into flag[me] on every peer
//   * wait until flag[p] == k for every peer p (their payloads have landed)
// The measured time is the all-gather step latency as seen by this chip. The receive side
// also checks that the first and last payload word carry k when the flag is seen, which
// catches any data-after-flag reordering on the PCIe -> NoC path.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t me = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t pcie_lo = get_arg_val<uint32_t>(i++);  // this chip's PCIe core NoC XY | (1 << 60)
    const uint32_t pcie_hi = get_arg_val<uint32_t>(i++);
    const uint32_t nocb_lo = get_arg_val<uint32_t>(i++);  // outbound base: peer p lives at nocb + p * 2 MiB
    const uint32_t nocb_hi = get_arg_val<uint32_t>(i++);
    const uint32_t FLAGS = get_arg_val<uint32_t>(i++);     // L1: n flag slots of 64 B
    const uint32_t FLAG_SRC = get_arg_val<uint32_t>(i++);  // L1: 64 B staging for my flag write
    const uint32_t RESULTS = get_arg_val<uint32_t>(i++);
    const uint32_t SRC = get_arg_val<uint32_t>(i++);     // L1: my payload
    const uint32_t RECV = get_arg_val<uint32_t>(i++);    // L1: n receive slots
    const uint32_t stride = get_arg_val<uint32_t>(i++);  // receive slot stride
    const uint32_t size = get_arg_val<uint32_t>(i++);
    const uint32_t iters = get_arg_val<uint32_t>(i++);
    const uint32_t timeout_lo = get_arg_val<uint32_t>(i++);
    const uint32_t timeout_hi = get_arg_val<uint32_t>(i++);
    const uint32_t flag_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t warmup = get_arg_val<uint32_t>(i++);  // iterations excluded from the steady-state period

    const uint64_t pcie = (static_cast<uint64_t>(pcie_hi) << 32) | pcie_lo;
    const uint64_t nocb = (static_cast<uint64_t>(nocb_hi) << 32) | nocb_lo;
    const uint64_t timeout = (static_cast<uint64_t>(timeout_hi) << 32) | timeout_lo;
    auto peer = [&](uint32_t p, uint32_t l1_addr) -> uint64_t {
        return pcie | (nocb + (static_cast<uint64_t>(p) << 21) + l1_addr);
    };

    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FLAGS);
    volatile tt_l1_ptr uint32_t* seq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FLAG_SRC);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SRC);
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RECV);
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RESULTS);
    const uint32_t last_w = (size / 4) - 1;

    res[0] = 0;  // status: 1 ok, 2 timeout
    res[1] = 0;  // iterations completed
    res[2] = 0;  // ordering violations (flag seen before payload)
    // res[4..5]: 64-bit cycles from the start of iteration warmup+1 to the end of the last iteration

    uint32_t violations = 0;
    uint64_t t_period_start = 0;
    for (uint32_t k = 1; k <= iters; ++k) {
        const uint64_t t0 = get_timestamp();
        if (k == warmup + 1) {
            t_period_start = t0;
        }
        const uint32_t par = k & 1;  // receive slots are double-buffered by iteration parity
        src[0] = k;
        src[last_w] = k;
        for (uint32_t p = 0; p < n; ++p) {
            if (p == me) {
                continue;
            }
            noc_async_write(SRC, peer(p, RECV + (me * 2 + par) * stride), size);
        }
        seq[0] = k;
        for (uint32_t p = 0; p < n; ++p) {
            if (p == me) {
                continue;
            }
            noc_async_write(FLAG_SRC, peer(p, FLAGS + me * 64), flag_bytes);
        }
        for (uint32_t p = 0; p < n; ++p) {
            if (p == me) {
                continue;
            }
            while (flags[p * 16] != k) {
                if (get_timestamp() - t0 > timeout) {
                    res[0] = 2;
                    res[1] = k - 1;
                    res[2] = violations;
                    res[3] = p;  // peer we were waiting on
                    noc_async_write_barrier();
                    return;
                }
            }
        }
        const uint64_t t1 = get_timestamp();
        for (uint32_t p = 0; p < n; ++p) {
            if (p == me) {
                continue;
            }
            const uint32_t base_w = ((p * 2 + par) * stride) / 4;
            violations += (recv[base_w] != k) + (recv[base_w + last_w] != k);
        }
        res[8 + (k - 1)] = static_cast<uint32_t>(t1 - t0);
        noc_async_writes_flushed();  // SRC / FLAG_SRC are free to be rewritten
    }
    const uint64_t t_end = get_timestamp();
    noc_async_write_barrier();
    res[4] = static_cast<uint32_t>(t_end - t_period_start);
    res[5] = static_cast<uint32_t>((t_end - t_period_start) >> 32);
    res[0] = 1;
    res[1] = iters;
    res[2] = violations;
}
