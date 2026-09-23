// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The send leg: one tt_uva_put_signal per iteration. Framing, fencing and the commit all
// live behind the verb, so this file holds only what is specific to the benchmark.
#include <stdint.h>

#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"

#include "tt_metal/distributed/host_uva_layout.hpp"
#include "api/tt_uva.h"

namespace ex = tt::tt_metal::experimental;

void kernel_main() {
    constexpr uint32_t stage_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t payload_addr = get_compile_time_arg_val(2);
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t iterations = get_compile_time_arg_val(4);
    constexpr uint32_t grid_width = get_compile_time_arg_val(5);
    constexpr uint32_t host = get_compile_time_arg_val(6);
    constexpr uint32_t chip = get_compile_time_arg_val(7);
    constexpr uint32_t chips_per_host = get_compile_time_arg_val(8);
    // 0 disables tt_uva_sync(); a bandwidth run needs only tt_uva_fin().
    constexpr uint32_t consumed_addr = get_compile_time_arg_val(9);
    // Resolves sig_addr to a wire offset. The L1 map is identical on every core, so the
    // address this core holds names the same word on the target.
    constexpr uint32_t l1_base = get_compile_time_arg_val(10);
    // The receiver's signal word, chosen by the test -- 0 sends unsignalled frames.
    constexpr uint32_t sig_addr = get_compile_time_arg_val(11);
    // Where this core reports its own loop window; 0 collects nothing. The host reads it
    // after Finish(), so the rate is the device's own and no host clock is involved.
    constexpr uint32_t result_addr = get_compile_time_arg_val(12);
    // Iterations to run before the steady-state stamp, so the rate excludes the ramp.
    constexpr uint32_t warmup_iters = get_compile_time_arg_val(13);

    // RUNTIME, not compile-time: each core owns its own D2HSocket and each config buffer is
    // a separate allocation, so baking one in would point every core at one core's ring.
    const uint32_t cfg_addr = get_arg_val<uint32_t>(0);
    // Where the bytes are going is DATA and legitimately supplied; who this core is is
    // derived. That split is why tt_uva_self() takes no identity argument.
    const uint32_t dest_selector = get_arg_val<uint32_t>(1);
    const uint32_t dest_offset = get_arg_val<uint32_t>(2);

    ex::tt_uva_ini(
        cfg_addr,
        0,
        page_size,
        stage_addr,
        ex::tt_uva_self(grid_width, host, chip, chips_per_host),
        0,
        l1_base,
        0,
        consumed_addr);

    volatile tt_l1_ptr uint32_t* const payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr);
    const uint32_t me = ex::tt_uva_core_index(get_absolute_logical_x(), get_absolute_logical_y(), grid_width);

    // The receiving side is the witness, so the payload carries a per-sender pattern.
    // Written once: refilling it per iteration would put a memset in the timed loop.
    if (iterations != 0) {
        const uint32_t b = 0x40u + (me & 0x1Fu);
        const uint32_t w = b | (b << 8) | (b << 16) | (b << 24);
        for (uint32_t k = 1; k < payload_bytes / sizeof(uint32_t); ++k) {
            payload[k] = w;
        }
    }

    const ex::tt_uva_t dst = ex::tt_uva_t6_from_selector(dest_selector, dest_offset);

    const uint64_t t_begin = ex::tt_uva_clock();
    uint64_t t_steady = t_begin;
    for (uint32_t i = 0; i < iterations; ++i) {
        if (i == warmup_iters) {
            t_steady = ex::tt_uva_clock();
        }
        // Word 0 is the iteration stamp: without it a host re-reading a stale slot cannot
        // be told from one that got a fresh push.
        payload[0] = i;
        // ADD 1, so the target's word counts frames and a consumer can wait on the i-th.
        // Compile-time, so the unsignalled build carries none of the signal path.
        if constexpr (sig_addr != 0) {
            ex::tt_uva_put_signal(payload_addr, dst, payload_bytes, sig_addr, 1, ex::kSignalAdd);
        } else {
            ex::tt_uva_put(payload_addr, dst, payload_bytes);
        }
    }
    // Before fin(): that drains the tail, and a posting rate should not carry the drain.
    const uint64_t t_end = ex::tt_uva_clock();
    ex::tt_uva_fin();

    // test_d2h_bw.cpp:273 is where the host reads the value
    if constexpr (result_addr != 0) {
        volatile tt_l1_ptr uint32_t* const r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);
        r[0] = static_cast<uint32_t>(t_begin);
        r[1] = static_cast<uint32_t>(t_begin >> 32);
        r[2] = static_cast<uint32_t>(t_end);
        r[3] = static_cast<uint32_t>(t_end >> 32);
        r[4] = iterations;
        r[5] = static_cast<uint32_t>(t_steady);
        r[6] = static_cast<uint32_t>(t_steady >> 32);
    }
}
