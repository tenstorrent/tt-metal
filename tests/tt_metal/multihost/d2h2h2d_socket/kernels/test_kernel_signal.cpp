// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The receive leg. apply_signal runs inside land_one, so the signal advances by exactly one
// per landed frame: asking tt_uva_test for seen+1 IS the per-frame hook.
#include <stdint.h>

#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"

#include "api/tt_uva.h"

namespace ex = tt::tt_metal::experimental;

void kernel_main() {
    // page_size, not payload_bytes: the whole frame lands, so a put_signal's trailer
    // arrives with its payload rather than behind it.
    constexpr uint32_t landing_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    // The senders' ADD-1 lands here, so the word counts frames and `expect` ends the run.
    constexpr uint32_t sig_addr = get_compile_time_arg_val(2);
    constexpr uint32_t expect = get_compile_time_arg_val(3);
    // Bounds a signal offset that arrived from a peer, and resolves it to an address.
    constexpr uint32_t l1_base = get_compile_time_arg_val(4);
    constexpr uint32_t l1_size = get_compile_time_arg_val(5);
    // Payload bytes, i.e. page_size minus the trailer: what the sender actually patterned.
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(6);
    // 0 for a bandwidth run, so that build carries none of the compare loop.
    constexpr uint32_t verify = get_compile_time_arg_val(7);
    // l1.verify_addr: its own line, read by the host after Finish(). NOT stop_addr, which
    // the host polls mid-run -- one writer per doorbell line.
    constexpr uint32_t result_addr = get_compile_time_arg_val(8);

    // RUNTIME, not compile-time: each core owns its own H2DSocket and each config buffer is
    // a separate allocation, so baking one in would point every core at one core's ring.
    const uint32_t cfg_addr = get_arg_val<uint32_t>(0);
    const uint32_t enabled = get_arg_val<uint32_t>(1);

    // A core with no traffic is still launched so the L1 map is identical everywhere.
    // Returning before touching the socket is safe; after would arm a ring nobody writes.
    if (enabled == 0) {
        return;
    }

    // L1 is not zeroed between runs, so a stale count would satisfy the wait immediately.
    // Safe here: a frame cannot reach the word until this core starts polling below.
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sig_addr) = 0;

    volatile tt_l1_ptr uint32_t* const result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);
    result[0] = 0;
    result[1] = 0;

    ex::tt_uva_ini(0, cfg_addr, page_size, 0, 0, landing_addr, l1_base, l1_size);

    uint32_t seen = 0;
    uint32_t bad = 0;
    while (seen < expect) {
        // seen+1, so a true return means one more frame landed -- and it is the one sitting in
        // the landing buffer. Requires every frame to be signalled, which put_signal does.
        if (!ex::tt_uva_test(sig_addr, seen + 1)) {
            continue;
        }
        ++seen;
        if constexpr (verify != 0) {
            const auto* t =
                reinterpret_cast<const volatile ex::FrameTrailer*>(landing_addr + page_size - ex::kFrameTrailerBytes);
            const uint32_t b = 0x40u + (ex::tt_uva_t6_selector_core(t->origin) & 0x1Fu);
            const uint32_t want = b | (b << 8) | (b << 16) | (b << 24);
            const auto* p = reinterpret_cast<const volatile uint32_t*>(landing_addr);
            // Word 0 is the sender's per-frame stamp. Frames from one origin arrive in
            // order, so anything but seen-1 is a stale slot or a duplicated page.
            if (p[0] != seen - 1) {
                ++bad;
                continue;
            }
            for (uint32_t k = 1; k < payload_bytes / sizeof(uint32_t); ++k) {
                if (p[k] != want) {
                    ++bad;
                    break;
                }
            }
        }
    }

    result[0] = bad;
    result[1] = seen;
    ex::tt_uva_fin();
}
