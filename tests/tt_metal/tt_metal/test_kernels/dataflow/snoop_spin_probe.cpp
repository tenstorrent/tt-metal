// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Direct probe of the host-write snoop bit, independent of DFB (issue #55788).
//
//   1. clear `data` in SRAM through the UNCACHED alias, so the starting state is unambiguous
//   2. read it through the CACHED view, pulling the line into this core's cache holding zero
//   3. publish a ready flag UNCACHED so the host's NOC read sees it immediately
//   4. spin on the CACHED view until the host's write becomes visible, or the host aborts
//   5. publish what was seen, uncached
//
// Without snoop the host's write lands in L1 SRAM while this core keeps reading its own cached
// zero, so only the abort ends the spin and the result is 0. With snoop the cached line is acted on
// and the written value appears.
//
// data/flag/result/abort are spaced a cache line apart so the uncached publishes cannot disturb
// the line under test.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // The host requests one thread; gate anyway so exactly one core runs the probe.
    if (get_my_thread_id() != 0) {
        return;
    }

    constexpr std::uint32_t data_addr = get_arg(args::data_addr);
    constexpr std::uint32_t flag_addr = get_arg(args::flag_addr);
    constexpr std::uint32_t result_addr = get_arg(args::result_addr);
    constexpr std::uint32_t abort_addr = get_arg(args::abort_addr);

    constexpr std::uint32_t kUncached = MEM_L1_UNCACHED_BASE - MEM_L1_BASE;

    volatile tt_l1_ptr std::uint32_t* const data_cached =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(data_addr);
    volatile tt_l1_ptr std::uint32_t* const data_unc =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(data_addr + kUncached);
    volatile tt_l1_ptr std::uint32_t* const flag_unc =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(flag_addr + kUncached);
    volatile tt_l1_ptr std::uint32_t* const result_unc =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(result_addr + kUncached);
    // Read UNCACHED, so the host can always end the spin regardless of the cached view. Making
    // termination independent of the value under test is what keeps a stale read from hanging the
    // emulator: an iteration-bounded spin is unusable there (fine on silicon, forever on the emulator).
    volatile tt_l1_ptr std::uint32_t* const abort_unc =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(abort_addr + kUncached);

    *result_unc = 0xFFFFFFFFu;  // sentinel: kernel did not reach the end
    *data_unc = 0u;

    (void)*data_cached;  // cache the line, holding zero
    __asm__ __volatile__("fence" ::: "memory");

    *flag_unc = 1u;  // host may write now

    std::uint32_t seen = 0u;
    while (true) {
        seen = *data_cached;  // the thing under test: cached view
        if (seen != 0u) {
            break;
        }
        if (*abort_unc != 0u) {  // always visible: guarantees termination
            // Re-read before reporting. The host write can land between the data read above and
            // this abort read; without this the loop would exit reporting 0 even though the value
            // had arrived.
            seen = *data_cached;
            break;
        }
    }
    *result_unc = seen;
}
