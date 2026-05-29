// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sequence:
//   1. Reserve 1 entry on dfb::scratch.
//   2. Stamp every byte with 0xFF; verify it landed (catches infra bugs that
//      would otherwise let a no-op kernel look like a pass).
//   3. noc.async_write_zeros(dfb, total_bytes) + write_zeros_l1_barrier().
//   4. Verify every byte is now 0x00.
//   5. Write pass/fail status to flag_addr, then push_back the (now zero-filled)
//      DFB entry for the consumer to use as DRAM scratch.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {
constexpr uint32_t kStatusOk = 0xCAFEBABEu;
constexpr uint32_t kStatusStampFail = 0xDEAD0001u;
constexpr uint32_t kStatusZeroFail = 0xDEAD0002u;
}  // namespace

// Write the status word and flush the L2 cache (on Quasar) before returning, so the
// host's ReadFromDeviceL1 observes the most recent write. The host pre-stamps the flag
// with a sentinel (0xBAADF00D) before dispatch, so an early crash leaves the sentinel
// in place and the host correctly reports failure — no optimistic stamp needed here.
FORCE_INLINE void report(uintptr_t flag_addr, uint32_t status) {
    CoreLocalMem<volatile uint32_t> flag(flag_addr);
    flag[0] = status;
#ifdef ARCH_QUASAR
    flush_l2_cache_line(flag_addr);
#endif
}

void kernel_main() {
    const uint32_t total_bytes = get_arg(args::total_bytes);
    const uint32_t flag_addr = get_arg(args::flag_addr);

    DataflowBuffer dfb(dfb::scratch);
    dfb.reserve_back(1);
    CoreLocalMem<volatile uint32_t> buf(dfb.get_write_ptr());
    const uint32_t num_words = total_bytes / sizeof(uint32_t);

    // Stamp 0xFF into every byte and verify it landed before the zero-API call.
    for (uint32_t i = 0; i < num_words; ++i) {
        buf[i] = 0xFFFFFFFFu;
    }
    for (uint32_t i = 0; i < num_words; ++i) {
        if (buf[i] != 0xFFFFFFFFu) {
            report(flag_addr, kStatusStampFail);
            dfb.push_back(1);
            return;
        }
    }

    Noc noc;
    noc.async_write_zeros(dfb, total_bytes);
    noc.write_zeros_l1_barrier();

    // Verify every byte is now 0.
    for (uint32_t i = 0; i < num_words; ++i) {
        if (buf[i] != 0u) {
            report(flag_addr, kStatusZeroFail);
            dfb.push_back(1);
            return;
        }
    }

    dfb.push_back(1);
    report(flag_addr, kStatusOk);
}
