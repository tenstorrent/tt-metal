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
    const uint32_t buf_addr = dfb.get_write_ptr();
    CoreLocalMem<volatile uint32_t> buf(buf_addr);
    const uint32_t num_words = total_bytes / sizeof(uint32_t);

    // Stamp 0xFF into every byte and verify it landed before the zero-API call.
    {
        auto lock = buf.scoped_lock(num_words);
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
    }

    // TODO(buffer-unlock-eviction): unlock should evict the buffer's dirty lines from
    // this core's cache. Until that is implemented, evict explicitly here so the iDMA
    // zero (which writes TL1, bypassing the cache) is not shadowed by the stale 0xFF
    // stamp on the readback below. Remove this loop once scoped_lock release evicts.
#ifdef ARCH_QUASAR
    for (uint32_t off = 0; off < total_bytes; off += L2_CACHE_LINE_SIZE) {
        invalidate_l2_cache_line(buf_addr + off);
    }
#endif

    Noc noc;
#ifdef ZERO_NUM_CHUNKS
    // Issue ZERO_NUM_CHUNKS disjoint async_write_zeros into this one entry, then barrier
    // ONCE.
    const uint32_t chunk_bytes = total_bytes / ZERO_NUM_CHUNKS;  // host picks an exact divisor
    for (uint32_t c = 0; c < ZERO_NUM_CHUNKS; ++c) {
        noc.async_write_zeros(dfb, chunk_bytes, {.offset_bytes = c * chunk_bytes});
    }
    noc.write_zeros_l1_barrier();
#else
    noc.async_write_zeros(dfb, total_bytes);
    noc.write_zeros_l1_barrier();
#endif

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
