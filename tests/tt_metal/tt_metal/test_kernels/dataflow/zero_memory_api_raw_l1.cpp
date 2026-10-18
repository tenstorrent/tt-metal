// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Noc::async_write_zeros' local-L1 overload on the raw L1 handles.
//
// One kernel body, one handle type selected by the ZERO_TARGET define, so the verification logic is
// byte-identical across types and any difference in outcome is attributable to the handle:
//
//   ZERO_TARGET_CORE_LOCAL_MEM        CoreLocalMem<volatile uint32_t> over a host-known L1 address
//   ZERO_TARGET_SCRATCHPAD            Scratchpad<uint32_t> from a scratchpad binding token
//   ZERO_TARGET_LOCAL_TENSOR_ACCESSOR LocalTensorAccessor<uint32_t> from a tensor binding token
//
// Sequence (distinct status word per invariant, so the host names the broken one):
//   1. CPU-stamp the WHOLE region with a per-word pattern, then read it back. A per-word pattern
//      (not a constant) means a shifted or duplicated window is detectable.
//   2. Zero only a SUB-WINDOW [offset_bytes, offset_bytes + window_bytes).
//   3. Verify the window is zero AND every word outside it still holds its stamp. That flank
//      invariance is what catches an ignored offset, a mis-scaled offset, and a run-long zero --
//      a zero-the-whole-region bug passes a window-only check but fails this.
//
// Each phase is bracketed in its own scoped_lock, mirroring how zero_memory_api_l1_producer.cpp uses
// dfb.scoped_write_lock / scoped_read_lock.
//
// TODO: remove the QUASAR_MANUAL_CACHE_MAINTENANCE blocks below once CoreLocalMem::scoped_lock does
// automatic cache management.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#if defined(ZERO_TARGET_SCRATCHPAD)
#include "api/scratchpad.h"
#elif defined(ZERO_TARGET_LOCAL_TENSOR_ACCESSOR)
#include "api/tensor/local_tensor_accessor.h"
#endif
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {
constexpr uint32_t kStatusOk = 0xCAFEBABEu;
constexpr uint32_t kStatusStampFail = 0xDEAD0001u;
constexpr uint32_t kStatusZeroFail = 0xDEAD0002u;
constexpr uint32_t kStatusFlankFail = 0xDEAD0003u;

constexpr uint32_t kPatternBase = 0xA5A50000u;

// Write the status word so the host's read observes it. The status word lives outside the
// region under test, so neither the lock nor the flush can perturb the zeroed window.
FORCE_INLINE void report(uintptr_t flag_addr, uint32_t status) {
    CoreLocalMem<volatile uint32_t> flag(flag_addr);
    const auto flag_lock = flag.scoped_lock(1);
    flag[0] = status;
#ifdef ARCH_QUASAR  // QUASAR_MANUAL_CACHE_MAINTENANCE
    flush_l2_cache_line(flag_addr);
#endif
}
}  // namespace

void kernel_main() {
    const uint32_t region_bytes = get_arg(args::region_bytes);
    const uint32_t offset_bytes = get_arg(args::offset_bytes);
    const uint32_t window_bytes = get_arg(args::window_bytes);
    const uint32_t flag_addr = get_arg(args::flag_addr);
    const uint32_t report_addr = get_arg(args::report_addr);

    // ---- Construct the handle under test, and a uniform volatile CPU view of the same region ----
#if defined(ZERO_TARGET_CORE_LOCAL_MEM)
    const uint32_t region_addr = get_arg(args::region_addr);
    CoreLocalMem<volatile uint32_t> handle(static_cast<uintptr_t>(region_addr));
#elif defined(ZERO_TARGET_SCRATCHPAD)
    Scratchpad<uint32_t> handle(scratch::pad);
    const uint32_t region_addr = handle.get_base_address();
#elif defined(ZERO_TARGET_LOCAL_TENSOR_ACCESSOR)
    LocalTensorAccessor<uint32_t> handle(tensor::region);
    const uint32_t region_addr = handle.get_bank_base_address();
#else
#error "define exactly one ZERO_TARGET_*"
#endif

    // Report where the region actually is, so the host can read it back independently of the kernel.
    report(static_cast<uintptr_t>(report_addr), region_addr);

    CoreLocalMem<volatile uint32_t> mem(static_cast<uintptr_t>(region_addr));
    const uint32_t num_words = region_bytes / sizeof(uint32_t);
    const uint32_t window_first = offset_bytes / sizeof(uint32_t);
    const uint32_t window_last = (offset_bytes + window_bytes) / sizeof(uint32_t);  // exclusive

    Noc noc;
    uint32_t status = kStatusOk;

    // ---- 1. Stamp the whole region and verify the stamp landed ----
    {
        const auto stamp_lock = mem.scoped_lock(num_words);
        for (uint32_t i = 0; i < num_words; ++i) {
            mem[i] = kPatternBase + i;
        }
        for (uint32_t i = 0; i < num_words; ++i) {
            if (mem[i] != kPatternBase + i) {
                status = kStatusStampFail;
                break;
            }
        }
#ifdef ARCH_QUASAR  // QUASAR_MANUAL_CACHE_MAINTENANCE
        flush_l2_cache_range(static_cast<uintptr_t>(region_addr), region_bytes);
#endif
    }

    // ---- 2. Zero only the sub-window ----
    if (status == kStatusOk) {
        const auto zero_lock = mem.scoped_lock(num_words);
#ifdef ZERO_NUM_CHUNKS
        // ZERO_NUM_CHUNKS disjoint zeros covering the same window, then ONE barrier.
        const uint32_t chunk_bytes = window_bytes / ZERO_NUM_CHUNKS;  // host picks an exact divisor
        for (uint32_t c = 0; c < ZERO_NUM_CHUNKS; ++c) {
            noc.async_write_zeros(handle, chunk_bytes, {.offset_bytes = offset_bytes + c * chunk_bytes});
        }
        noc.write_zeros_l1_barrier();
#else
        noc.async_write_zeros(handle, window_bytes, {.offset_bytes = offset_bytes});
        noc.write_zeros_l1_barrier();
#endif
    }

    // ---- 3. Window must be zero; everything outside it must still hold its stamp ----
    if (status == kStatusOk) {
        const auto verify_lock = mem.scoped_lock(num_words);
#ifdef ARCH_QUASAR  // QUASAR_MANUAL_CACHE_MAINTENANCE
        invalidate_l2_cache_range(static_cast<uintptr_t>(region_addr), region_bytes);
#endif
        for (uint32_t i = window_first; i < window_last; ++i) {
            if (mem[i] != 0u) {
                status = kStatusZeroFail;
                break;
            }
        }
        if (status == kStatusOk) {
            for (uint32_t i = 0; i < num_words; ++i) {
                if (i >= window_first && i < window_last) {
                    continue;
                }
                if (mem[i] != kPatternBase + i) {
                    status = kStatusFlankFail;
                    break;
                }
            }
        }
    }

    report(static_cast<uintptr_t>(flag_addr), status);
}
