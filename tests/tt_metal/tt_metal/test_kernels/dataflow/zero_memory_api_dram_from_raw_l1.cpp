// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: the DRAM overload of Noc::async_write_zeros sourcing its zeros from a raw L1 handle.
//
// One kernel body, the scratch handle selected by the ZERO_TARGET define (same names as
// zero_memory_api_raw_l1.cpp), so all three raw-L1 types are exercised as the `scratch` argument:
//
//   ZERO_TARGET_CORE_LOCAL_MEM        CoreLocalMem<volatile uint32_t> over a host-known L1 address
//   ZERO_TARGET_SCRATCHPAD            Scratchpad<uint32_t> from a scratchpad binding token
//   ZERO_TARGET_LOCAL_TENSOR_ACCESSOR LocalTensorAccessor<uint32_t> from a tensor binding token
//
//   1. CPU-stamp the scratch region non-zero and verify the stamp.
//   2. Zero it via overload (1) + write_zeros_l1_barrier().
//   3. CPU-verify it is now all zero.
//   4. Loop overload (2) over the DRAM tensor's pages using that region as the zeros source, then
//      write_zeros_dram_barrier() and report status.
//
// Step 1 is what makes steps 2-3 meaningful: a freshly allocated Scratchpad or L1 tensor may already
// be zero, so without stamping first the all-zero check would pass even if overload (1) did nothing.
// The host separately stamps the DRAM tensor with 0xFFFFFFFF and checks it reads back all zeros.
//
// TODO: remove the QUASAR_MANUAL_CACHE_MAINTENANCE blocks once CoreLocalMem::scoped_lock does
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
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {
constexpr uint32_t kStatusOk = 0xCAFEBABEu;
constexpr uint32_t kStatusStampFail = 0xDEAD0001u;
constexpr uint32_t kStatusScratchNotZero = 0xDEAD0011u;

constexpr uint32_t kPatternBase = 0xA5A50000u;

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
    const uint32_t page_start = get_arg(args::page_start);
    const uint32_t page_end = get_arg(args::page_end);
    const uint32_t page_size = get_arg(args::page_size);
    const uint32_t flag_addr = get_arg(args::flag_addr);
    const uint32_t region_bytes = get_arg(args::region_bytes);

    // ---- Construct the scratch handle under test ----
#if defined(ZERO_TARGET_CORE_LOCAL_MEM)
    const uint32_t region_addr = get_arg(args::region_addr);
    CoreLocalMem<volatile uint32_t> scratch(static_cast<uintptr_t>(region_addr));
#elif defined(ZERO_TARGET_SCRATCHPAD)
    Scratchpad<uint32_t> scratch(scratch::pad);
    const uint32_t region_addr = scratch.get_base_address();
#elif defined(ZERO_TARGET_LOCAL_TENSOR_ACCESSOR)
    LocalTensorAccessor<uint32_t> scratch(tensor::region);
    const uint32_t region_addr = scratch.get_bank_base_address();
#else
#error "define exactly one ZERO_TARGET_*"
#endif

    const auto out = TensorAccessor(tensor::out);
    CoreLocalMem<volatile uint32_t> mem(static_cast<uintptr_t>(region_addr));
    const uint32_t num_words = region_bytes / sizeof(uint32_t);

    Noc noc;
    uint32_t status = kStatusOk;

    // ---- 1. Stamp the scratch region non-zero, so the all-zero check below proves overload (1)
    // actually ran. A freshly allocated Scratchpad or L1 tensor may already be zero, in which case
    // skipping this would let the test pass even if the zero did nothing. ----
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

    // ---- 2. Zero the whole scratch region (overload 1) ----
    if (status == kStatusOk) {
        const auto zero_lock = mem.scoped_lock(num_words);
        noc.async_write_zeros(scratch, region_bytes);
        noc.write_zeros_l1_barrier();
    }

    // ---- 3. Verify the L1 scratch really is zero before streaming it outward. ----
    if (status == kStatusOk) {
        const auto verify_lock = mem.scoped_lock(num_words);
#ifdef ARCH_QUASAR  // QUASAR_MANUAL_CACHE_MAINTENANCE
        invalidate_l2_cache_range(static_cast<uintptr_t>(region_addr), region_bytes);
#endif
        for (uint32_t i = 0; i < num_words; ++i) {
            if (mem[i] != 0u) {
                status = kStatusScratchNotZero;
                break;
            }
        }
    }

    // ---- 4. Stream those zeros to every DRAM page (overload 2, raw L1 handle as scratch) ----
    if (status == kStatusOk) {
        for (uint32_t p = page_start; p < page_end; ++p) {
            noc.async_write_zeros(out, page_size, {.page_id = p}, scratch);
        }
        noc.write_zeros_dram_barrier();
    }

    report(static_cast<uintptr_t>(flag_addr), status);
}
