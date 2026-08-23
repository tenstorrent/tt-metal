// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: the DRAM overload of Noc::async_write_zeros sourcing its zeros from a Scratchpad.
//
// Before the raw-L1 widening, overload (2)'s `scratch` argument had to be a CircularBuffer or
// DataflowBuffer, so a kernel with only a scratchpad could not zero DRAM at all -- and on Quasar a DM
// kernel cannot self-loop a DFB, so there was no single-kernel way to do this. This kernel is the
// end-to-end proof of the widened form, and contains no CB/DFB of any kind.
//
// It also exercises the contract simplification: for a CB/DFB, overload (1) writes the WRITE pointer
// while overload (2) reads the READ pointer, which is why the CB/DFB flow needs a
// push_back/wait_front handoff. A Scratchpad has one address, so "zero it, barrier, pass the same
// handle" is all that is required.
//
//   1. Zero the whole scratchpad via overload (1) + write_zeros_l1_barrier().
//   2. Loop overload (2) over the DRAM tensor's pages, using that scratchpad as the zeros source.
//   3. write_zeros_dram_barrier(), then report status.
//
// The host stamps the DRAM tensor with 0xFFFFFFFF beforehand and checks it reads back all zeros, so a
// kernel that silently did nothing cannot pass.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

namespace {
constexpr uint32_t kStatusOk = 0xCAFEBABEu;
constexpr uint32_t kStatusScratchNotZero = 0xDEAD0011u;

// Held under a scoped_lock like every other access in this kernel: once scoped_lock does automatic
// cache management, its release publishes the word and the manual flush below goes away.
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

    Scratchpad<uint32_t> pad(scratch::pad);
    const auto out = TensorAccessor(tensor::out);
    Noc noc;

    uint32_t status = kStatusOk;

    // ---- 1. Pre-zero the scratchpad (overload 1) ----
    {
        const auto zero_lock = pad.local_mem().scoped_lock(pad.size());
        noc.async_write_zeros(pad, pad.size_in_bytes());
        noc.write_zeros_l1_barrier();
    }

    // Confirm the L1 scratch really is zero before streaming it outward, so a DRAM-all-zeros result
    // cannot be credited to a scratch that was never zeroed in the first place.
    {
        const auto verify_lock = pad.local_mem().scoped_lock(pad.size());
        CoreLocalMem<volatile uint32_t> mem(static_cast<uintptr_t>(pad.get_base_address()));
        const uint32_t num_words = pad.size();
        for (uint32_t i = 0; i < num_words; ++i) {
            if (mem[i] != 0u) {
                status = kStatusScratchNotZero;
                break;
            }
        }
    }

    // ---- 2. Stream those zeros to every DRAM page (overload 2, Scratchpad as scratch) ----
    if (status == kStatusOk) {
        for (uint32_t p = page_start; p < page_end; ++p) {
            noc.async_write_zeros(out, page_size, {.page_id = p}, pad);
        }
        noc.write_zeros_dram_barrier();
    }

    report(static_cast<uintptr_t>(flag_addr), status);
}
