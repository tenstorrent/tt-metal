// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Evidence for the one ordering claim fused/model makes across program boundaries (PLAN.md section 6): an L1 range
// that program k reads on this chip is written by fabric arrivals of program k + 1 only after its last reader in
// program k has finished. This kernel sits on a core of its own inside program k and changes nothing: it polls
// (bounded) a word of this chip that says "the last reader has finished", then reads the arrival counters program
// k + 1's senders count into on this chip. Any of them non-zero at that moment means data of k + 1 landed while k
// could still have been reading: counted in the canary words, which the host reads after the run.
//
// Data movement only: NoC reads inside this chip, no circular buffer, no semaphore, no fabric, no unbounded wait.
//
// Runtime args:
//   0  NoC x | y << 16 of the core that holds the "finished" word
//   1  address of that word
//   2  mask: finished when (word & mask) == mask
//   3  polls at most
//   4  address of this core's canary words (8 uint32): 0 checks done, 1 checks that found an arrival, 2 the counters
//      seen at the last check that found one (sum), 3 polls that ran out, 4 polls of the last check, 5 runs
//   5  address of a 32-byte scratch on this core, 16-byte aligned
//   6  number of counters n, then n x (NoC x | y << 16, address)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
// The 16-byte line of another core's L1 that holds `address`, into `scratch`; returns the word.
FORCE_INLINE uint32_t peek(uint32_t core, uint32_t address, uint32_t scratch) {
    const uint32_t line = address & ~15u;
    noc_async_read(get_noc_addr(core & 0xFFFF, core >> 16, line), scratch, 16);
    noc_async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[(address - line) >> 2];
}
}  // namespace

void kernel_main() {
    const uint32_t done_core = get_arg_val<uint32_t>(0);
    const uint32_t done_address = get_arg_val<uint32_t>(1);
    const uint32_t mask = get_arg_val<uint32_t>(2);
    const uint32_t limit = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    const uint32_t scratch = get_arg_val<uint32_t>(5);
    const uint32_t n = get_arg_val<uint32_t>(6);

    out[5] = out[5] + 1;
    uint32_t polls = 0;
    bool finished = false;
    for (; polls < limit; ++polls) {
        if ((peek(done_core, done_address, scratch) & mask) == mask) {
            finished = true;
            break;
        }
    }
    out[4] = polls;
    if (!finished) {
        out[3] = out[3] + 1;
        return;
    }
    uint32_t seen = 0;
    for (uint32_t i = 0; i < n; ++i) {
        seen += peek(get_arg_val<uint32_t>(7 + 2 * i), get_arg_val<uint32_t>(8 + 2 * i), scratch);
    }
    out[0] = out[0] + 1;
    if (seen != 0) {
        out[1] = out[1] + 1;
        out[2] = seen;
    }
}
