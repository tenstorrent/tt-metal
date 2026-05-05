// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ASan positive-control kernel: writes 1 byte INSIDE a freshly-allocated L1
// buffer. With the per-buffer alloc hook running, the byte at offset 0 must
// be unpoisoned and the store must succeed without firing ASan. If this
// kernel trips ASan, the alloc hook isn't actually unpoisoning the buffer's
// region — meaning OOB tests are firing on the initial blanket poison
// rather than per-buffer poisoning.
//
// Runtime args:
//   0: L1 buffer base address (uint32_t)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Single inline reinterpret_cast<...>(get_arg_val<uint32_t>(0)) so the
    // host-build regex rewriter inserts __emule_local_l1_to_ptr(...).
    auto* base = reinterpret_cast<volatile uint8_t*>(get_arg_val<uint32_t>(0));
    base[0] = 0x42;  // inside the buffer; must NOT trip ASan.
}
