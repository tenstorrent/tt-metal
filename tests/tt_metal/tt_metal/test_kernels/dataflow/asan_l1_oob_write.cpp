// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ASan negative test kernel: writes 1 byte one past the end of a small L1
// buffer.  The runner's regex rewriter turns
//     reinterpret_cast<volatile uint8_t*>(get_arg_val<uint32_t>(0))
// into
//     reinterpret_cast<volatile uint8_t*>(
//         (uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>(0)))
// so the kernel ends up writing into the host-side L1 mirror.  Adding `size`
// bytes lands in the poisoned guard region above the buffer; the store traps
// in ASan and the JIT'd .so aborts.
//
// Runtime args:
//   0: L1 buffer base address (uint32_t)
//   1: L1 buffer size in bytes (uint32_t)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    std::uint32_t size = get_arg_val<uint32_t>(1);

    // Single inline reinterpret_cast<...>(get_arg_val<uint32_t>(0)) so the
    // host-build regex rewriter inserts __emule_local_l1_to_ptr(...).
    auto* base = reinterpret_cast<volatile uint8_t*>(get_arg_val<uint32_t>(0));
    base[size] = 0x42;  // 1 byte past the buffer end; poisoned -> ASan trap.
}
