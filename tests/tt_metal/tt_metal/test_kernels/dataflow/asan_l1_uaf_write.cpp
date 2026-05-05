// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ASan negative test kernel: writes to the first byte of an L1 address that
// has already been freed host-side (Buffer destroyed before EnqueueProgram).
// AllocatorImpl::deallocate_buffer reaches __emule_buffer_free, which
// repoisons the freed range.  The runner's regex rewriter turns the
//     reinterpret_cast<volatile uint8_t*>(get_arg_val<uint32_t>(0))
// expression into a call to __emule_local_l1_to_ptr(...) so we land in the
// host-side L1 mirror; the store below then traps in ASan.
//
// Runtime args:
//   0: captured L1 address of the freed buffer (uint32_t)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    auto* freed = reinterpret_cast<volatile uint8_t*>(get_arg_val<uint32_t>(0));
    freed[0] = 0x42;
}
