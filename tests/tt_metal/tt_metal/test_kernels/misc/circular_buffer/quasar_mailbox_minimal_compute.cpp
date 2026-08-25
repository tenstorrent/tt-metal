// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "dev_mem_map.h"

// Canonical mailbox usage for synchronizing threads with one another: each thread only ever
// touches a slot other than its own. (Loopback -- a thread using its OWN slot -- is legal
// hardware behavior, but must be avoided for inter-thread synchronization; an accidental
// self-loopback is what tripped the Watcher IB-interrupt fault, 0x19.) TRISC0 (UNPACK) writes
// into TRISC1's (MathThreadId) mailbox, and TRISC1 (MATH) reads TRISC0's (UnpackThreadId)
// mailbox.
void kernel_main() {
    constexpr std::uint32_t kValue = 0xfaceface;

    UNPACK({ ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, kValue); })

    MATH({
        const std::uint32_t result_l1_addr = get_arg_val<std::uint32_t>(0);
        const std::uint32_t v = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        volatile tt_l1_ptr std::uint32_t* const result =
            reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(result_l1_addr);
        result[0] = v;
    })
}
