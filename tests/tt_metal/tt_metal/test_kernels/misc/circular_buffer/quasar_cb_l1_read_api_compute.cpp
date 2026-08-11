// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "dev_mem_map.h"

// Minimal exercise of ckernel::read_tile_value / ckernel::get_tile_address (cb_api.h) on Quasar.
//
// Both APIs are implemented as a cross-thread mailbox handshake: UNPACK computes the value and
// mailbox_write()s it into MATH's and PACK's slots, which each mailbox_read() it back. Two
// consequences shape this kernel:
//   1. All three threads MUST call the APIs unconditionally. Gating the call site behind
//      #ifdef TRISC_UNPACK (as the earlier version of this test did) means MATH/PACK never execute
//      their mailbox_read() halves, so UNPACK's writes are never drained and the handshake this
//      test exists to cover is not exercised at all.
//   2. Each thread records what IT observed into its own slice of the result buffer, so the host
//      can verify MATH and PACK actually received UNPACK's broadcast -- checking UNPACK's copy
//      alone would pass even if the mailbox delivered nothing.
// ISOLATE_SFPU (TRISC3) also runs this kernel, but participates in neither half of the handshake
// (every UNPACK/MATH/PACK block is empty for it), so it keeps RESULT_SLOT_NONE and writes nothing.

constexpr std::uint32_t VALUES_PER_THREAD = 5;
constexpr int RESULT_SLOT_NONE = -1;

void kernel_main() {
    const std::uint32_t buf_id = get_compile_time_arg_val(0);

    // Reading tile_index 1 exercises the per-tile stride term that is always zero for tile 0.
    const std::uint32_t v0 = read_tile_value(buf_id, 0 /*tile_index*/, 0 /*element_offset*/);
    const std::uint32_t v1 = read_tile_value(buf_id, 0 /*tile_index*/, 1 /*element_offset*/);
    const std::uint32_t v2 = read_tile_value(buf_id, 1 /*tile_index*/, 0 /*element_offset*/);
    const std::uint32_t v3 = read_tile_value(buf_id, 1 /*tile_index*/, 1 /*element_offset*/);
    // get_tile_address must resolve to the same tile read_tile_value reads, and must be a usable
    // L1 address on the receiving threads too -- not merely a matching integer.
    const std::uint32_t tile1_addr = get_tile_address(buf_id, 1 /*tile_index*/);

    int slot = RESULT_SLOT_NONE;
    UNPACK(slot = 0;)
    MATH(slot = 1;)
    PACK(slot = 2;)

    if (slot != RESULT_SLOT_NONE) {
        const std::uint32_t result_l1_addr = get_arg_val<std::uint32_t>(0);
        volatile tt_l1_ptr std::uint32_t* const result =
            reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(result_l1_addr) + slot * VALUES_PER_THREAD;
        result[0] = v0;
        result[1] = v1;
        result[2] = v2;
        result[3] = v3;
        result[4] = *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(tile1_addr);
    }
}
