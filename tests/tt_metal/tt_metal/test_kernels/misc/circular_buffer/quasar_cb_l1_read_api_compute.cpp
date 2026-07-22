// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "dev_mem_map.h"

#include <cstdint>

// Unit test for ckernel::get_tile_address / ckernel::read_tile_value on Quasar.
// The host preloads two known tiles into the DFB ring; the UNPACK thread reads them
// back via both APIs. Reading tile_index 1 exercises the per-tile stride term that is
// always zero for tile_index 0.
void kernel_main() {
    const uint32_t buf_id = get_compile_time_arg_val(0);

#ifdef TRISC_UNPACK
    const uint32_t result_l1_addr = get_arg_val<uint32_t>(0);

    const uint32_t v0 = read_tile_value(buf_id, 0, 0);
    const uint32_t v1 = read_tile_value(buf_id, 0, 1);
    const uint32_t v2 = read_tile_value(buf_id, 1, 0);
    const uint32_t v3 = read_tile_value(buf_id, 1, 1);
    // get_tile_address must resolve to the same tile read_tile_value reads.
    const uint32_t v4 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(buf_id, 1));

    volatile tt_l1_ptr uint32_t* const result =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr + MEM_L1_UNCACHED_BASE);
    result[0] = v0;
    result[1] = v1;
    result[2] = v2;
    result[3] = v3;
    result[4] = v4;
#endif
}
