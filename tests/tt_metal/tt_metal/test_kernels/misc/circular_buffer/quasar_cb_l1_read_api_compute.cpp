// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "dev_mem_map.h"

#include <cstdint>

// Unit test for ckernel::get_tile_address / ckernel::read_tile_value on Quasar.
// No producer/consumer data movement is involved: the UNPACK thread seeds a known
// tile in-place (using the address get_tile_address returns) and reads it back via
// read_tile_value. This exercises the DFB address math + L1 read in cb_api.h directly.
void kernel_main() {
    const uint32_t buf_id = get_compile_time_arg_val(0);

#ifdef TRISC_UNPACK
    const uint32_t result_l1_addr = get_arg_val<uint32_t>(0);
    const uint32_t val0 = get_arg_val<uint32_t>(1);
    const uint32_t val1 = get_arg_val<uint32_t>(2);

    const uint32_t tile_addr = get_tile_address(buf_id, 0);
    volatile tt_l1_ptr uint32_t* const tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
    tile[0] = val0;
    tile[1] = val1;

    const uint32_t v0 = read_tile_value(buf_id, 0, 0);
    const uint32_t v1 = read_tile_value(buf_id, 0, 1);

    volatile tt_l1_ptr uint32_t* const result =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr + MEM_L1_UNCACHED_BASE);
    result[0] = v0;
    result[1] = v1;
    result[2] = tile[0];
#endif
}
