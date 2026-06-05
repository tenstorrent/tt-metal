// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Compute kernel for QuasarCbL1ReadApi: exercises cb_api.h read_tile_value and get_tile_address on Quasar.
 * Reads the page the dataflow reader pushed, then writes three uint32_t checks to the host L1 result buffer.
 */

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"

#include <cstdint>

void kernel_main() {
    const uint32_t buf_id = get_compile_time_arg_val(0);

    DataflowBuffer cb(buf_id);
    cb.wait_front(1);

    const uint32_t v0 = read_tile_value(buf_id, 0, 0);
    const uint32_t v1 = read_tile_value(buf_id, 0, 1);
    const uint32_t addr = get_tile_address(buf_id, 0);
    const uint32_t v0_direct = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);

#ifdef TRISC_UNPACK
    uint32_t* const result = get_arg_val<uint32_t*>(0);
    result[0] = v0;
    result[1] = v1;
    result[2] = v0_direct;
#endif

    cb.pop_front(1);
}
