// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Dataflow kernel for QuasarCbL1ReadApi.
 * Writes two known uint32_t values into the buffer; compute reads them via read_tile_value / get_tile_address.
 */

#include "api/dataflow/dataflow_buffer.h"

#include <cstdint>

namespace {
constexpr uint32_t VAL0 = 0xA5A5A5A5u;  // First uint32 in the page; host expects this at result[0].
constexpr uint32_t VAL1 = 0x11111111u;  // Second uint32 in the page; host expects this at result[1].
}  // namespace

void kernel_main() {
    const uint32_t buf_id = get_compile_time_arg_val(0);

    DataflowBuffer cb(buf_id);
    cb.reserve_back(1);
    auto* ptr = reinterpret_cast<volatile uint32_t*>(cb.get_write_ptr());
    ptr[0] = VAL0;
    ptr[1] = VAL1;
    cb.push_back(1);
}
