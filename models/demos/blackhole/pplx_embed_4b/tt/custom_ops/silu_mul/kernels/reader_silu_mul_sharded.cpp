// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// silu(a)*b reader, block-sharded a / b: CBs 0 / 1 are this core's shards, so the reader only publishes them.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);
    CircularBuffer ca(0), cbb(1);
    ca.reserve_back(n_tiles);
    ca.push_back(n_tiles);
    cbb.reserve_back(n_tiles);
    cbb.push_back(n_tiles);
}
