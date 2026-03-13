// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    uint32_t lock_addr = get_arg_val<uint32_t>(0);
    uint32_t lock_num_elements = get_arg_val<uint32_t>(1);

    experimental::CoreLocalMem<uint32_t> buffer(lock_addr);

    {
        [[maybe_unused]] auto lock = buffer.scoped_lock(lock_num_elements);
    }
}
