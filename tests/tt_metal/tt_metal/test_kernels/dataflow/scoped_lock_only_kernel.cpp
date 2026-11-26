// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t lock_addr = get_arg(args::lock_addr);
    uint32_t lock_num_elements = get_arg(args::lock_num_elements);

    CoreLocalMem<uint32_t> buffer(lock_addr);

    {
        [[maybe_unused]] auto lock = buffer.scoped_lock(lock_num_elements);
    }
}
