// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t l1_buffer_addr = get_arg(args::l1_buffer_addr);
    uint32_t num_elements = get_arg(args::num_elements);
    uint32_t writer_noc_x = get_arg(args::writer_noc_x);
    uint32_t writer_noc_y = get_arg(args::writer_noc_y);

    Semaphore lock_state_changed(sem::lock_state_changed);  // the writer waits on its own instance; we remote-up it
    Semaphore written(sem::written);                        // we wait on ours; the writer remote-ups it
    Noc noc;
    CoreLocalMem<uint32_t> buffer(l1_buffer_addr);

    {
        // Lock this buffer for num_elements
        auto lock = buffer.scoped_lock(num_elements);
        lock_state_changed.up(noc, writer_noc_x, writer_noc_y, 1);

        written.down(1);
    }

    // Unlocked
    lock_state_changed.up(noc, writer_noc_x, writer_noc_y, 1);
}
