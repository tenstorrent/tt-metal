// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t buffer_addr_a = get_arg(args::buffer_addr_a);
    uint32_t num_elements_a = get_arg(args::num_elements_a);
    uint32_t buffer_addr_b = get_arg(args::buffer_addr_b);
    uint32_t num_elements_b = get_arg(args::num_elements_b);
    uint32_t writer_noc_x = get_arg(args::writer_noc_x);
    uint32_t writer_noc_y = get_arg(args::writer_noc_y);

    Semaphore lock_state_changed(sem::lock_state_changed);  // the writer waits on its own instance; we remote-up it
    Semaphore written(sem::written);                        // we wait on ours; the writer remote-ups it
    Noc noc;
    CoreLocalMem<uint32_t> buf_a(buffer_addr_a);
    CoreLocalMem<uint32_t> buf_b(buffer_addr_b);

    {
        auto lock_a = buf_a.scoped_lock(num_elements_a);
        auto lock_b = buf_b.scoped_lock(num_elements_b);
        lock_state_changed.up(noc, writer_noc_x, writer_noc_y, 1);

        written.down(1);
    }

    lock_state_changed.up(noc, writer_noc_x, writer_noc_y, 1);
}
