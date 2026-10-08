// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t pages = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t bytes = 1088;
    const uint32_t x = get_arg_val<uint32_t>(0);
    const uint32_t y = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t blocks = (count + pages - 1) / pages;
    const uint64_t ready = get_noc_addr(x, y, get_semaphore(0));
    auto* consumed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t slot = block % depth;
        // A receive slot is reusable only after its previous consumer has
        // inspected/copied it. Source CB availability alone is insufficient.
        if (block >= depth) {
            noc_semaphore_wait_min(consumed, block - depth + 1);
        }
        cb_wait_front(slot, pages);
        const uint32_t offset = block * pages;
        const uint32_t valid = count - offset < pages ? count - offset : pages;
        const uint32_t src = get_read_ptr(slot);
        noc_async_write(src, get_noc_addr(x, y, src), valid * bytes);
        // Receiver notification must follow payload visibility, and the local
        // reader must not overwrite the source while the NoC is reading it.
        noc_async_write_barrier();
        noc_semaphore_inc(ready, 1);
        cb_pop_front(slot, pages);
    }
    noc_async_atomic_barrier();
    noc_semaphore_wait_min(consumed, blocks);
}
