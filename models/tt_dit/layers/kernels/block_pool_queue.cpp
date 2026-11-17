// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool unit queue, a kernel of the block_topk_route program on the first sender core: pool core ci starts on
// unit ci, so the queue hands out the units from NCORES on in their numbering (block_pool_units.hpp), one per request,
// then UNITS once per pool core that had a unit (a core without one never requests). Pool core ci requests by
// writing (request number << 16) | (noc y << 8) | noc x to its word in this core's sender page (SenderPage::REQ_WORD +
// ci); the reply (request number << 16) | unit goes to that core's CLAIM_SEM by a NoC write from its reply word. A core
// has at most one request outstanding, so a reply word is rewritten only after the core has read the previous reply.
// The page is zeroed by the sender kernel of this core, which then sets QUEUE_GO_SEM.
//
// Compile-time args: 0 H, 1 RT, 2 NBPT, 3 NCORES.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "block_pool_units.hpp"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t NBPT = get_compile_time_arg_val(2);
    constexpr uint32_t NCORES = get_compile_time_arg_val(3);
    using Layout = block_pool::Layout<H, RT, NBPT>;
    using Page = block_pool::SenderPage<Layout::KV_UNITS, NCORES>;
    constexpr uint32_t UNITS = Layout::UNITS;
    constexpr uint32_t cb_page = 3;

    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(block_pool::QUEUE_GO_SEM)), 1);
    const uint32_t page = CircularBuffer(cb_page).get_write_ptr();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page);
    const uint32_t reply_dst = get_semaphore(block_pool::CLAIM_SEM);

    uint16_t served[NCORES] = {};
    uint32_t next = NCORES < UNITS ? NCORES : UNITS;
    uint32_t ended = NCORES - next;
    while (ended < NCORES) {
        invalidate_l1_cache();
        for (uint32_t ci = 0; ci < NCORES; ++ci) {
            const uint32_t req = words[Page::REQ_WORD + ci];
            if (req >> 16 == served[ci]) {
                continue;
            }
            ++served[ci];
            const uint32_t u = next < UNITS ? next++ : UNITS;
            ended += u == UNITS;
            const uint32_t slot = Page::REPLY_WORD + 4 * ci;
            words[slot] = (static_cast<uint32_t>(served[ci]) << 16) | u;
            noc_async_write(page + slot * 4, get_noc_addr(req & 0xFF, (req >> 8) & 0xFF, reply_dst), 4);
        }
    }
    noc_async_write_barrier();
}
