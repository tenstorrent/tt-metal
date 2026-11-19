// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Indexer phase streamer (ixp_common.hpp), BRISC of the streamer core: the pooled keys of the window, read ONCE per
// launch in KC-key-tile blocks into a two-slot staging CB and multicast into every member's CB_K slot b % DEPTH.
// Compile-time: KC, DEPTH, NMEM, then the keys' TensorAccessorArgs. Runtime: keys address, NKT (key tiles), member
// rectangle NoC (x0, y0, x1, y1).
#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/ix/ixp_common.hpp"

using namespace ixp;

void kernel_main() {
    constexpr uint32_t KC = get_compile_time_arg_val(0);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(1);
    constexpr uint32_t NMEM = get_compile_time_arg_val(2);
    constexpr auto k_args = TensorAccessorArgs<3>();
    const uint32_t k_addr = get_arg_val<uint32_t>(0);
    const uint32_t nkt = get_arg_val<uint32_t>(1);
    const uint32_t x0 = get_arg_val<uint32_t>(2), y0 = get_arg_val<uint32_t>(3);
    const uint32_t x1 = get_arg_val<uint32_t>(4), y1 = get_arg_val<uint32_t>(5);
    const auto kk = TensorAccessor(k_args, k_addr);
    constexpr uint32_t SLOT = 4 * KC * T16;
    const uint32_t nblk = (nkt + KC - 1) / KC;

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
    volatile tt_l1_ptr uint32_t* addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ADDR));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_VALID));
    cb_reserve_back(CB_ST, 2 * 4 * KC);  // the staging halves, used by address (never pushed)
    const uint32_t st = get_write_ptr(CB_ST);
    noc_semaphore_wait_min(addr, 1);  // member 0 published its CB_K base (identical on every member)
    const uint32_t kbase = *addr;
    auto issue = [&](uint32_t bb) {  // block bb's key tiles -> staging half bb & 1 (dim-major slot layout)
        const uint32_t half = st + (bb & 1) * SLOT;
        const uint32_t t0 = bb * KC;
        for (uint32_t j = 0; j < KC && t0 + j < nkt; ++j) {
            for (uint32_t d = 0; d < 4; ++d) {
                noc_async_read_page((t0 + j) * 4 + d, kk, half + (d * KC + j) * T16);
            }
        }
    };
    if (nblk) {
        issue(0);
    }
    for (uint32_t b = 0; b < nblk; ++b) {
        const uint32_t half = st + (b & 1) * SLOT;
        noc_async_read_barrier();  // block b staged
        if (b + 1 < nblk) {        // prefetch block b + 1 into the other half (its last multicast was flushed)
            issue(b + 1);
        }
        noc_semaphore_wait_min(ready, (b + 1) * NMEM);  // every member reserved slot b % DEPTH
        const uint64_t dst = get_noc_multicast_addr(x0, y0, x1, y1, kbase + (b % DEPTH) * SLOT);
        noc_async_write_multicast(half, dst, SLOT, NMEM);
        noc_async_writes_flushed();  // the data left before the VALID raise (and the half is reusable)
        *valid = b + 1;
        noc_semaphore_set_multicast(
            get_semaphore(SEM_VALID), get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(SEM_VALID)), NMEM);
        noc_async_writes_flushed();
    }
    noc_async_write_barrier();
}
