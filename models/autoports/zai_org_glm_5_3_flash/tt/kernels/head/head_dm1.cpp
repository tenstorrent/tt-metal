// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b model head, owner NCRISC (NoC1): the all-gather of the 64 owners' partial sums of squares
// (kernels/head/head_common.hpp). Runtime args: T, my index k, then the 64 owners' NoC (x, y) (k order).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/head_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"

using namespace head;

void kernel_main() {
    const uint32_t T = get_arg_val<uint32_t>(0);
    const uint32_t k = get_arg_val<uint32_t>(1);
    const uint32_t land = get_write_ptr(CB_LAND);  // the same address on every owner (one CB, one core range)
    const uint32_t rec = (get_write_ptr(CB_REC) + 15u) & ~15u;

    // this owner's partials (column 0, rows 0 .. T-1 of P) -> a 16-B record
    cb_wait_front(CB_P, 1);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_P));
        volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec);
        for (uint32_t t = 0; t < REC_BYTES / 4; ++t) {
            r[t] = t < T ? p[tile_off(t, 0)] : 0u;
        }
    }
    cb_pop_front(CB_P, 1);
    for (uint32_t o = 0; o < N_OWNERS; ++o) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * o), y = get_arg_val<uint32_t>(3 + 2 * o);
        noc_async_write(rec, get_noc_addr(x, y, land + k * REC_BYTES), REC_BYTES);
    }
    noc_async_write_barrier();  // every record landed before its owner's count is raised
    mk::signal_handoff(HEAD_SEM, 2, N_OWNERS);
    mk::wait_handoff(HEAD_SEM, N_OWNERS);

    // the 64 records -> G tiles (owner k: column k % 32 of tile k / 32; every other element 0)
    cb_reserve_back(CB_G, 2);
    const uint32_t g = get_write_ptr(CB_G);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < 2 * FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, g + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    {
        volatile tt_l1_ptr uint32_t* gt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g);
        volatile tt_l1_ptr uint32_t* l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land);
        for (uint32_t o = 0; o < N_OWNERS; ++o) {
            for (uint32_t t = 0; t < T; ++t) {
                gt[(o >> 5) * 1024u + tile_off(t, o & 31u)] = l[o * (REC_BYTES / 4) + t];
            }
        }
    }
    cb_push_back(CB_G, 2);
}
