// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// N-core writer: hands each normalised bf16 tile row to the paired R core.
//
// CB_NORMED is declared on every core, so it has the same L1 address on the R core as here; this
// core never uses its own copy except to read that address. The R core's ring has `depth` slots of
// one tile row each, filled in order, so row r goes to slot r % depth.
//
// Protocol per row r:
//   wait SEM_CREDIT (local) >= r + 1   the R reader reserved slot r % depth and granted it
//   write the 4 tiles to the R core's slot, wait for the write acks
//   SEM_READY (on the R core) += 1     the R reader may now push the row to its compute
//
// Runtime args: 0 num_rows, 1 r_noc_x, 2 r_noc_y, 3 depth

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "qprep_common.hpp"

using namespace qprep;

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t r_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t r_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t depth = get_arg_val<uint32_t>(3);

    constexpr uint32_t row_bytes = WT * get_tile_size(CB_NOUT);
    static_assert(get_tile_size(CB_NOUT) == get_tile_size(CB_NORMED), "CB_NOUT and CB_NORMED must match");

    const uint32_t normed_base = get_write_ptr(CB_NORMED);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_CREDIT));
    const uint64_t ready_noc = get_noc_addr(r_noc_x, r_noc_y, get_semaphore(SEM_READY));

    CircularBuffer cb(CB_NOUT);
    uint32_t slot = 0;
    for (uint32_t r = 0; r < num_rows; ++r) {
        cb.wait_front(WT);
        noc_semaphore_wait_min(credit, r + 1);
        noc_async_write(cb.get_read_ptr(), get_noc_addr(r_noc_x, r_noc_y, normed_base + slot * row_bytes), row_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(ready_noc, 1);
        cb.pop_front(WT);
        if (++slot == depth) {
            slot = 0;
        }
    }
    noc_async_atomic_barrier();
}
