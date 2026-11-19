// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C1 unit (tt/fabric_stream.FabricStreamC1.unit): STUB consumer on the MM cores, BRISC — stands
// in for P3 (prefill_megakernel.md H2): MM core j waits FS_ARRIVE >= SLICES (its block landed: local or remote alike),
// then writes the XS block to the gathered output y (DRAM, row tile j) for the unit's bitwise check.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TPB = get_compile_time_arg_val(0);
constexpr uint32_t SLICES = get_compile_time_arg_val(1);
constexpr uint32_t CB_XS = get_compile_time_arg_val(2);
constexpr uint32_t SEM_ARRIVE = get_compile_time_arg_val(3);
constexpr uint32_t TB = 2048;

void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<4>();
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);  // the block (row tile of y)
    const auto y = TensorAccessor(y_args, y_addr);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ARRIVE));
    noc_semaphore_wait_min(sem, SLICES);
    const uint32_t xs = get_write_ptr(CB_XS);
    for (uint32_t t = 0; t < TPB; t++) {
        noc_async_write_page(j * TPB + t, y, xs + t * TB);
    }
    noc_async_write_barrier();
}
