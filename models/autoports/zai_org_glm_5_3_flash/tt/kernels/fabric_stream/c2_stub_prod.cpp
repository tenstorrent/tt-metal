// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C2 unit (tt/fabric_stream.FabricStreamC2.unit): STUB producer on the MM cores, NCRISC — stands
// in for P7 (prefill_megakernel.md H5): MM core j reads this chip's partial block j (32 rows of P, DRAM) into its OUT
// (CB 42). For a block OWNED BY THE PEER it publishes OUTADDR / SEMADDR (= FS_REDUCED's address, the same on the owner)
// to its lane, then +1 the lane's LREADY + k per 16-tile slice.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

constexpr uint32_t TPB = get_compile_time_arg_val(0);
constexpr uint32_t SLICE_T = get_compile_time_arg_val(1);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
constexpr uint32_t SEM_REDUCED = get_compile_time_arg_val(3);
constexpr uint32_t TB = 2048;

void kernel_main() {
    constexpr auto p_args = TensorAccessorArgs<4>();
    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t j = get_arg_val<uint32_t>(1);       // the block (row tile of P)
    const uint32_t remote = get_arg_val<uint32_t>(2);  // owned by the peer: send it
    const uint32_t lane_x = get_arg_val<uint32_t>(3);
    const uint32_t lane_y = get_arg_val<uint32_t>(4);
    const uint32_t sync = get_arg_val<uint32_t>(5);
    const uint32_t k = get_arg_val<uint32_t>(6);
    const auto pp = TensorAccessor(p_args, p_addr);
    const uint32_t out = get_write_ptr(CB_OUT);
    if (remote) {
        noc_inline_dw_write(get_noc_addr(lane_x, lane_y, sync + fs::OUTADDR * fs::WORD), out);
        noc_inline_dw_write(get_noc_addr(lane_x, lane_y, sync + fs::SEMADDR * fs::WORD), get_semaphore(SEM_REDUCED));
        noc_async_write_barrier();
    }
    for (uint32_t s = 0; s < TPB / SLICE_T; s++) {
        for (uint32_t t = s * SLICE_T; t < (s + 1) * SLICE_T; t++) {
            noc_async_read_page(j * TPB + t, pp, out + t * TB);
        }
        noc_async_read_barrier();
        if (remote) {
            noc_semaphore_inc(get_noc_addr(lane_x, lane_y, sync + (fs::LREADY + k) * fs::WORD), 1);
        }
    }
    noc_async_atomic_barrier();
}
