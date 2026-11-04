// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C1 unit (tt/fabric_stream.FabricStreamC1.unit): STUB producer on the MM cores, NCRISC — stands
// in for the P1 mHC members (prefill_megakernel.md H1). A core holding a LOCAL block reads its 32 rows (TPB tiles of x,
// DRAM) into its XS (CB 35) slice by slice; first it publishes the XS / FS_ARRIVE addresses to its lane (XSADDR /
// SEMADDR, written before any LREADY increment), then per slice: +1 its own FS_ARRIVE and +1 the lane's LREADY + k.
// Runtime args: local, x, row, lane x / y, lane sync, k, my x / y, XS-FREE mode, XS-FREE lane x / y (12).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

namespace c1p {
constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t SLICE_T = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_XS = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t SEM_ARRIVE = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t TB = 2048;
}  // namespace c1p

void kernel_main() {
    using namespace c1p;
    constexpr auto x_args = TensorAccessorArgs<PH_CT0 + 4>();
    const uint32_t a0 = ph_rt0;
    const uint32_t local = get_arg_val<uint32_t>(a0 + 0);
    // PMK H0 (XS-FREE): +1 on the XSFREE word of this holder's lane (G % 2) once its XS is free for this use. Mode 1:
    // at once (the first use in the launch: XS is free at launch); mode 2: after the compute released the previous
    // use (cb_reserve_back of the whole CB: the stationary block was popped). Every holder credits, local or not.
    const uint32_t xsf_mode = get_arg_val<uint32_t>(a0 + 9);
    if (xsf_mode) {
        if (xsf_mode == 2) {
            cb_reserve_back(CB_XS, TPB);
        }
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(a0 + 10),
                get_arg_val<uint32_t>(a0 + 11),
                get_arg_val<uint32_t>(a0 + 5) + fs::XSFREE * fs::WORD),
            1);
        noc_async_atomic_barrier();
    }
    if (!local) {
        return;
    }
    const uint32_t x_addr = get_arg_val<uint32_t>(a0 + 1);
    const uint32_t row = get_arg_val<uint32_t>(a0 + 2);     // the block's row tile in x (this chip's rows)
    const uint32_t lane_x = get_arg_val<uint32_t>(a0 + 3);  // its lane (NoC 0) and the lane's sync shard
    const uint32_t lane_y = get_arg_val<uint32_t>(a0 + 4);
    const uint32_t sync = get_arg_val<uint32_t>(a0 + 5);
    const uint32_t k = get_arg_val<uint32_t>(a0 + 6);     // the block's index in its lane's list
    const uint32_t my_x = get_arg_val<uint32_t>(a0 + 7);  // this core, NoC 0
    const uint32_t my_y = get_arg_val<uint32_t>(a0 + 8);
    const auto x = TensorAccessor(x_args, x_addr);
    const uint32_t xs = get_write_ptr(CB_XS);
    const uint32_t sem = get_semaphore(SEM_ARRIVE);
    noc_inline_dw_write(get_noc_addr(lane_x, lane_y, sync + fs::XSADDR * fs::WORD), xs);
    noc_inline_dw_write(get_noc_addr(lane_x, lane_y, sync + fs::SEMADDR * fs::WORD), sem);
    noc_async_write_barrier();
    for (uint32_t s = 0; s < TPB / SLICE_T; s++) {
        for (uint32_t t = s * SLICE_T; t < (s + 1) * SLICE_T; t++) {
            noc_async_read_page(row * TPB + t, x, xs + t * TB);
        }
        noc_async_read_barrier();
#if defined(C1P_LANE_ACK)  // the LAST slice's own +1 comes from the lane after it read the whole block (c1_read
                           // C1R_ACK):
        // FS_ARRIVE >= SLICES then also certifies that the lane is done with this XS (W7 step 1: the reclaim drain)
        if (s + 1 < TPB / SLICE_T)
#endif
            noc_semaphore_inc(get_noc_addr(my_x, my_y, sem), 1);
        noc_semaphore_inc(get_noc_addr(lane_x, lane_y, sync + (fs::LREADY + k) * fs::WORD), 1);
    }
    noc_async_atomic_barrier();
}
