// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C1 / C3 (the prefill-megakernel all-gather of x̂ into the MM cores' XS, doc/ttnn/16-prefill-
// production/prefill_megakernel.md §4b H1 / H2), lane NCRISC (tt/fabric_stream.FabricStreamC1): for each of this lane's
// local blocks in order, wait until its producer slices have landed in the MM core's XS (the producers +1 this lane's
// word LREADY + k per 16-tile slice of block k), then pull the block (TPB tiles, contiguous in the MM core's XS) into
// the staging CB in NB-tile batches for the lane's BRISC (c1_send.cpp) to send. The MM core's XS base address arrives
// in sync word XSADDR (written by the producers before their first LREADY increment).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

namespace c1r {
constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);  // tiles per block (32 rows x H / 32)
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);   // tiles per batch
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t SLICES = get_compile_time_arg_val(PH_CT0 + 3);  // producer slices per block (each +1 LREADY)
constexpr uint32_t SRC_WORD =
    get_compile_time_arg_val(PH_CT0 + 4);  // the sync word holding the source CB's base address
constexpr uint32_t TB = 2048;
static_assert(TPB % NB == 0, "a batch never straddles a block");
}  // namespace c1r

void kernel_main() {
    using namespace c1r;
    const uint32_t sync = get_arg_val<uint32_t>(ph_rt0 + 0);  // this lane's sync shard
    const uint32_t n_blk = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t blk_xy = ph_rt0 + 2;  // n_blk x (MM core NoC x, y)
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN_NC);
    const uint32_t g = *gen + 1;
    for (uint32_t k = 0; k < n_blk; k++) {
        fs::wait_ge(fs::word(sync, fs::LREADY + k), g * SLICES);  // block k's own word: producers finish in any order
        const uint32_t xs = *fs::word(sync, SRC_WORD);  // XSADDR (C1: x̂ in XS) / OUTADDR (C2: the partial in OUT)
        const uint64_t src =
            get_noc_addr(get_arg_val<uint32_t>(blk_xy + 2 * k), get_arg_val<uint32_t>(blk_xy + 2 * k + 1), xs);
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_reserve_back(CB_STAGE, NB);
            noc_async_read(src + b * NB * TB, get_write_ptr(CB_STAGE), NB * TB);
            noc_async_read_barrier();
            cb_push_back(CB_STAGE, NB);
        }
#if defined(C1R_ACK)  // the holder's last-slice FS_ARRIVE +1 (c1_stub_prod C1P_LANE_ACK): its XS is read out
        invalidate_l1_cache();
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(blk_xy + 2 * k),
                get_arg_val<uint32_t>(blk_xy + 2 * k + 1),
                *fs::word(sync, fs::SEMADDR)),
            1);
#endif
    }
#if defined(C1R_ACK)
    noc_async_atomic_barrier();
#endif
    *gen = g;
}
