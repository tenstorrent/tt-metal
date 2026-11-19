// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream line RS of fp32 partials (tt/fabric_stream.FabricStreamRSL; TP8 stage 2 along the 4-chip row),
// lane NCRISC: for each (destination, block) of this lane's send list, in order, stream the block's TPB fp32 tiles of
// the input r1 (DRAM) into CB_SEND (NB-tile batches) for rsl_send.cpp. ready: wait R1RDY + i >= g first (the stage-1
// reducers of the TP8 RS raise it when r1 of that block is written; 0 = the unit's prefilled r1).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_SEND = get_compile_time_arg_val(PH_CT0 + 2);
#if defined(FS_R1_BF16)  // GLM53_PMK_RS8_S2_BF16 (tt/fabric_stream.FabricStreamRSL r1_dtype): r1 / r2 in bf16 tiles
constexpr uint32_t TB32 = 2048;
#else
constexpr uint32_t TB32 = 4096;
#endif
static_assert(TPB % NB == 0, "a batch never straddles a block");

void kernel_main() {
    constexpr auto r1_args = TensorAccessorArgs<PH_CT0 + 3>();
    size_t a = ph_rt0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t ready = get_arg_val<uint32_t>(a++);
    const uint32_t r1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t rows = a;  // n x the block's row tile in r1
    const auto r1 = TensorAccessor(r1_args, r1_addr);
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN_NC);
    const uint32_t g = *gen + 1;
    for (uint32_t i = 0; i < n; i++) {
        if (ready) {
            fs::wait_ge(fs::word(sync, fs::R1RDY + i), g);
        }
        const uint32_t row = get_arg_val<uint32_t>(rows + i);
        for (uint32_t b = 0; b < TPB / NB; b++) {
            cb_reserve_back(CB_SEND, NB);
            const uint32_t wp = get_write_ptr(CB_SEND);
            for (uint32_t t = 0; t < NB; t++) {
                noc_async_read_page(row * TPB + b * NB + t, r1, wp + t * TB32);
            }
            noc_async_read_barrier();
            cb_push_back(CB_SEND, NB);
        }
    }
#if !defined(FS_NO_ROW_BARRIER)
    {  // THE PER-HOP DRAIN (fs_common.hpp): every source slot's blocks for this lane landed here (the fused +1 of
        // each block's last packet on ARR + slot x 2 + h; per = n / nslots blocks per source per launch, as each
        // source sends this chip what this chip sends it)
        const uint32_t nslots = get_arg_val<uint32_t>(rows + n);
        const uint32_t h = get_arg_val<uint32_t>(rows + n + 1);
        if (nslots) {
            const uint32_t per = n / nslots;
            for (uint32_t s = 0; s < nslots; s++) {
                fs::wait_ge(fs::word(sync, fs::ARR + s * 2 + h), g * per);
            }
        }
        *fs::word(sync, fs::LANDED) = g;
    }
#endif
    *gen = g;
}
