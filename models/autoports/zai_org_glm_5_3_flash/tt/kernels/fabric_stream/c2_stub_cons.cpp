// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream C2 unit (tt/fabric_stream.FabricStreamC2.unit): STUB consumer on the MM cores, BRISC — stands
// in for P10 (prefill_megakernel.md H6): MM core j of an OWNED block waits FS_REDUCED >= SLICES (the peer's partial of
// block j landed in the landing buffer r). The layer's consumer then adds own OUT + landed in fixed chip order (fp32);
// the unit checks r on the host.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t SLICES = get_compile_time_arg_val(0);
constexpr uint32_t SEM_REDUCED = get_compile_time_arg_val(1);

void kernel_main() {
    if (get_arg_val<uint32_t>(0) == 0) {  // not an owned block
        return;
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_REDUCED));
    noc_semaphore_wait_min(sem, SLICES);
}
