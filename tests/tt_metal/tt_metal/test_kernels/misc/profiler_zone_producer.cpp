// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// A real streaming-profiler PRODUCER: a Tensix RISC that emits `num_zones` profiler zones through the
// ordinary DeviceZoneScopedN macro. Nothing here knows about the drainer -- that is the point. This is
// the unmodified producer path from kernel_profiler.hpp, used to prove a DRISC consumer can service it.
//
// The zone count is chosen by the host to OVERFLOW the ring several times over. One zone is 2 markers x
// 2 words = 4 words, and a ring is PROFILER_L1_VECTOR_SIZE (512) words, so 128 zones fill it exactly.
// Beyond that the producer BLOCKS in ring_ensure_room() until a consumer advances the head. So this
// kernel completing at all is the test: with no drainer it hangs forever, by design.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t num_zones = get_arg_val<uint32_t>(0);
    const uint32_t work_per_zone = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_zones; i++) {
        DeviceZoneScopedN("E2E-PRODUCER");
        // A little work so the zone has a non-degenerate duration and the two markers do not land in
        // the same clock tick. Volatile so it survives -O3.
        for (volatile uint32_t s = 0; s < work_per_zone; s++) {
        }
    }
}
