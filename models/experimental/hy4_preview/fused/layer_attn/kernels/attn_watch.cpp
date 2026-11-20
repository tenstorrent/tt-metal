// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Measurement only (fused/layer_attn/program.py, P1 built with timing and run alone): on a landing core of X3 it
// waits (bounded) for the arrivals program P2's relay would wait for, takes them off and leaves a stamp: when the
// last chip of the head group has landed its queries on this chip.
//
// Stamps (HY4L_TIMING): 0 start, 1 the group's queries complete.
// Runtime args: the arrivals to wait for.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "attn_common.hpp"

void kernel_main() {
    const uint32_t need = get_arg_val<uint32_t>(0);
    hy4l::stamp(0);
    if (need != 0 && hy4a::wait_word(hy4a::A_QARR, need, hy4a::ST_QARR)) {
        hy4a::take_word(hy4a::A_QARR, need);
    }
    hy4l::stamp(1);
}
