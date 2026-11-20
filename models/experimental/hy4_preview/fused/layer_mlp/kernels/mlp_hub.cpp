// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The hub core of a chip in the MLP layer program (fused/layer_mlp/program.py): the MoE piece's hub with the
// router (fused/moe/kernels/hub_router.cpp, included as it is: its kernel_main runs as a function here) between the
// layer's two cross-chip steps:
//
//   1. wait (bounded) for the layer input x in this core's x buffer: word W_X, counted by the root chip's sender
//      (fused/layer_common/transport.py); take the count off.
//   2. the MoE piece's hub: x to every core, router, union, meta page, release of the expert cores.
//   3. wait (bounded) until the chip's partial output is complete in this core's output buffer: the piece's word
//      SEM_OUT has one count per working down core.
//   4. hand the partial on (hy4l::feed_reduce): to this chip's sender core, or into this chip's own reduce cores'
//      stage-1 landing when the chip heads its mesh row.
// A wait that runs out leaves its code in this core's status word and the kernel carries on with the next step, so
// the program ends and the chips downstream are not kept waiting for longer than their own bounds.
//
// Stamps (HY4L_TIMING): 0 start, 1 x arrived, 2 the expert cores released, 3 partial complete, 4 partial handed on.
//
// Runtime args: 0..5 hub_router.cpp's (0..2 hub.cpp's), then: the number of working down cores, the output buffer's
// address, then hy4l::feed_reduce's arguments. Compile-time args and defines: hub_router.cpp's, and HY4L_*.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// HY4_MLP_GIVEN: the routing is given (the piece's hub.cpp, 3 runtime args): layer 0's dense FFN, whose slots are
// the same on every step and sit in the hub's meta buffer.
#define kernel_main hy4_moe_hub_main
#ifdef HY4_MLP_GIVEN
#include "hub.cpp"
constexpr uint32_t hy4_mlp_base = 3;
#else
#include "hub_router.cpp"
#ifdef HY4_MLP_BASE  // (the MoE hub's runtime arguments are longer: its bias follows the six addresses)
constexpr uint32_t hy4_mlp_base = HY4_MLP_BASE;
#else
constexpr uint32_t hy4_mlp_base = 6;
#endif
#endif
#undef kernel_main

#include "layer_common.hpp"

void kernel_main() {
    constexpr uint32_t base = hy4_mlp_base;
    hy4l::stamp(0);
    volatile tt_l1_ptr uint32_t* runs = hy4l::word(hy4l::W_RUNS);
    runs[0] = runs[0] + 1;
    if (hy4l::wait_word(hy4l::W_X, 1, hy4l::ST_X)) {
        hy4l::take_word(hy4l::W_X, 1);
    }
    hy4l::stamp(1);
    hy4_moe_hub_main();
    hy4l::stamp(2);
    hy4l::wait_address(hy4moe::sem(hy4moe::SEM_OUT), get_arg_val<uint32_t>(base), hy4l::ST_OUT);
    hy4l::stamp(3);
    hy4l::feed_reduce(base + 2, get_arg_val<uint32_t>(base + 1));
    volatile tt_l1_ptr uint32_t* fed = hy4l::word(hy4l::W_FED);
    fed[0] = fed[0] + 1;
    hy4l::stamp(4);
}
