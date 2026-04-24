// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compile-gate kernel for kernel_lib.
//
// Purpose: surface any kernel_lib template-body regression at the earliest
// possible moment — kernel JIT compile — before it reaches any agent-written op.
//
// What this kernel does:
//   - #includes eltwise_helpers.hpp AND reduce_helpers_compute.hpp in one TU
//     (the historical -Wtemplate-body failure combo; e.g. FillTileInt's
//     undeduced DataFormat arg that bit the 2026-04-24 softmax run).
//   - GCC -Wtemplate-body parses every template body in sfpu_helpers.inl /
//     eltwise_helpers.inl at #include time. Any malformed call to an LLK
//     function with undeduced template args, any unqualified dependent-member
//     lookup, any missing declaration — all fail here, before instantiation.
//   - Runtime payload: identity passthrough of `n_tiles` tiles via
//     eltwise_op + Load. Exercises the eltwise_op codegen path end-to-end.
//
// Wiring:
//   Add a TEST_F in this directory's test_eltwise_helpers.cpp that launches
//   this kernel with 1 input tile and asserts output == input. JIT compile of
//   this TU becomes the actual gate; the correctness check just confirms the
//   kernel launched. (Currently blocked on unrelated drift in the other
//   TEST_Ps in that file — wire up once the harness is repaired.)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"  // copy_tile, used by Load chain element
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    if (n_tiles == 0) {
        return;
    }

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    using namespace compute_kernel_lib;

    auto chain = sfpu_chain(Load<cb_in, Dst::D0>{});
    eltwise_op<cb_out>(chain, EltwiseTileShape::flat(n_tiles));
}
