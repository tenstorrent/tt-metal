// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device test for the SHARED tilize helper's streaming entry point.
//
// This exercises compute_kernel_lib::tilize<..., StreamMode::PerTile> (per-tile pack +
// push_back(1), output CB sized to just 2 tiles) against the default StreamMode::Atomic
// path (output CB sized to the full tile-row). The host harness in
// streaming_tilize.cpp compares both against a host tilize golden and against each
// other, at W=4 and W=128, and asserts bit-exactness.
//
// Compile-time args:
//   0: cb_in  1: cb_out  2: W (block width in tiles)  3: streaming flag

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t streaming = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib::tilize_config;
    if constexpr (streaming != 0) {
        // STREAMING: packs one tile at a time, so cb_out is a 2-tile double-buffer.
        // StreamMode is the trailing template param, so spell out the intervening defaults.
        compute_kernel_lib::tilize<
            W,
            cb_in,
            cb_out,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
            Fp32Mode::Fast,
            RemapMode::Configure,
            StreamMode::PerTile>(/*num_blocks=*/1);
    } else {
        // ATOMIC REFERENCE: cb_out holds all W tiles.
        compute_kernel_lib::tilize<W, cb_in, cb_out>(/*num_blocks=*/1);
    }
}
