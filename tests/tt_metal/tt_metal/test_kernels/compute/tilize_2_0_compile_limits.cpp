// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/experimental/2_0/tilize.h"

// Compiles the id-free (2.0) tilize input-geometry contract; never executed on a device. The input L1 format,
// face grid and the entry point under test (`mode`: 0 = tilize_init, 1 = tilize_block, 2 = tilize_uninit) are
// compile-time args so the host test can probe, per entry point, which shapes the API accepts: the unpack
// tilize datapath has no single-face program, and its 8-bit sequence cannot address a narrow (single face
// column) tile.
void kernel_main() {
    constexpr auto in_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr std::uint32_t faces_r = get_compile_time_arg_val(1);
    constexpr std::uint32_t faces_c = get_compile_time_arg_val(2);
    constexpr std::uint32_t mode = get_compile_time_arg_val(3);

    using InOp = experimental::LLKOperand<in_format, TensorShape{16, 16, faces_r, faces_c}>;
    using OutOp = experimental::LLKOperand<in_format, TensorShape{16, 16, faces_r, faces_c}>;

    if constexpr (mode == 0) {
        experimental::tilize_init(InOp(0), 1 /*block*/, OutOp(0));
    } else if constexpr (mode == 1) {
        experimental::tilize_block(InOp(0), 1 /*block*/, OutOp(0));
    } else {
        experimental::tilize_uninit(InOp(0), OutOp(0));
    }
}
