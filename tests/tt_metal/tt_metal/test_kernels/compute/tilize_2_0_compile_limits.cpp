// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/2_0/tilize.h"

// Compiles the id-free (2.0) tilize operand-geometry contract; never executed on a device. The input L1 format,
// the face grid, the face height (shared by the input and output tile: tilize keeps the tile geometry) and the
// entry point under test (`mode`: 0 = tilize_init, 1 = tilize_block, 2 = tilize_uninit) are compile-time args so
// the host test can probe, per entry point, which operands the API accepts: the unpack tilize datapath has no
// single-face program, its 8-bit sequence cannot address a narrow (single face column) tile, and the pack tilize
// program for one-row output faces exists for two faces only.
void kernel_main() {
    constexpr auto in_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr std::uint32_t faces_r = get_compile_time_arg_val(1);
    constexpr std::uint32_t faces_c = get_compile_time_arg_val(2);
    constexpr std::uint32_t mode = get_compile_time_arg_val(3);
    constexpr std::uint32_t face_r_dim = get_compile_time_arg_val(4);

    using InOp = experimental::LLKOperand<in_format, TensorShape{face_r_dim, 16, faces_r, faces_c}>;
    using OutOp = experimental::LLKOperand<in_format, TensorShape{face_r_dim, 16, faces_r, faces_c}>;

    // Never dereferenced: the program is only compiled.
    constexpr std::uint32_t l1_address = 0;

    if constexpr (mode == 0) {
        experimental::tilize_init(InOp(l1_address), 1 /*block*/, OutOp(l1_address));
    } else if constexpr (mode == 1) {
        experimental::tilize_block(InOp(l1_address), 1 /*block*/, OutOp(l1_address));
    } else {
        experimental::tilize_uninit(InOp(l1_address), OutOp(l1_address));
    }
}
