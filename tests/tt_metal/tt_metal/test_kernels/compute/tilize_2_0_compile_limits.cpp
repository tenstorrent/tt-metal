// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/2_0/tilize.h"

// Compiles the id-free (2.0) tilize operand-geometry contract; never executed on a device. The input L1 format,
// the input face grid, the face height (shared by the input and output tile), the output face grid and the entry
// point under test (`mode`: 0 = tilize_init, 1 = tilize_block, 2 = tilize_uninit) are compile-time args so the
// host test can probe, per entry point, which operands the API accepts: tilize keeps the tile geometry, the
// datapath has no single-face and no narrow (single face column) program, and the pack tilize program used for a
// non-8-bit input handles faces shorter than 16 rows for a two-face tile only.
void kernel_main() {
    constexpr auto in_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr std::uint32_t faces_r = get_compile_time_arg_val(1);
    constexpr std::uint32_t faces_c = get_compile_time_arg_val(2);
    constexpr std::uint32_t mode = get_compile_time_arg_val(3);
    constexpr std::uint32_t face_r_dim = get_compile_time_arg_val(4);
    constexpr std::uint32_t out_faces_r = get_compile_time_arg_val(5);
    constexpr std::uint32_t out_faces_c = get_compile_time_arg_val(6);

    using InOp = experimental::LLKOperand<in_format, TensorShape{face_r_dim, 16, faces_r, faces_c}>;
    using OutOp = experimental::LLKOperand<in_format, TensorShape{face_r_dim, 16, out_faces_r, out_faces_c}>;

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
