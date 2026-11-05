// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Example: exp on SrcS through the SFPU op-type compute API. Everything runs on the SFPU thread (TRISC3): per
// tile, unpacker 2 moves it from L1 into SrcS, the SFPU computes exp, packer 1 writes it back to L1.
//
// The input and output are plain L1 regions of num_tiles consecutive tiles, not DataflowBuffers, so there is no
// per-tile handshake: the producer must have written every input tile before this kernel runs.
//
// Compile-time args: 0 = input L1 address (16-byte units), 1 = output L1 address (16-byte units),
// 2 = number of tiles, 3 = L1 data format of both regions, 4 = SrcS register format
// (Float16, Float16_b or Float32; equal to the L1 format for the common case).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/sfpu.h"

void kernel_main() {
    constexpr std::uint32_t in_addr_16B = get_compile_time_arg_val(0);
    constexpr std::uint32_t out_addr_16B = get_compile_time_arg_val(1);
    constexpr std::uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr DataFormat l1_format = static_cast<DataFormat>(get_compile_time_arg_val(3));
    constexpr DataFormat srcs_format = static_cast<DataFormat>(get_compile_time_arg_val(4));

    // SFPU thread only; nothing on UNPACK, MATH and PACK.
    sfpu_init<op::Exp<>, SfpuReg::SrcS, srcs_format>(in_addr_16B, l1_format, out_addr_16B, l1_format);
    sfpu_tiles<op::Exp<>, SfpuReg::SrcS, srcs_format>(num_tiles);  // unpack -> exp -> pack, per tile
}
