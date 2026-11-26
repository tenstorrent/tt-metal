// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Example: exp on SrcS over plain L1 regions, all on TRISC3. No DataflowBuffers, so no per-tile handshake: every
// input tile must be in L1 before this kernel runs.
// Compile-time args: 0 = input L1 address, 1 = output L1 address (16-byte units), 2 = number of tiles,
// 3 = L1 format of both regions, 4 = SrcS register format.

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
    using ExpOp = SfpuOp<op::Exp<>, SfpuReg::SrcS, srcs_format>;
    ExpOp::init();
    ExpOp::tiles(unpack_from(in_addr_16B, l1_format), pack_to(out_addr_16B, l1_format), num_tiles);
}
