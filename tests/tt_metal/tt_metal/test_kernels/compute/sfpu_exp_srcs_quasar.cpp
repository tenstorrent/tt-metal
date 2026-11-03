// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Example: exp on SrcS through the SFPU op-type compute API (api/compute/experimental/sfpu.h).
//
// The whole op runs on the SFPU thread (TRISC3): per tile, unpacker 2 moves it from L1 into SrcS,
// the SFPU computes exp over the tile's slices, packer 1 writes it back to L1. UNPACK, MATH and PACK
// do nothing in this kernel; the two calls below expand to nothing on those threads.
//
// The input and output are plain L1 regions of num_tiles consecutive tiles, not DataflowBuffers:
// TRISC3 has no DataflowBuffer interface yet (issue #63416), so there is no per-tile handshake. The
// producer must have written all num_tiles input tiles before this kernel runs, and the consumer
// reads the output after it completes. A DataflowBuffer overload replaces the addresses when the
// interface lands; the two calls keep their shape.
//
// Compile-time args: 0 = input L1 address (16-byte units), 1 = output L1 address (16-byte units),
// 2 = number of tiles, 3 = L1 data format of both regions, 4 = SrcS register format
// (Float16, Float16_b, Float32 or Int32; equal to the L1 format for the common case).

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
    sfpu_init<op::Exp, SfpuReg::SrcS>(in_addr_16B, l1_format, out_addr_16B, l1_format, srcs_format);
    sfpu_tiles<op::Exp, SfpuReg::SrcS>(srcs_format, num_tiles);  // unpack -> exp -> pack, per tile

    // Once TRISC3 owns DataflowBuffers (issue #63416) the two calls above take the buffers instead of
    // addresses and formats:
    //
    //     DataflowBuffer in(dfb::in), out(dfb::out);
    //     sfpu_init<op::Exp, SfpuReg::SrcS>(in, out);
    //     sfpu_tiles<op::Exp, SfpuReg::SrcS>(in, out, num_tiles);
    //
    // The ring bases and formats then come from the buffers, and the tile loop inside sfpu_tiles gains
    // the handshake this version lacks: in.wait_front / out.reserve_back before each tile's unpack,
    // in.pop_front / out.push_back after its pack. The op type, the slice loop and the exp math do not
    // change; this address form stays for callers that own raw L1 regions.
}
