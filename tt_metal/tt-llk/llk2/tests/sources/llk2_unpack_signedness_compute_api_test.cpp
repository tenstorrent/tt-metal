// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: a matmul of unsigned 8-bit integers and a matmul of signed ones, issued back to back ROUNDS times
// with nothing between them. The two differ on the unpack thread in ALU_FORMAT_SPEC_REG0 SrcAUnsigned / SrcBUnsigned,
// which the unpackers read for as long as an unpack lasts (tt-isa-documentation, Unpackers/FormatConversion.md: they
// "need to be held constant for the entire duration of the unpack operation", unlike the other configuration an
// UNPACR uses). llk_lib/unpack.h writes them behind a wait for both unpackers; without that wait the previous
// matmul's last tiles would be converted with the other signedness. One kernel body for all three threads.
//
// u and s are the same [2, 2] tiles of buffer A, read as UInt8 and as Int8 (in L1 an Int8 is sign and magnitude: bit
// 7 and bits 6..0), so every byte above 127 is a different number in the two matmuls. (One buffer because the
// harness takes one integer format for both of its inputs.) Round r packs u * u to output tiles [8r, 8r + 4) and
// s * s to [8r + 4, 8r + 8), as Int32.
//
// MATH_SPIN > 0: the math thread spins on the RISC-V before every op, so the Src banks stay full and the previous
// op's last unpacks stand still in the unpackers while the next op's configure is issued.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32     is_fp32_dest_acc_en
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"

constexpr DataFormat u_format   = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat s_format   = DataFormat::Int8;
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
static_assert(u_format == DataFormat::UInt8, "buffer A holds bytes, declared unsigned");
constexpr std::uint32_t side  = 2; // every tensor is [side, side] tiles
constexpr std::uint32_t tiles = side * side;

inline constexpr auto& u_spec   = llk2::l1_spec<side * llk2::default_tile_dim, side * llk2::default_tile_dim, u_format>;
inline constexpr auto& s_spec   = llk2::l1_spec<side * llk2::default_tile_dim, side * llk2::default_tile_dim, s_format>;
inline constexpr auto& out_spec = llk2::l1_spec<side * llk2::default_tile_dim, side * llk2::default_tile_dim, out_format>;
// Integers accumulate as Int32 in the 32-bit Dest (resolver/matmul.h), not in the kernel's float Dest format.
inline constexpr auto& acc_spec = llk2::dst_spec<side * llk2::default_tile_dim, side * llk2::default_tile_dim, DataFormat::Int32>;
using AccRing                   = llk2::DestRing<acc_spec>;

/// Keeps the math thread away from its Src banks for a while (MATH_SPIN > 0), then squares x and packs it.
template <class X, class Out>
inline __attribute__((always_inline)) void square(const X& x, const Out& out)
{
#ifdef LLK_TRISC_MATH
    for (volatile std::uint32_t spin = 0; spin < MATH_SPIN; spin = spin + 1)
    {
    }
#endif
    const auto y = llk2::matmul(x, x, llk2::reserve_back<AccRing>());
    llk2::pack(y, out);
    llk2::release(y);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto u = llk2::Tensor<u_spec>(params.buffer_A[0]);
    const auto s = llk2::Tensor<s_spec>(params.buffer_A[0]);
    for (std::uint32_t round = 0; round < ROUNDS; round++)
    {
        square(u, llk2::Tensor<out_spec>(params.buffer_Res[2 * round * tiles]));
        square(s, llk2::Tensor<out_spec>(params.buffer_Res[(2 * round + 1) * tiles]));
    }
}
