// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: two ops whose unpack configuration differs, issued back to back ROUNDS times with nothing
// between them, so every op boundary on the unpack thread rewrites unpacker state (transpose, the X end of a
// broadcast operand, held against streamed, which operand goes to which unpacker) right behind the previous op's
// UNPACRs. One kernel body for all three threads; only Tensors, the Compute API (matmul, add, pack) and a Dest ring.
//
// a and b are [2, 2] tiles. SEQ picks the pair (first op, second op):
//   0  matmul with in1 transposed, plain matmul      unpacker 0 transposes, then must not
//   1  matmul with in1 transposed, plain add         and held + streamed becomes two streams
//   2  add with b's first tile row as a row bias, plain add     unpacker 1 reads half a tile, then a whole one
//   3  add with a row bias, matmul with in1 transposed
// Round r packs the first op's result to output tiles [8r, 8r + 4) and the second's to [8r + 4, 8r + 8).
//
// MATH_SPIN > 0: the math thread spins on the RISC-V before every op, so the Src banks stay full and the previous
// op's last unpacks stand still in the unpackers while the next op's configure is issued: the widest window for a
// configuration word to reach an unpack in flight (see llk2_unpack_latch_probe_test.cpp, which measures per word
// which ones an unpack in flight reads).

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
#include "compute_api/eltwise_binary.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr std::uint32_t side    = 2; // a, b and every result are [side, side] tiles
constexpr std::uint32_t tiles   = side * side;

inline constexpr auto& in_spec  = llk2::l1_spec<side * llk2::default_tile_dim, side * llk2::default_tile_dim, in_format>;
inline constexpr auto& out_spec = llk2::l1_spec<side * llk2::default_tile_dim, side * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<side * llk2::default_tile_dim, side * llk2::default_tile_dim, llk2::kernel_dest_format>;

using AccRing = llk2::DestRing<acc_spec>;

enum class Op
{
    MatmulTransposed,
    Matmul,
    Add,
    AddRowBias,
};

constexpr Op first_op  = SEQ == 0 || SEQ == 1 ? Op::MatmulTransposed : Op::AddRowBias;
constexpr Op second_op = SEQ == 0 ? Op::Matmul : SEQ == 3 ? Op::MatmulTransposed : Op::Add;

/// One op of a and b into acc.
template <Op O, class A, class B, class Acc>
inline __attribute__((always_inline)) auto compute(const A& a, const B& b, const Acc& acc)
{
    if constexpr (O == Op::MatmulTransposed)
    {
        return llk2::matmul<llk2::matmul_modifiers {.transpose_in1 = true}>(a, b, acc);
    }
    else if constexpr (O == Op::Matmul)
    {
        return llk2::matmul(a, b, acc);
    }
    else if constexpr (O == Op::AddRowBias)
    {
        return llk2::add<llk2::add_modifiers {.bcast = llk2::Broadcast::Row}>(a, llk2::row(b, 0), acc);
    }
    else
    {
        return llk2::add(a, b, acc);
    }
}

/// Keeps the math thread away from its Src banks for a while (MATH_SPIN > 0), then runs op O and packs it.
template <Op O, class A, class B, class Out>
inline __attribute__((always_inline)) void run_op(const A& a, const B& b, const Out& out)
{
#ifdef LLK_TRISC_MATH
    for (volatile std::uint32_t spin = 0; spin < MATH_SPIN; spin = spin + 1)
    {
    }
#endif
    const auto y = compute<O>(a, b, llk2::reserve_back<AccRing>());
    llk2::pack(y, out);
    llk2::release(y);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto a = llk2::Tensor<in_spec>(params.buffer_A[0]);
    const auto b = llk2::Tensor<in_spec>(params.buffer_B[0]);
    for (std::uint32_t round = 0; round < ROUNDS; round++)
    {
        run_op<first_op>(a, b, llk2::Tensor<out_spec>(params.buffer_Res[2 * round * tiles]));
        run_op<second_op>(a, b, llk2::Tensor<out_spec>(params.buffer_Res[(2 * round + 1) * tiles]));
    }
}
