// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: pack's writes to config words it shares with the math thread, issued while the math thread's
// next op runs. pack_configure writes STACC_RELU ApplyRelu (the byte that also holds the Src / Dest zero-flag bits
// the FPU reads) and ALU_FORMAT_SPEC_REG2 Dstacc (the byte that also holds Fp32_enabled, SFPU_Fp32_enabled and
// INT8_math_enabled) after a wait for the packer only. Per the ISA only the packer reads the two fields
// (Packers/ReLU.md, Packers/FormatConversion.md) and RMWCIB is atomic per byte (RMWCIB.md); this test is the
// device-side check: ROUNDS ops back to back over a Dest ring of two sections, so op r + 1 runs on the FPU or the
// SFPU while pack r is configured, and the packs alternate ReLU off / on, so every configure changes the byte.
// Every round must come out as its own ReLU setting says: a math op that saw the packer's ReLU, or a pack that lost
// its own, shows as negatives where there must be none or the other way round.
//
// SFPU_OP: the op is a copy and a silu on the math thread (SFPLOAD / SFPSTORE on Dest); otherwise a matmul.

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
#include "compute_api/copy.h"
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "compute_api/vec.h"

constexpr DataFormat in0_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr std::uint32_t tile    = llk2::default_tile_dim;

inline constexpr auto& in0_spec = llk2::l1_spec<tile, (SFPU_OP ? TILES : KT_DIM) * tile, in0_format>; // SFPU_OP: the tensor itself
inline constexpr auto& in1_spec = llk2::l1_spec<KT_DIM * tile, TILES * tile, in1_format>;
inline constexpr auto& out_spec = llk2::l1_spec<tile, TILES * tile, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<tile, TILES * tile, llk2::kernel_dest_format>;

constexpr llk2::pack_modifiers plain_pack {.relu = false};
constexpr llk2::pack_modifiers relu_pack {.relu = true};

/// The round's op into acc: a copy of a and a silu on the math thread's SFPU (Sfpu), or the matmul a b.
template <bool Sfpu, class A, class B, class Acc>
inline __attribute__((always_inline)) auto math_op(const A& a, const B& b, const Acc& acc)
{
    if constexpr (Sfpu)
    {
        return llk2::vec_silu<llk2::vec_modifiers {.thread = llk2::Thread::Math}>(llk2::copy(a, acc));
    }
    else
    {
        return llk2::matmul(a, b, acc);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto a = llk2::Tensor<in0_spec>(params.buffer_A[0]);
    const auto b = llk2::Tensor<in1_spec>(params.buffer_B[0]);

    using AccRing = llk2::DestRing<acc_spec>;
    for (std::uint32_t round = 0; round < ROUNDS; round++)
    {
        const auto out = llk2::Tensor<out_spec>(params.buffer_Res[round * TILES]);
        const auto run = [&](const auto& y)
        {
            if (round % 2 == 0)
            {
                llk2::pack<plain_pack>(y, out);
            }
            else
            {
                llk2::pack<relu_pack>(y, out);
            }
            llk2::release(y);
        };
        run(math_op<SFPU_OP>(a, b, llk2::reserve_back<AccRing>()));
    }
}
