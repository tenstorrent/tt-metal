// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::pack modifiers (ReLU, L1 accumulation) behind a matmul, over PASSES Dest sections.
// Pass 0 packs (overwrites) the result; every later pass packs it again with l1_acc, adding into L1. With PASSES > 2
// the sections wrap around, so the test also covers the Dest sync API's double buffering. Only Tensors, the Compute
// API and the Dest sync API.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32     is_fp32_dest_acc_en
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"

constexpr DataFormat in0_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format  = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format  = static_cast<DataFormat>(PACK_OUT);
constexpr DataFormat dest_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;

inline constexpr auto& in0_spec = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, KT_DIM * llk2::default_tile_dim, in0_format>;
inline constexpr auto& in1_spec = llk2::l1_spec<KT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, in1_format>;
inline constexpr auto& out_spec = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, dest_format>;

constexpr llk2::pack_modifiers first_pack {.relu = PACK_RELU};
constexpr llk2::pack_modifiers accumulating_pack {.relu = PACK_RELU, .l1_acc = PASSES > 1}; // with one pass it never runs

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto x   = llk2::Tensor<in0_spec>(params.buffer_A[0]);
    const auto w   = llk2::Tensor<in1_spec>(params.buffer_B[0]);
    const auto out = llk2::Tensor<out_spec>(params.buffer_Res[0]);

    for (std::uint32_t pass = 0; pass < PASSES; pass++)
    {
        const auto acc = llk2::dest_acquire<acc_spec>();
        llk2::matmul(x, w, acc);
        llk2::dest_commit(acc);
        if (pass == 0)
        {
            llk2::pack<first_pack>(acc, out);
        }
        else
        {
            llk2::pack<accumulating_pack>(acc, out);
        }
        llk2::dest_release(acc);
    }
}
