// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK API test: the Blackhole matmul layer called directly, with a hand-filled matmul_params and no Tensor or
// Resolver. Unpack and math each call configure, then execute once per K step, the way today's LLK tests call
// init and the per-K execute: execute does one K step. Math-to-pack Dest sync and the pack use today's LLKs.

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

#include "llk_api/matmul.h"

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#endif
#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#endif

namespace bh = llk2::blackhole;

constexpr DataFormat in0_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format = static_cast<DataFormat>(UNPACK_B_IN);

constexpr bh::matmul_params params_mm {
    .in0_l1         = in0_format,
    .in0_reg        = static_cast<DataFormat>(UNPACK_A_OUT),
    .in1_l1         = in1_format,
    .in1_reg        = static_cast<DataFormat>(UNPACK_B_OUT),
    .in0_face_r_dim = 16,
    .in0_num_faces  = 4,
    .in1_face_r_dim = 16,
    .in1_num_faces  = 4,
    .in0_tile_size  = llk2::tile_bytes(in0_format, 32 * 32) / 16,
    .in1_tile_size  = llk2::tile_bytes(in1_format, 32 * 32) / 16,
    .transpose      = false,
    .rt_dim         = RT_DIM,
    .ct_dim         = CT_DIM,
    .kt_dim         = KT_DIM,
    .math_fidelity  = static_cast<std::uint32_t>(MATH_FIDELITY),
    .dest_fp32      = is_fp32_dest_acc_en,
    .math_format    = static_cast<DataFormat>(MATH_FORMAT),
};

void run_kernel(RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] const Operand& buffer_A   = params.buffer_A;
    [[maybe_unused]] const Operand& buffer_B   = params.buffer_B;
    [[maybe_unused]] const Operand& buffer_Res = params.buffer_Res;

#ifdef LLK_TRISC_UNPACK
    LLK_ASSERT(params.TILE_SIZE_UNPACK_A == params_mm.in0_tile_size && params.TILE_SIZE_UNPACK_B == params_mm.in1_tile_size, "tile sizes differ from the harness");
    bh::unpack_configure<params_mm>();
    for (std::uint32_t k = 0; k < KT_DIM; k++)
    {
        bh::unpack_execute<params_mm>(buffer_A[k], buffer_B[k * CT_DIM]);
    }
#endif

#ifdef LLK_TRISC_MATH
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    bh::math_configure<params_mm>();
    _llk_math_wait_for_dest_available_<dest_sync>();
    for (std::uint32_t k = 0; k < KT_DIM; k++)
    {
        bh::math_execute<params_mm>(0);
    }
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
#endif

#ifdef LLK_TRISC_PACK
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < RT_DIM * CT_DIM; i++)
    {
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
#endif
}
