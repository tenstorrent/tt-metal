// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::copy on Tensors, one kernel body for all three threads.
// Math-to-pack Dest sync and the pack itself still use today's LLKs.

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

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/copy.h"

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#endif
#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#endif

constexpr DataFormat src_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat dest_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b; // every format here is a B format
constexpr llk2::copy_modifiers mods {.transpose = TRANSPOSE};

// The Resolver must make the choice the harness made for the same test: the register format, and whether the
// tiles go straight to Dest.
inline constexpr auto& src_spec         = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, src_format>;
inline constexpr const llk2::Plan& plan = llk2::resolved<llk2::copy_op, src_spec, src_spec, mods>.plan;
static_assert(plan.in0.writes == static_cast<DataFormat>(UNPACK_A_OUT));
static_assert((plan.in0.to == llk2::Where::Dest) == unpack_to_dest);

void run_kernel(RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] const Operand& buffer_A   = params.buffer_A;
    [[maybe_unused]] const Operand& buffer_Res = params.buffer_Res;

    const auto src = llk2::Tensor<src_spec>(buffer_A[0]);
    const auto acc = llk2::dst_tensor<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, dest_format>(0); // from Dest row 0

#ifdef LLK_TRISC_MATH
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<dest_sync>();
#endif

    llk2::copy<mods>(src, acc); // acc = src, or src with each tile transposed

#ifdef LLK_TRISC_MATH
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
#endif

#ifdef LLK_TRISC_PACK
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < TILES_Y * TILES_X; i++)
    {
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
#endif
}
