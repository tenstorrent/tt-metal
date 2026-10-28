// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: llk2::add and llk2::sub on Tensors, plain and with a row broadcast, one kernel body for all three
// threads. Math-to-pack Dest sync and the pack itself still use today's LLKs.
//
// a is [RT_DIM, CT_DIM] tiles. b has a's shape, or with BCAST_ROW is one row of CT_DIM tiles (a bias) whose row 0 is
// added to every row of a's matching tile column.

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
#include "compute_api/add.h"
#include "compute_api/sub.h"

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#endif
#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#endif

constexpr DataFormat in0_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format  = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat dest_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
constexpr llk2::Broadcast bcast  = BCAST_ROW ? llk2::Broadcast::Row : llk2::Broadcast::None;
constexpr std::uint32_t in1_rows = BCAST_ROW ? 1 : RT_DIM; // tile rows of b

inline constexpr auto& in0_spec = llk2::l1_spec<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, in0_format>;
inline constexpr auto& in1_spec = llk2::l1_spec<in1_rows * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, in1_format>;

// The Resolver must pick the register formats the harness picked for the same test.
static_assert(llk2::resolved<llk2::add_op, in0_spec, in1_spec, llk2::add_modifiers {bcast}>.plan.in0.writes == static_cast<DataFormat>(UNPACK_A_OUT));
static_assert(llk2::resolved<llk2::add_op, in0_spec, in1_spec, llk2::add_modifiers {bcast}>.plan.in1.writes == static_cast<DataFormat>(UNPACK_B_OUT));

void run_kernel(RUNTIME_PARAMETERS params)
{
    [[maybe_unused]] const Operand& buffer_A   = params.buffer_A;
    [[maybe_unused]] const Operand& buffer_B   = params.buffer_B;
    [[maybe_unused]] const Operand& buffer_Res = params.buffer_Res;

    const auto a   = llk2::Tensor<in0_spec>(buffer_A[0]);
    const auto b   = llk2::Tensor<in1_spec>(buffer_B[0]);
    const auto acc = llk2::dst_tensor<RT_DIM * llk2::default_tile_dim, CT_DIM * llk2::default_tile_dim, dest_format>(0); // a's shape in Dest from row 0

#ifdef LLK_TRISC_MATH
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<dest_sync>();
#endif

    if constexpr (IS_SUB)
    {
        llk2::sub<llk2::sub_modifiers {.bcast = bcast}>(a, b, acc); // acc = a - b
    }
    else
    {
        llk2::add<llk2::add_modifiers {.bcast = bcast}>(a, b, acc); // acc = a + b
    }

#ifdef LLK_TRISC_MATH
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
