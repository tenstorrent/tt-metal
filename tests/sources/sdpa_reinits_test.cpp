// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "llk_defs.h"
#include "operand.h"
#include "tensix_types.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define UNUSED __attribute__((unused))

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_AB_reduce_custom.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    // Operation 0: Fused Unpack
    UNUSED const Operand buffer_A0(0x1a000, 2048);
    UNUSED const Operand buffer_B0(0x1a800, 2048);
    UNUSED const std::uint32_t unpack_a_src_format0 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_a_dst_format0 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_src_format0 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_dst_format0 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_unpack_hw_configure_<false, false>(unpack_a_src_format0, unpack_b_src_format0, unpack_a_dst_format0, unpack_b_dst_format0, 16, 16, 4, 4, 128, 128);
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_unpack_AB_matmul_init_<>(false, 1, 1, 1, 16, 16);
        {
            std::uint32_t mt = batch;
            for (std::uint32_t kt = 0; kt < 1; ++kt)
            {
                _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A0[0]), L1_ADDRESS(buffer_B0[0]), mt * 1 + kt, kt * 1, 128, 128, false, false, 1, 1, 1);
            }
        }
    }
    // Operation 1: Fused Unpack
    UNUSED const Operand buffer_A1(0x1a000, 2048);
    UNUSED const Operand buffer_B1(0x1a800, 2048);
    UNUSED const std::uint32_t unpack_a_src_format1 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_a_dst_format1 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_src_format1 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_dst_format1 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_unpack_reconfig_data_format_srca_impl_<false, false>(unpack_a_src_format1, unpack_a_dst_format1, 128);
    _llk_unpack_reconfig_data_format_srcb_impl_<false, false>(unpack_b_src_format1, unpack_b_dst_format1, 128);
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_unpack_AB_reduce_block_max_row_init_<1, false>();
        if ((batch * 1 + 0) % 1 == 0)
        {
            _llk_unpack_AB_reduce_block_max_row_(L1_ADDRESS(buffer_A1[batch * 1 + 0]), L1_ADDRESS(buffer_B1[batch * 1 + 0]));
        }
        _llk_unpack_AB_reduce_block_max_row_uninit_(16, 16);
    }
    // Operation 2: Fused Unpack
    UNUSED const Operand buffer_A2(0x1a000, 2048);
    UNUSED const Operand buffer_B2(0x1a800, 2048);
    UNUSED const std::uint32_t unpack_a_src_format2 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_a_dst_format2 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_src_format2 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_dst_format2 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_unpack_reconfig_data_format_srca_impl_<false, false>(unpack_a_src_format2, unpack_a_dst_format2, 128);
    _llk_unpack_reconfig_data_format_srcb_impl_<false, false>(unpack_b_src_format2, unpack_b_dst_format2, 128);
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_unpack_AB_init_<BroadcastType::COL>(16, 4, false, false);
        _llk_unpack_AB_<BroadcastType::COL>(L1_ADDRESS(buffer_A2[batch * 1 + 0]), L1_ADDRESS(buffer_B2[batch * 1 + 0]));
    }
    // Operation 3: Fused Unpack
    UNUSED const Operand buffer_A3(0x1a000, 2048);
    UNUSED const Operand buffer_B3(0x1a800, 2048);
    UNUSED const std::uint32_t unpack_a_src_format3 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_a_dst_format3 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_src_format3 = ckernel::to_underlying(DataFormat::Float16_b);
    UNUSED const std::uint32_t unpack_b_dst_format3 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_unpack_reconfig_data_format_srca_impl_<false, false>(unpack_a_src_format3, unpack_a_dst_format3, 128);
    _llk_unpack_reconfig_data_format_srcb_impl_<false, false>(unpack_b_src_format3, unpack_b_dst_format3, 128);
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_unpack_AB_matmul_init_<>(false, 1, 1, 1, 16, 16);
        {
            std::uint32_t mt = batch;
            for (std::uint32_t kt = 0; kt < 1; ++kt)
            {
                _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A3[0]), L1_ADDRESS(buffer_B3[0]), mt * 1 + kt, kt * 1, 128, 128, false, false, 1, 1, 1);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_reduce_custom.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    // Operation 0: Math Setup
    const std::uint32_t math_format0 = ckernel::to_underlying(DataFormat::Float16_b);
    constexpr DstSync dest_sync0     = DstSync::SyncHalf;
    _llk_math_hw_configure_<false>(math_format0, math_format0);
    _llk_math_pack_sync_init_<dest_sync0, false>();
    _llk_math_reduce_block_max_row_init_<1, false>();

    // Operation 0: Matmul FPU - Using experimental custom no-mop API
    _llk_math_matmul_init_no_mop_<0, 0>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, 1, 1);

    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_math_wait_for_dest_available_<dest_sync0>();
        for (std::uint32_t kt = 0; kt < 1; kt++)
        {
            _llk_math_matmul_no_mop_<0, 0>(0, 1, 1);
        }
        _llk_math_dest_section_done_<dest_sync0, false>();
    }
    // Operation 1: Math Setup
    const std::uint32_t math_format1 = ckernel::to_underlying(DataFormat::Float16_b);
    constexpr DstSync dest_sync1     = DstSync::SyncHalf;
    _llk_math_reconfig_data_format_<false, false>(math_format1, math_format1);
    _llk_math_pack_sync_init_<dest_sync1, false>();

    // Custom addr_mod reinit for reduce_block_max_row (full init done in Operation 0)
    reduce_max_row_configure_addrmod_reinit();

    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_math_wait_for_dest_available_<dest_sync1>();
        _llk_math_reduce_block_max_row_<1, false>(0);
        _llk_math_dest_section_done_<dest_sync1, false>();
    }

    _llk_math_reduce_block_max_row_uninit_();

    // Operation 2: Math Setup
    const std::uint32_t math_format2 = ckernel::to_underlying(DataFormat::Float16_b);
    constexpr DstSync dest_sync2     = DstSync::SyncHalf;
    _llk_math_reconfig_data_format_<false, false>(math_format2, math_format2);
    _llk_math_pack_sync_init_<dest_sync2, false>();

    // Operation 2: Eltwise ELWSUB FPU
    // REDUCE -> SUB TRANSITION REINIT NEEDS TO BE DONE MOSTLY FULL BECAUSE OF MOP RECONFIG
    _llk_math_eltwise_binary_init_<ckernel::EltwiseBinaryType::ELWSUB, BroadcastType::COL, ckernel::MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(4, 0);

    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_math_wait_for_dest_available_<dest_sync2>();
        _llk_math_eltwise_binary_<ELWSUB, BroadcastType::COL, dest_sync2, false, ckernel::MathFidelity::LoFi, EltwiseBinaryReuseDestType::NONE>(4, 0, false);
        _llk_math_dest_section_done_<dest_sync2, false>();
    }
    // Operation 3: Math Setup
    const std::uint32_t math_format3 = ckernel::to_underlying(DataFormat::Float16_b);
    constexpr DstSync dest_sync3     = DstSync::SyncHalf;
    _llk_math_reconfig_data_format_<false, false>(math_format3, math_format3);
    _llk_math_pack_sync_init_<dest_sync3, false>();

    // Operation 3: Matmul FPU - Using experimental custom no-mop API
    // _llk_math_matmul_init_<0, 0>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, 1, 1);

    // TEST MATMUL REINIT FOR JUST 2 ADDR_MODS AFTER ELTWISE BINARY
    // SO THIS ELWSUB BINARY -> MATMUL REINIT STEP§
    matmul_configure_addrmod_reinit();

    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_math_wait_for_dest_available_<dest_sync3>();

        for (std::uint32_t kt = 0; kt < 1; kt++)
        {
            _llk_math_matmul_no_mop_<0, 0>(0, 1, 1);
        }
        _llk_math_dest_section_done_<dest_sync3, false>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "perf.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    // Operation 0: Packer
    const Operand buffer_Res0(0x1b000, 2048);
    const std::uint32_t pack_src_format0 = ckernel::to_underlying(DataFormat::Float16_b);
    const std::uint32_t pack_dst_format0 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_pack_hw_configure_<false, false, false>(pack_src_format0, pack_dst_format0, 128);
    _llk_pack_init_<false, false, false>(pack_dst_format0, pack_dst_format0, 16, TILE_C_DIM, 4, false, false);
    _llk_pack_dest_init_<DstSync::SyncHalf, false>();
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t i = 0; i < 1; ++i)
        {
            std::uint32_t tile_idx = batch * 1 + i;
            _llk_pack_<DstSync::SyncHalf, false, false>(i, L1_ADDRESS(buffer_Res0[tile_idx]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
    }
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // Operation 1: Packer
    const Operand buffer_Res1(0x1b800, 2048);
    const std::uint32_t pack_src_format1 = ckernel::to_underlying(DataFormat::Float16_b);
    const std::uint32_t pack_dst_format1 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_pack_reconfig_data_format_<false, false>(pack_src_format1, pack_dst_format1, 128);
    _llk_pack_init_<false, false, false>(pack_dst_format1, pack_dst_format1, 16, TILE_C_DIM, 4, false, false);
    _llk_pack_dest_init_<DstSync::SyncHalf, false>();
    _llk_pack_reduce_mask_config_<false, ckernel::ReduceDim::REDUCE_ROW>();
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t i = 0; i < 1; ++i)
        {
            std::uint32_t tile_idx = batch * 1 + i;
            _llk_pack_<DstSync::SyncHalf, false, false>(i, L1_ADDRESS(buffer_Res1[tile_idx]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
    }
    t6_semaphore_post<>(semaphore::PACK_DONE);

    _llk_pack_reduce_mask_clear_();
    // Operation 2: Packer
    const Operand buffer_Res2(0x1c000, 2048);
    const std::uint32_t pack_src_format2 = ckernel::to_underlying(DataFormat::Float16_b);
    const std::uint32_t pack_dst_format2 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_pack_reconfig_data_format_<false, false>(pack_src_format2, pack_dst_format2, 128);
    _llk_pack_init_<false, false, false>(pack_dst_format2, pack_dst_format2, 16, TILE_C_DIM, 4, false, false);
    _llk_pack_dest_init_<DstSync::SyncHalf, false>();
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t i = 0; i < 1; ++i)
        {
            std::uint32_t tile_idx = batch * 1 + i;
            _llk_pack_<DstSync::SyncHalf, false, false>(i, L1_ADDRESS(buffer_Res2[tile_idx]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
    }
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // Operation 3: Packer
    const Operand buffer_Res3(0x1c800, 2048);
    const std::uint32_t pack_src_format3 = ckernel::to_underlying(DataFormat::Float16_b);
    const std::uint32_t pack_dst_format3 = ckernel::to_underlying(DataFormat::Float16_b);
    _llk_pack_reconfig_data_format_<false, false>(pack_src_format3, pack_dst_format3, 128);
    _llk_pack_init_<false, false, false>(pack_dst_format3, pack_dst_format3, 16, TILE_C_DIM, 4, false, false);
    _llk_pack_dest_init_<DstSync::SyncHalf, false>();
    for (std::uint32_t batch = 0; batch < 1; ++batch)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t i = 0; i < 1; ++i)
        {
            std::uint32_t tile_idx = batch * 1 + i;
            _llk_pack_<DstSync::SyncHalf, false, false>(i, L1_ADDRESS(buffer_Res3[tile_idx]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
    }
}

#endif
