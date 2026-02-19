// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = params->TILE_CNT;

    if (unpack_to_dest)
    {
        // Unpacking to DEST directly
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = L1_ADDRESS(params->buffer_A[0]);
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_src);
    bd_val.f.x_dim       = params->TEST_FACE_C_DIM;
    bd_val.f.y_dim       = params->TEST_FACE_R_DIM;
    bd_val.f.z_dim       = params->num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    if (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        // If Dst fmt is 32b and operation is Mov2D, we need both SrcA/B fmts to be configured since Mov2D will be implemented via ELWADD
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles_per_unpack);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0);

    if (unpack_to_dest)
    {
        _llk_unpack_dest_dvalid_section_done_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "params.h"

// Include all necessary SFPU headers
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_tanh.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// Helper to dispatch SFPU operations based on SFPU_UNARY_OPERATION
// Uses template metaprogramming to convert constexpr to template parameter
template <SfpuType op>
struct sfpu_op_dispatcher;

template <>
struct sfpu_op_dispatcher<SfpuType::exponential>
{
    static void call(int tile_idx, int num_sfpu_iterations)
    {
        _llk_math_eltwise_unary_sfpu_params_<false>(_calculate_exp_<true>, tile_idx, num_sfpu_iterations);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::relu>
{
    static void call(int tile_idx, int num_sfpu_iterations)
    {
        _llk_math_eltwise_unary_sfpu_params_<false>(_calculate_relu_, tile_idx, num_sfpu_iterations);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::reciprocal>
{
    static void call(int tile_idx, int num_sfpu_iterations)
    {
        _llk_math_eltwise_unary_sfpu_params_<false>(_calculate_reciprocal_<true>, tile_idx, num_sfpu_iterations);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::sqrt>
{
    static void call(int tile_idx, int num_sfpu_iterations)
    {
        _llk_math_eltwise_unary_sfpu_params_<false>(_calculate_sqrt_<true>, tile_idx, num_sfpu_iterations);
    }
};

template <>
struct sfpu_op_dispatcher<SfpuType::tanh>
{
    static void call(int tile_idx, int num_sfpu_iterations)
    {
        _llk_math_eltwise_unary_sfpu_params_<false>(_calculate_tanh_<true>, tile_idx, num_sfpu_iterations);
    }
};

// Convert constexpr SFPU_UNARY_OPERATION to template parameter using tag dispatch
inline void call_sfpu_operation_quasar(int tile_idx, int num_sfpu_iterations)
{
    // The compiler should optimize this to a direct call based on SFPU_UNARY_OPERATION
    constexpr SfpuType op = SFPU_UNARY_OPERATION;
    switch (op)
    {
        case SfpuType::exponential:
            sfpu_op_dispatcher<SfpuType::exponential>::call(tile_idx, num_sfpu_iterations);
            break;
        case SfpuType::relu:
            sfpu_op_dispatcher<SfpuType::relu>::call(tile_idx, num_sfpu_iterations);
            break;
        case SfpuType::reciprocal:
            sfpu_op_dispatcher<SfpuType::reciprocal>::call(tile_idx, num_sfpu_iterations);
            break;
        case SfpuType::sqrt:
            sfpu_op_dispatcher<SfpuType::sqrt>::call(tile_idx, num_sfpu_iterations);
            break;
        case SfpuType::tanh:
            sfpu_op_dispatcher<SfpuType::tanh>::call(tile_idx, num_sfpu_iterations);
            break;
        default:
            break;
    }
}

void run_kernel(const volatile struct RuntimeParams *params)
{
    // Setup dvalid for MATH kernel
    if (unpack_to_dest)
    {
        // Chain must match UNPACK's chain: {UNPACK, SFPU, PACK}
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    const int num_sfpu_iterations = params->TEST_FACE_R_DIM / ckernel::math::SFP_ROWS;

    if (!unpack_to_dest)
    {
        const std::uint32_t num_rows = params->num_faces * params->TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);

        // Datacopy all tiles from SRC to DEST
        for (int i = 0; i < params->TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(num_rows, i);
        }

        _llk_math_set_dvalid_<p_cleardvalid::FPU>();
    }

    _llk_math_eltwise_unary_sfpu_init_();

    // Apply SFPU operation to all tiles using compile-time dispatch
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        call_sfpu_operation_quasar(i, num_sfpu_iterations);
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU>();

    // Wait for all operations to complete
    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = params->TILE_CNT;

    // Setup dvalid for PACK
    if (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = params->buffer_Res[0] / 16;
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params->TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params->TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params->num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_<p_pacr::PACK0>(buf_desc_id, num_tiles_per_pack);
    _llk_pack_<p_pacr::PACK0>(params->DST_INDEX, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
