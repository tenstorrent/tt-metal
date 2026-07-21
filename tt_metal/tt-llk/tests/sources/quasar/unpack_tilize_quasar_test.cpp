// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    tdma_descriptor_t td_val;
    const std::uint32_t buf_desc_id = 0;

    // Setup data valid scheme
    constexpr auto dest_producer = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_producer, dest_dvalid_client::PACK});

    if constexpr (unpack_to_dest && is_fp32_dest_acc_en)
    {
        const bool int32_dest = static_cast<DataFormat>(formats.unpack_A_src) == DataFormat::Int32;
        // Dst is in 32b mode (and we unpack directly to dest) determine whether it's Float32 or Int32 from the unpack source format.
        if (int32_dest)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false, true>();
        }
        else
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, true, false>();
        }
    }
    else if constexpr (unpack_to_dest)
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false, false>();
    }

    const auto tensor_shape = tensor_shape_from_params(params);

    unsigned l1_addr_16B;
    if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_A || UNPACKER_ENGINE_SEL == p_unpacr::UNP_DEST)
    {
        l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    }
    else if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
    {
        l1_addr_16B = L1_ADDRESS(params.buffer_B[0]);
    }

    if (tensor_shape.face_r_dim <= ckernel::unpack::UNPACR_STRIDE_MAX_ROWS)
    {
        td_val = ckernel::trisc::construct_tdma_desc<L1AccessMode::Strided>(tensor_shape, l1_addr_16B, formats.unpack_A_src, buf_desc_id, formats.unpack_A_dst);
    }
    else
    {
        td_val = ckernel::trisc::construct_tdma_desc(tensor_shape, l1_addr_16B, formats.unpack_A_src, buf_desc_id, formats.unpack_A_dst);
    }

    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        // If Dst fmt is 32b and operation is Mov2D, we need both SrcA/B fmts to be configured since Mov2D will be implemented via ELWADD
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    std::uint32_t y_stride_external = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;
    if constexpr (unpack_to_dest)
    {
        // Batched tilize directly into DEST using block API.
        // DST_Z_STRIDE=num_faces so each Dst_Z_Cntr_inc advances DEST by one full tile.
        // L1 and DEST counters are set once per row.
        _llk_unpack_tilize_block_init_<FULL_CT_DIM, BLOCK_CT_DIM>(buf_desc_id, tensor_shape);
        for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
        {
            _llk_unpack_tilize_block_(y * y_stride_external, y * BLOCK_CT_DIM);
        }
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
    else if (tensor_shape.face_r_dim < FACE_R_DIM)
    {
        _llk_unpack_tilize_strided_init_small_faces_<UNPACKER_ENGINE_SEL, is_fp32_dest_acc_en, FULL_CT_DIM>(buf_desc_id, tensor_shape);
        for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
        {
            _llk_unpack_tilize_strided_small_faces_<UNPACKER_ENGINE_SEL, FULL_CT_DIM>(tensor_shape, y * FULL_CT_DIM);
        }
    }
    else
    {
        _llk_unpack_tilize_init_<UNPACKER_ENGINE_SEL, is_fp32_dest_acc_en>(buf_desc_id, FULL_CT_DIM, BLOCK_CT_DIM, tensor_shape);
        for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
        {
            _llk_unpack_tilize_<UNPACKER_ENGINE_SEL>(y * y_stride_external);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (!unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

        DataFormat src_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(
            params.num_faces * params.TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(i);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = params.TILE_CNT;

    constexpr auto dest_producer = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_producer, dest_dvalid_client::PACK});

    const auto tensor_shape = tensor_shape_from_params(params);

    tdma_descriptor_t tdma_desc =
        ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);

    _llk_pack_init_(buf_desc_id, tensor_shape, num_tiles_per_pack);
    _llk_pack_(0, 0, tensor_shape);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
