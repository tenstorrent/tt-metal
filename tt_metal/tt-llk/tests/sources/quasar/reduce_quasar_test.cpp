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
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_reduce.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    tdma_descriptor_t td_val_A;
    tdma_descriptor_t td_val_B;
    const std::uint32_t buf_desc_id_a = 0;
    const std::uint32_t buf_desc_id_b = 1;

    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    const auto tensor_shape_A = tensor_shape_from_params(params);

    td_val_A = ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, buf_desc_id_a, formats.unpack_A_dst);
    td_val_B = ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(params.buffer_B[0]), formats.unpack_B_src, buf_desc_id_b, formats.unpack_B_dst);

    _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
    _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);

    _llk_unpack_reduce_init_<POOL_TYPE, REDUCE_DIM>(buf_desc_id_a, buf_desc_id_b, tensor_shape_A, 1 /*num_tiles_per_unpack*/);

    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_unpack_reduce_(i, 0, tensor_shape_A);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Setup data valid scheme
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);

    const auto tensor_shape_A = tensor_shape_from_params(params);

    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /* int32 dest */>(src_format, src_format);
    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, MATH_FIDELITY>(tensor_shape_A);
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_reduce_(i);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
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
    std::uint32_t const buf_desc_id = 8;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    const auto tensor_shape_A = tensor_shape_from_params(params);

    tdma_descriptor_t tdma_desc =
        ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, tensor_shape_A, 1 /*num_tiles_per_pack*/);
    _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape_A);
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_pack_(i, i, tensor_shape_A);
    }
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    _llk_pack_reduce_mask_clear_();
}
#endif
