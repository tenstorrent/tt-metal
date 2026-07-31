// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standalone tt-llk test for the experimental Quasar block reduce_max_row kernel. Drives the LLK
// lib directly. A block of params.TILE_CNT operand tiles is row-max reduced into a
// single result tile.
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

#include "experimental/llk_unpack_AB_reduce_custom_runtime.h"
#include "llk_unpack_common.h"
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

    // no op for unpack thread.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    // Configure the buffer descriptors for the two unpackers (SrcA operand tiles, SrcB single face tile). The unpacker MOP config will use these buffer
    // descriptors to read the input data from L1.
    const auto tensor_shape_A = tensor_shape_from_params(params);

    td_val_A = ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, buf_desc_id_a, formats.unpack_A_dst);
    td_val_B = ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(params.buffer_B[0]), formats.unpack_B_src, buf_desc_id_b, formats.unpack_B_dst);

    _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
    _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);

    // Configure unpacker engines with buffer descriptors.
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);

    // Block of TILE_CNT operand tiles (SrcA) + one scaler face (SrcB) -> one reduced result tile.
    _llk_unpack_AB_reduce_block_max_row_init_runtime_(params.TILE_CNT, false /*respect_trigger*/, buf_desc_id_a, buf_desc_id_b, tensor_shape_A);
    _llk_unpack_AB_reduce_block_max_row_runtime_(params.TILE_CNT, 0 /*operand tile start*/, 0 /*scaler tile*/, buf_desc_id_b, tensor_shape_A);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_reduce_runtime_custom.h"
#include "llk_math_common.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);

    const auto tensor_shape_A = tensor_shape_from_params(params);

    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(src_format, src_format);

    _llk_math_reduce_block_max_row_init_runtime_<is_fp32_dest_acc_en>(params.TILE_CNT, tensor_shape_A);
    _llk_math_reduce_block_max_row_runtime_<is_fp32_dest_acc_en>(0 /*dst_index*/, tensor_shape_A);

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
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc, ckernel::ReluConfig::none());
    _llk_pack_init_(buf_desc_id, tensor_shape_A, 1 /*num_tiles_per_pack*/);
    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_ROW>(tensor_shape_A);

    // Block reduce produces a single result tile.
    _llk_pack_(0, 0, tensor_shape_A);

    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    _llk_pack_reduce_mask_clear_();
}
#endif
