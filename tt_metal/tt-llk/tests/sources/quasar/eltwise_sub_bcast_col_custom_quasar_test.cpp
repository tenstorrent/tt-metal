// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SDPA blocked bcast-col SUB with SrcB reuse (Quasar).
//
// Per block, UNPACK unpacks ONE SrcB tile in COL layout and holds it, then NUM_TILES_IN_BLOCK
// SrcA column tiles; MATH subtracts the reused SrcB from each SrcA tile into its own dest slot.
// NUM_TILES_IN_BLOCK is the ct_dim of the op, NUM_BLOCKS is the number of dest sections.

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id_a = 0;
    const std::uint32_t buf_desc_id_b = 1;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    const auto tensor_shape = tensor_shape_from_params(params);

    const tdma_descriptor_t td_val_A =
        ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, buf_desc_id_a, formats.unpack_A_dst);
    const tdma_descriptor_t td_val_B =
        ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(params.buffer_B[0]), formats.unpack_B_src, buf_desc_id_b, formats.unpack_B_dst);

    _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
    _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A.reg_data_format, td_val_B.reg_data_format);

    _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);

    const std::uint32_t ct_dim     = params.INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks = static_cast<std::uint32_t>(params.INPUT_NUM_BLOCKS);

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        // SrcB is the same single tile for every block, so its L1 tile index stays 0.
        _llk_unpack_AB_sub_bcast_col_custom_(buf_desc_id_a, buf_desc_id_b, block * ct_dim, 0 /*start_l1_tile_idx_1*/, ct_dim, tensor_shape);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_binary_custom.h"
#include "llk_math_common.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);

    const auto tensor_shape = tensor_shape_from_params(params);

    // The addr-mods are identical for every block, so init once outside the loop.
    _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE>(tensor_shape);

    const std::uint32_t ct_dim     = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks = static_cast<std::uint32_t>(params.OUTPUT_NUM_BLOCKS);

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        _llk_math_sub_bcast_cols_reuse_custom_(ct_dim, tensor_shape, 0 /*dst_index*/);
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
    const std::uint32_t buf_desc_id = 8;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    const auto tensor_shape = tensor_shape_from_params(params);

    const tdma_descriptor_t tdma_desc =
        ckernel::trisc::construct_tdma_desc(tensor_shape, L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc.reg_data_format, ckernel::ReluConfig::none());
    _llk_pack_init_(buf_desc_id, tensor_shape, 1 /*num_tiles_per_pack*/);

    const std::uint32_t ct_dim     = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks = static_cast<std::uint32_t>(params.OUTPUT_NUM_BLOCKS);

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        for (std::uint32_t tile = 0; tile < ct_dim; tile++)
        {
            _llk_pack_(tile, (block * ct_dim) + tile, tensor_shape);
        }
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
