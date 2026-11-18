// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Back-to-back re-programming of the fixed buffer descriptor of Unp0 and Pack0 (see ckernel_buf_desc.h).
// Each block programs the engine's descriptor BFD_REPROGRAMS (4) times in a row (the first
// BFD_REPROGRAMS - 1 point at other blocks as decoys), then inits and executes at tile 0, right after
// the previous block's MOP was issued. A stale or misordered descriptor reads/writes the wrong tiles.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Descriptor programs per block: BFD_REPROGRAMS - 1 decoys + the real one.
constexpr std::uint32_t BFD_REPROGRAMS = 4;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats               = params.formats;
    const Operand& buffer_A                   = params.buffer_A;
    const std::uint32_t num_blocks            = params.NUM_BLOCKS;
    const std::uint32_t tiles_in_block        = params.NUM_TILES_IN_BLOCK;
    const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);
    constexpr auto unp_res                    = ckernel::trisc::BFD_ID_UNP0;

    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_A_dst));

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        for (std::uint32_t k = 1; k < BFD_REPROGRAMS; k++)
        {
            const std::uint32_t decoy = (block + k) % num_blocks;
            ckernel::trisc::program_bfd<unp_res>(tensor_shape_A, L1_ADDRESS(buffer_A[decoy * tiles_in_block]), formats.unpack_A_src);
        }
        ckernel::trisc::program_bfd<unp_res>(tensor_shape_A, L1_ADDRESS(buffer_A[block * tiles_in_block]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_unpack = unp_res;
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(bfd_unpack, tensor_shape_A, tiles_in_block);
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(0 /*l1_tile_idx*/, tensor_shape_A);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats        = params.formats;
    const std::uint32_t num_blocks     = params.NUM_BLOCKS;
    const std::uint32_t tiles_in_block = params.NUM_TILES_IN_BLOCK;

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
    const DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format);
    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(
        params.num_faces * params.TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        for (std::uint32_t i = 0; i < tiles_in_block; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(params.DST_INDEX + i);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats               = params.formats;
    const Operand& buffer_Res                 = params.buffer_Res;
    const std::uint32_t num_blocks            = params.NUM_BLOCKS;
    const std::uint32_t tiles_in_block        = params.NUM_TILES_IN_BLOCK;
    const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);
    constexpr auto pack_res                   = ckernel::trisc::BFD_ID_PACK0;

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        for (std::uint32_t k = 1; k < BFD_REPROGRAMS; k++)
        {
            const std::uint32_t decoy = (block + k) % num_blocks;
            ckernel::trisc::program_bfd<pack_res>(tensor_shape_A, L1_ADDRESS(buffer_Res[decoy * tiles_in_block]), formats.pack_dst);
        }
        ckernel::trisc::program_bfd<pack_res>(tensor_shape_A, L1_ADDRESS(buffer_Res[block * tiles_in_block]), formats.pack_dst);
        constexpr std::uint8_t bfd_pack = pack_res;
        _llk_pack_init_(bfd_pack, tensor_shape_A, tiles_in_block);
        _llk_pack_(params.DST_INDEX, 0 /*start_l1_tile_idx*/, tensor_shape_A);
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
