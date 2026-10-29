// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unary broadcast with acc_to_dest: a plain broadcast pass of A seeds Dest, then a broadcast pass of B runs with
// acc_to_dest = ACC_TO_DEST. The result is bcast(A) + bcast(B) when accumulating, bcast(B) otherwise.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"

// perf.h declares PerfRunType, which the generated params.h uses.
#include "params.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

static_assert(!unpack_to_dest, "acc_to_dest is not supported when unpacking to dest");

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_broadcast_operands.h"

template <bool acc_to_dest>
inline void unpack_block(const std::uint32_t bfd_unpack, const std::uint32_t first_tile, const std::uint32_t tiles_in_block)
{
    _llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_B, BROADCAST_TYPE, is_fp32_dest_acc_en, false /*unpack_to_dest*/, acc_to_dest>(
        bfd_unpack, 1 /*num_tiles*/);
    for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
    {
        _llk_unpack_unary_broadcast_operands_<p_unpacr::UNP_B, false /*unpack_to_dest*/>(first_tile + tile);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(params.INPUT_NUM_BLOCKS);
    const std::uint32_t tiles_in_block = static_cast<std::uint32_t>(params.INPUT_NUM_TILES_IN_BLOCK);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();

    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
    const auto bfd_unpack_A =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(tensor_shape, L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);
    const auto bfd_unpack_B =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(tensor_shape, L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src);
    // The accumulating pass zeroes SrcA for ELWADD, so both unpackers carry the operand format.
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
        static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        unpack_block<false /*acc_to_dest*/>(bfd_unpack_A, block * tiles_in_block, tiles_in_block);
        unpack_block<ACC_TO_DEST>(bfd_unpack_B, block * tiles_in_block, tiles_in_block);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_unary_broadcast.h"

using namespace ckernel;
using namespace ckernel::math;

template <bool acc_to_dest>
inline void broadcast_block(const ckernel::TensorShape tensor_shape, const std::uint32_t tiles_in_block)
{
    _llk_math_eltwise_unary_broadcast_init_<BROADCAST_TYPE, is_fp32_dest_acc_en, false /*unpack_to_dest*/, acc_to_dest>(tensor_shape);
    for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
    {
        _llk_math_eltwise_unary_broadcast_<false /*unpack_to_dest*/>(tile);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t tiles_in_block      = static_cast<std::uint32_t>(params.OUTPUT_NUM_TILES_IN_BLOCK);
    const std::uint32_t num_blocks          = static_cast<std::uint32_t>(params.INPUT_NUM_BLOCKS);
    const DataFormat math_format            = static_cast<DataFormat>(formats.math);
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    _configure_default_alu_data_format_state_<is_fp32_dest_acc_en>(math_format, math_format);

    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        broadcast_block<false /*acc_to_dest*/>(tensor_shape, tiles_in_block);
        broadcast_block<ACC_TO_DEST>(tensor_shape, tiles_in_block);
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t output_num_blocks     = static_cast<std::uint32_t>(params.OUTPUT_NUM_BLOCKS);
    const std::uint32_t output_tiles_in_block = static_cast<std::uint32_t>(params.OUTPUT_NUM_TILES_IN_BLOCK);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
    const auto bfd_pack =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(tensor_shape, L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, tensor_shape, 1 /*num_tiles*/);

    for (std::uint32_t block = 0; block < output_num_blocks; block++)
    {
        for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
        {
            _llk_pack_(tile, (block * output_tiles_in_block) + tile, tensor_shape);
        }
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
