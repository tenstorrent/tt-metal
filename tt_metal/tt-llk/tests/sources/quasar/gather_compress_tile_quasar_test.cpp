// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "quasar_test_common.h"
#include "tensix.h"
#include "tensix_types.h"

constexpr std::uint32_t ROW_ELEMS    = ckernel::FACE_C_DIM;
constexpr std::uint32_t TILE_ELEMS   = ckernel::TILE_R_DIM * ckernel::TILE_C_DIM;
constexpr std::uint32_t TILE_VECTORS = TILE_ELEMS / GATHER_COMPRESS_ELEM_CNT;
constexpr std::uint32_t VECTOR_ROWS  = GATHER_COMPRESS_ELEM_CNT / ROW_ELEMS; // 16-element L1 rows per vector

// Gather pushes one index byte per element. Compress pushes GATHER_COMPRESS_ELEM_CNT 1-bit mask values packed
// into the vec FIFO entry, i.e. GATHER_COMPRESS_ELEM_CNT/8 bytes.
constexpr std::uint32_t ARG_PUSH_BYTES =
    (GATHER_COMPRESS_MODE == ckernel::GatherCompressModeSelect::COMPRESS) ? GATHER_COMPRESS_ELEM_CNT / 8 : GATHER_COMPRESS_ELEM_CNT;

using namespace ckernel;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_vector.h"
#include "experimental/llk_unpack_gather_compress.h"
#include "llk_bfd_alloc.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
    const Operand& buffer_B = params.buffer_B;
#endif
    auto unpack_arg_mbox = (std::uint8_t*)UNPACKER_ARG_MAILBOX_BASE1;

    // UNPACR_STRIDE indexes 16-element L1 rows as tiles.
    constexpr ckernel::TensorShape src_shape =
        ckernel::make_tensor_shape(1 /*face_r_dim*/, ROW_ELEMS /*face_c_dim*/, 1 /*num_faces_r_dim*/, 1 /*num_faces_c_dim*/);

    const std::uint8_t src_bfd_id = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1, ckernel::trisc::L1AccessMode::Strided>(
        src_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_B_src);

    _llk_unpack_hw_configure_<p_unpacr::UNP_B>(static_cast<DataFormat>(formats.unpack_B_dst));
    _llk_unpack_gather_compress_init_<GATHER_COMPRESS_MODE>(src_bfd_id, static_cast<DataFormat>(formats.unpack_B_dst));

    // One ARG FIFO entry's worth of index/mask bytes, in as few registers as hold it.
    constexpr std::uint32_t VREG_BYTES = 16; // one RVV vector register at VLEN=128
    constexpr vregmult LMUL            = (ARG_PUSH_BYTES == 4 * VREG_BYTES) ? M4 : (ARG_PUSH_BYTES == 2 * VREG_BYTES) ? M2 : M1;
    (void)vsetvl<E8, LMUL, ARG_PUSH_BYTES>();

    // Per vector, queue the MOP first, then feed its ARG FIFO entry: the queued UNPACR1_STRIDE stalls until the entry arrives.
    for (std::uint32_t vector = 0; vector < TILE_VECTORS; vector++)
    {
        vector_load<V0>(reinterpret_cast<const std::uint8_t*>(buffer_B[0] + vector * ARG_PUSH_BYTES));
        _llk_unpack_gather_compress_(vector * VECTOR_ROWS /*l1_row_16_idx*/);
        vector_store<V0>(unpack_arg_mbox);
    }

    _llk_unpack_gather_compress_uninit_();
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_gather_compress.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
    const DataFormat src_reg_format = static_cast<DataFormat>(formats.unpack_B_dst);
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en>(src_reg_format, src_reg_format);

    _llk_math_gather_compress_init_<is_fp32_dest_acc_en>(src_reg_format);

    for (std::uint32_t vector = 0; vector < TILE_VECTORS; vector++)
    {
        _llk_math_gather_compress_();
        _llk_math_set_dvalid_<p_cleardvalid::FPU, DstSync::SyncHalf>();
    }

    _llk_math_gather_compress_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_gather_compress.h"
#include "llk_bfd_alloc.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res;
#endif

    constexpr ckernel::TensorShape res_shape =
        ckernel::make_tensor_shape(GATHER_COMPRESS_ELEM_CNT / ROW_ELEMS /*face_r_dim*/, ROW_ELEMS /*face_c_dim*/, 1 /*num_faces_r_dim*/, 1 /*num_faces_c_dim*/);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(res_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_gather_compress_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>());

    // Vector v packs to L1 slot v, so packing all TILE_VECTORS vectors rebuilds the row-major tile.
    for (std::uint32_t vector = 0; vector < TILE_VECTORS; vector++)
    {
        _llk_pack_gather_compress_(vector /*l1_vector_idx*/);
        _llk_pack_dest_dvalid_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif
