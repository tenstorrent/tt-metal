// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// AI-generated — run_id: 2026-04-29_binary_max_min_quasar_dualpath
//
// Three-operand SFPU test for binary max/min on Quasar.
//
// Two execution paths, selected at runtime by `unpack_to_dest`:
//   * unpack_to_dest=true   — UNPACK → Dest directly. Used for 32-bit Dest formats
//                             (Float32, Int32 with dest_acc=Yes).
//   * unpack_to_dest=false  — UNPACK → SrcA → FPU datacopy → Dest. Required for
//                             non-32-bit and MX block formats (Float16_b, MxFp8R,
//                             MxFp8P), which the unpacker cannot route directly to
//                             Dest because the block-exponent conversion happens
//                             on the FPU datacopy side.
//
// Buffer layout (params.TILE_CNT == 2):
//   buffer_A[0] = in0 tile   → Dest[DST_INDEX + 0]
//   buffer_A[1] = in1 tile   → Dest[DST_INDEX + 1]
//   SFPU writes              → Dest[DST_INDEX + 2]
//   buffer_Res[0] = result   ← Dest[DST_INDEX + 2] (1 tile packed)
//
// Test flow (per kernel run):
//   T0 unpack: stage both input tiles from buffer_A in L1 into Dest.
//              unpack_to_dest=true  → write Dest tiles 0 and 1 directly.
//              unpack_to_dest=false → write into SrcA; T1 will datacopy.
//   T1 math:   on the FPU path, datacopy SrcA → Dest tiles 0 and 1.
//              Then run the SFPU max/min kernel: read Dest tiles 0 and 1,
//              compute element-wise max or min, write Dest tile 2.
//   T2 pack:   pack the single result tile (Dest tile 2) out to buffer_Res in L1.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id = 0;               // T0 source descriptor slot for buffer_A
    const std::uint32_t num_tiles   = params.TILE_CNT; // 2 input tiles (in0, in1) packed in buffer_A

    // DEST DVALID handshake: T0 is the producer; T1 (FPU or SFPU) and T2 (PACK) are the consumers.
    // The producer field differs per path: UNPACK on the unpack-to-dest path, FPU on the datacopy path.
    // The hw_configure call only applies to the unpack-to-dest path — it programs Dest to receive
    // unpacker writes directly. On the FPU path, srcAB hw_configure on T1 covers it instead.
    if constexpr(unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    // Source descriptor: buffer_A in L1, L1-side format = formats.unpack_A_src,
    // face geometry from the harness. buffer_A holds both input tiles back-to-back.
    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    // TDMA descriptor: bind the buffer descriptor to slot 0; reg_data_format =
    // unpack_A_dst is the Dest-side (post-conversion) format.
    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    // For 32-bit Dest with the FPU-datacopy path, Mov2D is implemented via ELWADD,
    // so both UNP_A and UNP_B must be configured with the same descriptor.
    // All other variants use the unary configuration on a single unpacker engine.
    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    // Configure unpacker → init for `num_tiles` consecutive tiles → unpack starting at L1 tile 0.
    // The single _llk_unpack_unary_operand_ call walks both tiles internally because init was
    // primed with num_tiles=2; tile 0 lands in Dest tile 0, tile 1 in Dest tile 1.
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);

    // Release the Dest section to the next consumer (SFPU on the unpack-to-dest path).
    // On the FPU path, T1's _llk_math_set_dvalid_<FPU> handles the release instead.
    if constexpr (unpack_to_dest)
    {
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // DEST DVALID handshake: T1 plays two roles on the FPU path (it's both the FPU
    // producer for the datacopy step and the SFPU producer for the max/min step), so
    // both clients must be registered. On the unpack-to-dest path T1 is only the SFPU
    // client; the producer is T0/UNPACK.
    if constexpr (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat math_format     = static_cast<DataFormat>(formats.math);
    DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);

    // srcAB hw_configure: srcA and srcB share the math format; Dest mode follows the
    // int / float split (int32 for integer formats, otherwise is_fp32_dest_acc_en).
    if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
    }
    else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
    }
    else
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
    }

    // FPU-datacopy path: move both input tiles from SrcA into Dest at DST_INDEX + i.
    // Required for non-32-bit and MX formats; skipped when the unpacker has already
    // placed data directly in Dest. After the moves, release Dest to the SFPU step.
    if constexpr (!unpack_to_dest)
    {
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    // SFPU step: read Dest tile 0 (in0) and Dest tile 1 (in1), write Dest tile 2 (out).
    // dst_index_in0/in1/out are tile offsets relative to params.DST_INDEX.
    _llk_math_eltwise_sfpu_init_();
    _init_binary_max_min_();

    // All integer formats route through the Int32 path (sfpmem::INT32); float and MX
    // formats use the DEFAULT path (sfpmem::DEFAULT, HW derives the mode from ACC_CTRL).
    if (math_format == DataFormat::Int32)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_binary_max_min<DataFormat::Int32, IS_MAX_OP, 8 /*ITERATIONS*/>,
            params.DST_INDEX,
            VectorMode::RC,
            /* dst_index_in0 */ 0U,
            /* dst_index_in1 */ 1U,
            /* dst_index_out */ 2U);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_binary_max_min<DataFormat::Float32, IS_MAX_OP, 8 /*ITERATIONS*/>,
            params.DST_INDEX,
            VectorMode::RC,
            /* dst_index_in0 */ 0U,
            /* dst_index_in1 */ 1U,
            /* dst_index_out */ 2U);
    }

    // Hand Dest off to PACK.
    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Drain SFPU/FPU/MOP queues before this thread returns, so PACK doesn't see
    // stale state from in-flight math instructions.
    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    std::uint32_t const buf_desc_id = 8; // T2 destination descriptor slot for buffer_Res
    // Only the SFPU output tile (Dest tile 2) is packed out; the two input tiles
    // staged in Dest tiles 0/1 are not part of the result.
    const std::uint32_t num_tiles_per_pack = 1;

    // PACK is the final consumer of the Dest DVALID chain. The producer field
    // tracks which path wrote Dest (UNPACK on the unpack-to-dest path, FPU otherwise).
    if constexpr (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    // Destination descriptor: buffer_Res in L1, L1-side format = formats.pack_dst,
    // face geometry from the harness.
    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = params.buffer_Res[0] / 16;
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    // TDMA descriptor: bind buffer_Res to slot 8; reg_data_format = pack_src is the
    // Dest-side format the packer reads.
    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    // Configure pack engine 0 → init for one tile → pack Dest tile 2 (the SFPU output)
    // into buffer_Res → release the Dest section.
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX + 2, 0, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif // LLK_TRISC_PACK
