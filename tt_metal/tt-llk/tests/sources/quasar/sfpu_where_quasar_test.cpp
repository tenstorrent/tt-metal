// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Ternary SFPU where test for Quasar — mirrors the Blackhole ttnn_where_test
// pattern: three independent L1 input buffers (buffer_A = condition,
// buffer_B = true_val, buffer_C = false_val), routed through DEST via the
// FPU-datacopy or unpack-to-dest path, then dispatched through the shared
// `_llk_math_eltwise_ternary_sfpu_params_` wrapper.
//
// Buffer layout (params.TILE_CNT == 1 per input):
//   buffer_A[0] = condition tile → Dest[DST_INDEX + 0]
//   buffer_B[0] = true_val  tile → Dest[DST_INDEX + 1]
//   buffer_C[0] = false_val tile → Dest[DST_INDEX + 2]
//   SFPU writes                  → Dest[DST_INDEX + 0]  (overwrites condition tile,
//                                                       matches Blackhole convention)
//   buffer_Res[0] = result       ← Dest[DST_INDEX + 0]
//
// Two execution paths, selected at runtime by `unpack_to_dest`:
//   * unpack_to_dest=true   — UNPACK → Dest directly. Used for 32-bit Dest formats
//                             (Float32 with dest_acc=Yes).
//   * unpack_to_dest=false  — UNPACK → SrcA → FPU datacopy → Dest. Required for
//                             non-32-bit formats.

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

    // DEST DVALID handshake: T0 is the producer on the unpack-to-dest path; on the
    // FPU-datacopy path T1 (FPU) is the producer instead.
    if constexpr (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    // One buf_desc per input. The hw_configure pass below only consumes one
    // tdma_descriptor — but the per-input MOP init reads from the matching
    // buf_desc_id, so all three must be in the buf_desc table. All three share
    // format / face geometry; only the L1 address (and slot id) differs.
    const std::uint32_t buf_desc_cond  = 0;
    const std::uint32_t buf_desc_true  = 1;
    const std::uint32_t buf_desc_false = 2;

    buffer_descriptor_u bd_cond = {0};
    bd_cond.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_cond.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_cond.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_cond.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_cond.f.z_dim             = params.num_faces;

    buffer_descriptor_u bd_true = bd_cond;
    bd_true.f.l1_addr_16B       = L1_ADDRESS(params.buffer_B[0]);

    buffer_descriptor_u bd_false = bd_cond;
    bd_false.f.l1_addr_16B       = L1_ADDRESS(params.buffer_C[0]);

    tdma_descriptor_t td_cond;
    td_cond.buf_desc        = bd_cond;
    td_cond.buf_desc_id     = buf_desc_cond;
    td_cond.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(buf_desc_cond, bd_cond);
    _configure_buf_desc_table_(buf_desc_true, bd_true);
    _configure_buf_desc_table_(buf_desc_false, bd_false);

    // For 32-bit Dest with the FPU-datacopy path, Mov2D is implemented via ELWADD,
    // so both UNP_A and UNP_B must be configured with the same descriptor. Use td_cond
    // (all three share format / face geometry — only the L1 address differs).
    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_cond, td_cond);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_cond);
    }

    // Issue three single-tile unpacks, one per input. The standard
    // `_llk_unpack_unary_operand_` resets the DST tile counter to 0 inside the
    // call, which would cause all three unpacks to write to DEST[0]. To put
    // the three inputs in DEST[0]/[1]/[2] we run the first via the wrapper
    // (which sets DST=0 and increments to 1 after the MOP), then for tiles 2
    // and 3 reconfigure the MOP for the new buf_desc_id and re-run it without
    // resetting the DST counter — the post-MOP TILE_INC chains the writes to
    // DEST[1] then DEST[2]. SRC counter is reset per tile (each L1 buffer is
    // independent, l1_tile_idx=0). `UNP_DEST` shares UNP_A's counters per the
    // wrapper convention.
    constexpr std::uint32_t COUNTER_UNP = (UNPACKER_ENGINE_SEL == p_unpacr::UNP_DEST) ? p_unpacr::UNP_A : UNPACKER_ENGINE_SEL;

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_cond, 1);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/); // → DEST[0], DST counter → 1

    _llk_unpack_unary_operand_mop_config_<UNPACKER_ENGINE_SEL, is_fp32_dest_acc_en>(buf_desc_true, 1);
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, COUNTER_UNP, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer); // → DEST[1], DST counter → 2

    _llk_unpack_unary_operand_mop_config_<UNPACKER_ENGINE_SEL, is_fp32_dest_acc_en>(buf_desc_false, 1);
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, COUNTER_UNP, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer); // → DEST[2], DST counter → 3

    if constexpr (unpack_to_dest)
    {
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "experimental/ckernel_sfpu_where.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (unpack_to_dest)
    {
        // Chain: UNPACK (writes DEST) → SFPU (reads/writes DEST) → PACK (reads DEST).
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        // Chain: UNPACK → SrcA → FPU datacopy (MOVA2D) → DEST → SFPU → PACK.
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    if constexpr (!unpack_to_dest)
    {
        // FPU path: datacopy all 3 tiles from SrcA into DEST at tile indices 0, 1, 2.
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);

        constexpr std::uint32_t kNumInputs = 3;
        for (std::uint32_t i = 0; i < kNumInputs; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
        }

        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    // Ternary SFPU dispatch: select(cond, true_val, false_val) → output.
    // Tile indices match the Blackhole `ttnn_where_test.cpp` convention (0, 1, 2, 0):
    // condition in DEST[0], true in DEST[1], false in DEST[2], output overwrites DEST[0].
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
    _init_where_();

    constexpr int k_where_iterations = 8; // 16 rows per face / 2 SFP rows per iteration.
    _llk_math_eltwise_ternary_sfpu_params_(
        ckernel::sfpu::_calculate_where_<k_where_iterations>,
        /* dst_index_in0 */ 0U,
        /* dst_index_in1 */ 1U,
        /* dst_index_in2 */ 2U,
        /* dst_index_out */ 0U);

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Wait for all math execution units this thread has driven to drain before PACK handover.
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
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = 1;

    constexpr auto unpack_dest = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({unpack_dest, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_Res[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK
