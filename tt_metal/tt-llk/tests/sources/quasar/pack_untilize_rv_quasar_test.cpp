// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RV_PACR untilize DEMO.
//
// Standalone proof-of-life for the RISC-V-descriptor pack path (RV_PACR,
// opcode 0x35). It deliberately does NOT touch the working _llk_pack_untilize_
// implementation: the UNPACK and MATH threads are copied verbatim from
// pack_untilize_quasar_test.cpp, and only the PACK thread swaps the
// MOP/PACR_UNTILIZE issue for a hand-built RV_PACR descriptor.
//
// Scope: a single ordinary 32x32 tile (4 faces), Packer 0. It uses HW-computed
// addressing (GPR2.inc_mode=1) so it reuses the exact addressing machinery of
// the working path (PACK_UNTILIZE_SRC/DST_Z_STRIDE + the FACE-select Z counters
// set by SET_SRC/DST_TILE_FACE_ROW_IDX); the only difference from the working
// test is that the packer op is issued via RV_PACR instead of the untilize MOP.

#include <algorithm>
#include <cstdint>
#include <cstdio>

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
    const std::uint32_t SELECTED_UNPACKER = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;
    tdma_descriptor_t td_val;
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = TILE_CNT;

    // Setup data valid scheme
    if constexpr (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});

        DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
        if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>();
        }
        else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
        }
        else
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>();
        }
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    }

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = params.buffer_A[0] / 16;
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim       = TEST_FACE_C_DIM;
    bd_val.f.y_dim       = TEST_FACE_R_DIM;
    bd_val.f.z_dim       = num_faces;

    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
    if constexpr (unpack_to_dest)
    {
        _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
        _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, BLOCK_CT_DIM);
        for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
        {
            _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block_rt * BLOCK_CT_DIM, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_unpack_dest_dvalid_section_done_<dest_sync>();
        }
    }
    else
    {
        if constexpr (is_fp32_dest_acc_en)
        {
            // If Dst fmt is 32b and operation is Mov2D, we need both SrcA/B fmts to be configured since Mov2D will be implemented via ELWADD
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
        }
        else
        {
            _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
        }
        _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
            buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_unpack);
        _llk_unpack_unary_operand_<SELECTED_UNPACKER>(0, ckernel::DEFAULT_TENSOR_SHAPE);
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
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (!unpack_to_dest)
    {
        // Setup data valid scheme
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

        DataFormat math_format     = static_cast<DataFormat>(formats.math);
        DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
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

        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
        for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
        {
            for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
            {
                _llk_math_eltwise_unary_datacopy_(num_faces * TEST_FACE_R_DIM /*num_rows_per_tile*/, block_ct);
            }
            _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

using namespace ckernel;

// RV_PACR feeds its 3 descriptor words from the RISC-V architectural registers
// a0/a1/a2 (x10/x11/x12), NOT the Tensix regfile[] MMIO namespace. This helper is
// a verbatim copy of the checked-in rv-pack.cpp reference: __attribute__((noinline))
// forces a real call so the first three uint32_t params land in a0/a1/a2 by the
// RISC-V calling convention, and the opcode (with hardcoded register indices
// 10/11/12) reads the words straight from those registers. `return a0 + a1 + a2`
// keeps the args live. There is no fence/sync on this path — the descriptor travels
// the per-thread instrn_reg FIFO, popped in program order with the RV opcode.
// Natural word order: a0 = GPR0, a1 = GPR1, a2 = GPR2.
__attribute__((noinline)) static std::uint32_t do_rv_pacr(std::uint32_t a0, std::uint32_t a1, std::uint32_t a2)
{
    // Index for RISCV register a0/a1/a2 is: 10/11/12
    TTI_RV_PACR(10, 11, 12);
    return a0 + a1 + a2;
}

// -----------------------------------------------------------------------------
// RV_PACR 3-GPR descriptor layout (Quasar), confirmed against RTL:
//   GPR0: [0] clr_dvalid | [10:1] input_addr | [28:11] l1_addr | [31:29] rows_to_untilize
//   GPR1: [7:0] output_format | [15:8] input_format | [31:16] untilize_stride
//   GPR2: [1:0] packer_sel | [19:2] buffer_addr | [22:20] tile_dim | [23] untilize
//         | [24] inc_mode | [25] inc_input_idx | [26] inc_output_idx | [31:27] reserved
// Bitfields are allocated LSB-first (little-endian), matching buffer_descriptor_u.
// -----------------------------------------------------------------------------
struct rv_pacr_gpr0_t
{
    std::uint32_t clr_dvalid       : 1;  // [0]
    std::uint32_t input_addr       : 10; // [10:1]  DEST/SrcS start row (16-datum rows)
    std::uint32_t l1_addr          : 18; // [28:11] untilize: 16-datum offset from buffer base
    std::uint32_t rows_to_untilize : 3;  // [31:29] encoded (0 => 16x16x4 / 32x32)
};

struct rv_pacr_gpr1_t
{
    std::uint32_t output_format   : 8;  // [7:0]  tile format in L1
    std::uint32_t input_format    : 8;  // [15:8] input (DEST/SrcS) format
    std::uint32_t untilize_stride : 16; // [31:16] row stride, 16-datum rows
};

struct rv_pacr_gpr2_t
{
    std::uint32_t packer_sel     : 2;  // [1:0]   0 => Packer[0]
    std::uint32_t buffer_addr    : 18; // [19:2]  output buffer base (16B addr)
    std::uint32_t tile_dim       : 3;  // [22:20] 0 => 16x16x4
    std::uint32_t untilize       : 1;  // [23]
    std::uint32_t inc_mode       : 1;  // [24]    1 => HW computes addr from base + Z counters
    std::uint32_t inc_input_idx  : 1;  // [25]
    std::uint32_t inc_output_idx : 1;  // [26]
    std::uint32_t reserved       : 5;  // [31:27]
};

union rv_pacr_gpr0_u
{
    rv_pacr_gpr0_t f;
    std::uint32_t val;
};

union rv_pacr_gpr1_u
{
    rv_pacr_gpr1_t f;
    std::uint32_t val;
};

union rv_pacr_gpr2_u
{
    rv_pacr_gpr2_t f;
    std::uint32_t val;
};

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    }

    tdma_descriptor_t tdma_desc;
    std::uint32_t const buf_desc_id = 31;

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = params.buffer_Res[0] / 16;
    bd_val.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim       = TEST_FACE_C_DIM;
    bd_val.f.y_dim       = TEST_FACE_R_DIM;
    bd_val.f.z_dim       = num_faces;

    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

    constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;

    // Keep the working setup: this programs valid packer HW state plus the
    // untilize Z strides (PACK_UNTILIZE_SRC/DST_Z_STRIDE) that RV_PACR untilize
    // still consumes from the THCON packer regs. RV_PACR overrides base addr,
    // formats, tile dim and untilize_stride from its descriptor, so the
    // buf-desc-table metadata is redundant but harmless here.
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM>(buf_desc_id, tensor_shape);

    // Build the RV_PACR descriptor for one 32x32 tile, HW-computed addressing.
    rv_pacr_gpr0_u g0     = {};
    g0.f.clr_dvalid       = 0; // dvalid handled by section_done handshake below
    g0.f.input_addr       = 0; // counters drive addressing in inc_mode=1
    g0.f.l1_addr          = 0;
    g0.f.rows_to_untilize = 0; // 0 => 16x16x4 (full 32x32 tile)

    rv_pacr_gpr1_u g1    = {};
    g1.f.output_format   = static_cast<std::uint8_t>(formats.pack_dst);
    g1.f.input_format    = static_cast<std::uint8_t>(formats.pack_src);
    g1.f.untilize_stride = tensor_shape.num_faces_c_dim * FULL_CT_DIM; // 16-datum rows (=2 for one 32x32 tile)

    rv_pacr_gpr2_u g2   = {};
    g2.f.packer_sel     = 0; // Packer[0]
    g2.f.buffer_addr    = params.buffer_Res[0] / 16;
    g2.f.tile_dim       = 0; // 16x16x4
    g2.f.untilize       = 1;
    g2.f.inc_mode       = 1; // use buffer base + Z counters (reuse working addressing path)
    g2.f.inc_input_idx  = 0;
    g2.f.inc_output_idx = 0;

    // Single 32x32 tile: DEST bank start (src) and output base (dst), face-select
    // counters at 0 (mirrors _llk_pack_untilize_(dest_idx=0, l1_tile_idx=0)).
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, 0);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, 0);

    // Feed the descriptor via the RISC-V register operand stream (a0/a1/a2), not
    // regfile[]. Natural word order (a0=GPR0, a1=GPR1, a2=GPR2) per rv-pack.cpp.
    volatile std::uint32_t rv_res = do_rv_pacr(g0.val, g1.val, g2.val);
    (void)rv_res;

    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
