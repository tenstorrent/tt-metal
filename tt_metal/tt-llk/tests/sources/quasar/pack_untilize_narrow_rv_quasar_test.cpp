// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NARROW-ROW untilize via RV_PACR (Quasar) DEMO.
//
// Produces a tight, contiguous ROW_NUM_DATUMS-wide untilized output for one
// 32x32 tile — the capability the pack-untilize config-stride path cannot express
// (face-granular / 16-datum floor). It exploits RV_PACR *tile mode* (untilize=0,
// tile_dim=16x1x1), whose GPR0.l1_addr is a 16-BYTE address: for a 16-bit format
// that is 8-datum granularity, so consecutive rows can be placed 8 datums apart.
//
// Scheme (overlap-overwrite), for a 16-bit format with ROW_NUM_DATUMS=8:
//   - one RV_PACR (16x1x1) per output row reads a 16-datum DEST face-row and
//     writes 16 datums to L1 at a 16B address advancing by 8 datums (16B) per row;
//   - row r's wanted low-8 overwrites row (r-1)'s garbage upper-8 (write in order);
//   - only the last row's upper-8 survive as tail (the full-tile result buffer has
//     room, so no separate reservation needed here).
//
// UNPACK/MATH threads are copied verbatim from pack_untilize_rv_quasar_test.cpp;
// only the PACK thread differs (per-row tile-mode RV_PACR instead of one untilize).

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
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

    {
        ZONE_SCOPED("INIT")
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
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, BLOCK_CT_DIM);
        }
        else
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
            }
            else
            {
                _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
            }
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_unpack);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (unpack_to_dest)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                {
                    _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block_rt * BLOCK_CT_DIM, ckernel::DEFAULT_TENSOR_SHAPE);
                    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_unary_operand_<SELECTED_UNPACKER>(0, ckernel::DEFAULT_TENSOR_SHAPE);
            }
        }
        PROFILER_SYNC();
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
        {
            ZONE_SCOPED("INIT")
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

            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
                num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
            PROFILER_SYNC();
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                {
                    for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
                    {
                        _llk_math_eltwise_unary_datacopy_(block_ct);
                    }
                    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                }
            }
            PROFILER_SYNC();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

using namespace ckernel;

// Narrow output row width in datums (< FACE_C_DIM). 16-bit format => 8 datums = 16B.
constexpr std::uint32_t ROW_NUM_DATUMS = 8;

// Verbatim rv-pack.cpp feed helper: noinline forces the 3 words into a0/a1/a2
// (x10/x11/x12); the opcode reads them by hardcoded indices 10/11/12.
__attribute__((noinline)) static std::uint32_t do_rv_pacr(std::uint32_t a0, std::uint32_t a1, std::uint32_t a2)
{
    // Index for RISCV register a0/a1/a2 is: 10/11/12
    TTI_RV_PACR(10, 11, 12);
    return a0 + a1 + a2;
}

// RV_PACR 3-GPR descriptor (confirmed layout). See pack_untilize_rv_quasar_test.cpp.
struct rv_pacr_gpr0_t
{
    std::uint32_t clr_dvalid       : 1;  // [0]
    std::uint32_t input_addr       : 10; // [10:1]  DEST/SrcS start row
    std::uint32_t l1_addr          : 18; // [28:11] tile mode: 16-BYTE L1 address
    std::uint32_t rows_to_untilize : 3;  // [31:29] (untilize-mode only)
};

struct rv_pacr_gpr1_t
{
    std::uint32_t output_format   : 8;  // [7:0]
    std::uint32_t input_format    : 8;  // [15:8]
    std::uint32_t untilize_stride : 16; // [31:16] (untilize-mode only)
};

struct rv_pacr_gpr2_t
{
    std::uint32_t packer_sel     : 2;  // [1:0]   0 => Packer[0]
    std::uint32_t buffer_addr    : 18; // [19:2]  (used when inc_mode=1)
    std::uint32_t tile_dim       : 3;  // [22:20] 0b101 => 16x1x1
    std::uint32_t untilize       : 1;  // [23]    0 => tile mode
    std::uint32_t inc_mode       : 1;  // [24]    0 => raw addresses from GPR0
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
    constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;

    // Single op, RAW addressing (inc_mode=0): input_addr = DEST row 0, l1_addr = first
    // 16 datums of the result buffer (absolute 16B addr). buffer_addr unused in raw mode.
    // Tests whether the raw input_addr/l1_addr path writes DEST row 0 -> L1 base.
    // GPR descriptors + the pack_row lambda live at function scope so both the INIT config
    // zone and the TILE_LOOP packing zone can see them (the profiler zone RAII objects only
    // bound the measurement window, not variable lifetime).
    rv_pacr_gpr0_u g0     = {};
    g0.f.clr_dvalid       = 0;
    g0.f.input_addr       = 0;                         // first DEST row (source)
    g0.f.l1_addr          = params.buffer_Res[0] / 16; // first 16 datums of target L1 (absolute 16B addr)
    g0.f.rows_to_untilize = 0;

    rv_pacr_gpr1_u g1    = {};
    g1.f.output_format   = static_cast<std::uint8_t>(formats.pack_dst);
    g1.f.input_format    = static_cast<std::uint8_t>(formats.pack_src);
    g1.f.untilize_stride = tensor_shape.num_faces_c_dim * FULL_CT_DIM;

    rv_pacr_gpr2_u g2   = {};
    g2.f.packer_sel     = 0;
    g2.f.buffer_addr    = params.buffer_Res[0] / 16;
    g2.f.tile_dim       = 0b101; // 16x1x1
    g2.f.untilize       = 0;
    g2.f.inc_mode       = 0; // RAW: use GPR0.input_addr / GPR0.l1_addr directly
    g2.f.inc_input_idx  = 0;
    g2.f.inc_output_idx = 0;

    // Multi-tile, row-major. FULL_CT_DIM tiles per tile-row; tiles 0..N-2 are packed
    // full-width (32 cols, all faces), the LAST tile is narrow (skip face 1 -> faces 0,2
    // = cols 0-15). Each op writes one DEST face-row (16 datums) to the output, and the
    // per-row advance is the MATRIX width W (not one tile), with tile t at column t*32.
    //
    //   output datum(tile t, out-row R, col-group g) = R*W + t*32 + g*16
    //   W = (FULL_CT_DIM-1)*32 + LAST_TILE_W_DATUMS
    //
    // A Quasar tile is 32x32 = 4 faces of 16x16 -> 64 DEST rows; tile t starts at DEST
    // row t*64 (src_z_stride). l1_addr is 16B-granular = 8 datums (bf16), so >>3.
    // LAST_TILE_W_DATUMS (16 or 8) is build-injected.
    constexpr std::uint32_t TILE_W_DATUMS = 32; // full tile width
    const std::uint32_t matrix_w_datums   = (FULL_CT_DIM - 1) * TILE_W_DATUMS + LAST_TILE_W_DATUMS;
    const std::uint32_t base_16B          = params.buffer_Res[0] / 16;
    const std::uint32_t last_t            = FULL_CT_DIM - 1;

    // Pack one DEST face-row (16 datums) of tile t to its untilized output slot.
    //
    // Both input_addr and l1_addr are recomputed in software every op ON PURPOSE — the
    // RV_PACR HW auto-increment (inc_mode=1 + inc_input_idx/inc_output_idx) cannot express
    // this mapping. Per the RTL (ws-tensix tt_pack_row.sv): inc_input/output_idx advance a
    // tile-INDEX counter by a hardcoded +1 (l.987/993), and that index maps to addresses by
    // fixed geometry only — source row = idx (l.565, 16x1x1) and output = idx<<tile_size (l.698,
    // contiguous 16-datum steps). That yields a plain linear tilized->linear walk. It cannot do
    // (a) the face de-interleave (remap: output row order is a bit-permutation of input rows),
    // (b) the custom matrix_w row stride (not a whole tile), or (c) the two-pass overwrite. HW
    // untilize=1 would do the de-interleave but forces full-tile geometry + face-granular stride
    // (the very thing this demo works around). So the software recompute is fundamental here.
    auto pack_row = [&](std::uint32_t t, std::uint32_t row)
    {
        g0.f.input_addr = t * 64 + row; // DEST row of tile t

        const std::uint32_t lo5  = row & 0x1F;
        const std::uint32_t rol  = ((lo5 << 1) | (lo5 >> 4)) & 0x1F;
        const std::uint32_t slot = (row & 0x20) | rol; // remap(row): output slot 0..63
        const std::uint32_t R    = slot >> 1;          // output tile-row 0..31
        const std::uint32_t g    = slot & 1;           // col group (0: cols0-15, 1: cols16-31)

        const std::uint32_t l1_datum = R * matrix_w_datums + t * TILE_W_DATUMS + g * 16;
        g0.f.l1_addr                 = base_16B + (l1_datum >> 3); // datums -> 16B units (bf16)

        volatile std::uint32_t rv_res = do_rv_pacr(g0.val, g1.val, g2.val);
        (void)rv_res;
    };

    {
        ZONE_SCOPED("INIT")
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

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM>(buf_desc_id, tensor_shape);

        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, 0);
        TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, 0);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        // PASS 1: narrow last tile FIRST. Column group g=0 (faces 0,2 -> cols 0-15) is always
        // needed; column group g=1 (faces 1,3 -> cols 16-31) is only needed when the kept width
        // exceeds 16. So skip faces 1,3 iff LAST_TILE_W_DATUMS <= 16 (reads only faces 0,2).
        // Each op writes a full 16-datum face-row; when the kept width is not a multiple of 16
        // (8 or 24) the upper spill datums of the boundary face-row land in the NEXT output
        // row's leading columns -- packing the narrow tile before the full tiles lets tile 0
        // overwrite that spill (for widths 16 and 32 the boundary write is face-aligned: no spill).
        constexpr bool last_needs_g1         = (LAST_TILE_W_DATUMS > 16);
        constexpr std::uint32_t last_row_end = last_needs_g1 ? 64u : 48u;
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t row = 0; row < last_row_end; row++)
            {
                if (!last_needs_g1 && row == 16)
                {
                    row += 16; // width<=16: g=0 only -> skip faces 1,3 (rows 16-31) -> jump to face 2
                }
                pack_row(last_t, row);
            }

            // PASS 2: full tiles 0..N-2 (all four faces, 32 cols each). These overwrite any spill
            // the narrow tile left in the leading columns.
            for (std::uint32_t t = 0; t < last_t; t++)
            {
                for (std::uint32_t row = 0; row < 64; row++)
                {
                    pack_row(t, row);
                }
            }

            _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif
