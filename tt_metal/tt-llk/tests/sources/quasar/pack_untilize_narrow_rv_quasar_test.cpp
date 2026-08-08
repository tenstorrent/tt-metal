// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// pack_untilize via RV_PACR (Quasar). Two PACK modes, selected by RV_WHOLE_TILE:
//
// RV_WHOLE_TILE = true  -- "normal" whole-tile untilize. One HW-streamed RV_PACR op per
//   tile (untilize=1, tile_dim=16x16x4, inc_mode=1) => byte-identical to the MOP/
//   PACR_UNTILIZE path. Proof-of-life for the RISC-V-descriptor pack path. Single-tile scope.
//
// RV_WHOLE_TILE = false -- NARROW-ROW untilize. Produces a tight, densely-packed output whose
//   LAST tile per tile-row may be NARROWER than a full tile — the narrow_row capability the HW
//   pack-untilize config-stride path cannot express on Quasar (its output stride is face-
//   granular / 16-datum, unlike WH/BH's 16-byte stride). It uses RV_PACR *tile mode*
//   (untilize=0, tile_dim=16x1x1): each op reads one 16-datum DEST face-row and writes it to a
//   16-BYTE-granular L1 address (= 8-datum granularity for a 16-bit format). Issuing one op per
//   output face-row, with input_addr (DEST row) and l1_addr (output) computed per op, lets us
//   place each face-row at an arbitrary offset -> face de-interleave + a custom row (matrix)
//   width. Last-tile widths not a multiple of 16 (8, 24) write a full 16-datum face-row whose
//   upper datums spill into the next output row; packing the narrow tile first lets the full
//   tiles overwrite that spill (see the PACK thread). Scope: one tile-row of FULL_CT_DIM 32x32
//   tiles, 16-bit formats, last-tile width in {8,16,24,32}.
//
// See test_pack_untilize_narrow_rv_quasar.py. Only the PACK thread depends on RV_WHOLE_TILE;
// the UNPACK (unary operand) and MATH (A2D datacopy) threads are shared by both modes.

#include <cstdint>

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
    // Runtime params (templates only under SPEED_OF_LIGHT). Kept as runtime args so they do not
    // trigger per-value recompiles.
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    // This test always runs the FPU (non-unpack-to-dest) path, so unpack goes to SrcA.
    const std::uint32_t SELECTED_UNPACKER       = p_unpacr::UNP_A;
    const std::uint32_t buf_desc_id             = 0;
    const std::uint32_t num_tiles_per_unpack    = TILE_CNT;
    constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;

    {
        ZONE_SCOPED("INIT")
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

        const tdma_descriptor_t td_val =
            ckernel::trisc::construct_tdma_desc(tensor_shape, params.buffer_A[0] >> 4, formats.unpack_A_src, buf_desc_id, formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        if constexpr (is_fp32_dest_acc_en)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
        }
        else
        {
            _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
        }
        _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, tensor_shape, num_tiles_per_unpack);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_unpack_unary_operand_<SELECTED_UNPACKER>(0, tensor_shape);
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
#endif
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

        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
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

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

using namespace ckernel;

// RV_PACR reads its 3 descriptor words from RISC-V registers a0/a1/a2 (x10/x11/x12) by the
// register indices baked into the opcode. Bind gpr0/1/2 to a0/a1/a2 via explicit register
// variables and pass them as "r" inputs so the compiler guarantees they are live at the .ttinsn.
__attribute__((noinline)) static std::uint32_t do_rv_pacr(std::uint32_t gpr0, std::uint32_t gpr1, std::uint32_t gpr2)
{
    register std::uint32_t a0 asm("a0") = gpr0;
    register std::uint32_t a1 asm("a1") = gpr1;
    register std::uint32_t a2 asm("a2") = gpr2;
    __asm__ __volatile__(".ttinsn %3" : : "r"(a0), "r"(a1), "r"(a2), "n"(TT_OP_RV_PACR(10 /*reg_idx2*/, 11 /*reg_idx1*/, 12 /*reg_idx0*/)));
    return a0 + a1 + a2;
}

// RV_PACR 3-GPR descriptor.
//   GPR0 = clr_valid | (input_idx << 1) | (l1_addr << 11) | (rows << 29)
//   GPR1 = out_fmt   | (in_fmt << 8)    | (untilize_stride << 16)
//   GPR2 = packer_sel | (buffer_addr << 2) | (tile_dim << 20) | (untilize << 23)
//        | (inc_mode << 24) | (inc_input_idx << 25) | (inc_output_idx << 26)
// i.e. clr_dvalid at bit 0, so input_addr @ [10:1] / l1_addr @ [28:11] as below. This is also
// validated empirically by this test (RV_WHOLE_TILE mode is byte-identical to the untilize
// golden, narrow mode places datums correctly — neither works if these offsets are wrong).
// NOTE: assembly.yaml's RV_PACR prose disagrees (it reads L1 addr as GPR0[17:0], buffer_addr as
// GPR1[24:7]), but that description is self-inconsistent (it also cites GPR[1] for a GPR2 field).
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

// RV_PACR GPR2.tile_dim ([22:20]) HW encodings (see assembly.yaml RV_PACR tile-dim table).
constexpr std::uint32_t TILE_DIM_16x16x4 = 0b000; // full 32x32 tile (whole-tile untilize)
constexpr std::uint32_t TILE_DIM_16x1x1  = 0b101; // one 16-datum face-row (narrow tile mode)

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t LAST_TILE_W_DATUMS = params.LAST_TILE_W_DATUMS;
#endif
    constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;

    // Whole-tile mode issues a single untilize op for one tile.
    // the multi-tile walk is narrow-mode only.
    static_assert(!(RV_WHOLE_TILE && FULL_CT_DIM != 1), "RV_WHOLE_TILE mode is single-tile only (use FULL_CT_DIM == 1 / input [32,32]).");

    // TODO: Once we introduce 32-bit format, change the assert to support 4 datums per row (32-bit format).
    if constexpr (!RV_WHOLE_TILE)
    {
        LLK_ASSERT(
            ckernel::trisc::SCALE_DATUM_SIZE(formats.pack_dst, 8) == 16, "narrow RV_PACR pack assumes a 16-bit format (8 datums == 16B for l1_addr >> 3)");
        LLK_ASSERT(LAST_TILE_W_DATUMS % 8 == 0 && LAST_TILE_W_DATUMS <= 32, "LAST_TILE_W_DATUMS must be a multiple of 8 (16B) and <= 32");
    }

    // RV_PACR base descriptor. g1 (formats/stride) and g2 (mode config) are constant across the
    // whole tile-row. In narrow mode g0.input_addr / g0.l1_addr are placeholders, recomputed per
    // op by pack_row (below). In whole-tile mode (inc_mode=1) input_addr is counter-driven and
    // l1_addr is the 16-datum OFFSET from the buffer base -> it must be 0 for a single tile at base
    // (the tile counter handles per-tile placement). It's kept at function scope so both the INIT config
    // zone and the TILE_LOOP packing zone can see them.
    rv_pacr_gpr0_u g0     = {};
    g0.f.clr_dvalid       = 0;
    g0.f.input_addr       = 0;
    g0.f.l1_addr          = 0;
    g0.f.rows_to_untilize = 0;

    rv_pacr_gpr1_u g1    = {};
    g1.f.output_format   = static_cast<std::uint8_t>(formats.pack_dst);
    g1.f.input_format    = static_cast<std::uint8_t>(formats.pack_src);
    g1.f.untilize_stride = tensor_shape.num_faces_c_dim * FULL_CT_DIM;

    rv_pacr_gpr2_u g2 = {};
    // RV_PACR GPR2.packer_sel encodes Packer[0] as 0 (NOT p_pacr::PACK0, which is 0b011 for the
    // PACR-instruction packer-sel field — a different encoding that would truncate to RESERVED here).
    g2.f.packer_sel     = 0;
    g2.f.buffer_addr    = params.buffer_Res[0] >> 4;
    g2.f.inc_input_idx  = 0;
    g2.f.inc_output_idx = 0;
    if constexpr (RV_WHOLE_TILE)
    {
        // Whole-tile "normal" RV_PACR untilize: one HW-streamed untilize op for the full
        // 32x32 tile, HW-computed addressing from buffer base + Z counters. Single-tile scope.
        g2.f.tile_dim = TILE_DIM_16x16x4; // full tile
        g2.f.untilize = 1;
        g2.f.inc_mode = 1; // HW computes L1 addr from buffer_addr + Z counters
    }
    else
    {
        // Narrow per-face-row: tile mode, RAW addressing (input_addr/l1_addr from GPR0 per op).
        g2.f.tile_dim = TILE_DIM_16x1x1; // one face-row
        g2.f.untilize = 0;
        g2.f.inc_mode = 0;
    }

    // Narrow-mode layout (inert / unused in whole-tile mode).
    // Multi-tile, row-major -> FULL_CT_DIM tiles per tile-row, tiles 0..N-2 are packed full-width (32 cols, all faces),
    // the LAST tile is narrow. Each op writes one DEST face-row (16 datums). The per-row advance
    // is the MATRIX width W (not one tile) with tile t at column t*32:
    //   output datum(tile t, out-row R, col-group g) = R*W + t*32 + g*16
    //   W = (FULL_CT_DIM-1)*32 + LAST_TILE_W_DATUMS   (LAST_TILE_W_DATUMS in {8,16,24,32})
    // A Quasar tile is 32x32 = 4 faces of 16x16 -> 64 DEST rows; tile t starts at DEST row t*64.
    // l1_addr is 16B-granular = 8 datums (16-bit format).
    [[maybe_unused]] constexpr std::uint32_t FACE_R_DIM    = tensor_shape.face_r_dim;                     // DEST rows per face (16)
    [[maybe_unused]] constexpr std::uint32_t FACE_C_DIM    = tensor_shape.face_c_dim;                     // datums per face-row (16)
    [[maybe_unused]] constexpr std::uint32_t ROWS_PER_TILE = tensor_shape.total_num_faces() * FACE_R_DIM; // DEST rows per tile (64)
    [[maybe_unused]] constexpr std::uint32_t TILE_W_DATUMS = tensor_shape.total_col_dim();                // datums per tile row (32)
    [[maybe_unused]] const std::uint32_t matrix_w_datums   = (FULL_CT_DIM - 1) * TILE_W_DATUMS + LAST_TILE_W_DATUMS;
    [[maybe_unused]] const std::uint32_t base_16B          = params.buffer_Res[0] >> 4;
    [[maybe_unused]] const std::uint32_t last_t            = FULL_CT_DIM - 1;

    // Pack one DEST face-row (16 datums) of tile t to its untilized output slot.
    //
    // Both input_addr and l1_addr are recomputed in software every op ON PURPOSE — the
    // RV_PACR HW auto-increment (inc_mode=1 + inc_input_idx/inc_output_idx) cannot express
    // this mapping. inc_input/output_idx advance a tile-INDEX counter by a hardcoded +1, and that index maps to addresses by
    // fixed geometry only — source row = idx and output = idx<<tile_size (contiguous 16-datum steps).
    // That yields a plain linear tilized->linear walk. It cannot do:
    // (a) the face de-interleave (remap: output row order is a bit-permutation of input rows),
    // (b) the custom matrix_w row stride (not a whole tile),
    // (c) the two-pass overwrite.
    // HW untilize=1 would do the de-interleave but it forces full-tile geometry + face-granular stride
    // (the very thing this demo works around). So the software recompute is fundamental here.
    [[maybe_unused]] auto pack_row = [&](std::uint32_t t, std::uint32_t row)
    {
        g0.f.input_addr = t * ROWS_PER_TILE + row; // DEST row of tile t

        const std::uint32_t lo5  = row & 0x1F;
        const std::uint32_t rol  = ((lo5 << 1) | (lo5 >> 4)) & 0x1F;
        const std::uint32_t slot = (row & 0x20) | rol; // remap(row): output slot 0..63
        const std::uint32_t R    = slot >> 1;          // output tile-row 0..31
        const std::uint32_t g    = slot & 1;           // col group (0: cols0-15, 1: cols16-31)

        const std::uint32_t l1_datum = R * matrix_w_datums + t * TILE_W_DATUMS + g * FACE_C_DIM;
        g0.f.l1_addr                 = base_16B + (l1_datum >> 3); // datums -> 16B units (bf16)

        volatile std::uint32_t rv_res = do_rv_pacr(g0.val, g1.val, g2.val);
        (void)rv_res;
    };

    {
        ZONE_SCOPED("INIT")
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

        const std::uint32_t buf_desc_id = 31;
        const tdma_descriptor_t tdma_desc =
            ckernel::trisc::construct_tdma_desc(tensor_shape, params.buffer_Res[0] >> 4, formats.pack_dst, buf_desc_id, formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc, ckernel::ReluConfig::none());

        if constexpr (RV_WHOLE_TILE)
        {
            // Whole-tile untilize uses untilize=1 + inc_mode=1: HW derives the L1 address from the
            // buffer base + the tile/face counters + the untilize Z-strides, so it needs the untilize
            // init and freshly-zeroed counters. Reset BOTH the tile and face selectors (the untilize
            // MOPs drive TILE_SEL for the tile index and FACE_SEL for the face/row offset) so a stale
            // tile index left by a prior op cannot offset the output.
            _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM>(buf_desc_id, tensor_shape);
            TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
            TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
            TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, 0);
            TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, 0);
        }
        // Narrow mode uses RV_PACR tile mode (untilize=0, inc_mode=0) with raw GPR0 addressing, so it
        // needs neither the untilize Z-strides nor the face/tile counters — nothing to configure here.
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (RV_WHOLE_TILE)
        {
            // Whole-tile "normal" untilize: one HW-streamed RV_PACR untilize op per iteration
            // (single 32x32 tile). Byte-identical to the MOP/PACR_UNTILIZE path.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                volatile std::uint32_t rv_res = do_rv_pacr(g0.val, g1.val, g2.val);
                (void)rv_res;
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        else
        {
            // PASS 1: narrow last tile FIRST. Column-group g=0 (faces 0 and 2 -> cols 0-15) is always
            // packed; g=1 (faces 1 and 3 -> cols 16-31) only when the kept width exceeds one face
            // (FACE_C_DIM). Each op writes a full 16-datum face-row, when the kept width is not a
            // multiple of 16 (8 or 24) the boundary face-row's upper datums spill into the next output
            // row's leading columns -- packing the narrow tile before the full tiles lets tile 0
            // overwrite that spill (widths 16/32 are face-aligned: no spill). Packing order within the
            // narrow tile doesn't matter: each DEST row maps to its own remapped output slot, g=0/g=1
            // write disjoint columns, and any g=1 spill lands in a full-tile column fixed by PASS 2.
            const bool last_needs_g1 = (LAST_TILE_W_DATUMS > FACE_C_DIM);
            // Pack all FACE_R_DIM rows of one DEST face (face f occupies DEST rows [f*FACE_R_DIM ..)).
            auto pack_face = [&](std::uint32_t face)
            {
                for (std::uint32_t r = 0; r < FACE_R_DIM; r++)
                {
                    pack_row(last_t, face * FACE_R_DIM + r);
                }
            };
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                pack_face(0); // g=0: cols 0-15
                pack_face(2);
                if (last_needs_g1)
                {
                    pack_face(1); // g=1: cols 16-31
                    pack_face(3);
                }

                // PASS 2: full tiles 0..N-2 (all four faces, 32 cols each). These overwrite any spill
                // the narrow tile left in the leading columns.
                for (std::uint32_t t = 0; t < last_t; t++)
                {
                    for (std::uint32_t row = 0; row < ROWS_PER_TILE; row++)
                    {
                        pack_row(t, row);
                    }
                }

                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
