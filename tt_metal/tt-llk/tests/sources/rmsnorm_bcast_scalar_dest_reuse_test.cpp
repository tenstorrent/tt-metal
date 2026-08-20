// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RMSNorm bcast-scalar DEST-reuse eltwise-binary LLK test (experimental, Blackhole only).
//
// Mirrors the ttnn compute-kernel flow of
//   ckernel::rmsnorm_bcast_scalar_reuse_tiles{,_fidelity} (hw/inc/api/compute/experimental/rmsnorm.h),
// expanded into the underlying experimental _llk_* calls so it runs in the tt-llk harness:
//   experimental/llk_unpack_A_rmsnorm.h   (unpack init + tile fetch)
//   experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h  (math init + per-tile op).
//
// --- What the header actually computes (this is how the golden is derived) ---
//
// The op reuses a value already resident in DEST as a *broadcast scalar* SrcB and
// fuses it with a freshly-unpacked input tile via an eltwise-binary FPU op:
//
//   _llk_math_rmsnorm_bcast_scalar_dest_reuse_(src_index, dst_index):
//     1. set_dst_write_addr(src_index)
//     2. rmsnorm_bcast_scalar_reuse_dest_as_src():
//          STALLWAIT(WAIT_SFPU|SRCB_VLD) then MOVD2B(SRC_ZERO_OFFSET+0, MOV_1_ROW)
//        -> moves row 0 of DEST[src_index] into SrcB row 0.
//     3. (optional) clear_dest: ZEROACC on DEST[dst_index] half/all.
//     4. set_dst_write_addr(dst_index)
//     5. Run the MOP: TT_OP_ELW{ADD,MUL}(..., broadcast_type = SRCB_BCAST_ALL, ...)
//        SrcA = the unpacked input tile, SrcB = the scalar from step 2 broadcast to
//        every output element (SRCB_BCAST_ALL). Result -> DEST[dst_index].
//     6. SETRWC(CLR_B) to clear SrcB after the mop.
//
// So per output tile:  DEST[dst_index][e] = A[e]  (op)  s     for every element e,
// where s is the scalar taken from DEST[src_index] and (op) is ELWADD or ELWMUL
// (the two ops promoted by the experimental header; ELWSUB is defined in the LLK
// but not surfaced by the rmsnorm compute API, so this test sweeps ADD and MUL).
//
// To make s a *known, uniform* value (the header assumes a prior op left the rms
// reciprocal in DEST), this harness seeds DEST[SRC_INDEX] with a single constant
// via the SFPU fill kernel before the first reuse op. Because the seed is uniform,
// SRCB_BCAST_ALL broadcasts the same s to every lane and every output lane is
// well-defined:  golden[e] = A[e] op SCALAR_SEED.  The Python golden computes
// exactly this (element-wise A op scalar, in the output format), and every lane is
// validated (no undefined lanes for the uniform-seed case).
//
// Axes swept by the Python driver (strategy §4): eltwise_binary_type {ELWADD,ELWMUL},
// num_tiles {1,2,3,7,8} bf16 / {1..4} fp32, math_fidelity {LoFi,HiFi2,HiFi4},
// clear_dest {F,T}, dest_acc {No,Yes}, num_faces {1,2,4}, unpack_full_transpose {F,T}.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// DEST slot that holds the broadcast scalar (seeded once) and DEST slot the fused
// results are written to. Each input tile i overwrites DEST[DST_BASE + i].
static constexpr std::uint32_t SRC_INDEX = 0;
static constexpr std::uint32_t DST_BASE  = 0;

// The value seeded into DEST[SRC_INDEX] and broadcast as SrcB. Must match
// RMSNORM_SCALAR_SEED in the Python golden bit-for-bit.
static constexpr float SCALAR_SEED = 0.5f;

#ifdef LLK_TRISC_UNPACK

// The promoted experimental header (#52709) declares several static-constexpr MOP
// opcodes and takes format parameters that only a subset of its num_faces/transpose
// branches use, so the unused ones trip the suite's -Werror=unused-{variable,parameter}.
// This is a header defect, not a use-site one; suppress locally around the include
// (same pattern as deepseek_moe_gate_test.cpp) until the header is cleaned up.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_rmsnorm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    const std::uint32_t num_faces           = tensor_shape.total_num_faces();

    // compute_kernel_hw_startup: configure the unpacker. The rmsnorm op moves the
    // scalar in from DEST (SrcB) inside MATH, so only operand A is streamed from L1
    // here; still configure both A/B slots so ALU formats are consistent.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        tensor_shape.face_r_dim,
        tensor_shape.face_r_dim,
        num_faces,
        num_faces,
        params.TILE_SIZE_UNPACK_A,
        params.TILE_SIZE_UNPACK_B);

    // rmsnorm_bcast_scalar_reuse_tiles_init{,_fidelity}: UNPACK side.
    // BroadcastType::SCALAR, acc_to_dest=true, DEST_TO_SRCB reuse. The blaze-only
    // axis unpack_full_transpose drives both transpose_of_faces and
    // within_face_16x16_transpose (matching the compute-API _fidelity variant).
    _llk_unpack_A_rmsnorm_init_<TILE_CNT, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        UNPACK_FULL_TRANSPOSE /* transpose_of_faces */,
        UNPACK_FULL_TRANSPOSE /* within_face_16x16_transpose */,
        tensor_shape.face_r_dim,
        num_faces,
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // The rmsnorm unpack MOP issues real UNPACR(SrcA, ...) ops to stream the input
    // tile into SrcA (the DEST scalar comes in on SrcB via MOVD2B in MATH). But the
    // op init above (llk_unpack_A_rmsnorm.h) programs the ADC X-end on unpacker B
    // (UNP_SEL = SCALAR ? UNP_B), never on unpacker A. Because _llk_unpack_hw_configure_
    // does NOT set the X-end (it is owned by the op init per llk_unpack_common.h:137),
    // UNP_A's X-end is left at its reset value and only the first datum of each SrcA
    // row is unpacked -> SrcA is zero except lane 0. In the real ttnn flow a prior
    // op's init leaves UNP_A's X-end at a full face; this standalone harness has no
    // such predecessor, so program the full-face X-end on UNP_A here before the fetch.
    // (Root cause is a defect in the promoted llk_unpack_A_rmsnorm.h init: its MOP
    // reads SrcA but it configures the X-end on SrcB -- see headerBugSuspected.)
    ckernel::unpacker::config_unpacker_x_end<p_setadc::UNP_A>(tensor_shape.face_r_dim);

    // Single unpack call streams all TILE_CNT tiles into SrcA: the rmsnorm unpack
    // MOP z-increments through num_tiles * num_faces faces from one L1 base
    // address, in lockstep with the FPU MOP consuming them. MATH pairs each face
    // with the DEST scalar (SrcB). buffer_A is contiguous (dense tiles).
    _llk_unpack_A_<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

// See the UNPACK-side note: the promoted experimental header trips the suite's
// -Werror=unused-{variable,parameter} in the branches this test does not exercise.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#pragma GCC diagnostic pop
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "params.h"

// The scalar seed is a float, so we only need the float SFPU-fill microcode.
// Do NOT pull in sfpu/ckernel_sfpu_fill.h: its _calculate_fill_int_ path names a
// non-existent sfpi::DataLayout, which the tt g++ front-end rejects at parse time
// (-Werror=template-body) even though this test never instantiates that overload.
// Reproduce the header's _calculate_fill_ float body locally instead; it is a
// verbatim copy of the float overload (see ckernel_sfpu_fill.h:16-27).
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _rmsnorm_calculate_fill_(const float value)
{
    sfpi::vFloat fill_val = value;
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = fill_val;
        sfpi::dst_reg++;
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // num_faces MUST be a compile-time constant for the rmsnorm init: the header's
    // addr-mod configurator folds it into the DEST increment, and only then can the
    // always-inline addr_mod_t::set() satisfy the immediate ("n") asm constraint on
    // the SETC16 register index. num_faces_r_dim_A/c_dim_A are file-scope constexprs
    // (emitted as templates), so this product is a constant expression. MATH needs no
    // TensorShape otherwise (all addressing is DEST-relative and MOP-driven).
    constexpr std::uint32_t num_faces   = static_cast<std::uint32_t>(num_faces_r_dim_A) * static_cast<std::uint32_t>(num_faces_c_dim_A);
    constexpr std::uint32_t ACC_TO_DEST = 0;

    // compute_kernel_hw_startup.
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // compute_kernel_hw_startup programs the SFPU config register once per kernel;
    // this standalone harness bypasses it, so run the idempotent once-init before
    // the _calculate_fill_ SFPU store that seeds the scalar.
    _llk_math_eltwise_unary_sfpu_init_once_();

    // rmsnorm_bcast_scalar_reuse_tiles_init{,_fidelity}: MATH side. Programs the
    // addr-mods and the ELW{ADD,MUL} MOP for TILE_CNT tiles at this fidelity.
    _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<ELTWISE_BINARY_OP, TILE_CNT, MATH_FIDELITY>(num_faces, ACC_TO_DEST);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Seed the broadcast scalar: fill DEST[SRC_INDEX] with SCALAR_SEED so the
    // MOVD2B inside the reuse op reads a known, uniform value. Mirrors the prior
    // op (e.g. rms reciprocal) that the header assumes has already run. The SFPU
    // params helper owns the DEST addressing for dst_index = SRC_INDEX.
    _llk_math_eltwise_unary_sfpu_params_(_rmsnorm_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, SRC_INDEX, VectorMode::RC, SCALAR_SEED);

    // Single fused reuse call processes all TILE_CNT tiles. The op's MOP walks
    // DEST from dst_index across num_tiles * num_faces faces (ADDR_MOD dest.incr=8),
    // pairing each streamed SrcA face with the seeded scalar broadcast in SrcB:
    //   DEST[DST_BASE + t][e] = A_t[e] (op) SCALAR_SEED   for t in [0, TILE_CNT).
    // src_index stays at SRC_INDEX so every tile reuses the same seeded scalar.
    LLK_ASSERT(
        ((DST_BASE + TILE_CNT - 1) < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "Output tile index exceeds maximum destination tiles");
    _llk_math_rmsnorm_bcast_scalar_dest_reuse_<ELTWISE_BINARY_OP, TILE_CNT, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY, CLEAR_DEST>(SRC_INDEX, DST_BASE);

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    // Blackhole-only test: call the pack LLKs directly (the _wrapper_ helpers exist
    // only to paper over the WH/BH signature split for dual-arch tests).
    // compute_kernel_hw_startup
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);

    // No-src init: packer strides are owned by the hw-configure above, so skip re-programming them here.
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, TILE_CNT /* num_tiles */, false /* skip_bh_tilize_workaround */);

    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
    {
        LLK_ASSERT(
            ((DST_BASE + i) < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
            "Output tile index exceeds maximum destination tiles");
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_BASE + i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
