// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// One-snapshot full-body Welford diagnostic.  TRACE_N selects exactly one
// post-row state.  For the HAND impls (0/1) no SFPSTORE occurs before that
// row, so the observed prefix is not perturbed by tracing.  The clean
// SEMANTIC impls (2/3/4) do store before the traced row — accumulator parks
// to private scratch (dst tile 10) and the traced block's input snapshot to
// its own trace slots — but none of those stores can perturb the observed
// values: they never touch the input tile, and the inputs they snapshot are
// immutable locals.  The remaining rows intentionally still exist in the
// linked program to keep code-generation context full-tile.

#include <array>
#include <cstdint>
#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context = 0;
std::uint32_t pack_sync_tile_dst_ptr = 0;
std::uint32_t math_sync_tile_dst_index = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
void run_kernel(RUNTIME_PARAMETERS params) {
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}
#endif

#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"
using namespace ckernel;
namespace {
constexpr std::array<std::uint32_t, 0> no_recip_lut{};

template <std::uint32_t N>
sfpi_inline void maybe_capture() {
    if constexpr (N == TRACE_N) {
        // Dedicated Dst locations, all distinct from input tile 0.  This is
        // emitted only after the selected state has been produced.
        TTI_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7,  64);
        TTI_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 128);
        TTI_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 192);
        TTI_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 256);
        TTI_SFPSTORE(p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 320);
        TTI_SFPSTORE(p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 384);
        TTI_SFPSTORE(p_sfpu::LREG6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 448);
        TTI_SFPSTORE(p_sfpu::LREG7, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 512);
        TTI_SFPSTORE(p_sfpu::LREG11, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, 576);
    }
}

template <std::uint32_t N, std::uint32_t Input>
sfpi_inline void raw_step() {
    _load_recip_of_idx_<0>(N - 1, no_recip_lut);
    _compute_welfords_row_<Input>();
    maybe_capture<N>();
}

// ---------------------------------------------------------------------------
// CLEAN SEMANTIC IMPLEMENTATIONS (impls 2/3/4) — plain typed sfpi C++ only.
//
// Rewritten (lane BB) after the LLK-pristine ruling removed the trusted
// __builtin_rvtt_sfprawlreg_access annotation from the (handwritten)
// _welfords_load_block_: the old semantic body read sfpi::l_reg[LReg0..3]
// populated by that helper's raw TTI SFPLOADs and pinned its accumulators in
// l_reg[LReg4/5] — both hand-isms.  Now: inputs come from sfpi::dst_reg[]
// directly, mean/m2 are plain vFloat locals carried across the whole body,
// and 1/N is a constexpr per-row constant.  The compiler proves everything;
// nothing asserts a register contract.
//
// Dst gather: a 32x32 tile row spans two faces (cols 0-15 and 16-31 live 16
// dst rows apart), so no single 32-lane dst_reg read is one tile row.  The
// four reads below fetch the block's 4 rows x 32 cols and subvec_transp —
// an EXECUTED SFPTRANSP data movement the algorithm requires to re-shape
// {2 half-rows per vector} into {1 full row per vector} — never a compiler
// hint.  Same offsets and the same transpose the hand load-block executes.
//
// Typed trace capture: the hand impls' capture snapshots raw LREG0-7/11 via
// TTI_SFPSTORE; the semantic impls have no LREG ABI, so vfloat_capture
// stores the typed locals themselves to the SAME trace Dst slots in typed
// code.  Slot map (what the harness labels these slots): 0-3 = the block's
// four input rows (NOTE: pristine values; the hand impl's LREG0-3 are
// partially clobbered by _compute_welfords_row_, and only slots 4/5 are
// asserted), 4 = mean (ASSERTED), 5 = m2 (ASSERTED), 6 = new-mean scratch
// (== mean after the row), 7 = 1/N, 8 = the -1 constant the hand ABI keeps
// in LREG11.
// ---------------------------------------------------------------------------

// Input-slot capture: stores the traced block's four (immutable) input rows
// to trace slots 0-3 immediately after the gather — the same values the
// post-row capture would store, emitted early so x0-x3 are not all forced
// live through every step of the block (they overflow the 8-lreg file
// together with the accumulators and the fold temporaries of impl 4).
template <std::uint32_t Base>
sfpi_inline void vfloat_capture_inputs(sfpi::vFloat x0, sfpi::vFloat x1, sfpi::vFloat x2, sfpi::vFloat x3)
{
    if constexpr (TRACE_N > Base && TRACE_N <= Base + 4)
    {
        // dst_reg index = SFPSTORE offset / 2 (SFP_DESTREG_STRIDE): the same
        // Dst locations the hand capture writes (offsets 64..256).
        sfpi::dst_reg[32]  = x0;
        sfpi::dst_reg[64]  = x1;
        sfpi::dst_reg[96]  = x2;
        sfpi::dst_reg[128] = x3;
    }
}

// Post-row capture: the asserted state (mean, m2) plus the diagnostic slots,
// stored after the traced row's update — trace offsets 320..576.
template <std::uint32_t N>
sfpi_inline void vfloat_capture(sfpi::vFloat mean, sfpi::vFloat m2)
{
    if constexpr (N == TRACE_N)
    {
        constexpr float recip = 1.0f / static_cast<float>(N);
        sfpi::dst_reg[160]    = mean;
        sfpi::dst_reg[192]    = m2;
        sfpi::dst_reg[224]    = mean;
        sfpi::dst_reg[256]    = sfpi::vFloat(recip);
        // The hand ABI keeps -1 in LREG11; here it is the architectural
        // constant register (no lreg is consumed materializing it).
        sfpi::dst_reg[288] = sfpi::vFloat(sfpi::vConstNeg1);
    }
}

template <std::uint32_t N, std::uint32_t Impl>
sfpi_inline void vfloat_step(sfpi::vFloat x, sfpi::vFloat& mean, sfpi::vFloat& m2)
{
    sfpi::vFloat delta = x - mean;
    constexpr float recip = 1.0f / static_cast<float>(N);
    if constexpr (Impl == 2) { // VFLOAT_DIRECT: normal source shape/literals.
        sfpi::vFloat next_mean = mean + delta * recip;
        sfpi::vFloat next_m2 = m2 + delta * (x - next_mean);
        mean = next_mean;
        m2 = next_m2;
    } else if constexpr (Impl == 3) { // VFLOAT_RESCUE.
        mean += delta * recip;
        sfpi::vFloat delta2 = x - mean;
        m2 += delta * delta2;
    } else { // VFLOAT_MANUAL_EARLY_FOLD.
        sfpi::vFloat delta2 = x - (mean + delta * recip);
        mean += delta * recip;
        m2 += delta * delta2;
    }
}

// Accumulator park slots (FP32 SFPLOAD/SFPSTORE address space, mod0=FMT_FP32):
// fp32 address A maps to physical dst rows ((A & 0x1F8) << 1) | (A & 7) plus
// the +8 pair rows, so A=320 -> physical rows 640-651 and A=328 -> 656-667:
// both inside physical dst tile 10, disjoint from the input tile (rows 0-63)
// and every trace-capture tile (rows 64-639).  dst_reg[] indexes are
// address/2: 160 and 164.  FP32 format round-trips the fp32 accumulators
// exactly (a FMT_SRCB park would re-round them to bf16 every block).
constexpr int VF_SCRATCH_MEAN = 160;
constexpr int VF_SCRATCH_M2   = 164;

template <std::uint32_t Impl, std::uint32_t Base, std::uint32_t I, std::uint32_t J>
sfpi_inline void vfloat_block(sfpi::vFloat& mean, sfpi::vFloat& m2)
{
    // ACCUMULATOR PARK (executed dst traffic, not a hint): the architectural
    // SFPTRANSP permutes BOTH four-register banks (Transpose4(0) and
    // Transpose4(4)); the hand load-block survives it by bracketing its raw
    // SFPLOADs in a TRANSP/TRANSP involution that re-descrambles the
    // accumulator bank.  Typed sfpi has no spelling of that involution
    // (subvec_transp models only its four operands — sfpi-gcc rvtt.md marks
    // the 4-operand tuple DELIBERATELY UNAUDITED for exactly this reason),
    // so any value the compiler leaves in the companion bank across the
    // transpose is destroyed.  The semantic body therefore parks mean/m2 in
    // private Dst scratch across every transpose; they are lreg-resident
    // only between transposes.
    sfpi::dst_reg[VF_SCRATCH_MEAN].mode<sfpi::DataLayout::F32>() = mean;
    sfpi::dst_reg[VF_SCRATCH_M2].mode<sfpi::DataLayout::F32>()   = m2;
    // Block (I,J) covers tile rows (I*16 + J*4)..+3.  SFPSTORE/SFPLOAD offset
    // units are 16-datum dst rows; dst_reg[] indexes 32-datum vectors, so
    // dst_reg index = offset / 2.  Offsets {o, o+2, o+16, o+18} with
    // o = I*32 + 4*J are the hand load-block's offsets exactly.
    constexpr int base = static_cast<int>(I * 16u + 2u * J);
    sfpi::vFloat x0    = sfpi::dst_reg[base + 0];
    sfpi::vFloat x1    = sfpi::dst_reg[base + 1];
    sfpi::vFloat x2    = sfpi::dst_reg[base + 8];
    sfpi::vFloat x3    = sfpi::dst_reg[base + 9];
    // Executed-instruction semantics: SFPTRANSP re-shapes the four vectors
    // from {4 rows x 8 same-parity cols} into {one tile row of 32 cols each}
    // — real data movement required by the per-column reduction, not a hint.
    sfpi::subvec_transp(x0, x1, x2, x3);
    mean = sfpi::dst_reg[VF_SCRATCH_MEAN].mode<sfpi::DataLayout::F32>();
    m2   = sfpi::dst_reg[VF_SCRATCH_M2].mode<sfpi::DataLayout::F32>();
    vfloat_capture_inputs<Base>(x0, x1, x2, x3);
    vfloat_step<Base + 1, Impl>(x0, mean, m2);
    vfloat_capture<Base + 1>(mean, m2);
    vfloat_step<Base + 2, Impl>(x1, mean, m2);
    vfloat_capture<Base + 2>(mean, m2);
    vfloat_step<Base + 3, Impl>(x2, mean, m2);
    vfloat_capture<Base + 3>(mean, m2);
    vfloat_step<Base + 4, Impl>(x3, mean, m2);
    vfloat_capture<Base + 4>(mean, m2);
}

template <std::uint32_t Impl, std::uint32_t Base, std::uint32_t I, std::uint32_t J>
sfpi_inline void block() {
    static_assert(Impl <= 1, "block() is the HAND path; semantic impls use vfloat_block()");
    _welfords_load_block_<I, J>();
    if constexpr (Impl == 0) { // HANDWRITTEN_DIRECT.
        raw_step<Base+1, p_sfpu::LREG0>(); raw_step<Base+2, p_sfpu::LREG1>();
        raw_step<Base+3, p_sfpu::LREG2>(); raw_step<Base+4, p_sfpu::LREG3>();
    }
    else
    { // HANDWRITTEN_REPLAY.
        _load_recip_of_idx_<0>(Base, no_recip_lut); _execute_welfords_row_replay_buffer_<p_sfpu::LREG0>(); maybe_capture<Base+1>();
        _load_recip_of_idx_<0>(Base+1, no_recip_lut); _execute_welfords_row_replay_buffer_<p_sfpu::LREG1>(); maybe_capture<Base+2>();
        _load_recip_of_idx_<0>(Base+2, no_recip_lut); _execute_welfords_row_replay_buffer_<p_sfpu::LREG2>(); maybe_capture<Base+3>();
        _load_recip_of_idx_<0>(Base + 3, no_recip_lut);
        _execute_welfords_row_replay_buffer_<p_sfpu::LREG3>();
        maybe_capture<Base + 4>();
    }
}

template <std::uint32_t Impl>
sfpi_inline void full_body() {
    if constexpr (Impl <= 1)
    { // Hand impls: unchanged (byte-identity gated).
        block<Impl, 0, 0, 0>();
        block<Impl, 4, 0, 1>();
        block<Impl, 8, 0, 2>();
        block<Impl, 12, 0, 3>();
        block<Impl, 16, 1, 0>();
        block<Impl, 20, 1, 1>();
        block<Impl, 24, 1, 2>();
        block<Impl, 28, 1, 3>();
    }
    else
    { // Clean semantic impls: accumulators are plain locals, zero-initialized
      // here (the typed equivalent of the hand path's LREG4/5 clear).
        sfpi::vFloat mean = 0.0f;
        sfpi::vFloat m2   = 0.0f;
        vfloat_block<Impl, 0, 0, 0>(mean, m2);
        vfloat_block<Impl, 4, 0, 1>(mean, m2);
        vfloat_block<Impl, 8, 0, 2>(mean, m2);
        vfloat_block<Impl, 12, 0, 3>(mean, m2);
        vfloat_block<Impl, 16, 1, 0>(mean, m2);
        vfloat_block<Impl, 20, 1, 1>(mean, m2);
        vfloat_block<Impl, 24, 1, 2>(mean, m2);
        vfloat_block<Impl, 28, 1, 3>(mean, m2);
        // The body's OUTPUT contract: deposit the final accumulators in the
        // Dst scratch slots.  The hand impls hand them to the next phase in
        // LREG4/5 (a register ABI the typed body deliberately has none of);
        // without an observable output the TRACE_N=0 perf build's last block
        // is dead code and the perf variant would under-measure the
        // algorithm (caught in the lane BB disassembly review).
        sfpi::dst_reg[VF_SCRATCH_MEAN].mode<sfpi::DataLayout::F32>() = mean;
        sfpi::dst_reg[VF_SCRATCH_M2].mode<sfpi::DataLayout::F32>()   = m2;
    }
}
}  // namespace
void run_kernel(RUNTIME_PARAMETERS params) {
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);
    _llk_math_welfords_sfpu_init_(); ckernel::sfpu::_clear_previous_mean_and_m2_();
    {
        START_PERF_MEASURE("WELFORD_BODY")
        _llk_math_welfords_sfpu_params_(full_body<TRACE_IMPL>, 0);
    }
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
void run_kernel(RUNTIME_PARAMETERS params) {
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>(); _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(2, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(3, L1_ADDRESS(params.buffer_Res[2]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(4, L1_ADDRESS(params.buffer_Res[3]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(5, L1_ADDRESS(params.buffer_Res[4]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(6, L1_ADDRESS(params.buffer_Res[5]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(7, L1_ADDRESS(params.buffer_Res[6]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(8, L1_ADDRESS(params.buffer_Res[7]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(9, L1_ADDRESS(params.buffer_Res[8]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
