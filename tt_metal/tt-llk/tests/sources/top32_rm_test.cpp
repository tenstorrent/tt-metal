// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Top32 row-major sort (the DeepSeek top32_rm family) -- Blackhole only.
//
// Unblocked by tt-metal #52713 merging, which is what put these headers on main:
//   llk_lib/experimental/llk_unpack_A_top32_rm.h      _llk_unpack_A_top32_rm_init_ / _
//   llk_lib/experimental/llk_math_top32_rm.h          _llk_math_top32_rm_init_ / _
//   sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h
//       _top32_rm_init_ (and _top32_rm_configure_addrmod_ through it),
//       _bitonic_top32_phases_steps_, _bitonic_top32_merge_, _bitonic_top32_rebuild_
// Before this file every one of them was uncalled from tests/sources -- the only occurrence
// of _top32_rm_init_ in the test tree was inside a comment in sort_headers_coexist_test.cpp.
//
// What the op does
// ----------------
// One row of `TOP32_ROW_ELEMENTS` values plus a parallel row of indices, in ROW-MAJOR L1
// (element i at byte i * datum_size -- not tilized), reduced to the 32 largest values in
// descending order together with the index each came from.
//
// The unpack is the whole reason this family has its own LLK. `_llk_unpack_A_top32_rm_`
// takes 64 consecutive row-major elements and lands them in the FIRST COLUMN of 64 Dest
// rows: 16 elements per face at a 16-row stride, transposed within the face. So one Dest
// tile holds a 64-element working set as a column, which is the layout the bitonic sort
// addresses (its distances are 8/16/32/64 Dest ROWS). Values live in Dest tile
// VALUE_TILE, indices at VALUE_TILE + 2 -- a fixed +128 Dest rows, which is where
// bitonic_top32_load16/store16 expect them (`dst_indices_offset = 128`).
//
// Sequence, mirroring tests/tt_metal/tt_metal/test_kernels/compute/top32_rm_dev_compute.cpp
// (the only in-tree consumer) statement for statement:
//
//   chunk 0            unpack 64 values + 64 indices -> tiles 0 / 2
//                      phases_steps(descending)   full bitonic sort of the 64-slot column
//                      merge(across_tiles=false)  keep the max half
//                      rebuild(descending)        make the survivors monotone again
//   each later chunk   unpack 64 (or 32) more     -> tiles 1 / 3
//                      phases_steps(descending), merge(false), rebuild(ASCENDING)
//                      merge(across_tiles=true)   tile 1's top32 against tile 0's
//                      rebuild(descending)        on tile 0, the running top32
//
// The last chunk of a row that is 32 (mod 64) is unpacked with num_faces=2. That is not a
// padding shortcut: `_llk_unpack_A_top32_rm_` clears SrcA to -infinity before unpacking
// (TTI_UNPACR_NOP ... CLR_SRC_NEGINF), so the two unfed faces are -inf and lose every
// comparison, while `_llk_math_top32_rm_` still moves all four faces.
//
// Formats, and why one format for both operands
// ---------------------------------------------
// The consumer runs bf16 values and uint32 indices, which needs a srcA format reconfig
// between the two unpacks (`reconfig_data_format_srca`) and a pack reconfig on the way
// out. This driver instead carries indices as bf16 floats holding the integer itself,
// which is exact for index < 256 (8 mantissa bits) and therefore for every row length
// this test sweeps -- the same trick test_topk.py uses, minus the bit reinterpretation.
// One format for both operands means no reconfig on either side, so what is measured is
// the sort, not the reconfig sequence. The reconfig path is C3's territory
// (experimental_reconfig_escape_test.cpp) and the uint32-index path needs Mode B's
// transpose route to be worth it -- both recorded as follow-ups.
//
// dest_acc is a real axis here rather than a formality: it picks the index word width in
// the sort (`InstrModLoadStore::INT32` vs `LO16` in bitonic_top32_load16/store16) AND the
// Dest-move opcode in `llk_math_top32_rm_configure_mop` (ELWADD against a zeroed SrcB at
// fp32 Dest, MOVA2D otherwise). The consumer only ever builds fp32 Dest, so the
// dest_acc=No cells are the first exercise the 16-bit half of this family has had.
//
// TOP32_MODE = 1: the pre-sorted 1024 path
// ----------------------------------------
// The second mode covers the other three SFPU entry points --
// _bitonic_top32_of_1024_rm_pre_sorted_{prep,combine,final}_ -- and mirrors
// top32_rm_dev_compute_v2.cpp, which is what the consumer runs at >= 1024 elements. It
// does NOT use this family's unpack: a whole 32x32 tile is transposed into Dest
// (transpose_tile's LLK sequence), so each Dest column holds 32 of the row's elements, and
// prep/combine/final reduce 16 columns at a time instead of one 64-slot column.
//
// The contract that mode carries, and the reason it needs its own stimuli: the input must
// already be sorted into descending runs of 32 ("pre_sorted" in the function names). prep
// only builds bitonic sequences out of runs that are already monotone; hand it unsorted
// data and it returns a wrong answer rather than failing. This driver leaves that to the
// python side, which generates the same shape the family's own dev test does (value keyed
// on i % 32, descending, plus a per-group tiebreak).
//
// Mode 1 is Float32-only, and that is forced rather than chosen: at >= 1024 elements the
// indices leave bf16's exactly-representable range, and so do value tiebreaks fine enough
// to keep 32 group leaders distinct. Float32 also routes the transpose through its 32-bit
// path (unpack-to-dest + _llk_math_transpose_dest_), which is the branch transpose_tile
// takes for uint32 index tiles in the consumer.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// Dest layout. The sort reads its index operand at a fixed +2 tiles from the value tile,
// so these four are not free choices.
static constexpr std::uint32_t VALUE_TILE       = 0; // running top32 values
static constexpr std::uint32_t INDEX_TILE       = 2; // running top32 indices  (VALUE_TILE + 2)
static constexpr std::uint32_t STAGE_VALUE_TILE = 1; // incoming chunk's values
static constexpr std::uint32_t STAGE_INDEX_TILE = 3; // incoming chunk's indices

// Elements one _llk_unpack_A_top32_rm_ call moves: FACE_R_DIM per face, 4 faces.
static constexpr std::uint32_t ELEMENTS_PER_CHUNK = 4 * 16;

// L1 addresses are 16-byte words, and one chunk is ELEMENTS_PER_CHUNK * datum bytes.
// This is the same arithmetic llk_unpack_A_top32_rm_api.h does with
// `(64 >> 4) * datum_size * tile_index`.
static constexpr std::uint32_t CHUNK_ADDR_STRIDE = (ELEMENTS_PER_CHUNK * TOP32_DATUM_BYTES) / 16;

// Faces the chunk starting at `first` feeds: 4 for a full 64, 2 for a trailing 32.
inline constexpr std::uint32_t chunk_num_faces(std::uint32_t first)
{
    return (first + ELEMENTS_PER_CHUNK > TOP32_ROW_ELEMENTS) ? 2 : 4;
}

// Mode 1 walks whole tiles first, then finishes any remainder in 64-element chunks.
static constexpr std::uint32_t ELEMENTS_PER_TILE = 32 * 32;
static constexpr std::uint32_t NUM_TILE_CHUNKS   = TOP32_ROW_ELEMENTS / ELEMENTS_PER_TILE;
static constexpr std::uint32_t TAIL_FIRST        = NUM_TILE_CHUNKS * ELEMENTS_PER_TILE;

static constexpr bool PRE_SORTED_MODE = (TOP32_MODE == 1);

static_assert(!PRE_SORTED_MODE || TOP32_ROW_ELEMENTS >= ELEMENTS_PER_TILE, "the pre-sorted mode needs at least one whole 1024-element chunk");

#ifdef LLK_TRISC_UNPACK

// The experimental top32_rm LLK headers carry unused parameters that predate this test:
// nothing outside the Metal JIT build compiled them, and that build does not treat
// -Wunused-* as errors. The tt-llk harness does (-Wall -Werror -Wunused-parameter), so the
// two diagnostics are suppressed across this include alone -- neither editing the kernel nor
// relaxing the flag for every test.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_top32_rm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

// One operand of one chunk. Re-initialising per call mirrors the consumer, which has to
// because it reconfigures srcA between the value and index operands.
//
// Two branches, picked exactly as llk_unpack_A_top32_rm_api.h picks them (there on the src
// format being Int32, here on the harness's unpack_to_dest, which is the same question asked
// of the whole variant):
//
//   16-bit  the unpacker does the within-face 16x16 transpose itself and clears SrcA to
//           -infinity first, so unfed faces lose every comparison.
//   32-bit  the datum is too wide for SrcA, so the tile goes straight to Dest and the
//           within-face transpose moves to the math thread. Nothing clears to -infinity:
//           unfed faces arrive as ZEROACC zeros, so a partially-filled chunk is only safe
//           for strictly positive inputs.
inline void unpack_chunk(std::uint32_t base_address, std::uint32_t chunk, std::uint32_t num_faces, std::uint32_t src_format, std::uint32_t dst_format)
{
    _llk_unpack_A_top32_rm_init_<unpack_to_dest>(unpack_to_dest ? 0 : 1 /* within_face_16x16_transpose */, src_format, dst_format);
    _llk_unpack_A_top32_rm_<unpack_to_dest>(num_faces, base_address + chunk * CHUNK_ADDR_STRIDE, src_format, dst_format);
    if constexpr (unpack_to_dest)
    {
        _llk_unpack_set_srcb_dummy_valid_();
    }
}

// Mode 1: transpose_tile's unpack half. The 32-bit branch leaves the within-face 16x16
// transpose to the math thread's _llk_math_transpose_dest_ and needs the SrcB dummy valid
// that transpose feeds on; the branch is picked by the format, exactly as
// api/compute/transpose.h picks it.
inline void unpack_transpose_tile(std::uint32_t l1_tile, std::uint32_t src_format, std::uint32_t dst_format)
{
    _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(l1_tile), src_format, dst_format);
    if constexpr (unpack_to_dest)
    {
        _llk_unpack_set_srcb_dummy_valid_();
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t src_format = formats.unpack_A_src;
    const std::uint32_t dst_format = formats.unpack_A_dst;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    if constexpr (PRE_SORTED_MODE)
    {
        for (std::uint32_t chunk = 0; chunk < NUM_TILE_CHUNKS; chunk++)
        {
            // transpose_init, per api/compute/transpose.h: transpose_of_faces always;
            // within-face 16x16 and acc_to_dest only on the non-32-bit path, where the
            // unpacker does the whole 32x32 by itself (acc_to_dest with unpack_to_dest is
            // static_asserted out in llk_unpack_A.h). Re-run per chunk to stay in step with
            // the math thread's own re-init.
            _llk_unpack_A_init_<BroadcastType::NONE, !unpack_to_dest /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                1 /* transpose_of_faces */, unpack_to_dest ? 0 : 1 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, src_format, dst_format);

            unpack_transpose_tile(params.buffer_A[chunk], src_format, dst_format);
            unpack_transpose_tile(params.buffer_B[chunk], src_format, dst_format);
        }
    }

    const std::uint32_t values_base  = L1_ADDRESS(params.buffer_A[0]);
    const std::uint32_t indices_base = L1_ADDRESS(params.buffer_B[0]);

    // Mode 0 walks the whole row; mode 1 only the remainder after its whole tiles, which is
    // how top32_rm_dev_compute_v2.cpp finishes a row that is not a multiple of 1024. Both use
    // the same per-chunk pair, which branches on unpack_to_dest -- see unpack_chunk.
    constexpr std::uint32_t FIRST_CHUNK = PRE_SORTED_MODE ? TAIL_FIRST : 0;

    for (std::uint32_t first = FIRST_CHUNK; first < TOP32_ROW_ELEMENTS; first += ELEMENTS_PER_CHUNK)
    {
        const std::uint32_t chunk     = first / ELEMENTS_PER_CHUNK;
        const std::uint32_t num_faces = chunk_num_faces(first);

        unpack_chunk(values_base, chunk, num_faces, src_format, dst_format);
        unpack_chunk(indices_base, chunk, num_faces, src_format, dst_format);
    }
}

#endif

#ifdef LLK_TRISC_MATH

// The experimental top32_rm LLK headers carry unused parameters that predate this test:
// nothing outside the Metal JIT build compiled them, and that build does not treat
// -Wunused-* as errors. The tt-llk harness does (-Wall -Werror -Wunused-parameter), so the
// two diagnostics are suppressed across this include alone -- neither editing the kernel nor
// relaxing the flag for every test.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_top32_rm.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h"

#if TOP32_VIA_WRAPPERS
// The Metal-side wrapper layer for this family: 7 entry points that had no caller anywhere in
// the tree. They are thin -- each is the same `_llk_math_eltwise_unary_sfpu_params_` call this
// driver makes directly -- so routing through them costs nothing and is the only way anything
// calls them. Reachable from here for the same reason deepseek_moe_gate_test.cpp can include
// llk_sfpu/ headers: the tt-llk test build has the Metal llk_api directory on its include path.
#include "experimental/llk_sfpu/llk_math_deepseek_top32_rm.h"
#endif

using namespace ckernel;

// The consumer's sort directions, named rather than passed as 0/1 like the kernel does.
static constexpr bool DESCENDING = false;
static constexpr bool ASCENDING  = true;

// Every sort step is one SFPU call on a Dest tile under VectorMode::RC_custom -- the mode
// the consumer uses, and the one this family needs: it does its own Dest addressing (see
// set_dst_write_addr_offset_test.cpp), so the standard per-face walk must not be applied.
// Spelled through _llk_math_eltwise_unary_sfpu_params_ rather than the Metal
// SFPU_UNARY_CALL macro, which is what the other tt-llk sort driver (topk_xl_test.cpp)
// does; the expansion is the same call.
inline void top32_phases_steps(std::uint32_t dst_tile, bool direction)
{
#if TOP32_VIA_WRAPPERS
    llk_math_deepseek_top32_rm_local_sort<false /* APPROXIMATE */, is_fp32_dest_acc_en>(dst_tile, static_cast<int>(direction));
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_phases_steps_<false /* APPROXIMATION_MODE */, is_fp32_dest_acc_en>,
        dst_tile,
        VectorMode::RC_custom,
        static_cast<int>(direction));
#endif
}

inline void top32_merge(std::uint32_t dst_tile, bool across_tiles)
{
#if TOP32_VIA_WRAPPERS
    llk_math_deepseek_top32_rm_merge<false /* APPROXIMATE */, is_fp32_dest_acc_en, TOP32_TOP_MIN>(dst_tile, across_tiles);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_merge_<false /* APPROXIMATION_MODE */, is_fp32_dest_acc_en, TOP32_TOP_MIN>,
        dst_tile,
        VectorMode::RC_custom,
        across_tiles);
#endif
}

inline void top32_rebuild(std::uint32_t dst_tile, bool direction, bool skip_second)
{
#if TOP32_VIA_WRAPPERS
    llk_math_deepseek_top32_rm_rebuild<false /* APPROXIMATE */, is_fp32_dest_acc_en>(dst_tile, direction, skip_second);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_rebuild_<false /* APPROXIMATION_MODE */, is_fp32_dest_acc_en>, dst_tile, VectorMode::RC_custom, direction, skip_second);
#endif
}

// Mode 1's three entry points. prep's third template argument is the direction its rebuild
// leaves each column in, and the consumer alternates it per chunk -- descending for the
// first chunk, ascending for every later one -- so combine sees one bitonic sequence across
// the two tiles. It is a template argument, so both polarities are instantiated.
template <bool top_min>
inline void top32_pre_sorted_prep(std::uint32_t dst_tile)
{
#if TOP32_VIA_WRAPPERS
    llk_math_deepseek_top32_of_1024_rm_pre_sorted_prep<false /* APPROXIMATE */, is_fp32_dest_acc_en, top_min>(dst_tile);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_of_1024_rm_pre_sorted_prep_<false /* APPROXIMATION_MODE */, is_fp32_dest_acc_en, top_min>,
        dst_tile,
        VectorMode::RC_custom,
        dst_tile);
#endif
}

inline void top32_pre_sorted_combine(std::uint32_t dst_tile)
{
#if TOP32_VIA_WRAPPERS
    llk_math_deepseek_top32_of_1024_rm_pre_sorted_combine<false /* APPROXIMATE */, is_fp32_dest_acc_en>(dst_tile);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_of_1024_rm_pre_sorted_combine_<false /* APPROXIMATION_MODE */, is_fp32_dest_acc_en>,
        dst_tile,
        VectorMode::RC_custom,
        dst_tile);
#endif
}

inline void top32_pre_sorted_final(std::uint32_t dst_tile)
{
#if TOP32_VIA_WRAPPERS
    llk_math_deepseek_top32_of_1024_rm_pre_sorted_final<false /* APPROXIMATE */, is_fp32_dest_acc_en>(dst_tile);
#else
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_bitonic_top32_of_1024_rm_pre_sorted_final_<false /* APPROXIMATION_MODE */, is_fp32_dest_acc_en>,
        dst_tile,
        VectorMode::RC_custom,
        dst_tile);
#endif
}

// Mode 1: transpose_init's math half.
inline void math_transpose_init(std::uint32_t math_format)
{
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, math_format);
    if constexpr (unpack_to_dest)
    {
        _llk_math_transpose_dest_init_<false /* transpose_of_faces */, true /* is_32bit */>();
    }
}

// Mode 1: transpose_tile's math half, per format branch, mirroring api/compute/transpose.h.
// On the 32-bit path the unpacker only reordered faces, so the within-face 16x16 transpose
// happens here; on the 16-bit path the unpacker did the whole 32x32 and this is a datacopy.
inline void math_transpose_tile(std::uint32_t dst_tile, std::uint32_t math_format)
{
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        dst_tile, math_format, math_format);
    if constexpr (unpack_to_dest)
    {
        _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, false /* transpose_of_faces */, true /* is_32bit */>(dst_tile);
    }
}

// Move one unpacked operand into its Dest tile, the math half of the pair above and split the
// same way (llk_math_top32_rm_api.h). On the 16-bit path this runs the MOP over all four faces
// always -- per the LLK's own comment, the faces the unpacker left at -infinity have to reach
// Dest too, or a trailing 32-element chunk would sort against stale data. On the 32-bit path
// the data is already in Dest and what is left is the within-face transpose.
inline void copy_chunk_to_dest(std::uint32_t dst_tile, std::uint32_t num_faces, std::uint32_t src_format, std::uint32_t dst_format)
{
    _llk_math_top32_rm_init_<is_fp32_dest_acc_en>(num_faces, dst_format);
    if constexpr (unpack_to_dest)
    {
        _llk_math_transpose_dest_init_<false /* transpose_of_faces */, true /* is_32bit */>();
        _llk_math_top32_rm_<DST_SYNC, is_fp32_dest_acc_en, true /* unpack_to_dest */>(dst_tile, src_format, dst_format, num_faces);
        _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, false /* transpose_of_faces */, true /* is_32bit */>(dst_tile);
    }
    else
    {
        _llk_math_top32_rm_<DST_SYNC, is_fp32_dest_acc_en, false /* unpack_to_dest */>(dst_tile, src_format, dst_format, num_faces);
    }
}

// Sort a freshly loaded 64-slot column down to its top 32, in `direction`.
// phases_steps is the full bitonic sort; merge keeps the max half; rebuild restores
// monotonicity. skip_second=true because only the surviving half is needed.
inline void sort_chunk(std::uint32_t dst_tile, bool direction)
{
    top32_phases_steps(dst_tile, direction);
    top32_merge(dst_tile, false /* across_tiles */);
    top32_rebuild(dst_tile, direction, true /* skip_second */);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t math_format = formats.math;

    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // _top32_rm_init_ programs this family's ADDR_MODs, MOP and REPLAY buffer. It cannot
    // coexist with _topk_xl_init_<K, fused>() in one kernel -- they overlap in all three
    // (measured on BH p100a: the math thread hangs) -- which is why nothing here touches
    // the topk_xl family.
    if constexpr (PRE_SORTED_MODE)
    {
        // Chunk 0 seeds the running top32. The transpose init is re-run before every
        // chunk's pair of transposes, as the consumer does: the SFPU family in between owns
        // ADDR_MOD_6 and the SFPU control register, the datacopy owns the MOP and the rest
        // of the ADDR_MODs, and only the datacopy half has to be reinstated.
        math_transpose_init(math_format);
        math_transpose_tile(VALUE_TILE, math_format);
        math_transpose_tile(INDEX_TILE, math_format);

#if TOP32_VIA_WRAPPERS
        llk_math_deepseek_top32_rm_init<false /* APPROXIMATE */>();
#else
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::sfpu::_top32_rm_init_();
#endif

        top32_pre_sorted_prep<false /* top_min: descending */>(VALUE_TILE);

        for (std::uint32_t chunk = 1; chunk < NUM_TILE_CHUNKS; chunk++)
        {
            math_transpose_init(math_format);
            math_transpose_tile(STAGE_VALUE_TILE, math_format);
            math_transpose_tile(STAGE_INDEX_TILE, math_format);

            // Staged chunks are left ASCENDING so the two tiles form one bitonic sequence,
            // which is what combine's across_tiles merge consumes.
            top32_pre_sorted_prep<true /* top_min: ascending */>(STAGE_VALUE_TILE);
            top32_pre_sorted_combine(VALUE_TILE);
        }

        // Reduce the 16 surviving columns of F0/F1 to one: the final top32, in column 0.
        top32_pre_sorted_final(VALUE_TILE);

        // Tail, for a row that is not a multiple of 1024: fold the leftover 64-element chunks
        // into the running top32 with the plain family's Dest moves. The step sequence is v2's,
        // and it is NOT the plain mode's: where mode 0 opens with phases_steps (a full bitonic
        // sort of an arbitrary 64), v2 opens with `rebuild(skip_second=false)`, which sorts a
        // *bitonic* 64 rather than an arbitrary one. That is sound only because this mode's
        // input contract already holds -- two adjacent descending runs of 32 are bitonic as a
        // cyclic sequence -- and it is the reason the tail cannot be shared with mode 0.
        for (std::uint32_t first = TAIL_FIRST; first < TOP32_ROW_ELEMENTS; first += ELEMENTS_PER_CHUNK)
        {
            const std::uint32_t num_faces = chunk_num_faces(first);

            copy_chunk_to_dest(STAGE_VALUE_TILE, num_faces, math_format, math_format);
            copy_chunk_to_dest(STAGE_INDEX_TILE, num_faces, math_format, math_format);

            top32_rebuild(STAGE_VALUE_TILE, DESCENDING, false /* skip_second */);
            top32_merge(STAGE_VALUE_TILE, false /* across_tiles */);
            top32_rebuild(STAGE_VALUE_TILE, ASCENDING, true /* skip_second */);

            top32_merge(VALUE_TILE, true /* across_tiles */);
            top32_rebuild(VALUE_TILE, DESCENDING, true /* skip_second */);
        }

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
        return;
    }

#if TOP32_VIA_WRAPPERS
    llk_math_deepseek_top32_rm_init<false /* APPROXIMATE */>();
#else
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_top32_rm_init_();
#endif

    for (std::uint32_t first = 0; first < TOP32_ROW_ELEMENTS; first += ELEMENTS_PER_CHUNK)
    {
        const std::uint32_t num_faces = chunk_num_faces(first);
        const bool is_first_chunk     = (first == 0);

        const std::uint32_t value_tile = is_first_chunk ? VALUE_TILE : STAGE_VALUE_TILE;
        const std::uint32_t index_tile = is_first_chunk ? INDEX_TILE : STAGE_INDEX_TILE;

        copy_chunk_to_dest(value_tile, num_faces, math_format, math_format);
        copy_chunk_to_dest(index_tile, num_faces, math_format, math_format);

        if (is_first_chunk)
        {
            // Seeds the running top32.
            sort_chunk(VALUE_TILE, DESCENDING);
            continue;
        }

        // The staged chunk is rebuilt ASCENDING so that the two columns form one bitonic
        // sequence for the across-tiles merge below.
        top32_phases_steps(STAGE_VALUE_TILE, DESCENDING);
        top32_merge(STAGE_VALUE_TILE, false /* across_tiles */);
        top32_rebuild(STAGE_VALUE_TILE, ASCENDING, true /* skip_second */);

        // Fold the staged top32 into the running one. across_tiles=true is what makes the
        // merge read its second operand a whole tile (64 Dest rows) away.
        top32_merge(VALUE_TILE, true /* across_tiles */);
        top32_rebuild(VALUE_TILE, DESCENDING, true /* skip_second */);
    }

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();

    // The result is a COLUMN: one datum per Dest row, 32 rows of it. Narrowing the packer's
    // x count to a single datum is what turns that column into 32 contiguous L1 elements,
    // and is exactly what the consumer does before its two pack_tile calls.
    TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0);

    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(VALUE_TILE, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(INDEX_TILE, L1_ADDRESS(params.buffer_Res[1]));

    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
