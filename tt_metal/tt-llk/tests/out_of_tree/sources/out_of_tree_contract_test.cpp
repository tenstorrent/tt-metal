
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Out-of-tree contract fixture. The kernel body is an ordinary unary datacopy
// (copied from sources/eltwise_unary_datacopy_test.cpp) so that a failure here
// means the out-of-tree wiring broke, not the LLK under it.
//
// This driver lives outside tests/sources/ on purpose: TestConfig receives it
// as an absolute path. The checks below fail the *compile*, so the contract is
// verified by observable build behaviour rather than by reaching into
// TestConfig internals.

#include <algorithm>
#include <cstdint>
#include <cstdio>

// Each #include below is itself the check that its search dir was registered:
// if the dir never reached the compiler, preprocessing dies right here with
// "No such file or directory" naming the header. There is deliberately no
// #error or static_assert guarding *resolution* — one could never fire, since
// the failing include stops the translation unit before any later line is
// evaluated.
//
// (1) Header from an out-of-tree -I dir, registered with add_include_dirs.
//     Two fixture dirs supply oot_probe.h with different ids, so which one
//     lands here reports the effective search-dir *precedence* — a question
//     the include alone cannot answer, which is why this one does carry a
//     check. OOT_PROBE_SHADOWED is asserted on by name from
//     test_consumer_contract.py's negative test; keep the token stable.
#include "oot_probe.h"

#if OOT_PROBE_ID != 2
#error "OOT_PROBE_SHADOWED: oot_probe.h resolved to the low-priority copy"
#endif

// (2) Header from an out-of-tree helpers tree (add_helpers_tree -> <tree>/include).
#include "oot_helpers.h"

// (3) Source pulled in via #include <foo.cpp> from <tree>/src, the
//     tests/helpers/src role (add_helpers_tree -> add_src_include_dirs).
#include <oot_src_probe.cpp>

// Both symbols come from the headers above and are compared against macros
// those same headers define, so these only confirm the definitions are usable
// in a constant expression — the resolution guarantee is the include itself.
static_assert(oot_helpers_marker() == OOT_EXPECTED_MARKER, "oot_helpers.h is not usable at compile time");
static_assert(oot_src_probe_value() == OOT_EXPECTED_SRC_VALUE, "oot_src_probe.cpp is not usable at compile time");

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if constexpr (!tilize_en)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);

        const std::uint32_t num_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;

        for (std::uint32_t i = 0; i < num_tiles; ++i)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
    else
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */);

        for (std::uint32_t i = 0; i < BLOCK_RT_DIM; i++)
        {
            const std::uint32_t read_offset = i * BLOCK_CT_DIM;
            for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
            {
                _llk_unpack_tilize_wrapper_(
                    L1_ADDRESS(params.buffer_A[read_offset]),
                    j,
                    formats.unpack_A_src,
                    formats.unpack_A_dst,
                    0 /* block_ct_dim */,
                    FACE_R_DIM,
                    4 /* num_faces */,
                    false /* narrow_tile */);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Hardware configuration first. This is the order the LLK sanitizer FSM
    // requires, not a preference: _llk_math_hw_configure_ performs the CONFIGURE
    // transition and the datacopy init performs INITIALIZED, and
    // common/sanitizer/impl.h asserts at ERROR level that "First transition must
    // be INITIAL -> CONFIGURED" ("the first operation in the kernel must be a
    // hardware configure").
    //
    // The datacopy driver this fixture mirrors calls init first, as do
    // eltwise_binary_test.cpp and reduce_test.cpp; matmul_test.cpp and
    // sfpu_sampling_test.cpp configure first. Those are latent rather than
    // broken today because the sanitizer is compiled out -- LLK_SAN_ENABLE is
    // not defined anywhere in this repo, so llk::san::* is the stub branch of
    // common/sanitizer/api.h.
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, tilize_en>>(params.num_faces, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            LLK_ASSERT(
                (params.DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "tile_num exceeds max dest tiles");
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                params.DST_INDEX + tile_num, formats.math, formats.math, params.num_faces);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            LLK_ASSERT(
                (params.DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "tile_num exceeds max dest tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                params.DST_INDEX + tile_num, L1_ADDRESS(params.buffer_Res[block_num * params.NUM_TILES_IN_BLOCK + tile_num]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}
#endif
