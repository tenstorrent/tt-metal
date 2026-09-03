// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Descriptor-ownership check for `_llk_unpack_tilize_uninit_` (tt-llk#1161).
//
// The cross-op restore test (`unpack_tilize_uninit_restore_test.cpp`) configures
// the SrcA baseline and tears down with the SAME `num_faces`, so both the old
// (descriptor-writing) and the new (descriptor-preserving) Wormhole teardown
// leave the same value behind — it cannot tell the two apart. This test can: it
// establishes a pre-tilize descriptor Z-dim that DIFFERS from the tilize
// operand's `num_faces`, runs tilize init + uninit with no geometry reconfig in
// between, and reads the tile-descriptor word back on-device.
//
// Flow (unpack thread only, no stimuli / golden needed — the register IS the
// deliverable, same shape as `unpack_canonical_baseline_check_test.cpp`):
//   1. `_llk_unpack_hw_configure_(..., pre_num_faces)` programs the canonical
//      SrcA baseline for a *different* operand: descriptor Z-dim = pre_num_faces.
//   2. Snapshot tile-descriptor word 1 (y_dim in [15:0], z_dim in [31:16]).
//   3. `_llk_unpack_tilize_init_` + `_llk_unpack_tilize_uninit_` with
//      `tilize_num_faces != pre_num_faces`. Deliberately NO
//      `_llk_unpack_reconfig_*` in between — a reconfig would reprogram the
//      geometry and mask the escape.
//   4. Read the word back and LLK_ASSERT the per-arch contract:
//        * Wormhole: the whole word is UNCHANGED. Tilize neither writes nor
//          mutates the descriptor, so teardown must leave it alone. The old
//          teardown stamped z_dim = tilize_num_faces here, which is exactly the
//          corruption reported in tt-metal#45179 / #47016 — this assert fires on
//          that code.
//        * Blackhole: tilize init DOES write the descriptor (z_dim = num_faces
//          unconditionally, plus x_dim/z_dim=1 on the non-8-bit whole-tile path),
//          so teardown must re-establish z_dim = tilize_num_faces. y_dim is still
//          untouched.
//   5. Wormhole only: `Tile_x_dim_cntx0` is owned by tilize init, so teardown must
//      put the canonical face_r_dim-derived value back. Deliberately NOT asserted on
//      Blackhole: `_llk_unpack_tilize_uninit_wrapper_` hardcodes MAX_FACE_R_DIM there
//      rather than threading this test's face_r_dim through, so the expected value
//      would not correspond to the operand under test. The Blackhole assertions above
//      cover the descriptor word, which is what this test exists for.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals referenced by the LLK config helpers (configure_unpack_AB / sync_regfile_write).
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"
#include "params.h"

// Drain the config writes before reading them back (mirrors
// unpack_canonical_baseline_check_test.cpp).
static inline void drain_cfg_writes()
{
    tensix_sync();
    for (std::uint32_t i = 0; i < 10; i++)
    {
        asm volatile("nop");
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t tilize_num_faces = params.num_faces;
    const std::uint32_t face_r_dim       = params.TEST_FACE_R_DIM;

    // Pre-tilize baseline deliberately differs from the tilize operand's num_faces,
    // so a teardown that stamps the tilize operand's value is detectable. Both
    // values are legal descriptor Z-dims (<1, 2, 4>).
    const std::uint32_t pre_num_faces = (tilize_num_faces == 4) ? 2 : 4;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, face_r_dim, pre_num_faces, pre_num_faces);

    drain_cfg_writes();

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    const std::uint32_t pre_desc_word = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];

    _llk_unpack_tilize_init_wrapper_(
        formats.unpack_A_src, formats.unpack_A_dst, 1 /* ct_dim */, face_r_dim, false /* narrow_tile */, tilize_num_faces);

#ifdef ARCH_WORMHOLE
    _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst, tilize_num_faces, face_r_dim);
#else
    _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst, tilize_num_faces);
#endif

    drain_cfg_writes();

    const std::uint32_t post_desc_word = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];
    const std::uint32_t post_z_dim     = post_desc_word >> 16;
    const std::uint32_t post_y_dim     = post_desc_word & 0xffff;
    const std::uint32_t post_tile_x    = cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32];

#ifdef ARCH_WORMHOLE
    // Tilize does not own the descriptor on WH: the word must be bit-identical.
    LLK_ASSERT(post_desc_word == pre_desc_word, "WH tilize uninit must leave the SrcA tile-descriptor Y/Z-dim word untouched (tt-llk#1161)");

    // Tile_x_dim_cntx0 IS tilize's to restore. Only checked on WH: the BH test wrapper
    // cannot thread face_r_dim into uninit (it hardcodes MAX_FACE_R_DIM), so a tiny-tile
    // expectation would not be meaningful there.
    LLK_ASSERT(post_tile_x == canonical_unpA_tile_x_dim_cntx(face_r_dim), "tilize uninit must restore the canonical Tile_x_dim_cntx0");
#else
    // BH tilize init writes the descriptor, so uninit must re-establish the
    // tilize operand's baseline; y_dim is still not tilize's to touch.
    LLK_ASSERT(post_z_dim == tilize_num_faces, "BH tilize uninit must re-establish descriptor Z-dim = tilize operand num_faces");
    LLK_ASSERT(post_y_dim == (pre_desc_word & 0xffff), "tilize uninit must not disturb the descriptor Y-dim");
#endif

    // Silence unused-variable warnings on the arch whose asserts do not use these.
    (void)post_desc_word;
    (void)post_z_dim;
    (void)post_y_dim;
    (void)post_tile_x;
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif
