// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"

#ifdef ARCH_BLACKHOLE
#include "experimental/llk_fast_untilize_common.h"
#endif
#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_math_fast_untilize_api.h"
#endif
#endif
#ifdef TRISC_UNPACK
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_unpack_fast_untilize_api.h"
#endif
#endif
#ifdef TRISC_PACK
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_pack_fast_untilize_api.h"
#endif
#endif

namespace ckernel {

// BH fast-untilize is the row-major counterpart to fast-tilize. Regular
// pack_untilize first loads/copies tiles into DEST tile-by-tile, then the
// packer reads two interfaces per PACR for a 4-face tile row. This path groups
// the row into 2/3/4-tile chunks, has math place the chunk in the exact order
// pack needs, and packs with ALL_INTF_ACTIVE + STRIDED_MODE so full-width PACRs
// emit wider row-major strips. That reduces pack-side PACR count and improves
// output bandwidth. The real unpack payload is unchanged, but the common 16-bit
// DEST path also avoids the generic zero-SrcB sideband; native fp32 DEST keeps
// that sideband because math copies with ELWADD. Compressed BFP input is the
// exception on the unpack side because each tile must still be addressed around
// its exponent section.

template <std::uint32_t full_ct_dim, bool configure_remap>
ALWI void fast_untilize_init_impl(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    static_assert(full_ct_dim > 0, "fast_untilize full_ct_dim must be greater than 0");

#ifdef ARCH_BLACKHOLE
    if constexpr (full_ct_dim == 1) {
        if constexpr (configure_remap) {
            pack_untilize_init<1, 1>(icb, ocb, call_line);
        } else {
            pack_untilize_init_skip_remap<1, 1>(icb, ocb, call_line);
        }
        return;
    }

    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

    constexpr std::uint32_t first_unit_dim = fast_untilize_next_unit_dim(full_ct_dim);

    // Fast-untilize can run immediately after other LLKs (for example matmul
    // bias pack_tile into the same CB). Re-enter a known math/pack sync
    // contract so stale dest offset/semaphore state cannot leak into the fast
    // packer.
    MATH((_llk_math_pack_sync_init_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));
    PACK((_llk_pack_dest_init_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));
    UNPACK((llk_unpack_fast_untilize_init(icb, first_unit_dim)));
    if constexpr (configure_remap) {
        MATH((llk_math_fast_untilize_init()));
    } else {
        MATH((llk_math_fast_untilize_init_skip_remap()));
    }
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_fast_untilize_init<FAST_UNTILIZE_MAX_UNIT_DIM, full_ct_dim>(ocb)));
    PACK((_llk_init_packer_dest_offset_registers_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>()));
#else
    if constexpr (configure_remap) {
        pack_untilize_init<full_ct_dim, full_ct_dim>(icb, ocb, call_line);
    } else {
        pack_untilize_init_skip_remap<full_ct_dim, full_ct_dim>(icb, ocb, call_line);
    }
#endif
}

// Default fast-untilize init configures BH DEST remap. Use the skip-remap variant
// only when the caller has already configured remap and no intervening op changes it.
template <std::uint32_t full_ct_dim>
ALWI void fast_untilize_init(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    fast_untilize_init_impl<full_ct_dim, true>(icb, ocb, call_line);
}

template <std::uint32_t full_ct_dim>
ALWI void fast_untilize_init_skip_remap(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    fast_untilize_init_impl<full_ct_dim, false>(icb, ocb, call_line);
}

template <std::uint32_t full_ct_dim>
ALWI void fast_untilize_block(
    uint32_t icb, uint32_t ocb, uint32_t input_tile_index = 0, uint32_t output_tile_index = 0) {
    static_assert(full_ct_dim > 0, "fast_untilize full_ct_dim must be greater than 0");

#ifdef ARCH_BLACKHOLE
    if constexpr (full_ct_dim == 1) {
        pack_untilize_block<1, 1>(icb, 1, ocb, 0);
        return;
    }

    // Keep the common 2/3/4-tile case as a direct path. Routing it through the
    // generic decomposition loop materializes unit_dims state in the hot kernel
    // and costs enough instructions to erase the small-width fast-path gain.
    if constexpr (full_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM) {
        constexpr std::uint32_t unit_dim = full_ct_dim;

        MATH((_llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>()));
        UNPACK((llk_unpack_fast_untilize_block(icb, input_tile_index, unit_dim)));
        MATH((llk_math_fast_untilize_block(0, unit_dim)));
        MATH((_llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));

        PACK((llk_packer_wait_for_math_done()));
        PACK((llk_pack_fast_untilize_block<FAST_UNTILIZE_MAX_UNIT_DIM>(ocb, output_tile_index, unit_dim)));
        PACK((_llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));
    } else {
        std::uint32_t tiles_done = 0;
        constexpr std::uint32_t first_unpack_unit_dim = fast_untilize_next_unit_dim(full_ct_dim);
        [[maybe_unused]] std::uint32_t prev_unpack_unit_dim = first_unpack_unit_dim;
        [[maybe_unused]] std::uint32_t prev_pack_unit_dim = 0;

        while (tiles_done < full_ct_dim) {
            const std::uint32_t remaining_tiles = full_ct_dim - tiles_done;
            const std::uint32_t unit_dim = fast_untilize_next_unit_dim(remaining_tiles);

            MATH((_llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>()));
            UNPACK(
                (llk_unpack_fast_untilize_block(icb, input_tile_index + tiles_done, unit_dim, prev_unpack_unit_dim)));
            MATH((llk_math_fast_untilize_block(0, unit_dim)));
            MATH((_llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));

            PACK((llk_packer_wait_for_math_done()));
            PACK((llk_pack_fast_untilize_block_strided<FAST_UNTILIZE_MAX_UNIT_DIM, full_ct_dim>(
                ocb, output_tile_index, tiles_done, unit_dim, prev_pack_unit_dim)));
            PACK((_llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));

            tiles_done += unit_dim;
        }

        // Preserve the init-time first-unit MOP invariant for the next stateless block call.
        UNPACK((llk_unpack_fast_untilize_restore_unit_dim(icb, first_unpack_unit_dim, prev_unpack_unit_dim)));
    }
#else
    pack_untilize_block<full_ct_dim, full_ct_dim>(icb, 1, ocb, 0);
#endif
}

template <std::uint32_t full_ct_dim>
ALWI void fast_untilize_uninit(uint32_t ocb) {
    static_assert(full_ct_dim > 0, "fast_untilize full_ct_dim must be greater than 0");

#ifdef ARCH_BLACKHOLE
    if constexpr (full_ct_dim == 1) {
        pack_untilize_uninit(ocb);
        return;
    }

    UNPACK((llk_unpack_fast_untilize_uninit()));
    MATH((llk_math_fast_untilize_uninit()));
    // Leave the math/pack semaphore in the mode requested by the kernel so
    // a following LLK does not inherit fast-untilize's private half-sync.
    if constexpr (DST_SYNC_MODE != FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE) {
        MATH((_llk_math_pack_sync_init_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
    }
    PACK((llk_init_packer_dest_offset_registers<PackMode::Default>()));
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_fast_untilize_uninit<FAST_UNTILIZE_MAX_UNIT_DIM, full_ct_dim>(ocb)));
#else
    pack_untilize_uninit(ocb);
#endif
}

}  // namespace ckernel
