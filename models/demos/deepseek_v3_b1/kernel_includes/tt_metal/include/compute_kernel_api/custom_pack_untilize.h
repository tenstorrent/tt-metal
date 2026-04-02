// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

// clang-format off
// This is a custom version that doesn't configure MATH, as this assumes it's already been configured before
/**
 * Performs the necessary hardware and software initialization for the pack untilize operation. This initialization
 * function should be used when the desired PACK input is already in DEST register - therefore, it doesn't
 * configure UNPACK and MATH threads for transferring data from circular buffers to DEST register (this is done with
 * `pack_untilize_init` function). Matching pack untilize operation for this initialization function is
 * `pack_untilize_dest`. In order for this untilization to be performed correctly, some other function must
 * place the tiles in the DEST register, e.g. `reduce_tile`, `copy_tile`, etc. This initialization function
 * should, therefore, be called right after the op-specific initialization function (`reduce_init`, `copy_init`, etc.).
 *
 * Since pack untilize works on a block of tiles, the user should specify the width of a single block (block_ct_dim),
 * and the width of the full block (`full_ct_dim`). It is not needed to provide the height of the block during the
 * initialization, since `pack_untilize_block` will loop over the height of the block. Note that the maximum size
 * of the block is limited by the size of the DEST and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * NOTE: This function allows the user to specify `face_r_dim` and `num_faces` through function parameters. Setting these
 * parameters results in an expensive MMIO write and cannot be avoided currently.
 * This should be addressed more systematically within the issue tt-metal#22820, since these two values can be inferred
 * from the circular buffer description, the same way as it is done in `llk_pack_hw_configure`. This
 * would remove the need for `llk_pack_untilize_hw_configure_disaggregated` altogether and we would pay the price
 * of the MMIO write only once, in `compute_kernel_hw_startup`.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                      | Type      | Valid Range               | Required              |
 * |------------|----------------|--------------------------------------------------|-----------|---------------------------|-----------------------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                 | uint32_t  | 1 to max (see note)       | False (default = 8)   |
 * | Template   | full_ct_dim    | Width of a full input in tiles                   | uint32_t  | Divisible by block_ct_dim | False                 |
 * | Template   | narrow_row     |  Whether the provided input is narrow            | bool      | true/false                | False                 |
 * | Template   | row_num_datums | Number of datums per row                         | uint32_t  | >= 1                      | False                 |
 * | Template   | dense          | Packs two 2 face tiles in a single 4 face region | bool      | true/false                | False (default false) |
 * | Function   | ocb            | Output circular buffer identifier                | uint32_t  | 0 to 31                   | True                  |
 * | Function   | face_r_dim     | Face height in rows                              | uint32_t  | 1, 8 or 16                | False (default = 16)  |
 * | Function   | num_faces      | Number of faces                                  | uint32_t  | 1, 2 or 4                 | False (default = 4)   |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
ALWI void custom_pack_untilize_dest_init(
    uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4, uint32_t call_line = __builtin_LINE()) {
    state_configure<Operand::PACK>(ocb, call_line);
    // TODO NC: A workaround for tt-metal#17132. Should be addressed more systematically in tt-llk#989
    PACK(
        (llk_pack_untilize_hw_configure_disaggregated<DST_ACCUM_MODE, false /*untilize*/>(ocb, face_r_dim, num_faces)));
    PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim, false, narrow_row, row_num_datums, dense>(
        ocb, face_r_dim, num_faces)));
    PACK((llk_init_packer_dest_offset_registers<true, false>()));
}

}  // namespace ckernel
