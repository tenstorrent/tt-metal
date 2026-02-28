// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_unpack_AB_custom_mm.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <bool read_transposed = false>
inline void _llk_unpack_AB_sdpa_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_mask,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1,
    const bool mask_chunk = false) {
    volatile uint* cfg = get_cfg_pointer();
    const std::uint32_t block_increment = read_transposed ? kt_dim * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? tile_size_a : ct_dim * tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    if (mask_chunk) {
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = base_address_mask;
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcB, 0b00000000, 1, 1);
    }

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, block_increment, inner_increment, kt_dim, ct_dim);
}
