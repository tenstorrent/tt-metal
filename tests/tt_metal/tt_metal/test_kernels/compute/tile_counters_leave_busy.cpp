// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deliberately leaves this Neo's Tensix-only tile counters holding entries. The kernel binds no DFB, so
// the firmware's kernel-exit drain does not wait for those entries.

#include <cstdint>
#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
#include "internal/tt-2xx/quasar/overlay/remapper_common.hpp"

void kernel_main() {
#ifdef UCK_CHLKC_UNPACK
    constexpr std::uint32_t capacity = get_arg(args::capacity);
    constexpr std::uint32_t num_entries = get_arg(args::num_entries);
    static_assert(num_entries != 0 && num_entries <= capacity);
    constexpr std::uint32_t num_pairs = dfb::NUM_TENSIX_ONLY_TILE_COUNTERS / 2;
    const std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    const std::uint32_t first_pair = dfb::REMAPPER_ONE_TO_ONE_PAIR_START + neo_id * num_pairs;
    const std::uint32_t client_id = static_cast<std::uint32_t>(overlay::NEO_0) + neo_id;

    // Arm the pairs; ClientL carries the valid bits, so it is written last. Writes are posted, so read one back.
    for (std::uint32_t i = 0; i < num_pairs; i++) {
        const std::uint32_t tc = dfb::TC_TENSIX_POOL_START + 2 * i;
        WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(first_pair + i), client_id | ((tc + 1) << 3));
        WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(first_pair + i), client_id | (tc << 3) | (1u << 8));
    }
    (void)READ_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(first_pair + num_pairs - 1));

    for (std::uint32_t i = 0; i < num_pairs; i++) {
        const std::uint32_t tc = dfb::TC_TENSIX_POOL_START + 2 * i;
        ckernel::trisc::tile_counters[tc].f.buf_capacity = capacity;
        ckernel::trisc::tile_counters[tc].f.posted = num_entries;  // increments by the value written
    }
    // Wait until every post has reached its counter, so it went through its pair before the pair is cleared.
    for (std::uint32_t i = 0; i < num_pairs; i++) {
        const std::uint32_t tc = dfb::TC_TENSIX_POOL_START + 2 * i;
        while ((ckernel::trisc::tile_counters[tc].f.posted & 0xFFFFu) != num_entries) {
        }
    }

    // Disarm the pairs, so the stale state the test checks is in the counters alone.
    for (std::uint32_t i = 0; i < num_pairs; i++) {
        WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(first_pair + i), 0u);
    }
    (void)READ_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(first_pair + num_pairs - 1));
#endif
}
