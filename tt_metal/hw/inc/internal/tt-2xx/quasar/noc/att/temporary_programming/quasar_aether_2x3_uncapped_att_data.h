// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/tt-2xx/quasar/noc/att/temporary_programming/att_program_types.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_uncapped_att_config.h"

/**
 * @file
 * @brief The ATT tables the firmware writes at boot on the 2x3 emulator for the
 * uncapped map. The base map's tables plus a DRAM window in mask slot 2 and
 * its two endpoint rows.
 */
namespace quasar_aether_2x3_uncapped_att_program {

inline constexpr noc_att::MaskEntry MASKS[] = {
    {.slot = 0, .window = quasar_aether_2x3_uncapped_att_config::LOCAL_WINDOW, .bar = 0},
    {.slot = 1, .window = quasar_aether_2x3_uncapped_att_config::REMOTE_WINDOW, .bar = 0},
    {.slot = 2, .window = quasar_aether_2x3_uncapped_att_config::DRAM_WINDOW, .bar = 0},
};

inline constexpr noc_att::EndpointEntry ENDPOINTS[] = {
    // Remote table starts at index 1. Index 0 is patched to the issuing tile
    // for the local window by noc_att::program_for_test().
    {.index = 1, .x = 0, .y = 1},  // remote selector 0: left Tensix
    {.index = 2, .x = 1, .y = 1},  // remote selector 1: right Tensix
    {.index = 3, .x = 0, .y = 0},  // remote selector 2: left DRAM (64 MiB alias)
    {.index = 4, .x = 1, .y = 0},  // remote selector 3: right DRAM (64 MiB alias)
    {.index = 5, .x = 0, .y = 2},  // remote selector 4: dispatch in 2x3_DISPATCH RTL
    {.index = 6, .x = 1, .y = 2},  // remote selector 5: NOC2AXI in 2x3_DISPATCH RTL
    // DRAM table starts at index 7.
    {.index = 7, .x = 0, .y = 0},  // DRAM selector 0: bank 0
    {.index = 8, .x = 1, .y = 0},  // DRAM selector 1: bank 1
};

inline constexpr noc_att::Program PROGRAM_IMAGE{
    .masks = MASKS,
    .mask_count = sizeof(MASKS) / sizeof(MASKS[0]),
    .endpoints = ENDPOINTS,
    .endpoint_count = sizeof(ENDPOINTS) / sizeof(ENDPOINTS[0]),
    .local_endpoint_index = 0,
};

}  // namespace quasar_aether_2x3_uncapped_att_program
