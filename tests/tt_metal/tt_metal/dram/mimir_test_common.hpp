// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llrt/tt_cluster.hpp"
#include "tt_metal/tt_metal/context/mimir_contract.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal::mimir_test {

using mimir_contract::base_dram_view_for_mimir;
using mimir_contract::cce_index_for;
using mimir_contract::cce_param_name;
using mimir_contract::configured_soc_desc;
using mimir_contract::dram_views_for_channel;
using mimir_contract::local_dram_view_for_cce;
using mimir_contract::location_for_cce;
using mimir_contract::locations_per_channel;
using mimir_contract::logical_cce_core;
using mimir_contract::mimir_index_for_cce;
using mimir_contract::num_cces_from_soc;
using mimir_contract::soc_desc_of;
using mimir_contract::translated_dram_core;

// The status line a CCE test kernel writes once at exit at tl1_done_block(hart), with done_magic in word 0. The
// runners in mimir_dm_common.hpp fill its num_words words with poison before the launch.
struct CceStatusLine {
    uint32_t done_magic = 0;
    uint32_t num_words = 0;
    uint32_t poison = 0;
};

// The next CCE on the same Mimir; the CCE itself when the descriptor has one CCE per Mimir.
inline uint32_t sibling_cce(uint32_t cce_index) {
    return cce_index_for(
        mimir_index_for_cce(cce_index),
        (location_for_cce(cce_index) + 1) % locations_per_channel(configured_soc_desc()));
}

// The next Mimir, wrapping by the descriptor's Mimir count; the Mimir itself when the descriptor has one.
inline uint32_t next_mimir(uint32_t mimir_index) {
    return (mimir_index + 1) % static_cast<uint32_t>(configured_soc_desc().get_num_dram_channels());
}

// The CCE at the same location on the next Mimir.
inline uint32_t next_mimir_cce(uint32_t cce_index) {
    return cce_index_for(next_mimir(mimir_index_for_cce(cce_index)), location_for_cce(cce_index));
}

inline uint32_t mimir_of_view(const metal_SocDescriptor& soc_desc, uint32_t view) {
    return static_cast<uint32_t>(soc_desc.get_channel_for_dram_view(static_cast<int>(view)));
}

// The runtime's logical-to-virtual mapping, as IDevice::virtual_core_from_logical_core uses it.
inline CoreCoord cce_virtual_core(const Cluster& cluster, ChipId chip, uint32_t cce_index) {
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        chip, logical_cce_core(cluster.get_soc_desc(chip), cce_index), CoreType::DRAM);
}

}  // namespace tt::tt_metal::mimir_test
