// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar emulator models that back less DRAM than their soc descriptor declares, with the DRAM they
// really have per channel. tt-metal clamps the descriptor's DRAM size to this when the TT_METAL_SIMULATOR
// directory has one of these names. Models that are not listed keep their descriptor's value.
// TT_METAL_SIMULATOR_DRAM_UNCAPPED=1 makes Cluster::get_metal_desc_from_tt_desc skip the clamp.

#include <cstdint>
#include <string_view>

namespace tt::tt_metal::quasar_emu {

struct ModelInfo {
    std::string_view simulator_name;      // TT_METAL_SIMULATOR directory name, matched exactly
    std::uint64_t dram_size_per_channel;  // the DRAM the model backs per channel
};

inline constexpr std::uint64_t AETHER_DRAM_TILE_SIZE = std::uint64_t{64} << 20;  // 64 MiB

inline constexpr ModelInfo MODELS[] = {
    {"emu-quasar-1x3", AETHER_DRAM_TILE_SIZE},
    {"emu-quasar-2x3", AETHER_DRAM_TILE_SIZE},
    {"emu-quasar-2x3_DISPATCH", AETHER_DRAM_TILE_SIZE},
};

// The row whose simulator_name equals the given name exactly, or nullptr.
constexpr const ModelInfo* find_model(std::string_view simulator_name) {
    for (const ModelInfo& model : MODELS) {
        if (model.simulator_name == simulator_name) {
            return &model;
        }
    }
    return nullptr;
}

}  // namespace tt::tt_metal::quasar_emu
