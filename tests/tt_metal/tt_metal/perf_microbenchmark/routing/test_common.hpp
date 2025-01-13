// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include "routing/kernels/traffic_gen_test.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include <unordered_map>
#include <string>

static inline std::string to_string(pkt_dest_size_choices_t choice) {
    switch (choice) {
        case pkt_dest_size_choices_t::RANDOM: return "RANDOM";
        case pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE: return "RND_ROBIN_FIX";
        default: return "unexpected";
    }
}

static inline void log_phys_coord_to_json(nlohmann::json& config, const std::vector<CoreCoord>& phys_cores, const std::string& name) {
    for (int i = 0; i < phys_cores.size(); ++i) {
        config[fmt::format("{}_{}", name, i)] = fmt::format("({}, {})", phys_cores[i].x, phys_cores[i].y);
    }
}

static inline void log_phys_coord_to_json(nlohmann::json& config, const CoreCoord& phys_core, const std::string& name) {
    config[name] = fmt::format("({}, {})", phys_core.x, phys_core.y);
}

// Make buffer address for test. The address are created continguously in order of the stage_config
static inline std::map<std::string, std::vector<uint32_t>> make_buffer_addresses_for_test(uint32_t base_address, uint32_t per_buffer_size, const std::vector<std::pair<std::string, uint32_t>>& stage_config) {
    // Use a vector to make the addresses to keep everything continiguous
    std::vector<std::vector<uint32_t>> scratch_buffers;

    for (int stage = 0; stage < stage_config.size(); stage++) {
        assert(stage_config[stage].second > 0 && "Each stage must consist of at least 1 kernel");

        std::vector<uint32_t> addresses_for_stage;
        if (stage == 0) {
            addresses_for_stage.push_back(base_address);
        } else {
            addresses_for_stage.push_back(scratch_buffers.back().back() + per_buffer_size);
        }

        // Start at kernel 1 as the address for kernel 0 is already in the addresses_for_stage
        for (int kernel = 1; kernel < stage_config[stage].second; kernel++) {
            addresses_for_stage.push_back(addresses_for_stage.back() + per_buffer_size);
        }

        scratch_buffers.push_back(std::move(addresses_for_stage));
    }

    // Convert the vector to a map indexed by the stage name
    std::map<std::string, std::vector<uint32_t>> scratch_buffers_by_stage;
    for (int stage = 0; stage < stage_config.size(); stage++) {
        scratch_buffers_by_stage[stage_config[stage].first] = std::move(scratch_buffers[stage]);
    }

    return scratch_buffers_by_stage;
}
