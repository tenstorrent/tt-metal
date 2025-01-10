// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/llrt/llrt.hpp"

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
