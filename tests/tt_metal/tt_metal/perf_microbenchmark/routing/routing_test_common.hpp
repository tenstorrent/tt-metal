// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "llrt.hpp"

static inline std::string to_string(pkt_dest_size_choices_t choice) {
    switch (choice) {
        case pkt_dest_size_choices_t::RANDOM: return "RANDOM";
        case pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE: return "RND_ROBIN_FIX";
        default: return "unexpected";
    }
}

static inline void log_phys_coord_to_json(
    nlohmann::json& config, const std::vector<CoreCoord>& phys_cores, const std::string& name) {
    for (int i = 0; i < phys_cores.size(); ++i) {
        config[fmt::format("{}_{}", name, i)] = fmt::format("({}, {})", phys_cores[i].x, phys_cores[i].y);
    }
}

static inline void log_phys_coord_to_json(nlohmann::json& config, const CoreCoord& phys_core, const std::string& name) {
    config[name] = fmt::format("({}, {})", phys_core.x, phys_core.y);
}

inline uint64_t get_64b_result(uint32_t* buf, uint32_t index) {
    return (((uint64_t)buf[index]) << 32) | buf[index + 1];
}

inline uint64_t get_64b_result(const std::vector<uint32_t>& vec, uint32_t index) {
    return (((uint64_t)vec[index]) << 32) | vec[index + 1];
}

inline tt::tt_metal::CoreCoord get_active_ethernet_core(tt::tt_metal::IDevice* device) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& device_active_eth_cores = device->get_active_ethernet_cores();
    auto eth_core_iter = device_active_eth_cores.begin();
    for (; eth_core_iter != device_active_eth_cores.end(); eth_core_iter++) {
        if (cluster.is_ethernet_link_up(device->id(), *eth_core_iter)) {
            break;
        }
    }
    if (eth_core_iter == device_active_eth_cores.end()) {
        log_info(
            tt::LogTest,
            "No active ethernet link found on device {}. Need 1 active ethernet link for this test.",
            device->id());
        tt::tt_metal::CloseDevice(device);
        throw std::runtime_error("Test cannot run on specified device.");
    }

    return *eth_core_iter;
}
