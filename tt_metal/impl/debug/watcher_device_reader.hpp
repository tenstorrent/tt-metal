// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core_coord.hpp>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "debug_helpers.hpp"
#include "impl/context/metal_env_impl.hpp"

namespace tt::tt_metal {

constexpr uint64_t DEBUG_SANITIZE_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_SENTINEL_OK_16 = 0xbada;
constexpr uint8_t DEBUG_SANITIZE_SENTINEL_OK_8 = 0xda;

class WatcherDeviceReader {
public:
    WatcherDeviceReader(
        FILE* f,
        ChipId device_id,
        const std::vector<std::string>& kernel_names,
        tt::tt_metal::MetalEnvImpl& env,
        WatcherServer& watcher_server);
    ~WatcherDeviceReader();
    void Dump(FILE* file = nullptr);
    const std::vector<std::string>& get_cached_enable_symbols(HalProgrammableCoreType core_type) const {
        TT_FATAL(
            core_type < HalProgrammableCoreType::COUNT,
            "Invalid HalProgrammableCoreType {}",
            static_cast<int>(core_type));
        return symbols_info_cache_.at(core_type).symbols;
    }

private:
    class Core;
    struct DumpData;
    FILE* f;
    ChipId device_id;
    tt::tt_metal::MetalEnvImpl& env;
    WatcherServer& watcher_server;  // Reference to the parent object
    const std::vector<std::string>& kernel_names;
    std::map<CoreCoord, uint32_t> logical_core_to_eth_link_retraining_count;
    std::map<HalProgrammableCoreType, EnableSymbolsInfo> symbols_info_cache_;
};

}  // namespace tt::tt_metal
