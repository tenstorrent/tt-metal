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

namespace tt::tt_metal {

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;
constexpr uint8_t DEBUG_SANITIZE_NOC_SENTINEL_OK_8 = 0xda;

class WatcherDeviceReader {
public:
    WatcherDeviceReader(FILE* f, chip_id_t device_id, const std::vector<std::string>& kernel_names);
    ~WatcherDeviceReader();
    void Dump(FILE* file = nullptr);

private:
    class Core;
    struct DumpData;
    FILE* f;
    chip_id_t device_id;
    uint32_t num_erisc_cores{0};
    const std::vector<std::string>& kernel_names;
    std::map<CoreCoord, uint32_t> logical_core_to_eth_link_retraining_count;
};

}  // namespace tt::tt_metal
