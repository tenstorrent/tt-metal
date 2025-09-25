// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include <umd/device/types/cluster_descriptor_types.h>

class ProfilerOptionalMetadata {
    using RuntimeID = uint32_t;

public:
    ProfilerOptionalMetadata(std::map<std::pair<chip_id_t, RuntimeID>, std::string>&& runtime_map) :
        runtime_id_to_opname_(std::move(runtime_map)) {}

    const std::string& get_op_name(chip_id_t device_id, RuntimeID runtime_id) const {
        static const std::string empty_string;
        auto key = std::make_pair(device_id, runtime_id);
        auto it = runtime_id_to_opname_.find(key);
        if (it != runtime_id_to_opname_.end()) {
            return it->second;
        }
        return empty_string;
    }

private:
    std::map<std::pair<chip_id_t, RuntimeID>, std::string> runtime_id_to_opname_;
};
