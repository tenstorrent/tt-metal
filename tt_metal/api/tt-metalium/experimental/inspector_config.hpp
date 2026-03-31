// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Public API for registering external configuration providers with Inspector.
// TTNN uses this to register its config at library load time.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tt::tt_metal::inspector {

// Mirrors the ConfigurationScope enum in rpc.capnp.
enum class ConfigScope : uint8_t {
    Environment = 0,
    RtOptions = 1,
    TtnnConfig = 2,
    Unknown = 3,
};

struct ConfigurationEntry {
    std::string name;
    std::string value;
    ConfigScope scope;
};

using ConfigCallback = std::function<std::vector<ConfigurationEntry>()>;

// The storage for this callback is defined out-of-line in data.cpp
// (within the Inspector/Metalium library) to ensure all DSOs share the same instance.
ConfigCallback& ttnn_config_callback();

}  // namespace tt::tt_metal::inspector
