// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Public API for registering external configuration providers with Inspector.
// Any library (e.g. TTNN) can register a callback to expose its config at triage time.

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
};

struct ConfigurationEntry {
    std::string name;
    std::string value;
    ConfigScope scope;
};

using ConfigCallback = std::function<std::vector<ConfigurationEntry>()>;

// Register a configuration provider callback.
// Multiple callbacks can be registered — all are invoked when getConfiguration RPC is queried.
void add_config_callback(ConfigCallback callback);

}  // namespace tt::tt_metal::inspector
