// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Public API for registering external configuration providers with Inspector.
// TTNN uses this to register its config at library load time.

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace tt::tt_metal::inspector {

// Scope constants — used by producers and the serializer to avoid string typos.
inline constexpr const char* kScopeEnvironment = "Environment";
inline constexpr const char* kScopeRtOptions = "RtOptions";
inline constexpr const char* kScopeTtnnConfig = "TtnnConfig";

struct ConfigurationEntry {
    std::string name;
    std::string value;
    std::string scope;
};

using ConfigCallback = std::function<std::vector<ConfigurationEntry>()>;

// The storage for this callback is defined out-of-line in data.cpp
// (within the Inspector/Metalium library) to ensure all DSOs share the same instance.
ConfigCallback& ttnn_config_callback();

}  // namespace tt::tt_metal::inspector
