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

struct ConfigurationEntry {
    std::string name;
    std::string value;
    std::string scope;
};

using ConfigCallback = std::function<std::vector<ConfigurationEntry>()>;

// TTNN sets this at library load time via a file-scope static initializer in config.cpp.
// Inspector calls it (if set) when getConfiguration RPC is invoked.
inline ConfigCallback& ttnn_config_callback() {
    static ConfigCallback callback;
    return callback;
}

}  // namespace tt::tt_metal::inspector
