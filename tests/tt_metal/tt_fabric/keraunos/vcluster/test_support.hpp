// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What the virtual-cluster tests share.

#pragma once

#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace tt::tt_fabric::vcluster::test {

// The package descriptor, UMD descriptors/soc_descs/mmk.yaml, located from this source file so a test runs from any
// working directory. TT_VCLUSTER_PACKAGE_SOC_DESC overrides it, for a tree whose UMD checkout is elsewhere.
inline std::filesystem::path package_soc_desc() {
    if (const char* path = std::getenv("TT_VCLUSTER_PACKAGE_SOC_DESC"); path != nullptr && *path != '\0') {
        return path;
    }
    // This file is five directories below the tree's root.
    std::filesystem::path root = std::filesystem::path(__FILE__).parent_path();
    for (int up = 0; up < 5; ++up) {
        root = root.parent_path();
    }
    return root / "tt_metal/third_party/umd/descriptors/soc_descs/mmk.yaml";
}

// `text` with each `from` replaced by its `to`. Throws std::logic_error unless each `from` occurs exactly once, so an
// edit cannot land elsewhere.
inline std::string edited(
    std::string text, std::initializer_list<std::pair<std::string_view, std::string_view>> edits) {
    for (const auto& [from, to] : edits) {
        const size_t at = text.find(from);
        if (at == std::string::npos || text.find(from, at + 1) != std::string::npos) {
            throw std::logic_error("edited: '" + std::string(from) + "' is not in the text exactly once");
        }
        text.replace(at, from.size(), to);
    }
    return text;
}

}  // namespace tt::tt_fabric::vcluster::test
