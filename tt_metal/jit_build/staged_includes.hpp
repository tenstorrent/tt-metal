// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace tt::jit_build {

// observability/api, the headers device code shares with the host, relative to the tt-metal root and installed
// alongside the other JIT inputs. The host owns them; the device may replace any of them with a header of the same
// name, or add its own.
inline constexpr std::string_view OBSERVABILITY_API_HOST_DIR = "tt_metal/observability/api";
inline constexpr std::string_view OBSERVABILITY_API_DEVICE_DIR = "tt_metal/hw/observability/api";

// Device code never includes from the host tree, so it does not see either folder. It sees a copy of both, merged
// with the device's headers taking precedence, staged under |cache_root|/include/<hash>/observability/api. Return the
// include root to pass with -I. Throw if nothing can be staged.
std::string stage_observability_api(const std::filesystem::path& root, const std::filesystem::path& cache_root);

}  // namespace tt::jit_build
