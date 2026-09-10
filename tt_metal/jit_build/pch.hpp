// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace tt::jit_build {

// Internal JIT cache helpers. An empty result asks the caller to compile without a PCH.
// Defines must match the consumer's command line, with per-kernel -include pairs removed.
std::string ensure_pch(
    const std::string& gpp,
    const std::string& root,
    const std::string& out_root,
    const std::string& target_name,
    std::string_view umbrella_rel,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::vector<std::string>& defines);

void merge_pch_deps_into_kernel_d(
    const std::string& kernel_d_path, const std::string& kernel_obj, const std::string& pch_d_path);

void pch_cache_clear();

// Strict validation: require GCC's -H acceptance line for this exact header.
void require_pch_consumed(const std::string& log_path, const std::string& header);

}  // namespace tt::jit_build
