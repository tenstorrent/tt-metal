// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "types.hpp"

namespace tt::jit_build {

// Relative to the tt-metal root; installed alongside the other JIT inputs.
inline constexpr std::string_view PCH_UMBRELLA = "tt_metal/hw/inc/internal/pch.h";

// Throw if the umbrella or compiler version cannot be read; otherwise return the cached header
// path or empty if PCH setup fails.
// The STL-only umbrella uses the compiler's default include paths.
// Match the consuming compile's flags and optimization level for GCC PCH compatibility.
std::string ensure_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::filesystem::path& umbrella,
    const std::filesystem::path& pch_root);

// Returns the header of a PCH that includes headers in order (the standard-library umbrella, the
// processor's kernel prefix header, then the kernel's own precompiled header), or empty if it cannot be
// built. generated maps the names of generated per-kernel headers the prefix includes to the files
// staged under those names, such as CB_TABLE_DECLARATIONS_HEADER as chlkc_descriptors.h.
// The PCH is shared by every compile with the same flags, includes, defines, headers and generated
// contents. The second request in a process starts building it in the background; that compile and
// every compile until the PCH is ready get an empty header and parse the headers. Compile with
// -fpch-deps, so a kernel's dependency hash covers the headers in the PCH.
std::string ensure_kernel_pch(
    const std::string& gpp,
    const std::string& opt_level,
    const std::string& cflags,
    const std::string& includes,
    const std::vector<std::string>& defines,
    const std::vector<std::filesystem::path>& headers,
    const std::map<std::string, std::filesystem::path>& generated,
    const std::filesystem::path& pch_root);

// Waits for the kernel PCHs being built (ensure_kernel_pch) to be ready.
void wait_for_kernel_pch_builds();

// How a compile loads its PCH: force-includes to put first among its defines, and flags to append.
struct PchUse {
    std::vector<std::string> includes;
    std::string cflags;
};

// The PCH for a compile from this recipe, whose generated per-kernel files are in kernel_dir: the
// kernel's own PCH when the recipe names its headers (ensure_kernel_pch), otherwise the
// standard-library one. A kernel PCH whose headers or generated files are missing here (such as on
// a compile server on another host) is skipped, never an error. Empty when no PCH can be built.
PchUse select_pch(
    const std::string& gpp,
    const TargetRecipe& recipe,
    const std::filesystem::path& kernel_dir,
    const std::filesystem::path& pch_root);

}  // namespace tt::jit_build
