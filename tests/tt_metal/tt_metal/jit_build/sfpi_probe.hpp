// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdlib.h>
#include <unistd.h>

#include <cerrno>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "jit_build/jit_build_utils.hpp"

namespace tt::tt_metal::jit_build_test {

// The sfpi g++ the JIT build uses (<root_dir>runtime/sfpi, then system sfpi); empty when neither exists.
inline std::string sfpi_gxx_path(const std::string& root_dir) {
    for (const std::string& sfpi_root : {root_dir + "runtime/sfpi", std::string("/opt/tenstorrent/sfpi")}) {
        auto gxx = sfpi_root + "/compiler/bin/riscv-tt-elf-g++";
        if (std::filesystem::exists(gxx)) {
            return gxx;
        }
    }
    return {};
}

// True when the sfpi g++ exists and exits 0 on `flags`. Runs through the JIT build's shell-free launcher
// (posix_spawn with a fixed argv) in a scratch directory that is removed afterwards. Throws when the
// scratch directory cannot be made, so a broken TMPDIR fails the caller instead of reading as a toolchain
// that rejects the flags (callers skip on false).
inline bool sfpi_gxx_accepts(const std::string& root_dir, const std::vector<std::string>& flags) {
    const std::string gxx = sfpi_gxx_path(root_dir);
    if (gxx.empty()) {
        return false;
    }
    auto pattern = (std::filesystem::temp_directory_path() / "tt_metal_sfpi_probe_XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) {
        throw std::system_error(errno, std::generic_category(), "mkdtemp " + pattern);
    }
    const std::filesystem::path scratch = pattern;
    std::vector<std::string> args = {gxx};
    args.insert(args.end(), flags.begin(), flags.end());
    const bool ok = tt::jit_build::utils::exec_command(args, scratch.string(), (scratch / "probe.log").string());
    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
    return ok;
}

}  // namespace tt::tt_metal::jit_build_test
