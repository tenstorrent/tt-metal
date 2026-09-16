// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include <tt_stl/assert.hpp>

#include "llrt/rtoptions.hpp"

namespace tt::jit_build::test {

// Use the runtime's root selection and the same SFPI search order as JitBuildEnv.
struct JitTestTools {
    std::filesystem::path root = llrt::RunTimeOptions().get_root_dir();

    std::string compiler() const {
        for (const auto& sfpi : {root / "runtime/sfpi", std::filesystem::path("/opt/tenstorrent/sfpi")}) {
            const auto gxx = sfpi / "compiler/bin/riscv-tt-elf-g++";
            if (std::filesystem::is_regular_file(gxx)) {
                return gxx.string();
            }
        }
        TT_THROW("SFPI compiler missing: checked {}/runtime/sfpi and /opt/tenstorrent/sfpi", root.string());
    }

    std::string hw_include_dir() const {
        const auto inc = root / "tt_metal/hw/inc";
        TT_FATAL(
            std::filesystem::is_regular_file(inc / "api/named_compile_time_args.h"),
            "JIT headers missing from {}",
            inc.string());
        return inc.string();
    }
};

}  // namespace tt::jit_build::test
