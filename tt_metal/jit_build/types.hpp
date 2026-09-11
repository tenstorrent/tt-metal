// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::jit_build {

struct GeneratedFile {
    std::string name;
    std::vector<std::uint8_t> content;
};

struct TargetRecipe {
    std::string target_name;

    // Compile recipe.
    std::string cflags;
    std::vector<std::string> defines;
    std::string includes;
    std::string compiler_opt_level;
    std::vector<std::string> srcs;
    std::vector<std::string> objs;

    // Link recipe.
    std::string lflags;
    std::string extra_link_objs;
    std::string linker_script;
    std::string weakened_firmware_name;
    bool firmware_is_kernel_object = false;
    std::string linker_opt_level;

    // Hash of the effective build parameters (JitBuildState::build_state_hash_), used by an
    // out-of-process prewarm to write the .build_state gate file. 0 when not captured.
    std::uint64_t build_state_hash = 0;
};

}  // namespace tt::jit_build
