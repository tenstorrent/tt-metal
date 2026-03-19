// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::tt_metal::jit_server {

struct GeneratedFile {
    std::string name;
    std::vector<std::uint8_t> content;
};

struct TargetRecipe {
    std::string target_name;

    // Compile recipe.
    std::string cflags;
    std::string defines;
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
};

struct CompileRequest {
    std::uint64_t build_key = 0;
    std::string kernel_name;
    std::string gpp;
    std::vector<TargetRecipe> targets;
    std::vector<GeneratedFile> generated_files;
};

struct ElfBlob {
    std::string name;
    std::vector<std::uint8_t> data;
};

struct CompileResponse {
    bool success = false;
    std::string error_message;
    std::vector<ElfBlob> elf_blobs;
};

}  // namespace tt::tt_metal::jit_server
