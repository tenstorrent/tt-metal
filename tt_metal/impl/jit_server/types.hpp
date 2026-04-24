// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "jit_build/types.hpp"

namespace tt::tt_metal::jit_server {

// Convenience aliases so the RPC types below (and serialization code) can refer
// to these build-layer DTOs without full qualification.
using tt::jit_build::GeneratedFile;
using tt::jit_build::TargetRecipe;

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

struct FirmwareArtifact {
    std::string target_name;
    std::string file_name;
    bool is_kernel_object = false;
    std::vector<std::uint8_t> data;
};

struct UploadFirmwareRequest {
    std::uint64_t build_key = 0;
    std::vector<FirmwareArtifact> artifacts;
};

struct UploadFirmwareResponse {
    bool success = false;
    std::string error_message;
};

struct CompileResponse {
    bool success = false;
    std::string error_message;
    std::vector<ElfBlob> elf_blobs;
};

}  // namespace tt::tt_metal::jit_server
