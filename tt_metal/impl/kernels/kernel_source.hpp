// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

#include <tt_stl/unreachable.hpp>

namespace tt::tt_metal {

struct KernelSource {
    enum SourceType { FILE_PATH, SOURCE_CODE };

    std::string source_;
    SourceType source_type_;
    // if source_type_ is FILE_PATH, file pointed by path_ exists at time of construction
    std::filesystem::path path_;

    KernelSource(const std::string& source, const SourceType& source_type);

    std::string name() const {
        if (this->source_type_ == SourceType::FILE_PATH) {
            return this->path_.stem().string();
        }
        return "Kernel_Source_Code";
    }

    // Build-argument-independent identifier that is unique per distinct source: the full file path for
    // file kernels, a content hash for inline source. Unlike name() (just the file stem, which recurs
    // across ops), this uniquely identifies the source, so the profiler can assign one zone file-id per
    // source instead of one per compile variant. Contains no tab/newline, so it is registry-line safe.
    std::string profiler_zone_src_id() const {
        if (this->source_type_ == SourceType::FILE_PATH) {
            return this->path_.string();
        }
        return "inline:" + std::to_string(std::hash<std::string>{}(this->source_));
    }

    // Returns the actual source code (file content or source string)
    std::string get_content() const {
        switch (source_type_) {
            case SourceType::FILE_PATH: {
                std::ifstream file(path_);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open kernel source file: " + path_.string());
                }
                std::stringstream buffer;
                buffer << file.rdbuf();
                if (file.fail() && !file.eof()) {
                    throw std::runtime_error("Failed to read kernel source file: " + path_.string());
                }
                return buffer.str();
            }
            case SourceType::SOURCE_CODE: return source_;
        }
        ttsl::unreachable();
    }
};

}  // namespace tt::tt_metal
