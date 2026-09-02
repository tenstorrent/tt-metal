// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <impl/context/context_types.hpp>
#include <tt_stl/unreachable.hpp>

namespace tt::tt_metal {

struct KernelSource {
    enum SourceType { FILE_PATH, SOURCE_CODE };

    std::string source_;
    SourceType source_type_;
    // if source_type_ is FILE_PATH, file pointed by path_ exists at time of construction
    std::filesystem::path path_;

    // Resolve path using rtoptions from context_id and construct a FILE_PATH KernelSource.
    static KernelSource from_path(ContextId context_id, const std::filesystem::path& path);

    // Construct a SOURCE_CODE KernelSource from inline source.
    static KernelSource from_source(const std::string& source_code);

    std::string name() const {
        if (this->source_type_ == SourceType::FILE_PATH) {
            return this->path_.stem().string();
        }
        return "Kernel_Source_Code";
    }

    // Stable, variant-independent identity of this source, for the profiler zone tu-id registry.
    // The full PATH, not name(): name() is only the stem, and stems recur across ops
    // (reader_unary.cpp, writer_unary.cpp, ...), which would fuse unrelated TUs onto one id.
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

private:
    KernelSource(std::string source, SourceType source_type, std::filesystem::path path);
};

}  // namespace tt::tt_metal
