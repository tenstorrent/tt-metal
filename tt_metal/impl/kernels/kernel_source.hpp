// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
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
