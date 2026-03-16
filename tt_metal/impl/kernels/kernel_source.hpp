// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
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
        std::string name;
        if (this->source_type_ == SourceType::FILE_PATH) {
            const std::size_t start_pos_of_name = this->source_.rfind('/') + 1;
            const std::size_t pos_of_dot = this->source_.rfind('.');
            name = this->source_.substr(start_pos_of_name, (pos_of_dot - start_pos_of_name));
        } else {
            name = "Kernel_Source_Code";
        }
        return name;
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
