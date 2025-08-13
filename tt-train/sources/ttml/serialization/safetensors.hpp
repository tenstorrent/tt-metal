// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>
#include <filesystem>
#include <functional>
#include <span>
#include <string>

namespace ttml::serialization {

class SafetensorSerialization {
public:
    struct TensorInfo {
        std::string name;
        std::string dtype;  // "F16","BF16","F32","I64",...
        ttnn::Shape shape;  // dims
    };

    // Return false from callback to stop early.
    using TensorCallback = std::function<bool(const TensorInfo& info, std::span<const std::byte> bytes)>;

    static void visit_safetensors_file(const std::filesystem::path& path, const TensorCallback& cb);
};
}  // namespace ttml::serialization
