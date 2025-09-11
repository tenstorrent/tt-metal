// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
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

    using TensorCallback = std::function<bool(const TensorInfo& info, std::span<const std::byte> bytes)>;

    static void visit_safetensors_file(const std::filesystem::path& path, const TensorCallback& cb);

    /*
        Span can point to the unaligned memory, so it is not safe to copy it to the float buffer before using it.
    */
    static std::vector<float> bytes_to_floats_copy(std::span<const std::byte> bytes);
};
}  // namespace ttml::serialization
