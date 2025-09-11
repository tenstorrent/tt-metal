// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "serialization/safetensors.hpp"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

std::string get_test_data_dir() {
    const char* env_var = std::getenv("TEST_LOCAL_DATA_DIR");
    return (env_var) ? std::string(env_var) : std::string(TEST_DATA_DIR);
}

inline std::vector<float> bytes_to_floats_copy(std::span<const std::byte> bytes) {
    if (bytes.size_bytes() % sizeof(float) != 0) {
        throw std::runtime_error("bytes_to_floats_copy: size not multiple of sizeof(float)");
    }
    const std::size_t n = bytes.size_bytes() / sizeof(float);
    std::vector<float> out(n);
    if (n) {
        std::memcpy(out.data(), bytes.data(), n * sizeof(float));
    }
    return out;
}
// Disabled test because it requires a specific test file to be present.
// It doesn't work in CI for now :(
TEST(SafeTensorsTest, DISABLED_LoadSimpleMlp) {
    const std::unordered_map<std::string, std::vector<float>> test_params = {
        // net.0.weight: shape (4, 2), row-major
        {"net.0.weight", {0.3930f, 0.8285f, 0.8702f, 0.8824f, 0.1990f, -0.8696f, 0.0920f, -0.6256f}},
        // net.0.bias: shape (4,)
        {"net.0.bias", {0.0f, 0.0f, 0.0f, 0.0f}},
        // net.2.weight: shape (4, 4), row-major
        {"net.2.weight",
         {-0.8071f,
          0.7695f,
          0.6585f,
          -0.8639f,
          0.1621f,
          -0.1459f,
          -0.1425f,
          -0.3964f,
          0.3330f,
          -0.5129f,
          0.3175f,
          0.4380f,
          0.6200f,
          0.3238f,
          -0.8571f,
          -0.5618f}},
        // net.2.bias: shape (4,)
        {"net.2.bias", {0.0f, 0.0f, 0.0f, 0.0f}},
        // net.4.weight: shape (8, 4), row-major
        {"net.4.weight",
         {0.3531f,  0.1480f,  -0.5516f, -0.4072f, 0.6652f,  0.4765f,  -0.3083f, -0.1780f, -0.6736f, -0.0127f, -0.5325f,
          -0.5454f, -0.0390f, 0.1062f,  -0.2896f, 0.4196f,  -0.4303f, 0.6416f,  0.4846f,  -0.5963f, -0.1760f, 0.0319f,
          0.1032f,  0.1677f,  0.2775f,  0.0424f,  -0.3450f, 0.3346f,  -0.6783f, -0.4191f, -0.1770f, -0.3444f}},
        // net.4.bias: shape (8,)
        {"net.4.bias", {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    };

    std::filesystem::path path = get_test_data_dir() + "/tiny_mlp.safetensors";

    std::vector<ttml::serialization::SafetensorSerialization::TensorInfo> tensors;
    std::vector<std::vector<float>> tensors_data;

    ttml::serialization::SafetensorSerialization::visit_safetensors_file(
        path,
        [&tensors, &tensors_data](
            const ttml::serialization::SafetensorSerialization::TensorInfo& info, std::span<const std::byte> bytes) {
            tensors.emplace_back(info);
            auto floats = bytes_to_floats_copy(bytes);
            tensors_data.emplace_back(std::move(floats));

            return true;  // Continue visiting
        }

    );
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& info = tensors[i];
        const auto& tensor_data = tensors_data[i];
        fmt::print("Tensor: {}, dtype: {}, shape: {}\n", info.name, info.dtype, info.shape);
        fmt::print("  {} floats\n", tensor_data);
        EXPECT_EQ(tensor_data.size(), info.shape.volume());
        EXPECT_NEAR(tensor_data.size(), test_params.at(info.name).size(), 1e-6);
    }

    ASSERT_EQ(tensors.size(), 6);  // Assuming the MLP has 3 tensors
    EXPECT_EQ(tensors[0].name, "net.0.bias");
    EXPECT_EQ(tensors[1].name, "net.0.weight");
    EXPECT_EQ(tensors[2].name, "net.2.bias");
    EXPECT_EQ(tensors[3].name, "net.2.weight");
    EXPECT_EQ(tensors[4].name, "net.4.bias");
    EXPECT_EQ(tensors[5].name, "net.4.weight");
}
