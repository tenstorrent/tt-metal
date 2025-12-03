// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "serialization/flatbuffer_file.hpp"
#include "serialization/serialization.hpp"

namespace {
std::string generate_unique_temp_dir_name() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    constexpr int name_length = 16;
    std::string name = "tensor_test_";

    for (int i = 0; i < name_length; ++i) {
        name += "0123456789abcdef"[dis(gen)];
    }

    return name;
}

std::filesystem::path create_unique_temp_dir() {
    std::filesystem::path base_dir = std::filesystem::temp_directory_path();

    size_t max_attempts = 1024;
    while (--max_attempts > 0) {
        std::string random_name = generate_unique_temp_dir_name();
        std::filesystem::path temp_dir = base_dir / random_name;

        if (!std::filesystem::exists(temp_dir)) {
            std::filesystem::create_directories(temp_dir);
            return temp_dir;
        }
    }

    throw std::runtime_error("Failed to create unique temporary directory after maximum attempts");
}
}  // namespace

class TensorFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        temp_dir = create_unique_temp_dir();
        test_filename = temp_dir.string();  // Use directory path

        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();

        // Clean up temp directory after each test
        if (std::filesystem::exists(temp_dir)) {
            std::filesystem::remove_all(temp_dir);
        }
    }

    std::filesystem::path temp_dir;
    std::string test_filename;
};

TEST_F(TensorFileTest, SerializeDeserializeTensor) {
    ttml::serialization::FlatBufferFile serializer;
    // Set output directory before writing tensors
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttnn::Shape({1, 2, 32, 321});
    auto tensor_zeros = ttml::core::zeros(shape, device);
    auto tensor_ones = ttml::core::ones(shape, device);

    // Write tensor to file
    ttml::serialization::write_ttnn_tensor(serializer, "tensor", tensor_ones);
    // Use directory path for serialization
    std::filesystem::path output_dir = temp_dir / "model_data";
    serializer.serialize(output_dir.string());

    // Verify metadata file exists
    std::filesystem::path metadata_file = output_dir / "metadata.flatbuffer";
    ASSERT_TRUE(std::filesystem::exists(metadata_file)) << "Metadata file should exist: " << metadata_file;

    // Verify tensor file was created
    std::filesystem::path tensor_file = output_dir / "tensor.tensorbin";
    ASSERT_TRUE(std::filesystem::exists(tensor_file)) << "Tensor file should exist: " << tensor_file;

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(output_dir.string());

    // Read tensor from file
    tt::tt_metal::Tensor tensor_read = tensor_zeros;
    ttml::serialization::read_ttnn_tensor(deserializer, "tensor", tensor_read);

    auto read_vec = ttml::core::to_vector(tensor_read);

    for (auto& val : read_vec) {
        EXPECT_EQ(val, 1.F);
    }
}

bool compare_tensors(const tt::tt_metal::Tensor& tensor1, const tt::tt_metal::Tensor& tensor2) {
    auto vec1 = ttml::core::to_vector(tensor1);
    auto vec2 = ttml::core::to_vector(tensor2);
    return vec1 == vec2;
}

TEST_F(TensorFileTest, SerializeDeserializeNamedParameters) {
    ttml::serialization::FlatBufferFile serializer;
    // Set output directory before writing tensors
    auto model_params = ttml::modules::MultiLayerPerceptronParameters{
        .input_features = 128, .hidden_features = {256}, .output_features = 10};
    ttml::modules::MultiLayerPerceptron mlp_to_write(model_params);
    ttml::modules::MultiLayerPerceptron mlp_to_read(model_params);
    // Write tensor to file
    auto params_to_write = mlp_to_write.parameters();
    ttml::serialization::write_named_parameters(serializer, "mlp", params_to_write);
    serializer.serialize(test_filename);
    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);
    auto params_to_read = mlp_to_read.parameters();
    ttml::serialization::read_named_parameters(deserializer, "mlp", params_to_read);

    EXPECT_EQ(params_to_read.size(), params_to_write.size());
    for (const auto& [key, value] : params_to_read) {
        EXPECT_TRUE(compare_tensors(value->get_value(), params_to_write.at(key)->get_value()));
    }
}
