// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "serialization/msgpack_file.hpp"
#include "serialization/serialization.hpp"

class TensorFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Remove test file if it exists
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }

        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        // Clean up test file after each test
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
        ttml::autograd::ctx().close_device();
    }

    const std::string test_filename = "/tmp/test_tensor.msgpack";
};

TEST_F(TensorFileTest, SerializeDeserializeTensor) {
    ttml::serialization::MsgPackFile serializer;
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttml::core::create_shape({1, 2, 32, 321});
    auto tensor_zeros = ttml::core::zeros(shape, device);
    auto tensor_ones = ttml::core::ones(shape, device);

    // Write tensor to file
    ttml::serialization::write_ttnn_tensor(serializer, "tensor", tensor_ones);
    serializer.serialize(test_filename);
    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(test_filename);

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
    ttml::serialization::MsgPackFile serializer;
    auto* device = &ttml::autograd::ctx().get_device();
    auto model_params = ttml::modules::MultiLayerPerceptronParameters{
        .input_features = 128, .hidden_features = {256}, .output_features = 10};
    ttml::modules::MultiLayerPerceptron mlp_to_write(model_params);
    ttml::modules::MultiLayerPerceptron mlp_to_read(model_params);
    // Write tensor to file
    auto params_to_write = mlp_to_write.parameters();
    ttml::serialization::write_named_parameters(serializer, "mlp", params_to_write);
    serializer.serialize(test_filename);
    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(test_filename);
    auto params_to_read = mlp_to_read.parameters();
    ttml::serialization::read_named_parameters(deserializer, "mlp", params_to_read);

    EXPECT_EQ(params_to_read.size(), params_to_write.size());
    for (const auto& [key, value] : params_to_read) {
        EXPECT_TRUE(compare_tensors(value->get_value(), params_to_write.at(key)->get_value()));
    }
}
