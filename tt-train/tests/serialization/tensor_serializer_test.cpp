// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "serialization/flatbuffer_file.hpp"
#include "serialization/serialization.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/tensor/tensor.hpp"

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
    ttnn::Tensor tensor_read = tensor_zeros;
    ttml::serialization::read_ttnn_tensor(deserializer, "tensor", tensor_read);

    auto read_vec = ttml::core::to_vector(tensor_read);

    for (auto& val : read_vec) {
        EXPECT_EQ(val, 1.F);
    }
}

bool compare_tensors(const ttnn::Tensor& tensor1, const ttnn::Tensor& tensor2) {
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

class CheckpointTrainingTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void SetUp() override {
        temp_dir = create_unique_temp_dir();
        ttml::autograd::ctx().set_seed(123U);
    }
    void TearDown() override {
        std::filesystem::remove_all(temp_dir);
    }

    static ttml::optimizers::AdamWConfig adamw_config() {
        ttml::optimizers::AdamWConfig config;
        config.lr = 1e-2F;
        return config;
    }

    static void step_with_random_grads(
        const ttml::serialization::NamedParameters& params, ttml::optimizers::OptimizerBase& optimizer, int steps) {
        auto& gen = ttml::autograd::ctx().get_generator();
        auto* device = &ttml::autograd::ctx().get_device();
        for (int step = 0; step < steps; ++step) {
            for (const auto& [name, param] : params) {
                const auto& shape = param->get_shape();
                const std::vector<std::size_t> dims(shape.cbegin(), shape.cend());
                auto grad = ttml::test_utils::make_uniform_xarray<float>(dims, -1.0F, 1.0F, gen());
                param->set_grad(ttml::core::from_xtensor(grad, device));
            }
            optimizer.step();
        }
    }

    std::filesystem::path save(
        const std::string& tag,
        const ttml::modules::ModuleBase& model,
        const ttml::optimizers::OptimizerBase& optimizer) {
        ttml::serialization::FlatBufferFile file;
        ttml::serialization::write_module(file, "model", &model);
        ttml::serialization::write_optimizer(file, "optimizer", &optimizer);
        auto path = temp_dir / tag;
        file.serialize(path.string());
        return path;
    }

    // The checkpoint must hold exactly the tensor the training step produced: the same dtype and the same values.
    static void expect_saved(
        ttml::serialization::FlatBufferFile& file, const std::string& key, const ttnn::Tensor& trained) {
        ttnn::Tensor saved;
        ttml::serialization::read_ttnn_tensor(file, key + "/value", saved);
        EXPECT_EQ(saved.dtype(), trained.dtype()) << key;
        EXPECT_TRUE(compare_tensors(saved, trained)) << key << " in the checkpoint does not match the trained value";
    }

    // Checks every model weight against its bf16 HALF view, and every tensor in the optimizer state entries
    // `state_keys` against its `state_precision` view. The defaults are AdamW's moments, which the fused kernel
    // updates in place in bf16; AdamWFullPrecision passes its fp32 master weights and moments with FULL.
    static void expect_checkpoint_matches(
        const std::filesystem::path& path,
        const ttml::serialization::NamedParameters& params,
        const ttml::optimizers::OptimizerBase& optimizer,
        const std::vector<std::string>& state_keys = {"exp_avg", "exp_avg_sq"},
        ttml::autograd::PreferredPrecision state_precision = ttml::autograd::PreferredPrecision::HALF) {
        ttml::serialization::FlatBufferFile file;
        file.deserialize(path.string());
        for (const auto& [name, param] : params) {
            expect_saved(file, "model/" + name, param->get_value(ttml::autograd::PreferredPrecision::HALF));
        }
        const auto state = optimizer.get_state_dict();
        for (const auto& key : state_keys) {
            const auto& tensors = std::get<ttml::serialization::NamedParameters>(state.at(key));
            ASSERT_FALSE(tensors.empty()) << key;
            for (const auto& [name, value] : tensors) {
                expect_saved(file, "optimizer/" + key + "/" + name, value->get_value(state_precision));
            }
        }
    }

    std::filesystem::path temp_dir;
};

TEST_F(CheckpointTrainingTest, CheckpointMatchesTrainedValuesOnFirstSave) {
    ttml::modules::LinearLayer model(32, 64);
    auto params = model.parameters();
    ttml::optimizers::AdamW optimizer(params, adamw_config());

    step_with_random_grads(params, optimizer, 3);
    expect_checkpoint_matches(save("only", model, optimizer), params, optimizer);
}

TEST_F(CheckpointTrainingTest, LaterCheckpointMatchesTrainedValues) {
    ttml::modules::LinearLayer model(32, 64);
    auto params = model.parameters();
    ttml::optimizers::AdamW optimizer(params, adamw_config());

    step_with_random_grads(params, optimizer, 1);
    save("first", model, optimizer);
    step_with_random_grads(params, optimizer, 3);
    expect_checkpoint_matches(save("second", model, optimizer), params, optimizer);
}

TEST_F(CheckpointTrainingTest, CheckpointMatchesTrainedValuesAfterResume) {
    ttml::modules::LinearLayer model(32, 64);
    auto params = model.parameters();
    ttml::optimizers::AdamW optimizer(params, adamw_config());
    step_with_random_grads(params, optimizer, 1);
    const auto first = save("first", model, optimizer);

    ttml::modules::LinearLayer resumed_model(32, 64);
    auto resumed_params = resumed_model.parameters();
    ttml::optimizers::AdamW resumed_optimizer(resumed_params, adamw_config());
    ttml::serialization::FlatBufferFile file;
    file.deserialize(first.string());
    ttml::serialization::read_module(file, "model", &resumed_model);
    ttml::serialization::read_optimizer(file, "optimizer", &resumed_optimizer);
    // The resumed model and optimizer hold what the checkpoint stored, before any further steps.
    expect_checkpoint_matches(first, resumed_params, resumed_optimizer);

    step_with_random_grads(resumed_params, resumed_optimizer, 3);
    expect_checkpoint_matches(save("resumed", resumed_model, resumed_optimizer), resumed_params, resumed_optimizer);
}

// AdamWFullPrecision keeps its master weights and moments as fp32-native tensors. The checkpoint must store
// them in fp32, bit for bit, not rounded down to the bf16 view, and the model weights as their bf16 copies.
TEST_F(CheckpointTrainingTest, CheckpointKeepsFullPrecisionOptimizerState) {
    ttml::modules::LinearLayer model(32, 64);
    auto params = model.parameters();
    ttml::optimizers::AdamWFullPrecisionConfig config;
    config.lr = 1e-2F;
    ttml::optimizers::AdamWFullPrecision optimizer(params, config);

    step_with_random_grads(params, optimizer, 3);
    expect_checkpoint_matches(
        save("only", model, optimizer),
        params,
        optimizer,
        {"master_weights", "exp_avg", "exp_avg_sq"},
        ttml::autograd::PreferredPrecision::FULL);
}
