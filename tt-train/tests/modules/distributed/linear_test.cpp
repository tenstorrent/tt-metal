// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/distributed/linear.hpp"

#include <gtest/gtest.h>
#include <umd/device/cluster.h>

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_utils.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed_mapping.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"

namespace {

auto check_board_is_n300() {
    return tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0) == BoardType::N300;
}

ttml::autograd::TensorPtr get_parameter(auto& parameters, const std::string& name_substring) {
    for (const auto& [name, parameter] : parameters) {
        if (name.find(name_substring) != std::string::npos) {
            return parameter;
        }
    }
    throw std::logic_error(fmt::format("Parameter for a given name substring {} not found", name_substring));
}

}  // namespace

class N300TensorParallelLinearTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!check_board_is_n300()) {
            GTEST_SKIP() << "Skipping N300 specific tests";
        }
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(N300TensorParallelLinearTest, RowParallelLinearHasBiasNotInputParallel) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = true;
    bool input_is_parallel = false;

    auto layer = ttml::modules::distributed::RowParallelLinear(in_features, out_features, has_bias, input_is_parallel);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");
    auto bias = get_parameter(parameters, "bias");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    EXPECT_TRUE(xt::allclose(output_xtensor[0], output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    ttml::core::MeshToXTensorVariant<float> concat_composer = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer);
    auto bias_xtensor = ttml::core::to_xtensor<float>(bias->get_value(), identity_composer);

    auto weight_xtensor_shape = weight_xtensor[0].shape();
    auto test_data_shape = test_data.shape();

    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));
    if (has_bias) {
        expected_output += bias_xtensor[0];
    }

    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, RowParallelLinearNoBiasNotInputParallel) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = false;
    bool input_is_parallel = false;

    auto layer = ttml::modules::distributed::RowParallelLinear(in_features, out_features, has_bias, input_is_parallel);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    EXPECT_TRUE(xt::allclose(output_xtensor[0], output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    ttml::core::MeshToXTensorVariant<float> concat_composer = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer);

    auto weight_xtensor_shape = weight_xtensor[0].shape();
    auto test_data_shape = test_data.shape();

    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));
    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, RowParallelLinearHasBiasInputParallel) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = true;
    bool input_is_parallel = true;

    auto layer = ttml::modules::distributed::RowParallelLinear(in_features, out_features, has_bias, input_is_parallel);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");
    auto bias = get_parameter(parameters, "bias");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> shard_composer = ttml::core::ShardXTensorToMesh<float>(mesh_shape, 3);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, shard_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    EXPECT_TRUE(xt::allclose(output_xtensor[0], output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    ttml::core::MeshToXTensorVariant<float> concat_composer = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer);
    auto bias_xtensor = ttml::core::to_xtensor<float>(bias->get_value(), identity_composer);
    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));
    if (has_bias) {
        expected_output += bias_xtensor[0];
    }

    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, RowParallelLinearNoBiasInputParallel) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = false;
    bool input_is_parallel = true;

    auto layer = ttml::modules::distributed::RowParallelLinear(in_features, out_features, has_bias, input_is_parallel);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> shard_composer = ttml::core::ShardXTensorToMesh<float>(mesh_shape, 3);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, shard_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    EXPECT_TRUE(xt::allclose(output_xtensor[0], output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    ttml::core::MeshToXTensorVariant<float> concat_composer = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer);
    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));

    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[0], /* rtol */ 1e-3, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, ColumnParallelLinearHasBiasAllGather) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = true;
    bool use_all_gather = true;

    auto layer = ttml::modules::distributed::ColumnParallelLinear(in_features, out_features, has_bias, use_all_gather);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");
    auto bias = get_parameter(parameters, "bias");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    EXPECT_TRUE(xt::allclose(output_xtensor[0], output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    ttml::core::MeshToXTensorVariant<float> concat_composer_2 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 2U);
    ttml::core::MeshToXTensorVariant<float> concat_composer_3 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer_2);
    auto bias_xtensor = ttml::core::to_xtensor<float>(bias->get_value(), concat_composer_3);

    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));
    if (has_bias) {
        expected_output += bias_xtensor[0];
    }

    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[1], /* rtol */ 1e-2, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, ColumnParallelLinearNoBiasAllGather) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = false;
    bool use_all_gather = true;

    auto layer = ttml::modules::distributed::ColumnParallelLinear(in_features, out_features, has_bias, use_all_gather);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    EXPECT_TRUE(xt::allclose(output_xtensor[0], output_xtensor[1], /* rtol */ 1e-3, /* atol */ 1e-2));

    ttml::core::MeshToXTensorVariant<float> concat_composer_2 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 2U);
    ttml::core::MeshToXTensorVariant<float> concat_composer_3 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer_2);
    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));

    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(expected_output, output_xtensor[1], /* rtol */ 1e-2, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, ColumnParallelLinearHasBiasNoAllGather) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = true;
    bool use_all_gather = false;

    auto layer = ttml::modules::distributed::ColumnParallelLinear(in_features, out_features, has_bias, use_all_gather);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");
    auto bias = get_parameter(parameters, "bias");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);

    ttml::core::MeshToXTensorVariant<float> concat_composer_2 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 2U);
    ttml::core::MeshToXTensorVariant<float> concat_composer_3 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer_2);
    auto bias_xtensor = ttml::core::to_xtensor<float>(bias->get_value(), concat_composer_3);

    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));
    expected_output = expected_output.reshape({1U, 1U, 1U, out_features});
    if (has_bias) {
        expected_output += bias_xtensor[0];
    }

    EXPECT_TRUE(xt::allclose(
        xt::view(expected_output, xt::all(), xt::all(), xt::all(), xt::range(0, out_features / 2)),
        output_xtensor[0],
        /* rtol */ 1e-2,
        /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        xt::view(expected_output, xt::all(), xt::all(), xt::all(), xt::range(out_features / 2, out_features)),
        output_xtensor[1],
        /* rtol */ 1e-2,
        /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, ColumnParallelLinearNoBiasNoAllGather) {
    uint32_t in_features = 64U;
    uint32_t out_features = 64U;
    bool has_bias = false;
    bool use_all_gather = false;

    auto layer = ttml::modules::distributed::ColumnParallelLinear(in_features, out_features, has_bias, use_all_gather);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto weight = get_parameter(parameters, "weight");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features}, 0.F, 1.F).reshape({1U, 1U, 1U, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);

    ttml::core::MeshToXTensorVariant<float> concat_composer_2 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 2U);
    ttml::core::MeshToXTensorVariant<float> concat_composer_3 = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    // (1, 1, out_features, in_features)
    auto weight_xtensor = ttml::core::to_xtensor<float>(weight->get_value(), concat_composer_2);

    auto expected_output = xt::linalg::dot(test_data, xt::transpose(weight_xtensor[0], {0, 1, 3, 2}));
    expected_output = expected_output.reshape({1U, 1U, 1U, out_features});

    EXPECT_TRUE(xt::allclose(
        xt::view(expected_output, xt::all(), xt::all(), xt::all(), xt::range(0, out_features / 2)),
        output_xtensor[0],
        /* rtol */ 1e-2,
        /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        xt::view(expected_output, xt::all(), xt::all(), xt::all(), xt::range(out_features / 2, out_features)),
        output_xtensor[1],
        /* rtol */ 1e-2,
        /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, RowParallelLinearHasBiasNanoGPT) {
    uint32_t batch_size = 64;
    uint32_t sequence_length = 256;
    uint32_t in_features = 384U;
    uint32_t out_features = 128U;
    bool has_bias = true;
    bool input_is_parallel = false;

    auto generator = ttml::autograd::ctx().get_generator();

    auto layer = ttml::modules::distributed::RowParallelLinear(in_features, out_features, has_bias, input_is_parallel);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto row_parallel_weight = get_parameter(parameters, "weight");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features * batch_size * sequence_length}, -1.F, 1.F)
                                      .reshape({batch_size, 1U, sequence_length, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);
    output->backward();

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto row_parallel_output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);

    auto row_parallel_input_gradients = ttml::core::to_xtensor<float>(tensor->get_grad(), identity_composer);

    ttml::core::MeshToXTensorVariant<float> concat_composer = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 3U);
    auto row_parallel_weight_gradients =
        ttml::core::to_xtensor<float>(row_parallel_weight->get_grad(), concat_composer);

    // set generator
    ttml::autograd::ctx().set_generator(generator);

    auto replicate_layer = ttml::modules::LinearLayer(in_features, out_features, has_bias);
    auto replicate_layer_parameters = replicate_layer.parameters();
    auto replicate_layer_weight = get_parameter(replicate_layer_parameters, "weight");
    auto replicate_layer_input = ttml::autograd::create_tensor(tt_tensor);
    auto replicate_layer_output = replicate_layer(replicate_layer_input);
    replicate_layer_output->backward();
    auto replicate_output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    auto replicate_layer_input_gradients =
        ttml::core::to_xtensor<float>(replicate_layer_input->get_grad(), identity_composer);
    auto replicate_layer_weight_gradients =
        ttml::core::to_xtensor<float>(replicate_layer_weight->get_grad(), identity_composer);

    EXPECT_TRUE(
        xt::allclose(replicate_output_xtensor[0], row_parallel_output_xtensor[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(
        xt::allclose(replicate_output_xtensor[1], row_parallel_output_xtensor[1], /* rtol */ 1e-2, /* atol */ 1e-2));

    EXPECT_TRUE(xt::allclose(
        replicate_layer_input_gradients[0], row_parallel_input_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        replicate_layer_input_gradients[1], row_parallel_input_gradients[1], /* rtol */ 1e-2, /* atol */ 1e-2));

    EXPECT_TRUE(xt::allclose(
        replicate_layer_weight_gradients[0], row_parallel_weight_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        replicate_layer_weight_gradients[1], row_parallel_weight_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, ColumnParallelLinearHasBiasNanoGPT) {
    uint32_t batch_size = 64;
    uint32_t sequence_length = 256;
    uint32_t in_features = 384U;
    uint32_t out_features = 128U;
    bool has_bias = true;
    bool use_all_gather = true;

    auto generator = ttml::autograd::ctx().get_generator();

    auto layer = ttml::modules::distributed::ColumnParallelLinear(in_features, out_features, has_bias, use_all_gather);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto column_parallel_weight = get_parameter(parameters, "weight");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features * batch_size * sequence_length}, -1.F, 1.F)
                                      .reshape({batch_size, 1U, sequence_length, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);
    output->backward();

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto column_parallel_output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);

    auto column_parallel_input_gradients = ttml::core::to_xtensor<float>(tensor->get_grad(), identity_composer);

    ttml::core::MeshToXTensorVariant<float> concat_composer = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 2U);
    auto column_parallel_weight_gradients =
        ttml::core::to_xtensor<float>(column_parallel_weight->get_grad(), concat_composer);

    // set generator
    ttml::autograd::ctx().set_generator(generator);

    auto replicate_layer = ttml::modules::LinearLayer(in_features, out_features, has_bias);
    auto replicate_layer_parameters = replicate_layer.parameters();
    auto replicate_layer_weight = get_parameter(replicate_layer_parameters, "weight");
    auto replicate_layer_input = ttml::autograd::create_tensor(tt_tensor);
    auto replicate_layer_output = replicate_layer(replicate_layer_input);
    replicate_layer_output->backward();
    auto replicate_output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    auto replicate_layer_input_gradients =
        ttml::core::to_xtensor<float>(replicate_layer_input->get_grad(), identity_composer);
    auto replicate_layer_weight_gradients =
        ttml::core::to_xtensor<float>(replicate_layer_weight->get_grad(), identity_composer);

    EXPECT_TRUE(
        xt::allclose(replicate_output_xtensor[0], column_parallel_output_xtensor[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(
        xt::allclose(replicate_output_xtensor[1], column_parallel_output_xtensor[1], /* rtol */ 1e-2, /* atol */ 1e-2));

    EXPECT_TRUE(xt::allclose(
        replicate_layer_input_gradients[0], column_parallel_input_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        replicate_layer_input_gradients[1], column_parallel_input_gradients[1], /* rtol */ 1e-2, /* atol */ 1e-2));

    EXPECT_TRUE(xt::allclose(
        replicate_layer_weight_gradients[0], column_parallel_weight_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        replicate_layer_weight_gradients[1], column_parallel_weight_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
};

TEST_F(N300TensorParallelLinearTest, ColumnParallelLinearNoBiasNanoGPT) {
    uint32_t batch_size = 64;
    uint32_t sequence_length = 256;
    uint32_t in_features = 384U;
    uint32_t out_features = 128U;
    bool has_bias = false;
    bool use_all_gather = true;

    auto generator = ttml::autograd::ctx().get_generator();

    auto layer = ttml::modules::distributed::ColumnParallelLinear(in_features, out_features, has_bias, use_all_gather);
    auto parameters = layer.parameters();
    EXPECT_EQ(parameters.size(), 1UL + static_cast<size_t>(has_bias));

    auto column_parallel_weight = get_parameter(parameters, "weight");

    auto* device = &ttml::autograd::ctx().get_device();
    auto mesh_shape = device->shape();

    xt::xarray<float> test_data = xt::random::rand({in_features * batch_size * sequence_length}, -1.F, 1.F)
                                      .reshape({batch_size, 1U, sequence_length, in_features});
    ttml::core::XTensorToMeshVariant<float> replicate_composer = ttml::core::ReplicateXTensorToMesh<float>(mesh_shape);
    auto tt_tensor = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(test_data, device, replicate_composer);
    auto tensor = ttml::autograd::create_tensor(tt_tensor);
    auto output = layer(tensor);
    output->backward();

    ttml::core::MeshToXTensorVariant<float> identity_composer = ttml::core::VectorMeshToXTensor<float>(mesh_shape);
    auto column_parallel_output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);

    auto column_parallel_input_gradients = ttml::core::to_xtensor<float>(tensor->get_grad(), identity_composer);

    ttml::core::MeshToXTensorVariant<float> concat_composer = ttml::core::ConcatMeshToXTensor<float>(mesh_shape, 2U);
    auto column_parallel_weight_gradients =
        ttml::core::to_xtensor<float>(column_parallel_weight->get_grad(), concat_composer);

    // set generator
    ttml::autograd::ctx().set_generator(generator);

    auto replicate_layer = ttml::modules::LinearLayer(in_features, out_features, has_bias);
    auto replicate_layer_parameters = replicate_layer.parameters();
    auto replicate_layer_weight = get_parameter(replicate_layer_parameters, "weight");
    auto replicate_layer_input = ttml::autograd::create_tensor(tt_tensor);
    auto replicate_layer_output = replicate_layer(replicate_layer_input);
    replicate_layer_output->backward();
    auto replicate_output_xtensor = ttml::core::to_xtensor<float>(output->get_value(), identity_composer);
    auto replicate_layer_input_gradients =
        ttml::core::to_xtensor<float>(replicate_layer_input->get_grad(), identity_composer);
    auto replicate_layer_weight_gradients =
        ttml::core::to_xtensor<float>(replicate_layer_weight->get_grad(), identity_composer);

    EXPECT_TRUE(
        xt::allclose(replicate_output_xtensor[0], column_parallel_output_xtensor[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(
        xt::allclose(replicate_output_xtensor[1], column_parallel_output_xtensor[1], /* rtol */ 1e-2, /* atol */ 1e-2));

    EXPECT_TRUE(xt::allclose(
        replicate_layer_input_gradients[0], column_parallel_input_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        replicate_layer_input_gradients[1], column_parallel_input_gradients[1], /* rtol */ 1e-2, /* atol */ 1e-2));

    EXPECT_TRUE(xt::allclose(
        replicate_layer_weight_gradients[0], column_parallel_weight_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
    EXPECT_TRUE(xt::allclose(
        replicate_layer_weight_gradients[1], column_parallel_weight_gradients[0], /* rtol */ 1e-2, /* atol */ 1e-2));
};
