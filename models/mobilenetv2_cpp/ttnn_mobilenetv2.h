// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MOBILENETV2_CPP_TTNN_MOBILENETV2
#define MOBILENETV2_CPP_TTNN_MOBILENETV2

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include "ttnn/types.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

class TtMobileNetV2Conv2D {
public:
    TtMobileNetV2Conv2D(
        const std::vector<int>& input_params,
        const std::pair<ttnn::Tensor, ttnn::Tensor>& parameters,
        std::shared_ptr<ttnn::MeshDevice> device,
        int batch_size,
        int groups = 1,
        int dilation = 1,
        bool act_block_h = false,
        bool block_shard = false,
        bool deallocate_activation = false,
        ttnn::Layout output_layout = ttnn::Layout::TILE,
        bool width_shard = false,
        int act_blocks = 32,
        bool enable_act_double_buffer = false,
        bool reshard_if_not_optimal = true,
        ttnn::DataType activation_dtype = ttnn::DataType::BFLOAT8_B,
        ttnn::TensorMemoryLayout shard_layout = ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
        std::optional<ttnn::operations::unary::UnaryWithParam> activation = std::nullopt);

    ttnn::Tensor operator()(const ttnn::Tensor& x, int& h, int& w);

private:
    ttnn::operations::conv::conv2d::Conv2dConfig initialize_conv_config();
    ttnn::DeviceComputeKernelConfig initialize_compute_config();

private:
    std::shared_ptr<ttnn::MeshDevice> device_;
    std::pair<ttnn::Tensor, ttnn::Tensor> parameters;
    ttnn::DataType activation_dtype;
    std::vector<int> input_params;
    int groups;
    int dilation;
    bool act_block_h;
    bool block_shard;
    bool deallocate_activation;
    ttnn::Layout output_layout;
    bool width_shard;
    int act_blocks;
    bool enable_act_double_buffer;
    bool reshard_if_not_optimal;
    int batch_size;
    ttnn::TensorMemoryLayout shard_layout;
    ttnn::operations::conv::conv2d::Conv2dConfig conv_config;
    ttnn::DeviceComputeKernelConfig compute_config;
    std::optional<ttnn::operations::unary::UnaryWithParam> activation;
};

class TtInvertedResidual {
public:
    TtInvertedResidual(
        const std::unordered_map<std::string, ttnn::Tensor>& model_params,
        std::shared_ptr<ttnn::MeshDevice> device,
        int batchsize,
        int expand_ratio,
        int stride,
        int in_channels,
        int out_channels,
        int id,
        bool block_shard = false);

    ttnn::Tensor operator()(const ttnn::Tensor& x);

private:
    std::shared_ptr<ttnn::MeshDevice> device_;
    int batchsize;
    int stride;
    int expand_ratio;
    int in_channels;
    int out_channels;
    bool block_shard;
    int id;
    bool use_res_connect;

    std::unique_ptr<TtMobileNetV2Conv2D> conv1;
    std::unique_ptr<TtMobileNetV2Conv2D> conv2;
    std::unique_ptr<TtMobileNetV2Conv2D> conv3;
};

class TtMobileNetV2 : public std::enable_shared_from_this<TtMobileNetV2> {
public:
    TtMobileNetV2(
        const std::unordered_map<std::string, ttnn::Tensor>& model_params,
        std::shared_ptr<ttnn::MeshDevice> device,
        int batchsize);

    ~TtMobileNetV2() { blocks.clear(); }

    ttnn::Tensor operator()(const ttnn::Tensor& x);

private:
    std::unique_ptr<TtInvertedResidual> define_inverted_residual_block(
        int expand_ratio, int stride, int in_channels, int out_channels, int id, bool block_shard);

    ttnn::Tensor process_blocks(ttnn::Tensor& tensor);

private:
    std::shared_ptr<ttnn::MeshDevice> device_;
    const std::unordered_map<std::string, ttnn::Tensor>& model_parameters;
    int batchsize;
    std::unique_ptr<TtMobileNetV2Conv2D> conv1, conv2, conv3, conv4;
    std::vector<std::unique_ptr<TtInvertedResidual>> blocks;
    ttnn::Tensor l1_weight, l1_bias;
};

#endif  // MOBILENETV2_CPP_TTNN_MOBILENETV2
