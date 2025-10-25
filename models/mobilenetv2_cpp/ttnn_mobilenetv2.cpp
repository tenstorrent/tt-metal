// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_mobilenetv2.h"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace TT_conv2d = ttnn::operations::conv::conv2d;
namespace TT_matmul = ttnn::operations::matmul;

TtMobileNetV2Conv2D::TtMobileNetV2Conv2D(
    const std::vector<int>& input_params,
    const std::pair<ttnn::Tensor, ttnn::Tensor>& parameters,
    std::shared_ptr<ttnn::MeshDevice> device,
    int batch_size,
    int groups /* = 1*/,
    int dilation /* = 1*/,
    bool act_block_h /* = false*/,
    bool block_shard /* = false*/,
    bool deallocate_activation /* = false*/,
    ttnn::Layout output_layout /* = ttnn::Layout::TILE*/,
    bool width_shard /* = false*/,
    int act_blocks /* = 32*/,
    bool enable_act_double_buffer /* = false*/,
    bool reshard_if_not_optimal /* = false*/,
    ttnn::DataType activation_dtype /* = ttnn::DataType::BFLOAT8_B*/,
    ttnn::TensorMemoryLayout shard_layout /* = ttnn::TensorMemoryLayout::HEIGHT_SHARDED*/,
    std::optional<ttnn::operations::unary::UnaryWithParam> activation /* = std::nullopt*/
    ) :
    device_(device),
    parameters(parameters),
    activation_dtype(activation_dtype),
    input_params(input_params),
    groups(groups),
    dilation(dilation),
    act_block_h(act_block_h),
    block_shard(block_shard),
    deallocate_activation(deallocate_activation),
    output_layout(output_layout),
    width_shard(width_shard),
    act_blocks(act_blocks),
    enable_act_double_buffer(enable_act_double_buffer),
    reshard_if_not_optimal(reshard_if_not_optimal),
    batch_size(batch_size),
    shard_layout(shard_layout),
    activation(activation) {
    if (block_shard) {
        shard_layout = ttnn::TensorMemoryLayout::BLOCK_SHARDED;
    }
    if (width_shard) {
        shard_layout = ttnn::TensorMemoryLayout::WIDTH_SHARDED;
    }

    conv_config = initialize_conv_config();
    compute_config = initialize_compute_config();
}

ttnn::Tensor TtMobileNetV2Conv2D::operator()(const ttnn::Tensor& x, int& h, int& w) {
    int input_height, input_width;
    const ttnn::Shape& logical_shape = x.logical_shape();
    if (logical_shape[1] != 1) {
        input_height = logical_shape[1];
        input_width = logical_shape[2];
    } else {
        input_height = static_cast<int>(std::sqrt(logical_shape[2] / batch_size));
        input_width = static_cast<int>(std::sqrt(logical_shape[2] / batch_size));
    }

    ttnn::Tensor output_tensor;
    auto conv2d_result = ttnn::conv2d(
        /*input_tensor=*/x,
        /*weight_tensor=*/parameters.first,
        device_.get(),
        /*in_channels=*/logical_shape[3],
        /*out_channels=*/input_params[3],
        batch_size,
        input_height,
        input_width,
        /*kernel_size=*/std::array<uint32_t, 2>{input_params[0], input_params[0]},
        /*stride=*/std::array<uint32_t, 2>{input_params[1], input_params[1]},
        /*padding=*/std::array<uint32_t, 2>{input_params[2], input_params[2]},
        /*dilation=*/std::array<uint32_t, 2>{dilation, dilation},
        groups,
        /*dtype=*/activation_dtype,
        /*bias_tensor=*/parameters.second,
        conv_config,
        compute_config,
        std::nullopt,
        std::nullopt,
        /*return_output_dim=*/true,
        /*return_weights_and_bias=*/true);
    std::pair<ttnn::Tensor, ttnn::Tensor> devparams(std::move(parameters));
    std::visit(
        tt::stl::overloaded(
            [&output_tensor, &h, &w, &devparams](std::tuple<
                                                 ttnn::Tensor,
                                                 std::tuple<uint32_t, uint32_t>,
                                                 std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>> result) {
                output_tensor = std::move(std::get<0>(result));
                auto h_w = std::get<1>(result);
                h = static_cast<int>(std::get<0>(h_w));
                w = static_cast<int>(std::get<1>(h_w));
                auto w_b = std::get<2>(result);
                devparams.first = std::move(std::get<0>(w_b));
                auto bias = std::get<1>(w_b);
                if (bias.has_value()) {
                    devparams.second = std::move(bias.value());
                }
            },
            [](auto&&) { throw std::runtime_error("Conv2d result type error!"); }),
        conv2d_result);
    parameters = std::move(devparams);
    return output_tensor;
}

TT_conv2d::Conv2dConfig TtMobileNetV2Conv2D::initialize_conv_config() {
    TT_conv2d::Conv2dConfig config;
    // config.dtype = activation_dtype;
    config.weights_dtype = ttnn::DataType::BFLOAT8_B;
    config.activation = activation;
    config.shard_layout = shard_layout;
    config.act_block_w_div = 1;
    config.deallocate_activation = deallocate_activation;
    config.enable_act_double_buffer = enable_act_double_buffer;
    config.output_layout = output_layout;
    config.reallocate_halo_output = false;
    config.reshard_if_not_optimal = reshard_if_not_optimal;
    config.enable_weights_double_buffer = true;

    if (act_block_h) {
        config.act_block_h_override = act_blocks;
    }

    if (block_shard) {
        config.shard_layout = ttnn::TensorMemoryLayout::BLOCK_SHARDED;
    }

    return config;
}

ttnn::DeviceComputeKernelConfig TtMobileNetV2Conv2D::initialize_compute_config() {
    return ttnn::init_device_compute_kernel_config(
        device_->arch(),
        std::nullopt,
        MathFidelity::LoFi,
        /*math_approx_mode=*/false);
}

TtInvertedResidual::TtInvertedResidual(
    const std::unordered_map<std::string, ttnn::Tensor>& model_params,
    std::shared_ptr<ttnn::MeshDevice> device,
    int batchsize,
    int expand_ratio,
    int stride,
    int in_channels,
    int out_channels,
    int id,
    bool block_shard /* = false*/
    ) :
    device_(device),
    batchsize(batchsize),
    stride(stride),
    expand_ratio(expand_ratio),
    in_channels(in_channels),
    out_channels(out_channels),
    block_shard(block_shard),
    id(id) {
    int hidden_dim = static_cast<int>(std::round(in_channels * expand_ratio));
    use_res_connect = (stride == 1 && in_channels == out_channels);

    if (expand_ratio != 1) {
        conv1 = std::make_unique<TtMobileNetV2Conv2D>(
            std::vector<int>{1, 1, 0, hidden_dim},
            std::make_pair(
                model_params.at(fmt::format("fused_conv_{}_weight", id * 2)),
                model_params.at(fmt::format("fused_conv_{}_bias", id * 2))),
            device_,
            batchsize,
            /*groups=*/1,
            /*dilation=*/1,
            /*act_block_h=*/false,
            /*block_shard=*/false,
            /*deallocate_activation=*/!use_res_connect,
            /*output_layout=*/ttnn::Layout::TILE,
            /*width_shard=*/false,
            /*act_blocks=*/32,
            /*enable_act_double_buffer=*/true,
            /*reshard_if_not_optimal=*/true,
            /*activation_dtype=*/ttnn::DataType::BFLOAT8_B,
            /*shard_layout=*/ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
            /*activation=*/ttnn::operations::unary::UnaryWithParam(ttnn::operations::unary::UnaryOpType::RELU6));
    }

    conv2 = std::make_unique<TtMobileNetV2Conv2D>(
        std::vector<int>{3, stride, 1, hidden_dim},
        std::make_pair(
            model_params.at(fmt::format("fused_conv_{}_weight", id * 2 + 1)),
            model_params.at(fmt::format("fused_conv_{}_bias", id * 2 + 1))),
        device_,
        batchsize,
        /*groups=*/hidden_dim,
        /*dilation=*/1,
        /*act_block_h=*/false,
        /*block_shard=*/block_shard,
        /*deallocate_activation=*/true,
        /*output_layout=*/ttnn::Layout::TILE,
        /*width_shard=*/false,
        /*act_blocks=*/32,
        /*enable_act_double_buffer=*/block_shard,
        /*reshard_if_not_optimal=*/true,
        /*activation_dtype=*/ttnn::DataType::BFLOAT8_B,
        /*shard_layout=*/ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
        /*activation=*/ttnn::operations::unary::UnaryWithParam(ttnn::operations::unary::UnaryOpType::RELU6));

    conv3 = std::make_unique<TtMobileNetV2Conv2D>(
        std::vector<int>{1, 1, 0, out_channels},
        std::make_pair(
            model_params.at(fmt::format("conv_{}_weight", id)), model_params.at(fmt::format("conv_{}_bias", id))),
        device_,
        batchsize,
        /*groups=*/1,
        /*dilation=*/1,
        /*act_block_h=*/false,
        /*block_shard=*/(10 <= id && id <= 16) ? false : block_shard,
        /*deallocate_activation=*/true,
        /*output_layout=*/ttnn::Layout::TILE,
        /*width_shard=*/false,
        /*act_blocks=*/32,
        /*enable_act_double_buffer=*/true);
}

ttnn::Tensor TtInvertedResidual::operator()(const ttnn::Tensor& x) {
    ttnn::Tensor identity = x;
    ttnn::Tensor out = x;
    int h, w;

    if (conv1) {
        out = (*conv1)(x, h, w);
    }
    out = (*conv2)(out, h, w);
    out = (*conv3)(out, h, w);

    if (use_res_connect) {
        if (identity.memory_config() != out.memory_config()) {
            identity = ttnn::to_memory_config(identity, out.memory_config());
        }
        auto tmp = ttnn::add(identity, out);
        identity.deallocate(true);
        out.deallocate(true);
        return tmp;
    }
    return out;
}

TtMobileNetV2::TtMobileNetV2(
    const std::unordered_map<std::string, ttnn::Tensor>& model_params,
    std::shared_ptr<ttnn::MeshDevice> device,
    int batchsize) :
    device_(device), model_parameters(model_params), batchsize(batchsize) {
    conv1 = std::make_unique<TtMobileNetV2Conv2D>(
        std::vector<int>{3, 2, 1, 32},
        std::make_pair(model_parameters.at("fused_conv_0_weight"), model_parameters.at("fused_conv_0_bias")),
        device_,
        batchsize,
        /*groups=*/1,
        /*dilation=*/1,
        /*act_block_h=*/false,
        /*block_shard=*/false,
        /*deallocate_activation=*/true,
        /*output_layout=*/ttnn::Layout::TILE,
        /*width_shard=*/false,
        /*act_blocks=*/32,
        /*enable_act_double_buffer=*/true,
        /*reshard_if_not_optimal=*/false,
        /*activation_dtype=*/ttnn::DataType::BFLOAT8_B,
        /*shard_layout=*/ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
        /*activation=*/ttnn::operations::unary::UnaryWithParam(ttnn::operations::unary::UnaryOpType::RELU6));

    conv2 = std::make_unique<TtMobileNetV2Conv2D>(
        std::vector<int>{3, 1, 1, 32},
        std::make_pair(model_parameters.at("fused_conv_1_weight"), model_parameters.at("fused_conv_1_bias")),
        device_,
        batchsize,
        /*groups=*/32,
        /*dilation=*/1,
        /*act_block_h=*/false,
        /*block_shard=*/false,
        /*deallocate_activation=*/true,
        /*output_layout=*/ttnn::Layout::TILE,
        /*width_shard=*/false,
        /*act_blocks=*/32,
        /*enable_act_double_buffer=*/true,
        /*reshard_if_not_optimal=*/false,
        /*activation_dtype=*/ttnn::DataType::BFLOAT8_B,
        /*shard_layout=*/ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
        /*activation=*/ttnn::operations::unary::UnaryWithParam(ttnn::operations::unary::UnaryOpType::RELU6));

    conv3 = std::make_unique<TtMobileNetV2Conv2D>(
        std::vector<int>{1, 1, 0, 16},
        std::make_pair(model_parameters.at("conv_0_weight"), model_parameters.at("conv_0_bias")),
        device_,
        batchsize,
        /*groups=*/1,
        /*dilation=*/1,
        /*act_block_h=*/false,
        /*block_shard=*/false,
        /*deallocate_activation=*/true,
        /*output_layout=*/ttnn::Layout::TILE,
        /*width_shard=*/false,
        /*act_blocks=*/32,
        /*enable_act_double_buffer=*/true);

    // Define InvertedResidual blocks
    blocks.push_back(define_inverted_residual_block(6, 2, 16, 24, 1, false));
    blocks.push_back(define_inverted_residual_block(6, 1, 24, 24, 2, false));
    blocks.push_back(define_inverted_residual_block(6, 2, 24, 32, 3, false));
    blocks.push_back(define_inverted_residual_block(6, 1, 32, 32, 4, false));
    blocks.push_back(define_inverted_residual_block(6, 1, 32, 32, 5, false));
    blocks.push_back(define_inverted_residual_block(6, 2, 32, 64, 6, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 64, 64, 7, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 64, 64, 8, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 64, 64, 9, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 64, 96, 10, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 96, 96, 11, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 96, 96, 12, true));
    blocks.push_back(define_inverted_residual_block(6, 2, 96, 160, 13, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 160, 160, 14, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 160, 160, 15, true));
    blocks.push_back(define_inverted_residual_block(6, 1, 160, 320, 16, true));

    conv4 = std::make_unique<TtMobileNetV2Conv2D>(
        std::vector<int>{1, 1, 0, 1280},
        std::make_pair(model_parameters.at("fused_conv_34_weight"), model_parameters.at("fused_conv_34_bias")),
        device_,
        batchsize,
        /*groups=*/1,
        /*dilation=*/1,
        /*act_block_h=*/false,
        /*block_shard=*/false,
        /*deallocate_activation=*/true,
        /*output_layout=*/ttnn::Layout::TILE,
        /*width_shard=*/false,
        /*act_blocks=*/32,
        /*enable_act_double_buffer=*/false,
        /*reshard_if_not_optimal=*/true,
        /*activation_dtype=*/ttnn::DataType::BFLOAT8_B,
        /*shard_layout=*/ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
        /*activation=*/ttnn::operations::unary::UnaryWithParam(ttnn::operations::unary::UnaryOpType::RELU6));

    l1_weight = model_parameters.at("classifier_1_weight");
    l1_bias = model_parameters.at("classifier_1_bias");
}

ttnn::Tensor TtMobileNetV2::operator()(const ttnn::Tensor& x) {
    int h, w;
    auto output_tensor = (*conv1)(x, h, w);
    output_tensor = (*conv2)(output_tensor, h, w);
    output_tensor = (*conv3)(output_tensor, h, w);

    // Process all the InvertedResidual blocks
    output_tensor = process_blocks(output_tensor);

    output_tensor = (*conv4)(output_tensor, h, w);

    output_tensor = ttnn::to_layout(output_tensor, ttnn::Layout::ROW_MAJOR);
    auto tensor_shape = output_tensor.logical_shape();
    output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{batchsize, h, w, tensor_shape[3]});
    if (output_tensor.is_sharded()) {
        output_tensor = ttnn::sharded_to_interleaved(output_tensor, ttnn::L1_MEMORY_CONFIG, std::nullopt);
    }

    output_tensor = ttnn::global_avg_pool2d(output_tensor);
    output_tensor = ttnn::reshape(
        output_tensor, tt::tt_metal::infer_dims_for_reshape(output_tensor, std::vector<int>{batchsize, -1}));

    auto compute_config = ttnn::init_device_compute_kernel_config(
        device_->arch(),
        std::nullopt,
        /*math_fidelity=*/MathFidelity::LoFi,
        /*math_approx_mode=*/true,
        /*fp32_dest_acc_en=*/false,
        /*packer_l1_acc=*/true);

    auto matmul_config = TT_matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = {8, 8},
        .in0_block_w = 1,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .out_block_h = 1,
        .out_block_w = 1,
        .per_core_M = 1,
        .per_core_N = 1,
        .fuse_batch = true,
        .fused_activation = std::nullopt,
        .mcast_in0 = true,
        .gather_in0 = false,
        .hop_cores = ttnn::CoreRangeSet(),
        .num_global_cb_receivers = 1};

    auto shard_grid = ttnn::CoreRangeSet({ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(7, 7))});
    auto shard_spec = tt::tt_metal::ShardSpec(shard_grid, {32, 32});
    auto width_sharded_mem_config =
        ttnn::MemoryConfig(ttnn::TensorMemoryLayout::WIDTH_SHARDED, ttnn::BufferType::L1, shard_spec);

    output_tensor = ttnn::to_memory_config(output_tensor, width_sharded_mem_config);
    output_tensor = ttnn::linear(
        output_tensor,
        l1_weight,
        /*bias=*/l1_bias,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/ttnn::L1_WIDTH_SHARDED_MEMORY_CONFIG,
        /*dtype=*/std::nullopt,
        /*program_config=*/matmul_config,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_config);

    return output_tensor;
}

std::unique_ptr<TtInvertedResidual> TtMobileNetV2::define_inverted_residual_block(
    int expand_ratio, int stride, int in_channels, int out_channels, int id, bool block_shard) {
    return std::make_unique<TtInvertedResidual>(
        model_parameters, device_, batchsize, expand_ratio, stride, in_channels, out_channels, id, block_shard);
}

ttnn::Tensor TtMobileNetV2::process_blocks(ttnn::Tensor& tensor) {
    for (auto& block : blocks) {
        tensor = (*block)(tensor);
    }
    return tensor;
}
