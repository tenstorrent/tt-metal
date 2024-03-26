// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/conv/optimized_conv_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/constants.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/auto_format.hpp"

#include "tensor/tensor_utils.hpp"
using namespace tt::constants;

namespace optimized_conv_op_utils {
using namespace tt;
using namespace tt::tt_metal;

pair<uint32_t, uint32_t> compute_opt_conv_output_face_shape(uint32_t conv_activation_h, uint32_t conv_activation_w, uint32_t filter_h, uint32_t filter_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t padding_for_32B_alignment) {
    uint32_t conv_output_h = ((conv_activation_h - filter_h + (2 * pad_h)) / stride_h) + 1;
    uint32_t conv_output_w = ((conv_activation_w - filter_w + (2 * pad_w) - padding_for_32B_alignment) / stride_w) + 1;
    return {conv_output_h, conv_output_w};
}
pair<vector<uint32_t>, vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(Shape conv_activation_shape, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t padding_for_32B_alignment) {
    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    auto [conv_output_h, conv_output_w] = compute_opt_conv_output_face_shape(conv_activation_shape[1], conv_activation_shape[2], filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, padding_for_32B_alignment);
    uint32_t batch_size = conv_activation_shape[0];
    // pad height
    uint32_t num_rows = (uint32_t) batch_size * conv_output_h * conv_output_w;
    uint32_t act_block_h_datums = act_block_h_ntiles * TILE_HEIGHT;
    uint32_t num_rows_padded = (uint32_t) (ceil((double) num_rows / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t num_cols = conv_activation_shape[3] * filter_h * filter_w;
    uint32_t num_cols_padded = round_up(conv_activation_shape[3] * filter_w, TILE_WIDTH) * filter_h;
    return {{1, num_rows_padded, num_cols_padded}, {1, num_rows, num_cols}};
}

} // optimized_conv_op_utils

namespace tt {

namespace tt_metal {

Tensor optimized_conv(const Tensor& a,
            const Tensor &b,
            std::optional<const Tensor> bias,
            const std::optional<const Tensor> conv_reader_indices,
            const vector<int> conv_params,
            uint32_t output_channels,
            bool untilize_out,
            bool has_bias,
            bool fuse_relu,
            MathFidelity math_fidelity,
            const OptimizedConvParallelizationConfig& parallelization_config,
            const OptimizedConvBlockConfig& block_config,
            uint32_t extra_padding_for_32B_alignment,
            std::optional<MemoryConfig> output_mem_config,
            std::optional<DataType> output_dtype,
            std::optional<std::array<std::uint32_t, 4>> input_tensor_shape,
            bool use_shallow_conv_variant,
            std::optional<const DeviceComputeKernelConfig> compute_kernel_config
) {
    //TT_ASSERT(!untilize_out, "Optimized conv only supports tiled out");
    TT_ASSERT(b.get_layout() == Layout::TILE); // Weights should already be formatted
    const auto& ashape = input_tensor_shape.has_value() ? Shape(input_tensor_shape.value()) : a.get_legacy_shape();
    auto padded_a_shape = Shape({ashape[0], ashape[1], ashape[2], round_up(ashape[3], 16)});
    FormatParams input_a_format_params = {.pad_shape=padded_a_shape, .pad_value=0.0, .target_layout=Layout::ROW_MAJOR};
    FormatParams input_b_format_params = {.pad_shape=b.get_legacy_shape(), .pad_value=0.0, .target_layout=Layout::TILE};
    FormatParams input_bias_format_params = {};
    if (has_bias) {
        input_bias_format_params = {.pad_shape=bias.value().get_legacy_shape(), .pad_value=0, .target_layout=Layout::TILE};
    }
    auto output_layout = untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (output_mem_config.has_value()) {
        TT_ASSERT((output_mem_config.value().is_sharded() || output_mem_config.value().memory_layout == TensorMemoryLayout::INTERLEAVED));
    }
    auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    bool fp32_accum = a.device()->arch() == ARCH::WORMHOLE_B0;  // && compute_kernel_config.has_value()) ? compute_kernel_config.value().fp32_dest_acc_en : false;
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::LoFi, true, fp32_accum, false);
    return operation::run_without_autoformat(
        OptimizedConv(conv_params, output_channels, untilize_out, has_bias, fuse_relu, math_fidelity, parallelization_config, block_config, extra_padding_for_32B_alignment, output_mem_config.value_or(a.memory_config()), output_dtype.value_or(a.get_dtype()), ashape, use_shallow_conv_variant, kernel_config_val
        ),
        {a, b},
        {bias, conv_reader_indices}).at(0);
}

void OptimizedConv::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // TODO: ...
    TT_FATAL(!input_tensor_b.memory_config().is_sharded());
    if (this->untilize_out) {
        TT_FATAL((this->output_dtype == DataType::BFLOAT16) || (this->output_dtype == DataType::FLOAT32));
    }
    if (this->output_mem_config.is_sharded()) {
        uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntiles;
        auto [act_matrix_shape, act_matrix_shape_unpadded] = optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(input_tensor_a.get_legacy_shape(), conv_params, out_block_h_ntiles, extra_padding_for_32B_alignment);
        uint32_t out_width_ntiles = this->compute_output_shapes(input_tensors).at(0)[-1] / TILE_WIDTH;
        if(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(this->parallelization_config.per_core_out_matrix_width_ntiles == out_width_ntiles);
            TT_FATAL(this->block_config.out_subblock_w_ntiles == out_width_ntiles || this->block_config.out_subblock_h_ntiles == 1);
        } else if (this->output_mem_config.memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            // For block sharded, out_width per core is shard width, and this is split along row
            // TODO: We should clean this up and relax constraints on out_subblock h and w
            out_width_ntiles /= this->parallelization_config.grid_size.y;
            TT_FATAL(this->block_config.out_subblock_w_ntiles == out_width_ntiles || this->block_config.out_subblock_h_ntiles == 1);
        }
    }
}

std::vector<Shape> OptimizedConv::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a_shape = this->input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];
    uint32_t conv_activation_h = input_tensor_a_shape[1];
    uint32_t conv_activation_w = input_tensor_a_shape[2];
    // TODO: clean up here
    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    auto [conv_output_h, conv_output_w] = optimized_conv_op_utils::compute_opt_conv_output_face_shape(conv_activation_h, conv_activation_w, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, extra_padding_for_32B_alignment);

    // Tiled output shape is padded shape. Padded to tile shape.
    auto shape_w = batch_size * conv_output_h * conv_output_w;
    auto shape_c = output_channels;
    auto padded_shape_w = round_up(shape_w, TILE_HEIGHT);
    auto padded_shape_c = round_up(this->output_channels, TILE_WIDTH);
    auto output_padding = Padding(
        {{0, 0}, {0, 0}, {0, (padded_shape_w - shape_w)}, {0, (padded_shape_c - shape_c)}}, Padding::PadValue::Zero);
    auto output_tensor_shape = Shape({1, 1, padded_shape_w, padded_shape_c}, output_padding);
    return {output_tensor_shape};
}

std::vector<Tensor> OptimizedConv::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    if (this->output_mem_config.is_sharded()) {
        auto output_shape = this->compute_output_shapes(input_tensors).at(0);
        if (this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            uint32_t total_height_tiles = tt_metal::compute_volume(output_shape) / output_shape[-1] / TILE_HEIGHT;
            uint32_t num_cores = total_height_tiles / this->parallelization_config.per_core_out_matrix_height_ntiles;
            CoreRangeSet shard_grid = num_cores_to_corerange_set(num_cores, this->parallelization_config.grid_size, true);

            std::array<uint32_t, 2> shard_shape = {this->parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT, output_shape[-1]};
            auto shard_spec = ShardSpec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(output_shape, this->output_dtype, output_layout, input_tensor.device(), mem_config)};
        } else {
            auto [act_matrix_shape, act_matrix_shape_unpadded] = optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(this->input_tensor_shape, conv_params, this->parallelization_config.per_core_out_matrix_height_ntiles, extra_padding_for_32B_alignment);
            uint32_t act_matrix_height = (uint32_t) act_matrix_shape[1];
            uint32_t act_matrix_height_ntiles = act_matrix_height / TILE_HEIGHT;
            uint32_t total_active_num_cores_per_weight_slice = act_matrix_height_ntiles / this->parallelization_config.per_core_out_matrix_height_ntiles;
            uint32_t weight_matrix_width = weight_tensor.get_legacy_shape()[-1];
            uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;
            uint32_t num_weight_slices_width = weight_matrix_width_ntiles / this->parallelization_config.per_core_out_matrix_width_ntiles ;
            uint32_t total_active_num_cores = total_active_num_cores_per_weight_slice * num_weight_slices_width;
            CoreRangeSet shard_grid = num_cores_to_corerange_set(total_active_num_cores, this->parallelization_config.grid_size, true);
            std::array<uint32_t, 2> shard_shape = {this->parallelization_config.per_core_out_matrix_height_ntiles * TILE_HEIGHT, this->parallelization_config.per_core_out_matrix_width_ntiles * TILE_WIDTH};
            auto shard_spec = ShardSpec{shard_grid, shard_shape, ShardOrientation::COL_MAJOR};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_sharded_device_tensor(output_shape, this->output_dtype, output_layout, input_tensor.device(), mem_config)};
        }

    }
    return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, output_layout, this->output_mem_config);
}

operation::ProgramWithCallbacks OptimizedConv::create_program(const std::vector<Tensor>& input_tensors,
                                                     const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                     std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_bias = optional_input_tensors.at(0);
    const auto& conv_reader_indices = optional_input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);
    // TODO: Clean up split between different conv types
    if (input_tensor_a.memory_config().is_sharded()) {
        // If conv_reader_indices is passed in, use v2 where we don't generate indices locally
        if (conv_reader_indices.has_value()) {
            return multi_core_optimized_conv_sharded_v2_(input_tensor_a, input_tensor_b, this->input_tensor_shape, input_tensor_bias, conv_reader_indices, conv_params, output_channels, untilize_out, has_bias, fuse_relu, parallelization_config, block_config, extra_padding_for_32B_alignment, this->use_shallow_conv_variant, output_tensor, this->compute_kernel_config);
        } else {
            return multi_core_optimized_conv_sharded_(input_tensor_a, input_tensor_b, this->input_tensor_shape, input_tensor_bias, conv_params, output_channels, untilize_out, has_bias, fuse_relu, math_fidelity, parallelization_config, block_config, extra_padding_for_32B_alignment, output_tensor);
        }
    } else {
        return multi_core_optimized_conv_(input_tensor_a, input_tensor_b, this->input_tensor_shape, input_tensor_bias, conv_params, output_channels, untilize_out, has_bias, fuse_relu, math_fidelity, parallelization_config, block_config, extra_padding_for_32B_alignment, output_tensor);
    }
}

operation::OpPerformanceModel OptimizedConv::create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a_shape = this->input_tensor_shape;
    uint32_t batch_size = input_tensor_a_shape[0];
    uint32_t conv_activation_h = input_tensor_a_shape[1];
    uint32_t conv_activation_w = input_tensor_a_shape[2];
    uint32_t conv_activation_c = input_tensor_a_shape[3];

    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];

    const auto& t = output_tensors.at(0);
    if(t.storage_type() != StorageType::DEVICE) {
        tt::log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    const int num_cores = (arch == ARCH::WORMHOLE_B0) ? 8 * 8 : 9 * 12;
    const int tensix_mul_adds_per_cycle_lofi = (arch == ARCH::WORMHOLE_B0) ? 4096 : 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    int output_height = std::floor((conv_activation_h - filter_h + 2 * pad_h) / stride_h + 1);
    int output_width = std::floor((conv_activation_w - filter_w + 2 * pad_w) / stride_w + 1);

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = conv_activation_c * filter_h * filter_w * 2; // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * this->output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil(((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) * (float)operation::OpPerformanceModel::fidelity_multiplier(this->math_fidelity));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);

#if 0
    tt::log_info(tt::LogOp, "OptimizedConv PerfModel:");
    tt::log_info(tt::LogOp, "\t Batch: {}", batch_size);
    tt::log_info(tt::LogOp, "\t In (H, W, C): ({}, {}, {})", conv_activation_h, conv_activation_w, conv_activation_c);
    tt::log_info(tt::LogOp, "\t Filter (H, W): ({}, {})", filter_h, filter_w);
    tt::log_info(tt::LogOp, "\t Filter Stride (H, W): ({}, {})", stride_h, stride_w);
    tt::log_info(tt::LogOp, "\t Pad (H, W): ({}, {})", pad_h, pad_w);
    tt::log_info(tt::LogOp, "\t Out (H, W, C): ({}, {}, {})", output_height, output_width, this->output_channels);
    tt::log_info(tt::LogOp, "\t ideal_dev_clock_cycles: {}", ideal_dev_clock_cycles);
#endif

    return result;
}

}  // namespace tt_metal

}  // namespace tt
