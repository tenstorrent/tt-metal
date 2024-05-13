// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/sliding_window_op_infra/halo_op.hpp"

namespace ttnn::operations::halo {

using namespace tt::tt_metal;

void Halo::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // validate input data tensor
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        // skip the untilize, only do halo
        log_debug(tt::LogOp, "Input is ROW_MAJOR, no need to untilize.");
    } else {
        TT_FATAL(input_tensor.volume() % TILE_HW == 0);
    }
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED, "Only height or block sharded tensors are supported.");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Shard spec should not be empty");
}

const operation::Hash Halo::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    return operation::hash_operation<Halo>(this->attribute_values());
}

std::vector<tt::tt_metal::Shape> Halo::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();
    tt::tt_metal::Shape output_shape = input_shape;

    uint32_t nbatch = input_shape[0];
    uint32_t total_nsticks = config_.num_cores_nhw_ * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    output_shape[2] = (uint32_t) ceil((float) total_nsticks / nbatch);

    log_debug(tt::LogOp, "output_shape: [{} {} {} {}]", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    log_debug(tt::LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
    log_debug(tt::LogOp, "num_cores_nhw: {}", config_.num_cores_nhw_);

    return {output_shape};
}

std::vector<Tensor> Halo::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);

    TT_FATAL(input_tensor.memory_config().memory_layout == output_memory_config_.memory_layout, input_tensor.memory_config(), output_memory_config_);

    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto input_core_range = *(input_tensor.memory_config().shard_spec->grid.ranges().begin());
        auto output_core_range = *(output_memory_config_.shard_spec->grid.ranges().begin());
        auto input_core_w = input_core_range.end.y - input_core_range.start.y + 1;
        auto output_core_w = output_core_range.end.y - output_core_range.start.y + 1;
        TT_FATAL(input_core_w == output_core_w);
    }

    auto out_mem_config = output_memory_config_;
    out_mem_config.shard_spec->shape[0] = div_up(output_shape[0] * output_shape[2], config_.num_cores_nhw_);
    out_mem_config.shard_spec->shape[1] = input_tensor.memory_config().shard_spec->shape[1];
    out_mem_config.shard_spec->halo = true;
    return {create_sharded_device_tensor(
        output_shape, output_dtype, Layout::ROW_MAJOR, input_tensor.device(), out_mem_config)};
}


operation::ProgramWithCallbacks Halo::create_program(const std::vector<Tensor>& inputs, std::vector<Tensor> &outputs) const {
    const auto& input_tensor = inputs.at(0);
    auto& output_tensor = outputs.at(0);

    auto device = input_tensor.device();

    auto pad_metadata = sliding_window::generate_pad_metadata(config_);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config_);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(config_, op_trace_metadata);
    auto tensor_metadata = sliding_window::generate_tensor_metadata(pad_metadata, config_, reshard_num_cores_nhw_);
    auto kernel_config = sliding_window::generate_halo_kernel_config_tensors(tensor_metadata, shard_boundaries, remote_read_, device);

    const auto& pad_config = std::get<0>(kernel_config);
    const auto& local_config = std::get<1>(kernel_config);
    const auto& remote_config = std::get<2>(kernel_config);
    uint32_t max_out_nsticks_per_core = std::get<3>(kernel_config);

    ParallelConfig p_config;
    p_config.grid = input_tensor.shard_spec().value().grid;
    p_config.shard_scheme = input_tensor.memory_config().memory_layout;
    p_config.shard_orientation = input_tensor.shard_spec().value().orientation;

    TT_ASSERT(p_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED || (p_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED && ((transpose_mcast_ && p_config.shard_orientation == ShardOrientation::COL_MAJOR) || ((!transpose_mcast_) && p_config.shard_orientation == ShardOrientation::ROW_MAJOR))), "Transpose mcast and shard orientation should be consistent.");

    auto pad_config_tensor = sliding_window::construct_on_device_config_tensor(pad_config, config_, p_config, device);
    auto local_config_tensor = sliding_window::construct_on_device_config_tensor(local_config, config_, p_config, device);
    auto remote_config_tensor = sliding_window::construct_on_device_config_tensor(remote_config, config_, p_config, device);

    Program program = CreateProgram();

    tt::tt_metal::detail::AddConfigTensor(program, pad_config_tensor);
    tt::tt_metal::detail::AddConfigTensor(program, local_config_tensor);
    tt::tt_metal::detail::AddConfigTensor(program, remote_config_tensor);

    return {untilize_with_halo_multi_core_v2(
        program,
        input_tensor,
        pad_val_,
        config_.num_cores_nhw_,
        max_out_nsticks_per_core,
        pad_config_tensor,
        local_config_tensor,
        remote_config_tensor,
        remote_read_,
        transpose_mcast_,
        output_tensor
    )};
}


} // namespace ttnn::operations::halo
