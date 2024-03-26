// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include <algorithm>
using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Sharded::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    } else if (this->sharded_op_type == ShardedOpType::ShardedToInterleaved) {
        TT_FATAL(input_tensor.memory_config().is_sharded());
        if (input_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            if (input_tensor.get_legacy_shape()[-1] % this->shard_spec.shape[1] != 0 ||
                ((input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % this->shard_spec.shape[0]) != 0) {
                TT_FATAL(input_tensor.shard_spec().value().grid.ranges().size() == 1);
            }
        }
    }
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE);
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(this->grid_size.x <= device_grid.x && this->grid_size.y <= device_grid.y);
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<Shape> Sharded::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> Sharded::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = this->shard_spec;
        return {create_sharded_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            this->output_dtype,
            input_tensor.get_layout(),
            input_tensor.device(),
            mem_config,
            true
            )};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Sharded::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        return interleaved_to_sharded_multi_core(input_tensor, output_tensor);
    } else {
        return sharded_to_interleaved_multi_core(input_tensor, output_tensor);
    }
}

std::string Sharded::get_type_name() const { return magic_enum::enum_name(this->sharded_op_type).data(); }

ShardedOpParallelizationStrategy Sharded::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    return ShardedOpParallelizationStrategy::MULTI_CORE;
}


void Reshard::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");
    TT_FATAL(this->output_mem_config.is_sharded(), "output must be sharded");
    if(input_tensor.get_layout() == Layout::ROW_MAJOR) {
        bool same_row_size = input_tensor.memory_config().shard_spec.value().shape[1] == this->output_mem_config.shard_spec.value().shape[1];
        TT_FATAL(same_row_size, "row major must have shard_spec[1] be the same on both input and output");
    }
}

std::vector<Shape> Reshard::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

operation::ProgramWithCallbacks Reshard::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    //each tensor has its respective shard_spec within its memory_config

    if(this->rt_type == ReshardRunTimeArgType::RUNTIME_ARGS) {
        return reshard_runtime_args_multi_core(input_tensor, output_tensor);
    }
    else {
        auto config_tensor = output_tensors.at(1);
        return reshard_config_tensor_multi_core(input_tensor, config_tensor, output_tensor);
    }


}

std::vector<Tensor> Reshard::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto mem_config = this->output_mem_config;



    auto output_tensor = create_sharded_device_tensor(
        this->compute_output_shapes(input_tensors).at(0),
        input_tensor.get_dtype(),
        input_tensor.get_layout(),
        input_tensor.device(),
        mem_config,
        true
        );
    if (this->rt_type == ReshardRunTimeArgType::RUNTIME_ARGS) {
        return {output_tensor};
    }
    else {
        auto sharded_mem_config = MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1};
        auto cores = corerange_to_cores(mem_config.shard_spec.value().grid);

        auto output_core_to_page_range_pair = get_core_page_ranges(input_tensor.buffer(), output_tensor.buffer());


        //get maximum number of page_range per core for sharded size
        uint32_t max_ranges = 0;
        auto data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
        uint32_t page_size;
        if (input_tensor.get_layout() == Layout::TILE) {
            page_size = tt_metal::detail::TileSize(data_format);
        } else {
            page_size = output_tensor.get_legacy_shape()[-1] * output_tensor.element_size();
        }


        for(auto core: cores) {
            auto page_range_vector = output_core_to_page_range_pair.at(core);
            max_ranges = std::max((uint32_t)page_range_vector.size(), max_ranges);
        }


        uint32_t aligned_max_ranges = round_up_to_mul32(2+(max_ranges*4));
        //std::vector<uint32_t> args;
        //args.reserve(cores.size()*(aligned_max_ranges));

        //for(auto core: cores) {
        //    auto page_range_vector = output_core_to_page_range_pair.at(core);
        //    auto physical_input_core = input_tensor.device()->worker_core_from_logical_core(core);
        //    for (const auto& [core, range] : page_range_vector) {
        //        args.push_back(physical_input_core.x);
        //        args.push_back(physical_input_core.y);
        //        args.push_back(range.start * page_size);
        //        args.push_back((range.end - range.start) * page_size);
        //    }
        //    //padding for max_ranges
        //    for(uint32_t i=((uint32_t)page_range_vector.size() * 4); i < aligned_max_ranges; i++) {
        //        args.push_back(0);
        //    }
        //}


        ShardSpec shard_spec(mem_config.shard_spec.value().grid,
                    {1, aligned_max_ranges},
                    mem_config.shard_spec.value().orientation);

        sharded_mem_config.shard_spec = shard_spec;
        auto config_tensor = create_sharded_device_tensor(
            Shape({1,1,((uint32_t)cores.size()), aligned_max_ranges}),
            DataType::UINT32,
            Layout::ROW_MAJOR,
            input_tensor.device(),
            sharded_mem_config,
            true
        );

        //tt::tt_metal::detail::WriteToBuffer(*config_tensor.buffer(), args);
        return {output_tensor, config_tensor};
    }

}


ShardedOpParallelizationStrategy Reshard::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    return ShardedOpParallelizationStrategy::MULTI_CORE;
}

}  // namespace tt_metal

}  // namespace tt
