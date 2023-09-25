// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Sharded::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr , "Operands to shard need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16);
    if (this->sharded_op_type == ShardedOpType::INTERLEAVED_TO_SHARDED) {
        TT_ASSERT(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    } else if (this->sharded_op_type == ShardedOpType::SHARDED_TO_INTERLEAVED) {
        TT_ASSERT(input_tensor.memory_config().is_sharded());
        TT_ASSERT(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    }
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<Shape> Sharded::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> Sharded::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->sharded_op_type == ShardedOpType::INTERLEAVED_TO_SHARDED) {
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), input_tensor.layout(), input_tensor.device(), this->output_mem_config, this->shard_spec)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Sharded::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    if (this->sharded_op_type == ShardedOpType::INTERLEAVED_TO_SHARDED) {
        return interleaved_to_sharded_multi_core(input_tensor, output_tensor);
    } else {
        return sharded_to_interleaved_multi_core(input_tensor, output_tensor);
    }
}

tt::stl::reflection::Attributes Sharded::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}


}  // namespace tt_metal

}  // namespace tt
