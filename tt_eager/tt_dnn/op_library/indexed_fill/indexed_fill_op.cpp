// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/indexed_fill/indexed_fill_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"


namespace tt {

namespace tt_metal {

void IndexedFill::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(1);
    const auto& input_tensor_b = input_tensors.at(2);
    const auto& batch_ids = input_tensors.at(0);
    auto input_tensor_a_shape = input_tensor_a.get_legacy_shape();
    auto input_tensor_b_shape = input_tensor_b.get_legacy_shape();
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input_tensor_b.get_layout() == input_tensor_a.get_layout(), "Inputs must be same layout");
    TT_FATAL(input_tensor_a_shape[1] == input_tensor_b_shape[1] &&
            input_tensor_a_shape[2] == input_tensor_b_shape[2] &&
            input_tensor_a_shape[3] == input_tensor_b_shape[3]
            , "Dims except batch dim must be the same on inputs");
    TT_FATAL(input_tensor_b_shape[0] == batch_ids.get_legacy_shape()[-1], "Second input and batch ids must be same outer dim");
    TT_FATAL(batch_ids.get_layout() == Layout::ROW_MAJOR, "Batch IDs must be ROW MAJOR");
    TT_FATAL(this->dim == 0, "Currently only supporting batch dimension");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to Index Fill need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to Index Fill need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Index Fill does not currently support sharding");
}

std::vector<Shape> IndexedFill::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(1);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> IndexedFill::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(1);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks IndexedFill::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& batch_ids = input_tensors.at(0);
    const auto& input_tensor_a = input_tensors.at(1);
    const auto& input_tensor_b = input_tensors.at(2);
    const auto& output_tensor = output_tensors.at(0);

    return indexed_fill_multi_core(batch_ids, input_tensor_a, input_tensor_b, output_tensor);
}


tt::stl::reflection::Attributes IndexedFill::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor indexed_fill(const Tensor &batch_ids, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config, std::int64_t dim) {
    return operation::run_without_autoformat(IndexedFill{output_mem_config, dim}, {batch_ids, input_a, input_b}).at(0);
}


}  // namespace tt_metal

}  // namespace tt
