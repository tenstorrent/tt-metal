// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/risc_v/risc_v_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

void ArgMax::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Input to argmax need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Input to argmax need to be allocated in buffers on device!");

    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Only INTERLEAVED memory layout is supported for inputs!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    TT_FATAL(this->output_dtype == DataType::UINT32, "Only UINT32 is supported for outputs!");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Only INTERLEAVED memory layout is supported for outputs!");

    if (this->dim.has_value()) {
        const uint32_t input_rank = input_tensor_a.get_legacy_shape().rank();
        const uint32_t normalized_dim = dim.value() < 0 ? dim.value() + input_rank : dim.value();
        TT_FATAL(normalized_dim >= 0, fmt::format("Invalid dim for argmax: {}!", dim.value()));
        TT_FATAL(normalized_dim < input_rank, fmt::format("Invalid dim for argmax: {}!", dim.value()));
    }
}

std::vector<Shape> ArgMax::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto input_shape = input_tensors[0].get_legacy_shape();
    if (this->dim.has_value()) {
        // TODO: There seems to be an underflow issue with directly modifying last two dims
        if (this->dim.value() == -1 or this->dim.value() == 3) {
            Shape output_shape({input_shape[0], input_shape[1], input_shape[2], 1});
            return {output_shape};
        } else if (this->dim.value() == -2 or this->dim.value() == 2) {
            Shape output_shape({input_shape[0], input_shape[1], 1, input_shape[3]});
            return {output_shape};
        } else {
            input_shape[this->dim.value()] = 1;
            return {input_shape};
        }
    } else {
        Shape output_shape({1, 1, 1, 1});
        return {output_shape};
    }
}

std::vector<Tensor> ArgMax::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors[0];
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks ArgMax::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    const auto &output_tensor = output_tensors.at(0);

    return argmax_multi_core(input_tensor, output_tensor, this->dim);
}

tt::stl::reflection::Attributes ArgMax::attributes() const {
    return {
        {"output_dtype", this->output_dtype},
        {"output_mem_config", this->output_mem_config},
        {"dim", this->dim},
    };
}

}  // namespace tt_metal

}  // namespace tt
