// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/fold/fold_op.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal {
void Fold::validate(const std::vector<Tensor> &input_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Expect input tensor to be allocated on a device buffer.");
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Expect input tensor in row-major layout.");
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Folding of sharded tensors is not supported.");

    TT_FATAL(input_tensor.get_legacy_shape()[1] % stride_h == 0);
    TT_FATAL(input_tensor.get_legacy_shape()[2] % stride_w == 0);
}

std::vector<Shape> Fold::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const Shape &input_shape = input_tensors.at(0).get_legacy_shape();

    // we concatenate (stride_h sticks in H-dim) * (stride_w in W-dim) into 1 stick along C-dim
    return {{
        input_shape[0],
        input_shape[1] / stride_h,
        input_shape[2] / stride_w,
        input_shape[3] * stride_h * stride_w,
    }};
}

std::vector<Tensor> Fold::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.get_dtype();

    return operation::generic_create_output_tensors(
        *this, input_tensors, output_dtype, Layout::ROW_MAJOR, input_tensor.memory_config());
}

operation::ProgramWithCallbacks Fold::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);
    Tensor &output_tensor = output_tensors.at(0);

    return fold_single_core(input_tensor, output_tensor, stride_h, stride_w);
}

Tensor fold(const Tensor &input_tensor_a, uint8_t stride_h, uint8_t stride_w) {
    return operation::run(Fold{.stride_h = stride_h, .stride_w = stride_w}, {input_tensor_a}).at(0);
}

}  // namespace tt::tt_metal
