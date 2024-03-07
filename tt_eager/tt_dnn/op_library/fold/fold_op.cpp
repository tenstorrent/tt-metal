// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/fold/fold_op.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt::tt_metal {
FoldOpParallelizationStrategy Fold::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (is_sharded) {
        return FoldOpParallelizationStrategy::SHARDED_MULTI_CORE;
    } else {
        return FoldOpParallelizationStrategy::SINGLE_CORE;
    }
}

void Fold::validate(const std::vector<Tensor> &input_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);

    const Shape &input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Fold: Expect input tensor in row-major layout.");
    if (is_sharded) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
            "Fold: Only height-sharded input tensors are supported.");

        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(shard_shape[0] % (input_shape[2] * stride_h * stride_w) == 0);
    } else {
        TT_FATAL(input_shape[1] % stride_h == 0);
        TT_FATAL(input_shape[2] % stride_w == 0);
    }
    TT_FATAL(
        (input_shape[-1] * input_tensor.element_size()) % 16 == 0,
        "Fold: Expect input tensor's pages to be multiples of 16 bytes.");
}

std::vector<Shape> Fold::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const Shape &input_shape = input_tensors.at(0).get_legacy_shape();

    // we concatenate (stride_h sticks in H-dim) * (stride_w in W-dim) into 1 stick along C-dim
    return {{
        1,
        1,
        input_shape[0] * input_shape[1] * input_shape[2] / (stride_h * stride_w),
        input_shape[3] * stride_h * stride_w,
    }};
}

std::vector<Tensor> Fold::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.get_dtype();

    if (is_sharded) {
        MemoryConfig mem_config = input_tensor.memory_config();
        mem_config.shard_spec->shape[0] /= stride_h * stride_w;
        mem_config.shard_spec->shape[1] *= stride_h * stride_w;

        return {create_sharded_device_tensor(
            compute_output_shapes(input_tensors).at(0),
            output_dtype,
            input_tensor.get_layout(),
            input_tensor.device(),
            mem_config,
            true)};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, output_dtype, Layout::ROW_MAJOR, input_tensor.memory_config());
    }
}

operation::ProgramWithCallbacks Fold::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const Tensor &input_tensor = input_tensors.at(0);
    Tensor &output_tensor = output_tensors.at(0);

    if (is_sharded) {
        return fold_multi_core(input_tensor, output_tensor, stride_h, stride_w);
    } else {
        return fold_single_core(input_tensor, output_tensor, stride_h, stride_w);
    }
}

Tensor fold(const Tensor &input_tensor, uint8_t stride_h, uint8_t stride_w) {
    bool is_sharded = input_tensor.is_sharded();

    return operation::run(Fold{.stride_h = stride_h, .stride_w = stride_w, .is_sharded = is_sharded}, {input_tensor})
        .at(0);
}

}  // namespace tt::tt_metal
