// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_test_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "matmul_test_program_factory.hpp"

namespace ttml::metal::ops::matmul_test::device {

MatmulTestDeviceOperation::program_factory_t MatmulTestDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return MatmulTestProgramFactory{};
}

void MatmulTestDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MatmulTestDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "MatmulTest operation requires {} to be on Device.",
            name);

        TT_FATAL(tensor.buffer() != nullptr, "MatmulTest operation requires {} to be allocated in buffers.", name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "MatmulTest operation requires {} to be in Tile layout.",
            name);

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "MatmulTest operation requires {} to be BFLOAT16.",
            name);

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "MatmulTest operation requires {} to have Interleaved memory layout.",
            name);
    };

    check_tensor(tensor_args.input_a, "input_a");
    check_tensor(tensor_args.input_b, "input_b");
}

spec_return_value_t MatmulTestDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U);

    output_specs.emplace_back(
        tensor_args.input_a.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input_a.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input_a.memory_config()));

    return output_specs;
}

tensor_return_value_t MatmulTestDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_specs[0], tensor_args.input_a.device());
}

ttsl::hash::hash_t MatmulTestDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_a;
    const auto& input_logical_shape = input_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<MatmulTestDeviceOperation>(
        args.test_case, program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

std::tuple<MatmulTestDeviceOperation::operation_attributes_t, MatmulTestDeviceOperation::tensor_args_t>
MatmulTestDeviceOperation::invoke(const ttnn::Tensor& input_a, const ttnn::Tensor& input_b, TestCase test_case) {
    return {operation_attributes_t{.test_case = test_case}, tensor_args_t{.input_a = input_a, .input_b = input_b}};
}

}  // namespace ttml::metal::ops::matmul_test::device
