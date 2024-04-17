// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/owned_buffer_functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace binary {

using BinaryOpType = tt::tt_metal::BinaryOpType;

struct BinaryProgramConfig {
    const BinaryOpType op_type;
    const std::optional<std::vector<UnaryWithParam>> fused_activations;
    const MemoryConfig memory_config;
    const DataType dtype;
    const bool in_place;

    static constexpr auto attribute_names = std::make_tuple("op_type", "fused_activations", "memory_config", "dtype");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->op_type),
            std::cref(this->fused_activations),
            std::cref(this->memory_config),
            std::cref(this->dtype),
            std::cref(this->in_place));
    }
};

struct Binary {
    static inline const std::vector<TensorSchema> input_schemas{
        ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, false},
        ttnn::TensorSchema{
            2, 4, {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b}, {ttnn::TILE_LAYOUT}, true, false, false}};

    const BinaryProgramConfig program_config;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("program_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->program_config), std::cref(this->compute_kernel_config));
    }
};

namespace detail {
template <BinaryOpType binary_op_type, bool in_place>
struct MakeBinary {
    Tensor operator()(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt) const {
        std::vector<Tensor> output_tensors = {
            Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};
        operation::launch_op(
            [fused_activations, memory_config, dtype](
                const std::vector<Tensor> &input_tensors,
                const std::vector<std::optional<const Tensor>> &optional_input_tensors) mutable -> std::vector<Tensor> {
                auto &&[input_tensor_a, input_tensor_b] = [](const auto &input_tensor_a_arg,
                                                             const auto &input_tensor_b_arg) {
                    // Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
                    if (tt::tt_metal::compute_volume(input_tensor_a_arg.get_shape()) <
                        tt::tt_metal::compute_volume(input_tensor_b_arg.get_shape())) {
                        return std::make_tuple(input_tensor_b_arg, input_tensor_a_arg);
                    }
                    return std::make_tuple(input_tensor_a_arg, input_tensor_b_arg);
                }(input_tensors.at(0), input_tensors.at(1));

                return operation::run(
                    Binary{BinaryProgramConfig{
                        binary_op_type,
                        fused_activations,
                        memory_config.value_or(
                            ttnn::get_memory_config(input_tensor_a).value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG)),
                        dtype.value_or(input_tensor_a.get_dtype()),
                        in_place}},
                    {input_tensor_a, input_tensor_b});
            },
            {input_tensor_a, input_tensor_b},
            output_tensors);
        return output_tensors.at(0);
    }

    // TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
    // Currently, this is exactly how tt::tt_metal::add_unary works
    Tensor operator()(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        std::optional<std::vector<UnaryWithParam>> fused_activations = std::nullopt) const {
        auto buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(TILE_HEIGHT * TILE_WIDTH));
        buffer[0] = scalar;
        Tensor scalar_tensor =
            Tensor(
                OwnedStorage{buffer},
                ttnn::Shape(std::array<std::uint32_t, 2>{1, 1}, std::array<std::uint32_t, 2>{TILE_HEIGHT, TILE_WIDTH}),
                DataType::BFLOAT16,
                Layout::TILE)
                .to(input_tensor_a.device());
        return this->operator()(input_tensor_a, scalar_tensor, memory_config, dtype, fused_activations);
    }
};
}  // namespace detail

constexpr auto add = detail::MakeBinary<BinaryOpType::ADD, false>{};
constexpr auto add_ = detail::MakeBinary<BinaryOpType::ADD, true>{};
constexpr auto subtract = detail::MakeBinary<BinaryOpType::SUB, false>{};
constexpr auto subtract_ = detail::MakeBinary<BinaryOpType::SUB, true>{};
constexpr auto multiply = detail::MakeBinary<BinaryOpType::MUL, false>{};
constexpr auto multiply_ = detail::MakeBinary<BinaryOpType::MUL, true>{};

template <typename InputBType>
ttnn::Tensor operator+(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return add(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator-(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return subtract(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator*(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return multiply(input_tensor_a, scalar);
}

}  // namespace binary

}  // namespace operations
using operations::binary::add;
}  // namespace ttnn
