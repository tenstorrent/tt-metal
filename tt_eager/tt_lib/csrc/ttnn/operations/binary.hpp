// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"

namespace py = pybind11;

namespace ttnn {

static const auto DRAM_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::DRAM};
static const auto L1_MEMORY_CONFIG = tt::tt_metal::MemoryConfig{
    .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::L1};

ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return ttnn::Tensor(tensor.value.reshape(shape.value()));
}

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
    const auto& tensor_shape = tensor.shape;
    const auto rank = tensor_shape.rank();
    if (rank == 4) {
        return tensor;
    }
    if (rank > 4) {
        TT_THROW("Tensor rank is greater than 4");
    }

    const auto tensor_shape_4D = tensor_shape.to_rank<4>();
    return ttnn::reshape(tensor, tensor_shape_4D);
}

namespace operations {
namespace binary {

void py_module(py::module& m_binary) {
    m_binary.def(
        "add",
        [](const ttnn::Tensor& input_tensor_a,
           const ttnn::Tensor& input_tensor_b,
           const tt::tt_metal::MemoryConfig& memory_config) {
            const auto& original_shape = input_tensor_a.shape;
            const auto& input_shape_b = input_tensor_b.shape;

            std::size_t height_b{};
            std::size_t width_b{};
            if (input_shape_b.rank() == 1) {
                height_b = 1;
                width_b = input_shape_b[-1];
            } else {
                height_b = input_shape_b[-2];
                width_b = input_shape_b[-1];
            }

            auto input_tensor_a_4D = ttnn::unsqueeze_to_4D(input_tensor_a);
            auto input_tensor_b_4D = ttnn::unsqueeze_to_4D(input_tensor_b);

            const auto& ttl_input_tensor_a = input_tensor_a_4D.value;
            const auto& ttl_input_tensor_b = input_tensor_b_4D.value;

            if (height_b == 1 or width_b == 1) {
                tt::tt_metal::BcastOpDim bcast_op_dim;
                if (height_b == 1 and width_b == 1) {
                    bcast_op_dim = tt::tt_metal::BcastOpDim::HW;
                } else if (height_b == 1) {
                    bcast_op_dim = tt::tt_metal::BcastOpDim::H;
                } else if (width_b == 1) {
                    bcast_op_dim = tt::tt_metal::BcastOpDim::W;
                } else {
                    TT_THROW("Invalid broadcasting dimensions");
                }
                auto ttl_output = tt::tt_metal::bcast(
                    ttl_input_tensor_a,
                    ttl_input_tensor_b,
                    tt::tt_metal::BcastOpMath::ADD,
                    bcast_op_dim,
                    memory_config);
                auto output = ttnn::Tensor(ttl_output);
                return ttnn::reshape(output, original_shape);
            } else {
                auto ttl_output =
                    tt::tt_metal::add(ttl_input_tensor_a, ttl_input_tensor_b, std::nullopt, memory_config);
                auto output = ttnn::Tensor(ttl_output);
                return ttnn::reshape(output, original_shape);
            }
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG

    );

    m_binary.def(
        "add",
        [](const ttnn::Tensor& input_tensor_a,
           const float input_tensor_b,
           const tt::tt_metal::MemoryConfig& memory_config) {
            const auto& original_shape = input_tensor_a.shape;

            auto input_tensor_a_4D = ttnn::unsqueeze_to_4D(input_tensor_a);
            const auto& ttl_input_tensor_a = input_tensor_a_4D.value;

            auto ttl_output = tt::tt_metal::add_unary(ttl_input_tensor_a, input_tensor_b, memory_config);
            auto output = ttnn::Tensor(ttl_output);
            return ttnn::reshape(output, original_shape);
        },
        py::arg("input_tensor_a"),
        py::arg("input_tensor_b"),
        py::kw_only(),
        py::arg("memory_config") = DRAM_MEMORY_CONFIG);
}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
