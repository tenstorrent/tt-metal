// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

enum class BcastOpMath { ADD = 0, SUB = 1, MUL = 2 };

enum class BcastOpDim { H = 0, W = 1, HW = 2 };

// TODO: Accept parallelization
enum class BcastOpParallelizationStrategy { MULTI_CORE_H = 0, MULTI_CORE_W = 1, MULTI_CORE_HW = 2, SINGLE_CORE = 3 };

operation::ProgramWithCallbacks bcast_single_core(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    Tensor &output_tensor,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim);
operation::ProgramWithCallbacks bcast_multi_core_h(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    Tensor &output_tensor,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim);
operation::ProgramWithCallbacks bcast_multi_core_w(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    Tensor &output_tensor,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim);
operation::ProgramWithCallbacks bcast_multi_core_hw(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    Tensor &output_tensor,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim);

struct EltwiseBinaryBroadcast {
    const BcastOpMath math_op;
    const BcastOpDim dim;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    BcastOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("math_op", "dim", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->math_op), std::cref(this->dim), std::cref(this->output_mem_config));
    }

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

inline Tensor bcast(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    if (bcast_dim == BcastOpDim::W) {
        TT_FATAL(input_tensor_a.get_legacy_shape()[2] == input_tensor_b.get_legacy_shape()[2]);
        if (input_tensor_b.get_layout() == Layout::TILE) {
            TT_FATAL(input_tensor_b.get_legacy_shape()[3] == TILE_WIDTH);
        } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
            TT_FATAL(input_tensor_b.get_legacy_shape()[3] == 1 || input_tensor_b.get_legacy_shape()[3] == TILE_WIDTH);
        } else {
            TT_FATAL(false, "Unsupported layout");
        }
    } else if (bcast_dim == BcastOpDim::H) {
        TT_FATAL(input_tensor_a.get_legacy_shape()[3] == input_tensor_b.get_legacy_shape()[3]);
        if (input_tensor_b.get_layout() == Layout::TILE) {
            TT_FATAL(input_tensor_b.get_legacy_shape()[2] == TILE_HEIGHT);
        } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
            TT_FATAL(input_tensor_b.get_legacy_shape()[2] == 1 || input_tensor_b.get_legacy_shape()[2] == TILE_HEIGHT);
        } else {
            TT_FATAL(false, "Unsupported layout");
        }
    } else if (bcast_dim == BcastOpDim::HW) {
        if (input_tensor_b.get_layout() == Layout::TILE) {
            TT_FATAL(
                input_tensor_b.get_legacy_shape()[2] == TILE_HEIGHT &&
                input_tensor_b.get_legacy_shape()[3] == TILE_WIDTH);
        } else if (input_tensor_b.get_layout() == Layout::ROW_MAJOR) {
            TT_FATAL(
                (input_tensor_b.get_legacy_shape()[2] == 1 && input_tensor_b.get_legacy_shape()[3] == 1) ||
                (input_tensor_b.get_legacy_shape()[2] == TILE_HEIGHT &&
                 input_tensor_b.get_legacy_shape()[3] == TILE_WIDTH));
        }
    }
    return operation::run_with_autoformat(
               EltwiseBinaryBroadcast{bcast_op, bcast_dim, output_mem_config}, {input_tensor_a, input_tensor_b})
        .at(0);
}

}  // namespace tt_metal

namespace operations {

namespace primary {

inline Tensor bcast(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return operation::run(EltwiseBinaryBroadcast{bcast_op, bcast_dim, mem_config}, {input_tensor_a, input_tensor_b})
        .at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt

namespace bcast_op_utils {

using namespace tt::tt_metal;

const char *get_reader_name(BcastOpDim bcast_dim, BcastOpParallelizationStrategy bcast_parallelization_strategy);

const char *get_compute_name(BcastOpDim bcast_dim);

const char *get_math_to_op_define(BcastOpMath bcast_math);

std::map<std::string, std::string> get_defines(BcastOpDim bcast_dim, BcastOpMath bcast_math);

}  // namespace bcast_op_utils
