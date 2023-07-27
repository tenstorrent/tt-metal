#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

struct BcastOpMath {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

struct BcastOpDim {
    enum Enum { H = 0, W = 1, HW = 2 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

// TODO: Accept parallelization
struct BcastOpParallelizationStrategy {
    enum Enum { MULTI_CORE_H = 0, MULTI_CORE_W = 1, MULTI_CORE_HW = 2, SINGLE_CORE = 3 };
    static const auto all() { return magic_enum::enum_values<Enum>(); }
};

operation::ProgramWithCallbacks bcast_single_core(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
operation::ProgramWithCallbacks bcast_multi_core_h(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
operation::ProgramWithCallbacks bcast_multi_core_w(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
operation::ProgramWithCallbacks bcast_multi_core_hw(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor& output_tensor, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);

struct EltwiseBinaryBroadcast {
    const BcastOpMath::Enum math_op;
    const BcastOpDim::Enum dim;
    const MemoryConfig& output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    BcastOpParallelizationStrategy::Enum get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor bcast(const Tensor &input_tensor_a, const Tensor &input_tensor_b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    if (bcast_dim == BcastOpDim::W) {
        TT_ASSERT(input_tensor_a.shape()[2] == input_tensor_b.shape()[2]);
        if (input_tensor_b.layout() == Layout::TILE) {
            TT_ASSERT(input_tensor_b.shape()[3] == TILE_WIDTH);
        } else if (input_tensor_b.layout() == Layout::ROW_MAJOR) {
            TT_ASSERT(input_tensor_b.shape()[3] == 1 || input_tensor_b.shape()[3] == TILE_WIDTH);
        } else {
            TT_ASSERT(false, "Unsupported layout");
        }
    }
    else if (bcast_dim == BcastOpDim::H) {
        TT_ASSERT(input_tensor_a.shape()[3] == input_tensor_b.shape()[3]);
        if (input_tensor_b.layout() == Layout::TILE) {
            TT_ASSERT(input_tensor_b.shape()[2] == TILE_HEIGHT);
        } else if (input_tensor_b.layout() == Layout::ROW_MAJOR) {
            TT_ASSERT(input_tensor_b.shape()[2] == 1 || input_tensor_b.shape()[2] == TILE_HEIGHT);
        } else {
            TT_ASSERT(false, "Unsupported layout");
        }
    } else if (bcast_dim == BcastOpDim::HW) {
        if (input_tensor_b.layout() == Layout::TILE) {
            TT_ASSERT(input_tensor_b.shape()[2] == TILE_HEIGHT && input_tensor_b.shape()[3] == TILE_WIDTH);
        } else if (input_tensor_b.layout() == Layout::ROW_MAJOR) {
            TT_ASSERT((input_tensor_b.shape()[2] == 1 && input_tensor_b.shape()[3] == 1) || (input_tensor_b.shape()[2] == TILE_HEIGHT && input_tensor_b.shape()[3] == TILE_WIDTH));
        }
    }
    return operation::run_with_autoformat(EltwiseBinaryBroadcast{bcast_op, bcast_dim, mem_config}, {input_tensor_a, input_tensor_b}).at(0);
}

inline Tensor bcast_without_autoformat(const Tensor &input_tensor_a, const Tensor &input_tensor_b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    return operation::run_without_autoformat(EltwiseBinaryBroadcast{bcast_op, bcast_dim, mem_config}, {input_tensor_a, input_tensor_b}).at(0);
}

}  // namespace tt_metal

}  // namespace tt

namespace bcast_op_utils {

using namespace tt::tt_metal;

const char* get_reader_name(BcastOpDim::Enum bcast_dim, BcastOpParallelizationStrategy::Enum bcast_parallelization_strategy);

const char* get_compute_name(BcastOpDim::Enum bcast_dim);

const char* get_math_to_op_define(BcastOpMath::Enum bcast_math);

void add_defines(ComputeKernel * bcast_kernel, BcastOpDim::Enum bcast_dim, BcastOpMath::Enum bcast_math);

} // namespace bcast_op_utils
