// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace bcast_op_utils {
using namespace tt::tt_metal;
using namespace tt::constants;

// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
const char* get_reader_name(BcastOpDim bcast_dim, BcastOpParallelizationStrategy bcast_parallelization_strategy) {
    if (bcast_parallelization_strategy == BcastOpParallelizationStrategy::SINGLE_CORE) {
        if (bcast_dim == BcastOpDim::H) {
            return "tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_h_interleaved.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_w_interleaved.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_hw_interleaved.cpp";
        }
    }
    else {
        if (bcast_dim == BcastOpDim::H) {
            return "tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_h_interleaved_input_rows_partitioned.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_w_interleaved_input_cols_partitioned.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_hw_interleaved_partitioned.cpp";
        }
    }
    TT_ASSERT(false && "Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastOpDim bcast_dim) {
    switch (bcast_dim) {
        case BcastOpDim::H:  return "tt_eager/tt_dnn/op_library/bcast/kernels/compute/bcast_h.cpp";
        case BcastOpDim::W:  return "tt_eager/tt_dnn/op_library/bcast/kernels/compute/bcast_w.cpp";
        case BcastOpDim::HW: return "tt_eager/tt_dnn/op_library/bcast/kernels/compute/bcast_hw.cpp";
        default:  TT_ASSERT(false && "Unexpected bcast_dim!");
    }
    return "";
}

std::map<std::string, std::string> get_defines(BcastOpDim bcast_dim, BcastOpMath bcast_math)
{
    std::map<std::string, std::string> defines;
    const char* math_to_op_define[] = { "add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast" };
    const char* math_to_llkop_define[] = {"ELWADD", "ELWSUB", "ELWMUL"};
    const char* bdim_to_llkdim_define[] = { "BroadcastType::ROW", "BroadcastType::COL", "BroadcastType::SCALAR" };
    defines["BCAST_OP"] = math_to_op_define[int(bcast_math)];
    defines["BCAST_LLKOP"] = math_to_llkop_define[int(bcast_math)];
    defines["BCAST_DIM"] = bdim_to_llkdim_define[int(bcast_dim)];
    return defines;
}

} // namespace bcast_op_utils


using namespace tt::tt_metal;
using namespace tt::constants;
using uint32_t = std::uint32_t;


namespace tt {

namespace tt_metal {

void EltwiseBinaryBroadcast::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    TT_FATAL(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to bcast need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() != nullptr and input_tensor_b.device() != nullptr, "Operands to bcast need to be on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to bcast need to be on the same device!");

    const auto input_shape_a = input_tensor_a.get_legacy_shape();
    const auto input_shape_b = input_tensor_b.get_legacy_shape();

    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE);
    TT_FATAL(input_tensor_b.get_layout() == Layout::TILE);
    TT_FATAL(input_tensor_a.get_dtype() == input_tensor_b.get_dtype());
    TT_FATAL(input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED && input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED && this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Bcast does not currently support sharding");

    auto batch_size_a = input_shape_a[0];
    auto num_channels_a = input_shape_a[1];
    auto height_a = input_shape_a[2];
    auto width_a = input_shape_a[3];
    auto batch_size_b = input_shape_b[0];
    auto num_channels_b = input_shape_b[1];
    auto height_b = input_shape_b[2];
    auto width_b = input_shape_b[3];

    TT_FATAL((batch_size_b * num_channels_b == 1 || (batch_size_b == batch_size_a && num_channels_b == num_channels_a)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");

    // validate input dimensions
    if (this->dim == BcastOpDim::W)
        TT_FATAL(height_a == height_b && width_b == TILE_WIDTH);
    if (this->dim == BcastOpDim::H)
        TT_FATAL(width_a == width_b && height_b == TILE_HEIGHT);
    if (this->dim == BcastOpDim::HW)
        TT_FATAL(width_b == TILE_WIDTH && height_b == TILE_HEIGHT);
}


std::vector<Shape> EltwiseBinaryBroadcast::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}


std::vector<Tensor> EltwiseBinaryBroadcast::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks EltwiseBinaryBroadcast::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy){
        case BcastOpParallelizationStrategy::MULTI_CORE_H:
            return bcast_multi_core_h(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
        case BcastOpParallelizationStrategy::MULTI_CORE_W:
            return bcast_multi_core_w(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
        case BcastOpParallelizationStrategy::MULTI_CORE_HW:
            return bcast_multi_core_hw(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
        case BcastOpParallelizationStrategy::SINGLE_CORE:
        default:
            return bcast_single_core(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->dim);
    }
}

const operation::Hash EltwiseBinaryBroadcast::compute_program_hash(
    const std::vector<Tensor> &input_tensors) const {
    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);
    bool bcast_scalar = (input_tensors.at(1).get_legacy_shape()[-2] * input_tensors.at(1).get_legacy_shape()[-1] == 1) && this->dim == BcastOpDim::HW;
    return operation::hash_operation<EltwiseBinaryBroadcast>(
        *this,
        parallelization_strategy,
        input_tensors.at(0).memory_config(),
        input_tensors.at(0).get_dtype(),
        input_tensors.at(1).memory_config(),
        input_tensors.at(1).get_dtype(),
        bcast_scalar);
}

BcastOpParallelizationStrategy EltwiseBinaryBroadcast::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);

    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    uint32_t Ht = input_tensor_a.get_legacy_shape()[2] / TILE_HEIGHT;
    uint32_t Wt = input_tensor_a.get_legacy_shape()[3] / TILE_WIDTH;

    if(Ht > 1 and this->dim == BcastOpDim::H){
        return BcastOpParallelizationStrategy::MULTI_CORE_H;
    }
    else if(Wt > 1 and this->dim == BcastOpDim::W){
        return BcastOpParallelizationStrategy::MULTI_CORE_W;
    }
    else if(num_tiles > 1 and this->dim == BcastOpDim::HW){
        return BcastOpParallelizationStrategy::MULTI_CORE_HW;
    }
    else{
        return BcastOpParallelizationStrategy::SINGLE_CORE;
    }
}

}  // namespace tt_metal

}  // namespace tt
