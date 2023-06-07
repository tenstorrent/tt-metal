#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace bcast_op_utils {
using namespace tt::tt_metal;
using namespace tt::constants;

// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
const char* get_reader_name(BcastOpDim::Enum bcast_dim, BcastOpParallelizationStrategy::Enum bcast_parallelization_strategy) {
    if (bcast_parallelization_strategy == BcastOpParallelizationStrategy::SINGLE_CORE) {
        if (bcast_dim == BcastOpDim::H) {
            return "tt_metal/kernels/dataflow/reader_bcast_h_8bank.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "tt_metal/kernels/dataflow/reader_bcast_w_8bank.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "tt_metal/kernels/dataflow/reader_bcast_hw_8bank.cpp";
        }
    }
    else {
        if (bcast_dim == BcastOpDim::H) {
            return "tt_metal/kernels/dataflow/reader_bcast_h_8bank_input_rows_partitioned.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "tt_metal/kernels/dataflow/reader_bcast_w_8bank_input_cols_partitioned.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "tt_metal/kernels/dataflow/reader_bcast_hw_8bank_partitioned.cpp";
        }
    }
    TT_ASSERT(false && "Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastOpDim::Enum bcast_dim) {
    switch (bcast_dim) {
        case BcastOpDim::H:  return "tt_metal/kernels/compute/bcast_h.cpp";
        case BcastOpDim::W:  return "tt_metal/kernels/compute/bcast_w.cpp";
        case BcastOpDim::HW: return "tt_metal/kernels/compute/bcast_hw.cpp";
        default:  TT_ASSERT(false && "Unexpected bcast_dim!");
    }
    return "";
}

void add_defines(ComputeKernel* k, BcastOpDim::Enum bcast_dim, BcastOpMath::Enum bcast_math)
{
    const char* math_to_op_define[] = { "add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast" };
    const char* math_to_llkop_define[] = {"ELWADD", "ELWSUB", "ELWMUL"};
    const char* bdim_to_llkdim_define[] = { "BroadcastType::ROW", "BroadcastType::COL", "BroadcastType::SCALAR" };
    k->add_define("BCAST_OP", math_to_op_define[int(bcast_math)]);
    k->add_define("BCAST_LLKOP", math_to_llkop_define[int(bcast_math)]);
    k->add_define("BCAST_DIM", bdim_to_llkdim_define[int(bcast_dim)]);
}

BcastOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &input_tensor_a, BcastOpDim::Enum bcast_dim){
    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    uint32_t Ht = input_tensor_a.shape()[2] / TILE_HEIGHT;
    uint32_t Wt = input_tensor_a.shape()[3] / TILE_WIDTH;

    if(Ht > 1 and bcast_dim == BcastOpDim::H){
        return BcastOpParallelizationStrategy::MULTI_CORE_H;
    }
    else if(Wt > 1 and bcast_dim == BcastOpDim::W){
        return BcastOpParallelizationStrategy::MULTI_CORE_W;
    }
    else if(num_tiles > 1 and bcast_dim == BcastOpDim::HW){
        return BcastOpParallelizationStrategy::MULTI_CORE_HW;
    }
    else{
        return BcastOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace bcast_op_utils


using namespace tt::tt_metal;
using namespace tt::constants;
using u32 = std::uint32_t;


namespace tt {

namespace tt_metal {

void EltwiseBinaryBroadcast::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();

    const auto input_shape_a = input_tensor_a.shape();;
    const auto input_shape_b = input_tensor_b.shape();

    if (this->dim == BcastOpDim::W) {
        TT_ASSERT(input_shape_a[2] == input_shape_b[2]);
    }
    else if (this->dim == BcastOpDim::H) {
        TT_ASSERT(input_shape_a[3] == input_shape_b[3]);
    }

    auto batch_size_a = input_shape_a[0];
    auto num_channels_a = input_shape_a[1];
    auto height_a = input_shape_a[2];
    auto width_a = input_shape_a[3];
    auto batch_size_b = input_shape_b[0];
    auto num_channels_b = input_shape_b[1];
    auto height_b = input_shape_b[2];
    auto width_b = input_shape_b[3];

    TT_ASSERT((batch_size_b * num_channels_b == 1 || (batch_size_b == batch_size_a && num_channels_b == num_channels_a)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");

    TT_ASSERT(width_a % TILE_WIDTH == 0 && height_a % TILE_HEIGHT == 0);
    TT_ASSERT(height_a > 0 && width_a > 0 && batch_size_a * num_channels_a > 0);
    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);

    // validate input dimensions
    if (this->dim == BcastOpDim::W)
        TT_ASSERT(height_a == height_b && width_b == TILE_WIDTH);
    if (this->dim == BcastOpDim::H)
        TT_ASSERT(width_a == width_b && height_b == TILE_HEIGHT);
    if (this->dim == BcastOpDim::HW)
        TT_ASSERT(width_b == TILE_WIDTH && height_b == TILE_HEIGHT);
}


std::vector<Shape> EltwiseBinaryBroadcast::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    return {input_tensor.shape()};
}


std::vector<Tensor> EltwiseBinaryBroadcast::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors);
}

operation::ProgramWithCallbacks EltwiseBinaryBroadcast::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = bcast_op_utils::get_parallelization_strategy(input_tensor_a, this->dim);

    op_profiler::set_parallelization_strategy(parallelization_strategy);

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

operation::Hash EltwiseBinaryBroadcast::compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();

    return fmt::format(
        "eltwise_binary_broadcast_{}_{}_{}_{}",
         magic_enum::enum_name(this->math_op),
         magic_enum::enum_name(this->dim),
         operation::hash_tensor(input_tensor_a),
         operation::hash_tensor(input_tensor_b)
    );
}

}  // namespace tt_metal

}  // namespace tt
