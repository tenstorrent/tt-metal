#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"
#include <limits>

using namespace tt::constants;

namespace reduce_op_utils {

using namespace tt::tt_metal;

string dim_to_kernel_name(ReduceOpDim::Enum reduce_dim, ReduceOpMath::Enum reduce_op){
    string kernel_name;
    switch(reduce_dim){
        case ReduceOpDim::H: kernel_name = "tt_metal/kernels/compute/reduce_h.cpp"; break;
        case ReduceOpDim::W: kernel_name = "tt_metal/kernels/compute/reduce_w.cpp"; break;
        case ReduceOpDim::HW: kernel_name = "tt_metal/kernels/compute/reduce_hw.cpp"; break;
        default: TT_ASSERT(false && "Undefined dim");
    }
    return kernel_name;
}

void add_defines(ComputeKernel * reduce_kernel, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim){
    // TOOD(AP): need a sync with Reduce::Max from HLK headers
    bool do_max = reduce_op == ReduceOpMath::MAX;
    reduce_kernel->add_define("REDUCE_OP", do_max ? "PoolType::MAX" : "PoolType::SUM");
    switch(reduce_dim) {
        case ReduceOpDim::W: reduce_kernel->add_define("REDUCE_DIM", "ReduceDim::REDUCE_ROW"); break;
        case ReduceOpDim::H: reduce_kernel->add_define("REDUCE_DIM", "ReduceDim::REDUCE_COL"); break;
        case ReduceOpDim::HW: reduce_kernel->add_define("REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"); break;
        default: TT_ASSERT(false && "Invalid reduce_op!");
    }
    return;
}

ReduceOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, ReduceOpDim::Enum reduce_dim){
    uint32_t num_tiles = a.volume() / TILE_HW;
    auto shape = a.shape();
    uint32_t Wt = shape[3]/TILE_WIDTH;
    uint32_t Ht = shape[2]/TILE_HEIGHT;
    uint32_t NC = shape[1]*shape[0];
    if(NC * Wt > 1 and reduce_dim == ReduceOpDim::H){
        return ReduceOpParallelizationStrategy::MULTI_CORE_H;
    }else if(NC * Ht > 1 and reduce_dim == ReduceOpDim::W){
        return ReduceOpParallelizationStrategy::MULTI_CORE_W;
    }else if(num_tiles > 1 and reduce_dim == ReduceOpDim::HW){
        return ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    }else{
        return ReduceOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace reduce_op_utils
namespace tt {
namespace tt_metal {

 std::vector<Shape> Reduce::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();

    auto output_shape = input_tensor.shape();
    switch (this->dim){
        case ReduceOpDim::H:
            output_shape[2] = 32;
            break;
        case ReduceOpDim::W:
            output_shape[3] = 32;
            break;
        case ReduceOpDim::HW:
            output_shape[2] = 32;
            output_shape[3] = 32;
            break;

    }
    return {output_shape};
}

std::vector<Tensor> Reduce::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    const auto output_shape = compute_output_shapes(input_tensors).at(0);
    std::vector<Tensor> output_tensors;
    output_tensors.emplace_back(tt_metal::Tensor(output_shape, input_tensor.dtype(), tt::tt_metal::Layout::TILE, input_tensor.device()));
    return output_tensors;
}

Program Reduce::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);

    switch (reduce_op_utils::get_parallelization_strategy(input_tensor, this->dim)){
        case ReduceOpParallelizationStrategy::MULTI_CORE_H:
            return reduce_multi_core_h(input_tensor, output_tensor, this->math_op, this->dim, this->scaler);
        case ReduceOpParallelizationStrategy::MULTI_CORE_W:
            return reduce_multi_core_w(input_tensor, output_tensor, this->math_op, this->dim, this->scaler);
        case ReduceOpParallelizationStrategy::SINGLE_CORE:
        default:
            return reduce_single_core(input_tensor, output_tensor, this->math_op, this->dim, this->scaler);
    }

}

Tensor reduce_(const Tensor &input_tensor, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler) {
    auto parallelization_strategy = reduce_op_utils::get_parallelization_strategy(input_tensor, reduce_dim);
    auto is_multicore_hw = parallelization_strategy == ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    if (is_multicore_hw) {
        const Tensor output_tensor = std::move(Reduce(reduce_math, ReduceOpDim::W, scaler).run({std::cref(input_tensor)}).at(0));
        return std::move(Reduce(reduce_math, ReduceOpDim::H, scaler).run({std::cref(output_tensor)}).at(0));
    } else {
        return std::move(Reduce(reduce_math, reduce_dim, scaler).run({std::cref(input_tensor)}).at(0));
    }
}

Tensor reduce(const Tensor &input_tensor, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler) {
    Device * device;
    if (input_tensor.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto padded_input_shape = AutoPad::pad_to_tile_shape(input_tensor.shape());
    auto output_shape = Reduce(reduce_math, reduce_dim, scaler).compute_output_shapes({std::cref(input_tensor)}).at(0);

    if (AutoPad::check_input_tensor_format(input_tensor, padded_input_shape)) {
        return reduce_(input_tensor, reduce_math, reduce_dim, scaler);
    } else {
        auto output = reduce_(AutoPad::format_input_tensor(input_tensor, device, padded_input_shape, 0), reduce_math, reduce_dim, scaler);
        AutoPad::format_output_tensor(input_tensor, output, output_shape, device);
        return output;
    }
}

}  // namespace tt_metal

}  // namespace tt
