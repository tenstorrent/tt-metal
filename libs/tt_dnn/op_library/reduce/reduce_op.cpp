#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

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

} // namespace reduce_op_utils
namespace tt {
namespace tt_metal {

void Reduce::validate(const std::vector<Tensor> &input_tensors) const {}

std::vector<Shape> Reduce::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

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

std::vector<Tensor> Reduce::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors);
}

operation::ProgramWithCallbacks Reduce::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy){
        case ReduceOpParallelizationStrategy::MULTI_CORE_H:
            return {reduce_multi_core_h(input_tensor, output_tensor, this->math_op, this->dim, this->scaler)};
        case ReduceOpParallelizationStrategy::MULTI_CORE_W:
            return {reduce_multi_core_w(input_tensor, output_tensor, this->math_op, this->dim, this->scaler)};
        case ReduceOpParallelizationStrategy::SINGLE_CORE:
        default:
            return {reduce_single_core(input_tensor, output_tensor, this->math_op, this->dim, this->scaler)};
    }

}

operation::Hash Reduce::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    return fmt::format(
        "reduce_{}_{}_{}_{}",
         magic_enum::enum_name(this->math_op),
         magic_enum::enum_name(this->dim),
         this->scaler,
         operation::hash_tensor(input_tensor)
    );
}

ReduceOpParallelizationStrategy::Enum Reduce::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    auto shape = input_tensor.shape();
    uint32_t Wt = shape[3]/TILE_WIDTH;
    uint32_t Ht = shape[2]/TILE_HEIGHT;
    uint32_t NC = shape[1]*shape[0];
    if(NC * Wt > 1 and this->dim == ReduceOpDim::H){
        return ReduceOpParallelizationStrategy::MULTI_CORE_H;
    }else if(NC * Ht > 1 and this->dim == ReduceOpDim::W){
        return ReduceOpParallelizationStrategy::MULTI_CORE_W;
    }else if(num_tiles > 1 and this->dim == ReduceOpDim::HW){
        return ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    }else{
        return ReduceOpParallelizationStrategy::SINGLE_CORE;
    }
}

Tensor reduce(const Tensor &input_tensor, ReduceOpMath::Enum reduce_math, ReduceOpDim::Enum reduce_dim, float scaler) {
    auto parallelization_strategy = Reduce{reduce_math, reduce_dim, scaler}.get_parallelization_strategy({input_tensor});
    auto is_multicore_hw = parallelization_strategy == ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    float pad_value = reduce_math == ReduceOpMath::MAX ? std::numeric_limits<float>::lowest() : 0;
    if (is_multicore_hw) {
        Device * device;

        // Get the device
        if (input_tensor.storage_type() == StorageType::HOST) {
            device = AutoFormat::GetDefaultDevice();
            TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
        } else {
            device = input_tensor.device();
        }
        auto input_tensor_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape());

        if (AutoFormat::check_input_tensor_format(input_tensor, input_tensor_pad_shape)) {
            const Tensor output_tensor = operation::run_without_autoformat(Reduce{reduce_math, ReduceOpDim::W, scaler}, input_tensor);
            return operation::run_without_autoformat(Reduce{reduce_math, ReduceOpDim::H, scaler}, output_tensor);
        } else {
            // We only need to format the input tensor in this case, no need to format the output tensor
            const Tensor output_tensor = operation::run_without_autoformat(Reduce{reduce_math, ReduceOpDim::W, scaler}, AutoFormat::format_input_tensor(input_tensor, device, input_tensor_pad_shape, pad_value));
            return operation::run_without_autoformat(Reduce{reduce_math, ReduceOpDim::H, scaler}, output_tensor);
        }
    } else {
        return operation::run_with_autoformat(Reduce{reduce_math, reduce_dim, scaler}, input_tensor, pad_value);
    }
}

}  // namespace tt_metal

}  // namespace tt
