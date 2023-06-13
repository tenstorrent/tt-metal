#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

using u32 = std::uint32_t;
using namespace tt::constants;

namespace transpose_op_utils {

using namespace tt::tt_metal;

TransposeOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, TransposeOpDim::Enum transpose_dim){
    auto ashape = a.shape();
    uint32_t num_tiles = a.volume() / TILE_HW;
    if (transpose_dim == TransposeOpDim::WH && num_tiles > 1) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
    } else if (transpose_dim == TransposeOpDim::HC && num_tiles > 1) { // Always true for legal shape until requirement on tile size IO is no longer required
        return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
    } else {
        return TransposeOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace transpose_op_utils

namespace tt {

namespace tt_metal {

void Transpose::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto shape = input_tensor_a.shape();
    u32 W = shape[3], H = shape[2], C = shape[3], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
    if (this->dim == TransposeOpDim::HC) {
        TT_ASSERT(C % TILE_HEIGHT == 0);
    }
}


std::vector<Shape> Transpose::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto out_shape = input_tensor_a.shape();
    switch (this->dim){
        case TransposeOpDim::CN:
            out_shape[0] = input_tensor_a.shape()[1];
            out_shape[1] = input_tensor_a.shape()[0];
            break;
        case TransposeOpDim::HC:
            out_shape[1] = input_tensor_a.shape()[2];
            out_shape[2] = input_tensor_a.shape()[1];
            break;
        case TransposeOpDim::WH:
            out_shape[2] = input_tensor_a.shape()[3];
            out_shape[3] = input_tensor_a.shape()[2];
            break;
    }
    return {out_shape};
}


std::vector<Tensor> Transpose::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return detail::generic_create_output_tensors(*this, input_tensors);
}

Program Transpose::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);

    switch (transpose_op_utils::get_parallelization_strategy(input_tensor_a, this->dim)) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            return transpose_wh_multi_core(input_tensor_a, output_tensor);
            break;
        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            return transpose_hc_multi_core(input_tensor_a, output_tensor);
            break;
        default:
            return transpose_single_core(input_tensor_a, output_tensor, this->dim);
    }
}

Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim) {

    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    switch (transpose_dim) {
        case TransposeOpDim::CN:
            if (a.shape()[0] == 1 && a.shape()[1] == 1) {
                return a;
            }
            break;
        case TransposeOpDim::HC:
            if (a.shape()[1] == 1 && a.shape()[2] == 1) {
                return a;
            }
            break;
        case TransposeOpDim::WH:
            if (a.shape()[2] == 1 && a.shape()[3] == 1) {
                return a;
            }
    }

    return detail::run_with_autopad(Transpose(transpose_dim), a, 0, transpose_dim == TransposeOpDim::HC);
}

}  // namespace tt_metal

}  // namespace tt
