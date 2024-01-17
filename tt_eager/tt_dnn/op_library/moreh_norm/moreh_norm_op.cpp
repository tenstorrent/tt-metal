// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {}  // namespace

void MorehNorm::validate(const std::vector<Tensor> &input_tensors) const {}

std::vector<Shape> MorehNorm::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehNorm::create_output_tensors(const std::vector<Tensor> &) const { return {}; }

// Tensor moreh_norm_h(const Tensor &input, float p, int64_t dim) {
//     const auto input_shape = input.shape();
//     const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

//     std::vector<uint32_t> reduced_shape;
//     reduced_shape.reserve(input_rank);

//     for (decltype(dim) i = 0; i < input_rank; ++i) {
//         if (i == input_rank - 2) {
//             reduced_shape.push_back(TILE_WIDTH);
//         } else {
//             reduced_shape.push_back(input_shape[i]);
//         }
//     }

//     // const auto h_padding = input_shape.padding()[input_rank - 2].back;
//     const auto w_padding = input_shape.padding()[input_rank - 1].back;

//     Padding padding{{{0, 0}, {0, 0}, {0, TILE_HEIGHT - 1}, {0, w_padding}}, Padding::PadValue::Zero};
//     Shape output_shape{reduced_shape, padding};

//     const auto &output = create_device_tensor(output_shape, input.dtype(), Layout::TILE, input.device());

//     operation::run(MorehNorm{.p = p, .dim = dim}, {input, output}, {});
//     return output;
// }

Tensor moreh_norm_w(const Tensor &input, float p, int64_t dim) {
    const auto input_shape = input.shape();
    const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

    std::vector<uint32_t> reduced_shape;
    reduced_shape.reserve(input_rank);

    for (decltype(dim) i = 0; i < input_rank; ++i) {
        if (i == input_rank - 1) {
            reduced_shape.push_back(TILE_HEIGHT);
        } else {
            reduced_shape.push_back(input_shape[i]);
        }
    }

    const auto h_padding = input_shape.padding()[input_rank - 2].back;
    // const auto w_padding = input_shape.padding()[input_rank - 1].back;

    Padding padding{{{0, 0}, {0, 0}, {0, h_padding}, {0, TILE_WIDTH - 1}}, Padding::PadValue::Zero};
    Shape output_shape{reduced_shape, padding};

    const auto &output = create_device_tensor(output_shape, input.dtype(), Layout::TILE, input.device());

    operation::run(MorehNorm{.p = p, .dim = dim}, {input, output});
    // return input;
    return output;

    // operation::run(MorehNorm{.p = p, .dim = dim}, {input});
    // return input;
}

// Tensor moreh_norm_other(const Tensor &input, float p, int64_t dim) {
//     const auto input_shape = input.shape();
//     const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

//     std::vector<uint32_t> reduced_shape;
//     reduced_shape.reserve(input_rank);

//     for (decltype(dim) i = 0; i < input_rank; ++i) {
//         if (i == dim) {
//             reduced_shape.push_back(1);
//         } else {
//             reduced_shape.push_back(input_shape[i]);
//         }
//     }

//     const auto padding = input_shape.padding();
//     Shape output_shape{reduced_shape, padding};

//     const auto &output = create_device_tensor(output_shape, input.dtype(), Layout::TILE, input.device());

//     operation::run(MorehNorm{.p = p, .dim = dim}, {input, output}, {});
//     return output;
// }

Tensor moreh_norm(const Tensor &input, float p, int64_t dim) {
    const auto input_shape = input.shape();
    const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

    return moreh_norm_w(input, p, dim);

    // if (dim == input_rank - 1) {
    //     return moreh_norm_w(input, p, dim);
    // } else if (dim == input_rank - 2) {
    //     return moreh_norm_h(input, p, dim);
    // } else {
    //     return moreh_norm_other(input, p, dim);
    // }
}

operation::ProgramWithCallbacks MorehNorm::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    const auto &output_tensor = input_tensors.at(1);

    const auto reduction_dim = this->dim;

    const auto input_shape = input_tensor.shape();
    const auto input_rank = static_cast<decltype(reduction_dim)>(input_shape.rank());

    return moreh_norm_w_impl(input_tensor, this->p, reduction_dim, output_tensor);

    // return moreh_norm_w_impl(input_tensor, this->p, reduction_dim, output_tensor);

    // if (reduction_dim == input_rank - 1) {
    //     return moreh_norm_w_impl(input_tensor, this->p, reduction_dim, output_tensor);
    // } else if (reduction_dim == input_rank - 2) {
    //     return moreh_norm_h_impl(input_tensor, this->p, reduction_dim, output_tensor);
    // } else {
    //     return moreh_norm_other_impl(input_tensor, this->p, reduction_dim, output_tensor);
    // }
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
