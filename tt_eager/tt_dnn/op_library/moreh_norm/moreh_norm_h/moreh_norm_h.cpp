// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

// namespace {

// std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
//     auto floored_p = std::floor(p);
//     auto decimal = p - floored_p;
//     const bool p_is_negative = floored_p < 0.0f;
//     if (p_is_negative) {
//         floored_p = -floored_p;
//     }
//     return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
// }

// }  // namespace

operation::ProgramWithCallbacks moreh_norm_h_impl(const Tensor &input, float p, int64_t dim, const Tensor &output) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // auto device = input.device();
    auto program = CreateProgram();

    // ////////////////////////////////////////////////////////////////////////////
    // //                         Parameters Setup
    // ////////////////////////////////////////////////////////////////////////////
    // const auto input_shape = input.shape();
    // const auto input_rank = static_cast<decltype(dim)>(input_shape.rank());

    // const auto N = input_shape[0];
    // const auto C = input_shape[1];
    // const auto H = input_shape[2];
    // const auto W = input_shape[3];

    // const auto Ht = H / TILE_HEIGHT;
    // const auto Wt = W / TILE_WIDTH;

    // const auto origin_h = input_shape.without_padding()[input_rank - 2];

    // auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    // auto [floored_recip_p, recip_p_decimal, recip_p_is_negative] =
    //     get_floored_p_and_decimal_and_p_is_negative(1.0f / p);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback =
        [](const Program &program, const std::vector<Buffer *> &, const std::vector<Buffer *> &) {};

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
