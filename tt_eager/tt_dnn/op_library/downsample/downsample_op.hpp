// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

struct Downsample {
    std::array<uint32_t, 5> downsample_params;
    DataType output_dtype;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
};

//operation::ProgramWithCallbacks downsample_multi_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks downsample_single_core(const Tensor &a, std::array<uint32_t, 5> downsample_params, Tensor& output);

Tensor downsample (const Tensor &a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> output_dtype=std::nullopt);

// namespace downsample_helpers {
// uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);
// }

}  // namespace tt_metal

}  // namespace tt
