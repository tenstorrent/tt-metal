// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

struct IndexedFill {
    const MemoryConfig output_mem_config;
    const int64_t dim;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks indexed_fill_multi_core(const Tensor &batch_ids, const Tensor &input_a, const Tensor& input_b, const Tensor &output);


Tensor indexed_fill(const Tensor &batch_ids, const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config, std::int64_t dim=0);


}  // namespace tt_metal

}  // namespace tt
