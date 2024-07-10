// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/common/constants.hpp"

namespace tt {

namespace tt_metal {

struct UnpadOnHost {
    const Shape output_tensor_start;
    const Shape output_tensor_end;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;
};

Tensor unpad_on_host(const Tensor &input_tensor, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
