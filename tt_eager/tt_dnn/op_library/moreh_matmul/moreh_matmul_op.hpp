/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

/*
 * GENERAL MATMUL
 */
operation::ProgramWithCallbacks moreh_matmul_multi_core(
    const Tensor &input_tensor,
    const Tensor &other_tensor,
    const Tensor &output_tensor,
    bool transpose_input,
    bool transpose_other,
    uint32_t input_start_tile_id,
    uint32_t other_start_tile_id,
    uint32_t output_start_tile_id);

struct MorehMatmul {
    bool transpose_a;
    bool transpose_b;
    uint32_t a_start_tile_id = 0;
    uint32_t b_start_tile_id = 0;
    uint32_t output_start_tile_id = 0;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

Tensor moreh_matmul(
    const Tensor &input_tensor,
    const Tensor &other_tensor,
    std::optional<std::reference_wrapper<const Tensor>> output_tensor = std::nullopt,
    bool transpose_input = false,
    bool transpose_other = false,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
