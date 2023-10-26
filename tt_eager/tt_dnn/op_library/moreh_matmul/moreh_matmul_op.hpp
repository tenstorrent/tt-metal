/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {
namespace tt_metal {

Tensor moreh_matmul(const Tensor &input_tensor, const Tensor &other_tensor, const MemoryConfig &mem_config);

}  // namespace tt_metal

namespace operations {

namespace primary {

/*
 * GENERAL MATMUL
 */
tt_metal::operation::ProgramWithCallbacks moreh_matmul_multi_core(
    const tt_metal::Tensor &input_tensor,
    const tt_metal::Tensor &other_tensor,
    const tt_metal::Tensor &output_tensor,
    bool transpose_input,
    bool transpose_other,
    uint32_t input_start_tile_id,
    uint32_t other_start_tile_id,
    uint32_t output_start_tile_id);

struct MorehMatmul {
    bool transpose_input;
    bool transpose_other;
    uint32_t input_start_tile_id = 0;
    uint32_t other_start_tile_id = 0;
    uint32_t output_start_tile_id = 0;
    void validate(const std::vector<tt_metal::Tensor> &input_tensors) const;
    std::vector<tt_metal::Shape> compute_output_shapes(const std::vector<tt_metal::Tensor> &input_tensors) const;
    std::vector<tt_metal::Tensor> create_output_tensors(const std::vector<tt_metal::Tensor> &input_tensors) const;
    tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<tt_metal::Tensor> &input_tensors, std::vector<tt_metal::Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

tt_metal::Tensor moreh_matmul(
    const tt_metal::Tensor &input_tensor,
    const tt_metal::Tensor &other_tensor,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> output_tensor = std::nullopt,
    bool transpose_input = false,
    bool transpose_other = false,
    const tt_metal::MemoryConfig &mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
