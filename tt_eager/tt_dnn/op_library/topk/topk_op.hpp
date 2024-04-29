// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/run_operation.hpp"
#include "ttnn/operations/core.hpp"

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks single_core_topk_interleaved(const Tensor &input_tensor, const uint32_t k, Tensor &value_tensor, Tensor &index_tensor);

struct TopK {
    uint32_t k;
    MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline std::tuple<Tensor, Tensor> topk(const Tensor &input_tensor, const uint32_t k, const MemoryConfig &output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor})),
                                        Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [k, output_mem_config] (std::vector<Tensor> input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_input_tensors_const) mutable -> std::vector<Tensor> {

            return operation::run(TopK{k, output_mem_config}, input_tensors);

        }, {input_tensor}, output_tensors);
    return {output_tensors.at(0), output_tensors.at(1)};
}

} // namespace tt_metal
} // namespace tt
