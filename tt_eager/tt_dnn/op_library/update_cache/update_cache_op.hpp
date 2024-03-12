// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include "third_party/magic_enum/magic_enum.hpp"

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class UpdateCacheOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

enum class UpdateCacheOpType {
    FILL = 0, UPDATE = 1
};

operation::ProgramWithCallbacks update_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t update_idx, const uint32_t batch_offset);
operation::ProgramWithCallbacks update_cache_single_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t update_idx, const uint32_t batch_offset);
operation::ProgramWithCallbacks fill_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t batch_idx, const uint32_t update_idx);
operation::ProgramWithCallbacks fill_cache_single_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t batch_idx, const uint32_t update_idx);

struct UpdateCache {
    const uint32_t batch_idx;
    const uint32_t update_idx;
    const uint32_t batch_offset;
    const UpdateCacheOpType op_type;

    UpdateCacheOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;


    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;

    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors) const;
};

inline Tensor fill_cache(const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t batch_idx) {
    operation::run(UpdateCache{batch_idx, 0, 0, UpdateCacheOpType::FILL}, {cache_tensor, input_tensor});
    return cache_tensor;
}

inline Tensor update_cache(const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t update_idx, const uint32_t batch_offset) {
    operation::run(UpdateCache{0, update_idx, batch_offset, UpdateCacheOpType::UPDATE}, {cache_tensor, input_tensor});
    return cache_tensor;
}

}  // namespace tt_metal

}  // namespace tt
