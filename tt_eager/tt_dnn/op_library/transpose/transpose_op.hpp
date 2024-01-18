// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class TransposeOpDim {
    WH = 0, HC = 1, CN = 2, NH = 3, NW = 4, CW = 5
};

enum class TransposeOpParallelizationStrategy {
    MULTI_CORE_WH = 0, MULTI_CORE_HC = 1, MULTI_CORE_CN = 2, SINGLE_CORE = 3
};

struct Transpose {
    const TransposeOpDim dim;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    TransposeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors) const;
};

// TODO: Accept parallelization
Tensor transpose_(const Tensor &a, TransposeOpDim transpose_dim, const MemoryConfig& output_mem_config);

// transpose with tensor and dimensions
Tensor transpose(const Tensor &a, std::int64_t dim1, std::int64_t dim2, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

operation::ProgramWithCallbacks transpose_single_core(const Tensor &a, Tensor &output, TransposeOpDim transpose_dim);
operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_cn_multi_core(const Tensor &a, Tensor &output);

}  // namespace tt_metal

}  // namespace tt
