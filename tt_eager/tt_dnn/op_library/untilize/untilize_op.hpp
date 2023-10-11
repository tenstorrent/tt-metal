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

enum class UntilizeOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct Untilize {
    const MemoryConfig output_mem_config;
    const bool use_multicore;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    UntilizeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

enum class UntilizeWithUnpaddingOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct UntilizeWithUnpadding {
    const Shape output_tensor_start;
    const Shape output_tensor_end;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    UntilizeWithUnpaddingOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks untilize_multi_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks untilize_single_core(const Tensor &a, Tensor& output);
operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);
operation::ProgramWithCallbacks untilize_with_unpadding_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end);

Tensor untilize (const Tensor &a, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, bool use_multicore = false);
Tensor untilize_with_unpadding(const Tensor &a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// NOTE: UntilizeWithHalo is only for sharded input/output
struct UntilizeWithHalo {
    const uint32_t pad_val_;
    const uint32_t in_b;
    const uint32_t in_h;
    const uint32_t in_w;
    const uint32_t out_shard_size_max_per_core;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};
Tensor untilize_with_halo(const Tensor &a, const uint32_t pad_val, const uint32_t &in_b, const uint32_t &in_h, const uint32_t &in_w, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

namespace untilize_helpers {
uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);
}

}  // namespace tt_metal

}  // namespace tt
