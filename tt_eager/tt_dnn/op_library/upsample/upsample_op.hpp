#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {
namespace tt_metal {

enum class UpSampleParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct UpSample{
    const int scale_factor_h_;
    const int scale_factor_w_;
    const MemoryConfig output_mem_config;
    const bool use_multicore;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    UpSampleParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "scale_factor_h",
        "scale_factor_w",
        "output_mem_config",
        "use_multicore");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->scale_factor_h_),
            std::cref(this->scale_factor_w_),
            std::cref(this->output_mem_config),
            std::cref(this->use_multicore));
    }
};

Tensor upsample(const Tensor &input,
                  int scale_factor_h,
                  int scale_factor_w,
                  const MemoryConfig& out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                  bool use_multicore = false);

operation::ProgramWithCallbacks upsample_single_core(const Tensor &a, Tensor& output, int scale_factor_h_, int scale_factor_w_);

}  // namespace tt_metal
}  // namespace tt
