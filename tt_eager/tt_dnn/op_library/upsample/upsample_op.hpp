#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {
namespace tt_metal {

enum class UpSampleParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct UpSample{
    //uint32_t in_n_; // nbatch
    //uint32_t in_h_, in_w_;
    const float scale_factor_;
    const MemoryConfig output_mem_config;
    //const int fake_value;
    const bool use_multicore;

    //uint32_t out_h_, out_w_;
    //const DataType output_dtype;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    UpSampleParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        /*"in_n",
        "in_h",
        "in_w",*/
        "scale_factor",
        "output_mem_config",
        "use_multicore");
    const auto attribute_values() const {
        return std::make_tuple(
            /*std::cref(this->in_n_),
            std::cref(this->in_h_),
            std::cref(this->in_w_),*/
            std::cref(this->scale_factor_),
            std::cref(this->output_mem_config),
            std::cref(this->use_multicore));
    }
};

Tensor upsample(const Tensor &input,
                  //uint32_t in_n, uint32_t in_h, uint32_t in_w,
                  float scale_factor,
                  const MemoryConfig& out_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                  bool use_multicore = false);

operation::ProgramWithCallbacks upsample_single_core(const Tensor &a, Tensor& output);

}  // namespace tt_metal
}  // namespace tt
