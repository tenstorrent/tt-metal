// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "tt_dnn/op_library/upsample/upsample_op.hpp"
#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"   // for reduce_op_utils
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "detail/util.hpp"

namespace tt {
namespace tt_metal {

void UpSample::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to copy need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
}

std::vector<Shape> UpSample::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // NOTE1: data is packed in { N, H , W, C }
    // NOTE2: Mapping it into in 2D format should be {N*H*W, C}
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.shape().without_padding();

    uint32_t out_n = input_shape[0];
    uint32_t out_h = input_shape[1] * scale_factor_h_;
    uint32_t out_w = input_shape[2] * scale_factor_w_;
    uint32_t out_c = input_shape[3];
    const auto out_dims = std::vector<uint32_t>({ out_n, out_h, out_w, out_c }); //in the NHWC format
    auto out_shape = Shape{out_dims};

    return {out_shape};
}

std::vector<Tensor> UpSample::create_output_tensors(const std::vector<Tensor> &inputs) const {
    const auto& input = inputs.at(0);
    return operation::generic_create_output_tensors(*this, inputs, input.dtype(), input.layout(), output_mem_config);
}

 operation::ProgramWithCallbacks UpSample::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
     const Tensor& input_tensor_0 = input_tensors.at(0);
     Tensor& output_tensor_0 = output_tensors.at(0);
     return upsample_single_core(input_tensor_0, output_tensor_0, scale_factor_h_, scale_factor_w_);
}

UpSampleParallelizationStrategy UpSample::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if(use_multicore){
      TT_FATAL(!use_multicore, "multicore is not yet supported for upsample");
    }
    return UpSampleParallelizationStrategy::SINGLE_CORE;
}

Tensor upsample(const Tensor &input,
                int scale_factor_h,
                int scale_factor_w,
                const MemoryConfig& out_mem_config,
                bool use_multicore) {
    return operation::run_without_autoformat(UpSample{scale_factor_h,
                                                      scale_factor_w,
                                                      out_mem_config,
                                                      use_multicore},
                                              {input}).at(0);
}

} // namespace tt_metal
} // namespace tt
