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

}

std::vector<Shape> UpSample::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // NOTE: Only for RM
    // NOTE2: Assuming { N, 1, H * W, C }
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.shape().without_padding();

    uint32_t out_h = input_shape[2]*scale_factor_;
    uint32_t out_w = input_shape[3]*scale_factor_;
    // need to pad the last dim to TILE_WIDTH
    uint32_t out_c = input_shape[1];
    // const auto padding = Padding({{0, 0},
    //                               {0, 0},
    //                               {0, 0},
    //                               {0, 0},
    //                              Padding::PadValue::NegativeInfinity);
    const auto out_dims = std::vector<uint32_t>({ 1, out_c, out_h, out_w });
    auto out_shape = Shape{out_dims};

    return {out_shape};
}

std::vector<Tensor> UpSample::create_output_tensors(const std::vector<Tensor> &inputs) const {
    const auto& input = inputs.at(0);
    return operation::generic_create_output_tensors(*this, inputs, input.dtype(), input.layout(), output_mem_config);

}

 operation::ProgramWithCallbacks UpSample::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
     //const auto& input = inputs.at(0);
     //auto& output = outputs.at(0);
     std::cout << "Log testing entering into upsample create_program" << std::endl;
     const Tensor& input_tensor_0 = input_tensors.at(0);
     Tensor& output_tensor_0 = output_tensors.at(0);
     return upsample_single_core(input_tensor_0, output_tensor_0);
     //return {};
}

UpSampleParallelizationStrategy UpSample::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    /*const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(0);
    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    if(num_tiles > 1 || input_tensor_a.memory_config().is_sharded() || input_tensor_b.memory_config().is_sharded() || this->output_mem_config.is_sharded()){
        return BinaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{*/
       return UpSampleParallelizationStrategy::SINGLE_CORE;
    //}
}

Tensor upsample(const Tensor &input,
                float scale_factor,
                const MemoryConfig& out_mem_config,
                bool use_multicore) {
    // calculate the H and W dims for output
    std::cout << "Log testing entering into upsample tensor" << std::endl;
    //uint32_t out_h = in_h * scale_factor;
    //uint32_t out_w = in_w * scale_factor;   // floor
    return operation::run_without_autoformat(UpSample{scale_factor,
                                                      out_mem_config,
                                                      use_multicore},
                                              {input}).at(0);
}

} // namespace tt_metal
} // namespace tt
