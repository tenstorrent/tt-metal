// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/repeat/repeat_op.hpp"

#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_eager/tt_dnn/op_library/pad/pad_op.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

RepeatOpParallelizationStrategy Repeat::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    std::cout<<"\n\n enetr parallel";
    return RepeatOpParallelizationStrategy::MULTI_CORE;
}

void Repeat::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    std::cout<<"\n\n enter validate";
    const auto &input_tensor = input_tensors[0];
    tt::tt_metal::Shape input_shape = input_tensor.get_legacy_shape();
    TT_FATAL(this->repeat_dim < input_shape.rank(), "Repeat dim specified is larger than input tensor rank.");

    if (input_tensor.get_layout() == Layout::ROW_MAJOR && this->repeat_dim == input_shape.rank() - 1) {
        TT_FATAL(
            (input_shape[this->repeat_dim] * input_tensor.element_size()) % ADDRESS_ALIGNMENT == 0,
            "Current repeat implementation requires aligned last dim when repeating on last dim");
    }
    TT_FATAL(this->num_repeats > 0, "Number of repeats should be greater than 0");
    TT_FATAL(input_tensor.buffer(), "Operand to repeat needs to be allocated in a buffer on device.");
    TT_FATAL(input_tensor.device(), "Operand to repeat needs to be on device.");
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Input to repeat must be interleaved.");
    // if(!output_tensors.empty() && output_tensors.at(0).has_value()){
    //     const auto output_shape_required = this->compute_output_shapes(input_tensors);u
    //     const auto& out_tensor = output_tensors.at(0).value();
    //     TT_FATAL(out_tensor.get_legacy_shape() == output_shape_required.at(0), fmt::format("The input tensors need a shape of {}, however the output tensor is only {}", output_shape_required,  out_tensor.get_legacy_shape()));
    // }
    auto out_mem_config = (!output_tensors.empty() && output_tensors.at(0).has_value()) ? output_tensors.at(0).value().memory_config() : this->output_mem_config;
    TT_FATAL(out_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Output of repeat must be interleaved.");
    std::cout<<"\n\n exit validate";

}

std::vector<tt::tt_metal::Shape> Repeat::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    std::cout<<"\n\n enetr compute shape";
    tt::tt_metal::Shape shape_out = input_tensors[0].get_legacy_shape();
    std::cout<<"\n\n out shape"<<shape_out;
    shape_out[this->repeat_dim] *= this->num_repeats;
    std::cout<<"\n\n exit compute shape"<<shape_out;
    return {shape_out};
}

std::vector<Tensor> Repeat::create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const Tensor &ref_in_tensor = input_tensors[0];
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        std::cout<<"\n\nusing optional";
        const Shape start_index = {0, 0, 0, 0};
        auto endss = ref_in_tensor.get_shape();
        const Shape end_index = {endss[0]-1, endss[1]-1, endss[2]-1, endss[3]-1};
        std::cout<<"\n\n endsss"<<endss[0]<<endss[1]<<endss[2]<<endss[3];
        // const Shape end_index = {1, 1, 32, 32};
        auto qwe = input_tensors[0];

        // auto new_unpad_tensor = pad(output_tensors.at(0).value(), start_index, end_index);
        auto new_unpad_tensor = unpad(output_tensors.at(0).value(), start_index, end_index);
        auto &ref_out_tensor = output_tensors[0];
        // auto new_unpad_tensor = unpad(qwe, start_index, end_index);
        std::cout<<"\n\n new shape"<<new_unpad_tensor.shape();
        return {new_unpad_tensor};
    }
    std::cout<<"\n\nqwreryeru"<<ref_in_tensor.get_shape();
    log_debug(tt::LogOp, "\n\n inputss{}", operation::generic_create_output_tensors(
        *this, input_tensors, ref_in_tensor.get_dtype(), ref_in_tensor.get_layout(), this->output_mem_config));
    return operation::generic_create_output_tensors(
        *this, input_tensors, ref_in_tensor.get_dtype(), ref_in_tensor.get_layout(), this->output_mem_config);
    // return inputs;
}

operation::ProgramWithCallbacks Repeat::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    // log_debug(tt::LogOp, "opt_output_tensor is not empty ");
    std::cout<<"\n\nenter create  program";
    switch (this->get_parallelization_strategy(input_tensors)) {
        case RepeatOpParallelizationStrategy::MULTI_CORE:
            std::cout<<"\n\n not enter defult MULTI_CORE ";
        default:
            std::cout<<"\n\n enter default";
            return repeat_multi_core(input_tensors[0], this->repeat_dim, this->num_repeats, output_tensors[0]);
    };
}

Tensor repeat(const Tensor &input_tensor, const Shape &shape, const MemoryConfig &output_mem_config, std::optional<Tensor> output_tensor) {
    std::cout<<"\n\n start of launch";
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [shape, output_mem_config, output_tensor] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            uint32_t input_rank = input_tensor.get_legacy_shape().rank();
            TT_FATAL(shape.rank() == input_rank, "Number of repeat dims must be equal to number of tensor dims");
            Tensor output = input_tensor;
            // int loop = 0;
            // int loop_counter = 1;
            // for(int i=0; i<shape.rank(); i++)
            // {
            //     if(shape[i] > 1)
            //     {
            //         loop = loop + 1;
            //     }
            // }
            for (uint32_t dim = 0; dim < shape.rank(); ++dim) {
                // std::cout<<"\n\n count loop"<<loop;
                if (shape[dim] == 1) {
                    continue;
                }
                TT_FATAL(shape[dim] > 0, "Number of repetitions along a dim must be greater than 0");
                if (input_tensor.get_layout() == Layout::ROW_MAJOR && dim == input_rank - 1) {
                    TT_FATAL(
                        (input_tensor.get_legacy_shape()[dim] * input_tensor.element_size()) % ADDRESS_ALIGNMENT == 0,
                        "Current repeat implementation requires aligned last dim when repeating on last dim");
                }
                    output = operation::run_without_autoformat(Repeat{dim, shape[dim], output_mem_config}, {output},{},{output_tensor}).at(0);
                // if(loop == loop_counter){
                //     if(output_tensor.has_value())
                //     {
                //     output = operation::run_without_autoformat(Repeat{dim, shape[dim], output_mem_config}, {output},{},{output_tensor}).at(0);
                //     }
                //     else{
                //         output = operation::run_without_autoformat(Repeat{dim, shape[dim], output_mem_config}, {output}).at(0);
                //     }
                // }
                // else{
                //     output = operation::run_without_autoformat(Repeat{dim, shape[dim], output_mem_config}, {output}).at(0);
                //     loop_counter = loop_counter + 1;
                // }
            }
            // if(loop == 0)
            // {
            //     if(output_tensor.has_value())
            //     {
            //         assign(input_tensor, output_tensor.value());
            //     }
            // }
            return {output};
        }, {input_tensor}, output_tensors, {}, {output_tensor});
    std::cout<<"\n\n end of launch";
    return output_tensors.at(0);
}



// with cq_id
Tensor repeat(uint8_t cq_id, const Tensor &input_tensor, const Shape &shape, const MemoryConfig &output_mem_config, std::optional<Tensor> output_tensor) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    operation::launch_op(
        [shape, output_mem_config, output_tensor, cq_id] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);
            uint32_t input_rank = input_tensor.get_legacy_shape().rank();
            TT_FATAL(shape.rank() == input_rank, "Number of repeat dims must be equal to number of tensor dims");
            Tensor output = input_tensor;
            for (uint32_t dim = 0; dim < shape.rank(); ++dim) {
                if (shape[dim] == 1) {
                    continue;
                }
                TT_FATAL(shape[dim] > 0, "Number of repetitions along a dim must be greater than 0");
                if (input_tensor.get_layout() == Layout::ROW_MAJOR && dim == input_rank - 1) {
                    TT_FATAL(
                        (input_tensor.get_legacy_shape()[dim] * input_tensor.element_size()) % ADDRESS_ALIGNMENT == 0,
                        "Current repeat implementation requires aligned last dim when repeating on last dim");
                }
                output = operation::run_without_autoformat(Repeat{dim, shape[dim], output_mem_config}, {output}, {}, {output_tensor}, cq_id).at(0);
            }
            return {output};
        }, {input_tensor}, output_tensors, {}, {output_tensor});
    return output_tensors.at(0);
}


}  // namespace tt_metal

}  // namespace tt
