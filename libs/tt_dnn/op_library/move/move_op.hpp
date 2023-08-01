#pragma once

#include <optional>
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;


namespace tt {

namespace tt_metal {

enum class MoveOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct Move {
    const MemoryConfig output_mem_config;
    const MoveOpParallelizationStrategy move_op_parallelization_strategy;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks move_multi_core(const Tensor &input, Tensor &output);
operation::ProgramWithCallbacks move_single_core(const Tensor &input, Tensor &output);

inline Tensor move(Tensor& input_tensor, std::optional<MemoryConfig>& mem_config) {
    TT_ASSERT(input_tensor.is_allocated(), "Expected input tensor to be allocated");
    auto input_mem_config = input_tensor.memory_config();
    auto input_address = input_tensor.buffer()->address();
    auto output_mem_config = mem_config.value_or(input_mem_config);

    // TODO: This will cause problems if there's another reference to this buffer
    DeallocateBuffer(*input_tensor.buffer());
    auto output_tensor = create_device_tensor(input_tensor.shape(), input_tensor.dtype(), input_tensor.layout(), input_tensor.device(), output_mem_config);

    // get_parallelization_strategy
    uint32_t num_tiles = input_tensor.volume() / TILE_HW;

    bool non_overlap;
    const auto num_banks = input_tensor.device()->num_banks(output_tensor.buffer()->buffer_type());
    // If DRAM, inverse logic because memory is allocated bottom up
    if (input_mem_config.buffer_type == tt_metal::BufferType::DRAM)
        non_overlap = output_tensor.buffer()->address() + output_tensor.buffer()->size() / num_banks <= input_address;
    else
        non_overlap = input_address + output_tensor.buffer()->size() / num_banks <= output_tensor.buffer()->address();

    MoveOpParallelizationStrategy move_op_parallelization_strategy = MoveOpParallelizationStrategy::SINGLE_CORE;
    if (num_tiles > 1 and (input_mem_config.buffer_type != output_mem_config.buffer_type or non_overlap)) {
        move_op_parallelization_strategy = MoveOpParallelizationStrategy::MULTI_CORE;
    }

    auto output = operation::run(Move{output_mem_config, move_op_parallelization_strategy}, {input_tensor, output_tensor}).at(0);
    input_tensor.deallocate();
    return output;
}

}  // namespace tt_metal

}  // namespace tt
