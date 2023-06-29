#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/split/split_tiled.hpp"
#include "tt_metal/host_api.hpp"
namespace tt {

namespace tt_metal {

struct SplitLastDimQKTiled : public SplitTiled {
    // setting dim = 3 (last dim)
    // num_chunks = 2
    SplitLastDimQKTiled(const MemoryConfig& mem_config) : SplitTiled{3, 2, mem_config} { ; }
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
};

std::vector<Tensor> split_last_dim_qk_tiled(const Tensor &a, const MemoryConfig& mem_config);

}  // namespace tt_metal

}  // namespace tt
