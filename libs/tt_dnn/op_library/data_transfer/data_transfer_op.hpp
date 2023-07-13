#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct DataTransferToHost {
    Host * host;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;
};

Tensor data_transfer_to_host (const Tensor &input_tensor, Host* host);

struct DataTransferToDevice {
    Device* device;
    const MemoryConfig mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;
};

Tensor data_transfer_to_device (const Tensor &input_tensor, Device* device, const MemoryConfig mem_config);

}  // namespace tt_metal

}  // namespace tt
