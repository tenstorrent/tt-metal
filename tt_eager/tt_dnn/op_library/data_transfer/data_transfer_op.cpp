// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/data_transfer/data_transfer_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include <fmt/ranges.h>

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void DataTransferToHost::validate(const std::vector<Tensor> &input_tensors) const {
}
std::vector<Shape> DataTransferToHost::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}
std::vector<Tensor> DataTransferToHost::compute_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return {input_tensor};
    } else {
        return {input_tensor.cpu() };
    }
}

tt::stl::reflection::Attributes DataTransferToHost::attributes() const {
    return {};
}

Tensor data_transfer_to_host(const Tensor &input_tensor) {
    return operation::run(DataTransferToHost(), {input_tensor}).at(0);
}

void DataTransferToDevice::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    switch (input_tensor.get_layout()) {
        case Layout::ROW_MAJOR: TT_FATAL(input_tensor.get_legacy_shape()[-1] * input_tensor.element_size() % sizeof(uint32_t) == 0); break;
        default: break;
    }
}
std::vector<Shape> DataTransferToDevice::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}
std::vector<Tensor> DataTransferToDevice::compute_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.storage_type() == StorageType::DEVICE && input_tensor.device() == this->device) {
        return {input_tensor};
    } else {
        return {input_tensor.to(this->device, this->mem_config)};
    }
}

tt::stl::reflection::Attributes DataTransferToDevice::attributes() const {
    return {
        {"device", this->device->id()},
        {"mem_config", this->mem_config},
    };
}

Tensor data_transfer_to_device(const Tensor &input_tensor, Device* device, const MemoryConfig &mem_config) {
    return operation::run(DataTransferToDevice{device, mem_config}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
