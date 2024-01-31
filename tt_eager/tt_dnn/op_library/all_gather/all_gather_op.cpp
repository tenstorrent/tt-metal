// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include "eth_l1_address_map.h"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt {

namespace tt_metal {

void AllGather::validate(const std::vector<Tensor> &input_tensors) const {
    constexpr uint32_t header_size = 32;
    constexpr uint32_t semaphore_offset = 32;
    constexpr uint32_t MAX_BUFFER = round_down((eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE - semaphore_offset) / 2, 32) - header_size;
    TT_FATAL(input_tensors.size() == 1);
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size <= MAX_BUFFER, "Page size too large");
    TT_FATAL(page_size % 32 == 0);

    // TODO: Validate ring
    TT_FATAL(input_tensors[0].storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensors[0].buffer() != nullptr , "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(input_tensors[0].memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    if (this->receiver_device_id == this->sender_device_id) {
        TT_FATAL(input_tensors[0].device()->get_ethernet_sockets(this->receiver_device_id).size() >= 2, "2 Device ring requires at least 2 eth connections for ring gather");
    } else {
        TT_FATAL(input_tensors[0].device()->get_ethernet_sockets(this->receiver_device_id).size() >= 1, "Requires at least 1 eth connection between sender and receiver device");
        TT_FATAL(input_tensors[0].device()->get_ethernet_sockets(this->sender_device_id).size() >= 1, "Requires at least 1 eth connection between sender and receiver device");
    }
}

std::vector<Shape> AllGather::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto shape = input_tensors[0].shape();
    shape[this->dim] *= this->ring_size;
    return std::vector<Shape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGather::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks AllGather::create_program(const std::vector<Tensor> & input_tensors, std::vector<Tensor> &output_tensors) const {
    return all_gather_multi_core(input_tensors.at(0), output_tensors.at(0), this->dim, this->ring_size, this->ring_index, this->receiver_device_id, this->sender_device_id);
}

tt::stl::reflection::Attributes AllGather::attributes() const {
    return {
        {"dim", this->dim},
        {"ring_size", this->ring_size},
        {"ring_index", this->ring_index},
        {"receiver_device_id", this->receiver_device_id},
        {"sender_device_id", this->sender_device_id},
        {"output_mem_config", this->output_mem_config},
    };
}

std::vector<Tensor> all_gather(const std::vector<Tensor>& input_tensors, uint32_t dim, const MemoryConfig& output_mem_config) {

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());

    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        chip_id_t receiver_device_id = input_tensors[(i + 1) % input_tensors.size()].device()->id();
        chip_id_t sender_device_id = input_tensors[i == 0 ? input_tensors.size() - 1 : i - 1].device()->id();
        output_tensors.push_back(operation::run(AllGather{dim, (uint32_t)input_tensors.size(), i, receiver_device_id, sender_device_id, output_mem_config}, {input_tensors.at(i)}).at(0));
    }
    return output_tensors;
}

}  // namespace tt_metal

}  // namespace tt
