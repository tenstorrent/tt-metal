// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

namespace all_gather_buffer_params {
    constexpr uint32_t num_buffers = 2;
    constexpr uint32_t semaphore_size = 32; // TODO: Remove this once dedicated semaphore space for user kernels are added
    constexpr uint32_t semaphore_offset = semaphore_size * num_buffers; // TODO: Remove this once dedicated semaphore space for user kernels are added
    constexpr uint32_t eth_buffer_size = round_down((eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE - semaphore_offset) / num_buffers, 32);
    constexpr uint32_t eth_sem_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    constexpr uint32_t eth_buffer_l1_byte_address = eth_sem_l1_byte_address + semaphore_offset;
    static_assert(num_buffers <= eth_l1_mem::address_map::MAX_NUM_CHANNELS);
    static_assert(all_gather_buffer_params::eth_buffer_size * all_gather_buffer_params::num_buffers + all_gather_buffer_params::semaphore_offset <= eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);
}

struct AllGather {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const uint32_t ring_index;
    const chip_id_t receiver_device_id;
    const chip_id_t sender_device_id;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks all_gather_multi_core_with_workers(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const chip_id_t receiver_device_id, const chip_id_t sender_device_id);

std::vector<Tensor> all_gather(const std::vector<Tensor> &input_tensors, const uint32_t dim, const uint32_t num_links = 1, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
