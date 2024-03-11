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


class AllGatherConfig {
   public:
    AllGatherConfig(Tensor const& input_tensor, uint32_t dim, uint32_t ring_size, uint32_t num_links) :
        semaphore_size(32),

        // enable_bidirectional - currently doesn't support batch dim and multi-link (some tests are flaky with those configs)
        erisc_handshake_address(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE),
        enable_bidirectional(dim != 0 && dim != 1)
    {
        constexpr uint32_t total_l1_buffer_space = eth_l1_mem::address_map::MAX_L1_LOADING_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

        this->num_buffers = this->enable_bidirectional ? 8 : 4;
        this->eth_sems_l1_base_byte_address = this->erisc_handshake_address + 16;
        this->semaphore_offset = this->semaphore_size * this->num_buffers; // TODO: Remove this once dedicated semaphore space for user kernels are added
        this->eth_buffers_l1_base_byte_address = this->eth_sems_l1_base_byte_address + this->semaphore_offset;

        uint32_t const page_size = input_tensor.buffer()->page_size();
        this->eth_buffer_size = round_down((total_l1_buffer_space - this->semaphore_offset) / this->num_buffers, page_size);

        TT_FATAL(eth_buffer_size == 0 or num_buffers <= eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS);
        TT_FATAL(this->eth_buffer_size * this->num_buffers + this->semaphore_offset <= total_l1_buffer_space);

        // FIXME: dynamically select the number and size of each buffer based on tensor attributes, link count, ring size, etc.
        // Erisc is able to keep up with workers up to around 17-20 GBps bidirectional (numbers still being locked down)
        // depending on payload size. In general, the smaller each eth buffer, the more overhead per send, and the larger each
        // buffer, the less overhead. However, larger buffers are less desirable for smaller tensors as the first send is delayed
        // until a larger percentage of the overall tensor has landed in the erisc buffer (which can impact latency negatively)
        // and for smaller tensors, latency can be more dominant than throughput with respect to end-to-end runtime.
        // Additionally, tensor layout and location can affect worker throughput. Based on loose empirical testing,
        // if workers are in RowMajor or DRAM tile layout (maybe L1 too - need to measure), it's preffered add more workers
        // (8) with smaller buffers so each worker can keep up with erisc.
    }

    uint32_t get_erisc_handshake_address() const { return this->erisc_handshake_address; }

    uint32_t get_semaphores_offset() const { return this->semaphore_offset; }
    uint32_t get_num_buffers() const { return this->num_buffers; }

    uint32_t get_eth_buffer_size() const { return this->eth_buffer_size; }

    uint32_t get_eth_sems_l1_base_byte_address() const { return this->eth_sems_l1_base_byte_address; }

    uint32_t get_eth_buffers_l1_base_byte_address() const { return this->eth_buffers_l1_base_byte_address; }

    uint32_t get_semaphore_size() const { return this->semaphore_size; }

    uint32_t get_num_buffers_in_clockwise_direction() const {
        return this->enable_bidirectional ?
            this->num_buffers / 2 :
            this->num_buffers;
    }
    bool is_buffer_in_clockwise_ring(const uint32_t buffer_index) const {
        // For now we split it as lower half => clockwise, upper half => counter-clockwise
        // This is slightly suboptimal since the non-full-chunks go to the upper half.
        // A more optimal split would be round robin
        return this->enable_bidirectional ?
            buffer_index < (this->num_buffers - get_num_buffers_in_clockwise_direction()) :
            true;
    }
    uint32_t get_num_buffers_in_counter_clockwise_direction() const {
        // return all_gather_buffer_params::enable_bidirectional ? all_gather_buffer_params::num_buffers - all_gather_buffer_params::num_buffers / 2 : 0;
        // Force all through counter-clockwise direction
        return this->num_buffers - this->get_num_buffers_in_clockwise_direction();
    }

    void print() const {
        std::stringstream ss;
        ss << "AllGatherConfig: {\n";
        ss << "\terisc_handshake_address: " << erisc_handshake_address << ",\n";
        ss << "\tnum_buffers: " << num_buffers << ",\n";
        ss << "\teth_buffer_size: " << eth_buffer_size << ",\n";
        ss << "\tsemaphore_size: " << semaphore_size << ",\n";
        ss << "\tsemaphore_offset: " << semaphore_offset << ",\n";
        ss << "\teth_buffers_l1_base_byte_address: " << eth_buffers_l1_base_byte_address << ",\n";
        ss << "\teth_sems_l1_base_byte_address: " << eth_sems_l1_base_byte_address << ",\n";
        ss << "\tenable_bidirectional: " << enable_bidirectional << "\n";
        ss << "}";
        std::cout << ss.str() << std::endl;
    }

   private:
    const uint32_t erisc_handshake_address;
    uint32_t num_buffers;
    uint32_t eth_buffer_size;
    uint32_t semaphore_size;
    uint32_t semaphore_offset;
    uint32_t eth_buffers_l1_base_byte_address;
    uint32_t eth_sems_l1_base_byte_address;
    const bool enable_bidirectional;
};

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
