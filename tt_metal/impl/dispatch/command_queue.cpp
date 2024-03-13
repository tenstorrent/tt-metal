// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include <algorithm>  // for copy() and assign()
#include <iterator>   // for back_inserter
#include <memory>
#include <string>

#include "allocator/allocator.hpp"
#include "debug_tools.hpp"
#include "dev_msgs.h"
#include "logger.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::unique_ptr;

std::mutex finish_mutex;
std::condition_variable finish_cv;

namespace tt::tt_metal {

uint32_t get_noc_unicast_encoding(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

// EnqueueReadBufferCommandSection
EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    void* dst,
    bool stall,
    SystemMemoryManager& manager,
    uint32_t event,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id),
    dst(dst),
    stall(stall),
    manager(manager),
    buffer(buffer),
    src_page_index(src_page_index),
    pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {
    this->device = device;
    this->event = event;
}

const DeviceCommand EnqueueReadShardedBufferCommand::create_buffer_transfer_instruction(
    uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;

    TT_ASSERT(is_sharded(this->buffer.buffer_layout()));
    uint32_t buffer_address = this->buffer.address();
    uint32_t dst_page_index = 0;

    uint32_t num_cores = this->buffer.num_cores();
    uint32_t shard_size = this->buffer.shard_spec().size();

    auto core_host_page_indices = this->buffer.core_host_page_indices();
    vector<uint32_t> num_pages_in_shards;
    num_pages_in_shards.reserve(num_cores);
    for(size_t core_id = 0; core_id < num_cores; core_id++) {
        num_pages_in_shards.push_back(core_host_page_indices[core_id].size());
    }

    vector<uint32_t> core_id_x;
    core_id_x.reserve(num_cores);
    vector<uint32_t> core_id_y;
    core_id_y.reserve(num_cores);
    auto all_cores = this->buffer.all_cores();
    for (const auto& core : all_cores) {
        CoreCoord physical_core = this->device->worker_core_from_logical_core(core);
        core_id_x.push_back(physical_core.x);
        core_id_y.push_back(physical_core.y);
    }
    command.add_buffer_transfer_sharded_instruction(
        buffer_address,
        dst_address,
        num_pages,
        padded_page_size,
        (uint32_t)this->buffer.buffer_type(),
        uint32_t(BufferType::SYSTEM_MEMORY),
        this->src_page_index,
        dst_page_index,
        num_pages_in_shards,
        core_id_x,
        core_id_y);

    command.set_buffer_type(DeviceCommand::BufferType::SHARDED);
    command.set_sharded_buffer_num_cores(num_cores);
    return command;
}

const DeviceCommand EnqueueReadInterleavedBufferCommand::create_buffer_transfer_instruction(
    uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;
    TT_ASSERT(not is_sharded(this->buffer.buffer_layout()));

    uint32_t buffer_address = this->buffer.address();
    uint32_t dst_page_index = 0;
    command.add_buffer_transfer_interleaved_instruction(
        buffer_address,
        dst_address,
        num_pages,
        padded_page_size,
        (uint32_t)this->buffer.buffer_type(),
        uint32_t(BufferType::SYSTEM_MEMORY),
        this->src_page_index,
        dst_page_index);
    command.set_buffer_type(DeviceCommand::BufferType::INTERLEAVED);
    command.set_sharded_buffer_num_cores(1);
    return command;
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(uint32_t dst_address) {
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t num_pages = this->pages_to_read;
    DeviceCommand command = this->create_buffer_transfer_instruction(dst_address, padded_page_size, num_pages);

    // Commands are tunneled through ethernet to consumer tensix cores for remote fast dispatch
    bool route_through_ethernet = not device->is_mmio_capable();

    uint32_t push_and_pull_cb_num_pages;

    if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
        if (route_through_ethernet) {
            uint32_t router_data_buffer_size = get_cq_data_buffer_size(true);
            uint32_t router_cb_num_pages = router_data_buffer_size / padded_page_size;
            uint32_t router_cb_size = router_cb_num_pages * padded_page_size;
            TT_ASSERT(padded_page_size <= router_cb_size, "Page is too large to fit in consumer buffer");

            uint32_t producer_consumer_multiple = get_cq_data_buffer_size(false) / router_data_buffer_size;

            command.set_router_cb_num_pages(router_cb_num_pages);
            command.set_router_cb_size(router_cb_size);
            command.set_router_transfer_num_pages(router_cb_num_pages);

            push_and_pull_cb_num_pages = router_cb_num_pages * producer_consumer_multiple;
        } else {
            // pull and relay kernel reads in half of push_and_pull_cb_num_pages and writes are half the number of pages read
            uint32_t pull_and_push_data_buffer_size = get_cq_data_buffer_size(false);
            uint32_t num_pages_in_data_buffer = pull_and_push_data_buffer_size / padded_page_size;
            // make sure push_and_pull_cb_num_pages is multiple of 4 because we read half this number of pages and write a quarter of this number of pages
            push_and_pull_cb_num_pages = (num_pages_in_data_buffer / 4)* 4;
        }

        uint32_t push_and_pull_cb_size = push_and_pull_cb_num_pages * padded_page_size;
        TT_ASSERT(padded_page_size <= push_and_pull_cb_size, "Page is too large to fit in push and pull buffer");

        command.set_pull_and_push_cb_size(push_and_pull_cb_size);
        command.set_pull_and_push_cb_num_pages(push_and_pull_cb_num_pages);
    } else {
        uint32_t consumer_cb_num_pages = (get_consumer_data_buffer_size() / padded_page_size);

        // Number of pages that are transferred in one shot from producer to consumer
        uint32_t producer_consumer_tx_num_pages = 1;
        if (consumer_cb_num_pages >= DeviceCommand::SYNC_NUM_PAGES) {
            producer_consumer_tx_num_pages = consumer_cb_num_pages / DeviceCommand::SYNC_NUM_PAGES;
            consumer_cb_num_pages = producer_consumer_tx_num_pages * DeviceCommand::SYNC_NUM_PAGES; // want num pages to be previous multiple of SYNC_NUM_PAGES
        }
        command.set_producer_consumer_transfer_num_pages(producer_consumer_tx_num_pages);

        uint32_t consumer_cb_size = consumer_cb_num_pages * padded_page_size;
        TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");

        uint32_t producer_cb_num_pages = consumer_cb_num_pages * 2;
        uint32_t producer_cb_size = producer_cb_num_pages * padded_page_size;

        command.set_producer_cb_size(producer_cb_size);
        command.set_consumer_cb_size(consumer_cb_size);
        command.set_producer_cb_num_pages(producer_cb_num_pages);
        command.set_consumer_cb_num_pages(consumer_cb_num_pages);
    }

    if (this->stall) {
        command.set_stall();
    }
    command.set_page_size(padded_page_size);

    command.set_num_pages(num_pages);
    command.set_completion_data_size(padded_page_size * num_pages + align(EVENT_PADDED_SIZE, 32));
    command.set_event(this->event);

    return command;
}

void EnqueueReadBufferCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t read_buffer_addr =
        this->manager.get_next_completion_queue_write_ptr(this->command_queue_id) + align(EVENT_PADDED_SIZE, 32);
    const DeviceCommand cmd = this->assemble_device_command(read_buffer_addr);

    this->manager.issue_queue_reserve_back(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, this->command_queue_id);
    this->manager.cq_write(cmd.data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(
        DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, detail::LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}

// EnqueueWriteBufferCommand section
EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    const Buffer& buffer,
    const void* src,
    SystemMemoryManager& manager,
    uint32_t event,
    uint32_t dst_page_index,
    std::optional<uint32_t> pages_to_write) :
    command_queue_id(command_queue_id),
    manager(manager),
    src(src),
    buffer(buffer),
    dst_page_index(dst_page_index),
    pages_to_write(pages_to_write.has_value() ? pages_to_write.value() : buffer.num_pages()) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");
    this->device = device;
    this->event = event;
}

const DeviceCommand EnqueueWriteInterleavedBufferCommand::create_buffer_transfer_instruction(
    uint32_t src_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;

    TT_ASSERT(not is_sharded(this->buffer.buffer_layout()));

    uint32_t buffer_address = this->buffer.address();
    uint32_t src_page_index = 0;
    command.add_buffer_transfer_interleaved_instruction(
        src_address,
        buffer_address,
        num_pages,
        padded_page_size,
        (uint32_t)BufferType::SYSTEM_MEMORY,
        (uint32_t)this->buffer.buffer_type(),
        src_page_index,
        this->dst_page_index);
    command.set_buffer_type(DeviceCommand::BufferType::INTERLEAVED);
    return command;
}

const DeviceCommand EnqueueWriteShardedBufferCommand::create_buffer_transfer_instruction(
    uint32_t src_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;

    TT_ASSERT(is_sharded(this->buffer.buffer_layout()));
    uint32_t buffer_address = this->buffer.address();
    uint32_t src_page_index = 0;

    uint32_t num_cores = this->buffer.num_cores();
    uint32_t shard_size = this->buffer.shard_spec().size();

    auto core_host_page_indices = this->buffer.core_host_page_indices();
    vector<uint32_t> num_pages_in_shards;
    num_pages_in_shards.reserve(num_cores);
    for(size_t core_id = 0; core_id < num_cores; core_id++) {
        num_pages_in_shards.push_back(core_host_page_indices[core_id].size());
    }

    vector<uint32_t> core_id_x;
    core_id_x.reserve(num_cores);
    vector<uint32_t> core_id_y;
    core_id_y.reserve(num_cores);
    auto all_cores = this->buffer.all_cores();
    for (const auto& core : all_cores) {
        CoreCoord physical_core = this->device->worker_core_from_logical_core(core);
        core_id_x.push_back(physical_core.x);
        core_id_y.push_back(physical_core.y);
    }
    command.add_buffer_transfer_sharded_instruction(
        src_address,
        buffer_address,
        num_pages,
        padded_page_size,
        (uint32_t)BufferType::SYSTEM_MEMORY,
        (uint32_t)this->buffer.buffer_type(),
        src_page_index,
        this->dst_page_index,
        num_pages_in_shards,
        core_id_x,
        core_id_y);

    command.set_buffer_type(DeviceCommand::BufferType::SHARDED);
    command.set_sharded_buffer_num_cores(num_cores);

    return command;
}

const DeviceCommand EnqueueWriteBufferCommand::assemble_device_command(uint32_t src_address) {
    uint32_t num_pages = this->pages_to_write;
    uint32_t padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) {  // should buffer.size() be num_pages * page_size
        padded_page_size = align(this->buffer.page_size(), 32);
    }

    DeviceCommand command = this->create_buffer_transfer_instruction(src_address, padded_page_size, num_pages);

    if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
        // Commands are tunneled through ethernet to consumer tensix cores for remote fast dispatch
        bool route_through_ethernet = not device->is_mmio_capable();
        uint32_t push_and_pull_cb_num_pages;
        if (route_through_ethernet) {
            uint32_t router_data_buffer_size = get_cq_data_buffer_size(true);
            uint32_t router_cb_num_pages = router_data_buffer_size / padded_page_size;
            uint32_t router_cb_size = router_cb_num_pages * padded_page_size;
            TT_ASSERT(padded_page_size <= router_cb_size, "Page is too large to fit in consumer buffer");

            uint32_t producer_consumer_multiple = get_cq_data_buffer_size(false) / router_data_buffer_size;

            command.set_router_cb_num_pages(router_cb_num_pages);
            command.set_router_cb_size(router_cb_size);
            command.set_router_transfer_num_pages(router_cb_num_pages);

            push_and_pull_cb_num_pages = router_cb_num_pages * producer_consumer_multiple;
        }
        else {
            uint32_t pull_and_push_data_buffer_size = get_cq_data_buffer_size(false);
            uint32_t num_pages_in_data_buffer = pull_and_push_data_buffer_size / padded_page_size;
            // make sure push_and_pull_cb_num_pages is multiple of 4 because we read half this number of pages and write a quarter of this number of pages
            push_and_pull_cb_num_pages = (num_pages_in_data_buffer / 4)* 4;
        }

        uint32_t push_and_pull_cb_size = push_and_pull_cb_num_pages * padded_page_size;
        TT_ASSERT(padded_page_size <= push_and_pull_cb_size, "Page is too large to fit in push and pull buffer");

        command.set_pull_and_push_cb_size(push_and_pull_cb_size);
        command.set_pull_and_push_cb_num_pages(push_and_pull_cb_num_pages);
    } else {
        uint32_t consumer_cb_num_pages = get_consumer_data_buffer_size() / padded_page_size;

        // Number of pages that are transferred in one shot from producer to consumer
        uint32_t producer_consumer_tx_num_pages = 1;
        if (consumer_cb_num_pages >= DeviceCommand::SYNC_NUM_PAGES) {
            producer_consumer_tx_num_pages = consumer_cb_num_pages / DeviceCommand::SYNC_NUM_PAGES;
            consumer_cb_num_pages = producer_consumer_tx_num_pages * DeviceCommand::SYNC_NUM_PAGES; // want num pages to be previous multiple of SYNC_NUM_PAGES
        }
        command.set_producer_consumer_transfer_num_pages(producer_consumer_tx_num_pages);

        uint32_t consumer_cb_size = consumer_cb_num_pages * padded_page_size;
        TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");

        uint32_t producer_cb_num_pages = consumer_cb_num_pages * 2;
        uint32_t producer_cb_size = producer_cb_num_pages * padded_page_size;

        command.set_producer_cb_size(producer_cb_size);
        command.set_consumer_cb_size(consumer_cb_size);
        command.set_producer_cb_num_pages(producer_cb_num_pages);
        command.set_consumer_cb_num_pages(consumer_cb_num_pages);
    }

    command.set_page_size(padded_page_size);
    command.set_num_pages(num_pages);
    command.set_issue_data_size(padded_page_size * num_pages);
    command.set_completion_data_size(align(EVENT_PADDED_SIZE, 32));
    command.set_event(this->event);
    return command;
}

void EnqueueWriteBufferCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);
    uint32_t data_size_in_bytes = cmd.get_issue_data_size();

    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);

    this->manager.cq_write(cmd.data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        // If page size is not 32B-aligned, we cannot do a contiguous write
        uint32_t src_address_offset = unpadded_src_offset;
        uint32_t padded_page_size = align(this->buffer.page_size(), 32);
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_in_bytes;
             sysmem_address_offset += padded_page_size) {
            this->manager.cq_write(
                (char*)this->src + src_address_offset,
                this->buffer.page_size(),
                system_memory_temporary_storage_address + sysmem_address_offset);
            src_address_offset += this->buffer.page_size();
        }
    } else {
        this->manager.cq_write(
            (char*)this->src + unpadded_src_offset, data_size_in_bytes, system_memory_temporary_storage_address);
    }
    this->manager.issue_queue_push_back(cmd_size, detail::LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    const Program& program,
    SystemMemoryManager& manager,
    uint32_t event,
    bool stall,
    std::optional<std::reference_wrapper<Trace>> trace) :
    command_queue_id(command_queue_id),
    manager(manager),
    program(program),
    stall(stall) {
    this->device = device;
    this->trace = trace;
    this->event = event;
}

const DeviceCommand EnqueueProgramCommand::assemble_device_command(uint32_t host_data_src) {
    DeviceCommand command;
    command.set_num_workers(this->program.program_device_map.num_workers);

    const Program& program = this->program;

    auto populate_program_data_transfer_instructions =
        [&command](const vector<uint32_t>& num_transfers_per_page, const vector<transfer_info>& transfers_in_pages) {
            uint32_t i = 0;
            for (uint32_t j = 0; j < num_transfers_per_page.size(); j++) {
                uint32_t num_transfers_in_page = num_transfers_per_page[j];
                command.write_program_entry(num_transfers_in_page);
                for (uint32_t k = 0; k < num_transfers_in_page; k++) {
                    const auto [num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group, linked] =
                        transfers_in_pages[i];
                    command.add_write_page_partial_instruction(
                        num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group, linked);
                    i++;
                }
            }
        };

    command.set_is_program();

    // Not used, since we specified that this is a program command, and the consumer just looks at the write program
    // info
    constexpr static uint32_t dummy_dst_addr = 0;
    constexpr static uint32_t dummy_buffer_type = 0;
    uint32_t num_runtime_arg_pages =
        program.program_device_map.num_transfers_in_runtime_arg_pages.at(PageTransferType::MULTICAST).size();
    uint32_t num_cb_config_pages =
        program.program_device_map.num_transfers_in_cb_config_pages.at(PageTransferType::MULTICAST).size();
    uint32_t num_program_multicast_binary_pages =
        program.program_device_map.num_transfers_in_program_pages.at(PageTransferType::MULTICAST).size();
    uint32_t num_program_unicast_binary_pages =
        program.program_device_map.num_transfers_in_program_pages.at(PageTransferType::UNICAST).size();
    uint32_t num_go_signal_multicast_pages =
        program.program_device_map.num_transfers_in_go_signal_pages.at(PageTransferType::MULTICAST).size();
    uint32_t num_go_signal_unicast_pages =
        program.program_device_map.num_transfers_in_go_signal_pages.at(PageTransferType::UNICAST).size();
    uint32_t num_host_data_pages = num_runtime_arg_pages + num_cb_config_pages;
    uint32_t num_cached_pages = num_program_multicast_binary_pages + num_go_signal_multicast_pages +
                                num_program_unicast_binary_pages + num_go_signal_unicast_pages;
    uint32_t total_num_pages = num_host_data_pages + num_cached_pages;

    command.set_page_size(DeviceCommand::PROGRAM_PAGE_SIZE);
    command.set_num_pages(DeviceCommand::TransferType::RUNTIME_ARGS, num_runtime_arg_pages);
    command.set_num_pages(DeviceCommand::TransferType::CB_CONFIGS, num_cb_config_pages);
    command.set_num_pages(DeviceCommand::TransferType::PROGRAM_MULTICAST_PAGES, num_program_multicast_binary_pages);
    command.set_num_pages(DeviceCommand::TransferType::PROGRAM_UNICAST_PAGES, num_program_unicast_binary_pages);
    command.set_num_pages(DeviceCommand::TransferType::GO_SIGNALS_MULTICAST, num_go_signal_multicast_pages);
    command.set_num_pages(DeviceCommand::TransferType::GO_SIGNALS_UNICAST, num_go_signal_unicast_pages);
    command.set_num_pages(total_num_pages);
    command.set_completion_data_size(align(EVENT_PADDED_SIZE, 32));

    command.set_issue_data_size(DeviceCommand::PROGRAM_PAGE_SIZE * num_host_data_pages);

    const uint32_t page_index_offset = 0;
    if (num_host_data_pages) {
        command.add_buffer_transfer_interleaved_instruction(
            host_data_src,
            dummy_dst_addr,
            num_host_data_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            uint32_t(BufferType::SYSTEM_MEMORY),
            dummy_buffer_type,
            page_index_offset,
            page_index_offset);

        if (num_runtime_arg_pages) {
            populate_program_data_transfer_instructions(
                program.program_device_map.num_transfers_in_runtime_arg_pages.at(PageTransferType::MULTICAST),
                program.program_device_map.runtime_arg_page_transfers.at(PageTransferType::MULTICAST));
        }

        if (num_cb_config_pages) {
            populate_program_data_transfer_instructions(
                program.program_device_map.num_transfers_in_cb_config_pages.at(PageTransferType::MULTICAST),
                program.program_device_map.cb_config_page_transfers.at(PageTransferType::MULTICAST));
        }
    }

    if (num_cached_pages) {
        command.add_buffer_transfer_interleaved_instruction(
            program.buffer->address(),
            dummy_dst_addr,
            num_cached_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            uint32_t(program.buffer->buffer_type()),
            dummy_buffer_type, page_index_offset, page_index_offset);

        if (num_program_multicast_binary_pages) {
            populate_program_data_transfer_instructions(
                program.program_device_map.num_transfers_in_program_pages.at(PageTransferType::MULTICAST),
                program.program_device_map.program_page_transfers.at(PageTransferType::MULTICAST));
        }

        if (num_program_unicast_binary_pages) {
            populate_program_data_transfer_instructions(
                program.program_device_map.num_transfers_in_program_pages.at(PageTransferType::UNICAST),
                program.program_device_map.program_page_transfers.at(PageTransferType::UNICAST));
        }

        if (num_go_signal_multicast_pages) {
            populate_program_data_transfer_instructions(
                program.program_device_map.num_transfers_in_go_signal_pages.at(PageTransferType::MULTICAST),
                program.program_device_map.go_signal_page_transfers.at(PageTransferType::MULTICAST));
        }
        if (num_go_signal_unicast_pages) {
            populate_program_data_transfer_instructions(
                program.program_device_map.num_transfers_in_go_signal_pages.at(PageTransferType::UNICAST),
                program.program_device_map.go_signal_page_transfers.at(PageTransferType::UNICAST));
        }
    }

    if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
        uint32_t push_and_pull_cb_num_pages;
        bool route_through_ethernet = not this->device->is_mmio_capable();
        if (route_through_ethernet) {
            uint32_t router_data_buffer_size = get_cq_data_buffer_size(true);
            uint32_t router_cb_num_pages = router_data_buffer_size / DeviceCommand::PROGRAM_PAGE_SIZE;

            uint32_t router_transfer_num_pages = router_cb_num_pages / DeviceCommand::SYNC_NUM_PAGES;
            router_cb_num_pages = router_transfer_num_pages * DeviceCommand::SYNC_NUM_PAGES; // want num pages to be previous multiple of SYNC_NUM_PAGES

            uint32_t router_cb_size = router_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

            uint32_t producer_consumer_multiple = get_cq_data_buffer_size(false) / router_data_buffer_size;

            command.set_router_cb_num_pages(router_cb_num_pages);
            command.set_router_cb_size(router_cb_size);
            command.set_router_transfer_num_pages(router_transfer_num_pages);

            push_and_pull_cb_num_pages = router_cb_num_pages * producer_consumer_multiple / 2 * 2;
        } else {
            uint32_t pull_and_push_data_buffer_size = get_cq_data_buffer_size(false) / 2 * 2;
            push_and_pull_cb_num_pages = pull_and_push_data_buffer_size / DeviceCommand::PROGRAM_PAGE_SIZE;
        }

        const uint32_t push_and_pull_cb_size = push_and_pull_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

        command.set_pull_and_push_cb_size(push_and_pull_cb_size);
        command.set_pull_and_push_cb_num_pages(push_and_pull_cb_num_pages);
    } else {
        const uint32_t producer_cb_num_pages = get_cq_data_buffer_size(false) / DeviceCommand::PROGRAM_PAGE_SIZE;
        const uint32_t producer_cb_size = producer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

        bool cmd_consumer_on_ethernet = not device->is_mmio_capable();
        const uint32_t consumer_cb_num_pages = get_consumer_data_buffer_size() / DeviceCommand::PROGRAM_PAGE_SIZE;
        const uint32_t consumer_cb_size = consumer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

        command.set_producer_cb_size(producer_cb_size);
        command.set_consumer_cb_size(consumer_cb_size);
        command.set_producer_cb_num_pages(producer_cb_num_pages);
        command.set_consumer_cb_num_pages(consumer_cb_num_pages);
    }

    // Should only ever be set if we are
    // enqueueing a program immediately
    // after writing it to a buffer
    if (this->stall) {
        command.set_stall();
    }

    // This needs to be quite small, since programs are small
    command.set_producer_consumer_transfer_num_pages(DeviceCommand::SYNC_NUM_PAGES);
    command.set_program_transfer_num_pages(DeviceCommand::SYNC_NUM_PAGES);
    command.set_event(this->event);

    return command;
}

void EnqueueProgramCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);

    uint32_t data_size_in_bytes = cmd.get_issue_data_size();
    const uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    this->manager.cq_write(cmd.data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);

    bool tracing = this->trace.has_value() and not this->trace->get().trace_complete;
    vector<uint32_t> trace_host_data;
    uint32_t start_addr = system_memory_temporary_storage_address;
    constexpr static uint32_t padding_alignment = 16;
    for (size_t kernel_id = 0; kernel_id < this->program.num_kernels(); kernel_id++) {
        std::shared_ptr<Kernel> kernel = detail::GetKernel(program, kernel_id);
        for (const auto& c : kernel->cores_with_runtime_args()) {
            const auto& core_runtime_args = kernel->runtime_args(c);
            this->manager.cq_write(
                core_runtime_args.data(),
                core_runtime_args.size() * sizeof(uint32_t),
                system_memory_temporary_storage_address);

            system_memory_temporary_storage_address = align(
                system_memory_temporary_storage_address + core_runtime_args.size() * sizeof(uint32_t),
                padding_alignment);

            if (tracing) {
                trace_host_data.insert(trace_host_data.end(), core_runtime_args.begin(), core_runtime_args.end());
                trace_host_data.resize(align(trace_host_data.size(), padding_alignment / sizeof(uint32_t)));
            }
        }
    }

    system_memory_temporary_storage_address =
        start_addr + align(system_memory_temporary_storage_address - start_addr, DeviceCommand::PROGRAM_PAGE_SIZE);

    array<uint32_t, 4> cb_data;
    TT_ASSERT(cb_data.size() * sizeof(uint32_t) <= padding_alignment, "cb_data size exceeds padding_alignment");
    for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers()) {
        for (const auto buffer_index : cb->buffer_indices()) {
            cb_data = {
                cb->address() >> 4,
                cb->size() >> 4,
                cb->num_pages(buffer_index),
                cb->size() / cb->num_pages(buffer_index) >> 4};
            this->manager.cq_write(cb_data.data(), padding_alignment, system_memory_temporary_storage_address);
            system_memory_temporary_storage_address += padding_alignment;
            if (tracing) {
                // No need to resize since cb_data size is guaranteed to be 16 bytes
                trace_host_data.insert(trace_host_data.end(), cb_data.begin(), cb_data.end());
            }
        }
    }

    this->manager.issue_queue_push_back(cmd_size, detail::LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
    if (tracing) {
        Trace::TraceNode trace_node = {
            .command = cmd,
            .data = trace_host_data,
            .command_type = this->type(),
            .num_data_bytes = cmd.get_issue_data_size()};
        Trace& trace_ = trace.value();
        trace_.record(trace_node);
    }
}

EnqueueWrapCommand::EnqueueWrapCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager) :
    command_queue_id(command_queue_id), manager(manager) {
    this->device = device;
}

// EnqueueWrapCommand section
EnqueueIssueWrapCommand::EnqueueIssueWrapCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager) :
    EnqueueWrapCommand(command_queue_id, device, manager) {}

const DeviceCommand EnqueueIssueWrapCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.set_wrap(DeviceCommand::WrapRegion::ISSUE);
    return command;
}

void EnqueueIssueWrapCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t space_left_in_bytes = this->manager.get_issue_queue_limit(this->command_queue_id) - write_ptr;
    // There may not be enough space in the issue queue to submit another command
    // In that case we write as big of a vector as we can with the wrap index (0) set to wrap type
    // To ensure that the issue queue write pointer does wrap, we need the wrap packet to be the full size of the issue
    // queue
    uint32_t wrap_packet_size_bytes = std::min(space_left_in_bytes, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);

    const DeviceCommand cmd = this->assemble_device_command(0);
    this->manager.issue_queue_reserve_back(wrap_packet_size_bytes, this->command_queue_id);
    this->manager.cq_write(cmd.data(), wrap_packet_size_bytes, write_ptr);

    this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
}

EnqueueCompletionWrapCommand::EnqueueCompletionWrapCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event) :
    EnqueueWrapCommand(command_queue_id, device, manager) {
    this->event = event;
}

const DeviceCommand EnqueueCompletionWrapCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.set_wrap(DeviceCommand::WrapRegion::COMPLETION);
    command.set_event(this->event);
    return command;
}

void EnqueueCompletionWrapCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t space_left_in_bytes = this->manager.get_issue_queue_limit(this->command_queue_id) - write_ptr;
    uint32_t wrap_packet_size_bytes = std::min(space_left_in_bytes, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
    const DeviceCommand cmd = this->assemble_device_command(0);
    this->manager.issue_queue_reserve_back(wrap_packet_size_bytes, this->command_queue_id);
    this->manager.cq_write(cmd.data(), wrap_packet_size_bytes, write_ptr);
    this->manager.wrap_next_completion_queue_wr_ptr(this->command_queue_id);
    this->manager.issue_queue_push_back(wrap_packet_size_bytes, detail::LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}


EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event):
    command_queue_id(command_queue_id), device(device), manager(manager), event(event) {
}


const DeviceCommand EnqueueRecordEventCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.set_issue_data_size(0); // No extra data just CMD.
    command.set_completion_data_size(align(EVENT_PADDED_SIZE, 32));
    command.set_event(this->event);
    if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
        command.set_stall(); // Ensure ordering w/ programs in FD1.3+ prefetcher.
    }
    return command;
}

void EnqueueRecordEventCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    const DeviceCommand cmd = this->assemble_device_command(0);
    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    this->manager.cq_write(cmd.data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(cmd_size, detail::LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event, const Event& sync_event):
    command_queue_id(command_queue_id), device(device), manager(manager), event(event), sync_event(sync_event) {
        // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
        TT_ASSERT(command_queue_id != sync_event.cq_id || event != sync_event.event_id,
            "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. Event ID: {} CQ ID: {}",
            event, command_queue_id);
}

const DeviceCommand EnqueueWaitForEventCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.set_issue_data_size(0); // No extra data just CMD.
    command.set_completion_data_size(align(EVENT_PADDED_SIZE, 32));
    command.set_event(this->event);
    if (tt::Cluster::instance().arch() == tt::ARCH::WORMHOLE_B0) {
        command.set_stall(); // Ensure ordering w/ programs in FD1.3+ prefetcher.
    }
    // #5529 - Cross chip sync needs to be implemented. Currently, we only support sync on the same chip.
    TT_ASSERT(this->sync_event.device == this->device,
            "EnqueueWaitForEvent() cross-chip sync not yet supported. Sync event device: {} this device: {}",
            this->sync_event.device->id(), this->device->id());

    auto &event_sync_hw_cq = this->sync_event.device->command_queue(this->sync_event.cq_id).hw_command_queue();
    auto event_sync_core = this->sync_event.device->worker_core_from_logical_core(event_sync_hw_cq.completion_queue_writer_core);

    // Let dispatcher know this is sync event, and what core/event_id to sync on.
    command.set_is_event_sync(true);
    command.set_event_sync_core_x(event_sync_core.x);
    command.set_event_sync_core_y(event_sync_core.y);
    command.set_event_sync_event_id(this->sync_event.event_id);

    return command;
}

void EnqueueWaitForEventCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    const DeviceCommand cmd = this->assemble_device_command(0);
    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    this->manager.cq_write(cmd.data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(cmd_size, detail::LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}

// HWCommandQueue section
HWCommandQueue::HWCommandQueue(Device* device, uint32_t id) : manager(device->sysmem_manager()), completion_queue_thread{} {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->num_issued_commands = 0;
    this->num_completed_commands = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();

    tt_cxy_pair issue_q_reader_location =
        dispatch_core_manager::get(device->num_hw_cqs()).issue_queue_reader_core(device->id(), channel, this->id);
    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::get(device->num_hw_cqs()).completion_queue_writer_core(device->id(), channel, this->id);

    this->issue_queue_reader_core = CoreCoord(issue_q_reader_location.x, issue_q_reader_location.y);
    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
    this->stall_before_read = false;
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {
        TT_ASSERT(
            this->issued_reads.size() == 0,
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->issued_completion_wraps.size() == 0,
            "There should be no completion wraps in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_issued_commands == this->num_completed_commands,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted commands: {}", this->num_issued_commands - this->num_completed_commands);
        this->exit_condition = true;
        this->completion_queue_thread.join();
    }
}

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking) {
    command.process();
    this->num_issued_commands++;
    if (blocking) {
        this->finish();
    }

    // If this command has side-effects, then the next scheduled read needs
    // to stall before fetching. Else, it can pre-fetch
    this->stall_before_read = command.has_side_effects();
}

// TODO: Currently converting page ordering from interleaved to sharded and then doing contiguous read/write
//  Look into modifying command to do read/write of a page at a time to avoid doing copy
void convert_interleaved_to_sharded_on_host(const void* host, const Buffer& buffer, bool read = false) {
    const uint32_t num_pages = buffer.num_pages();
    const uint32_t page_size = buffer.page_size();

    const uint32_t size_in_bytes = num_pages * page_size;

    void* temp = malloc(size_in_bytes);
    memcpy(temp, host, size_in_bytes);

    const void* dst = host;
    std::set<uint32_t> pages_seen;
    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {

        if (read) {
            auto host_page_id = page_id;
            auto dev_page_id = buffer.get_dev_to_host_mapped_page_id(host_page_id);
            TT_ASSERT(dev_page_id < num_pages and dev_page_id >= 0);
            memcpy((char*)dst + dev_page_id * page_size, (char*)temp + host_page_id * page_size, page_size);
        } else {
            auto dev_page_id = page_id;
            auto host_page_id = buffer.get_host_to_dev_mapped_page_id(dev_page_id);
            TT_ASSERT(host_page_id < num_pages and host_page_id >= 0);
            memcpy((char*)dst + host_page_id * page_size, (char*)temp + dev_page_id * page_size, page_size);
        }
    }
    free(temp);
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void* dst, bool blocking) {
    this->enqueue_read_buffer(*buffer, dst, blocking);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion region
void HWCommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("HWCommandQueue_read_buffer");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    uint32_t read_buffer_command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;

    const uint32_t command_completion_limit = this->manager.get_completion_queue_limit(this->id);
    while (total_pages_to_read) {
        if ((this->manager.get_issue_queue_write_ptr(this->id)) + read_buffer_command_size >=
            this->manager.get_issue_queue_limit(this->id)) {
            this->issue_wrap();
        }

        uint32_t completion_queue_write_ptr = this->manager.get_next_completion_queue_write_ptr(this->id);
        if (completion_queue_write_ptr + padded_page_size + align(EVENT_PADDED_SIZE, 32) >= command_completion_limit) {
            uint32_t wrap_completion_event = this->manager.get_next_event(this->id);

            this->issued_completion_wraps.emplace(wrap_completion_event, wrap_completion_event);
            this->completion_wrap(wrap_completion_event);
        }

        uint32_t num_pages_available =
            (command_completion_limit - this->manager.get_next_completion_queue_write_ptr(this->id) -
             align(EVENT_PADDED_SIZE, 32)) /
            padded_page_size;
        uint32_t pages_to_read = std::min(total_pages_to_read, num_pages_available);
        TT_ASSERT(pages_to_read, "There should always be pages to read");
        uint32_t read_event = this->manager.get_next_event(this->id);
        this->issued_reads.emplace(
            read_event,
            detail::IssuedReadData(buffer, padded_page_size, dst, unpadded_dst_offset, pages_to_read, src_page_index));

        if (is_sharded(buffer.buffer_layout())) {
            auto command = EnqueueReadShardedBufferCommand(
                this->id, this->device, buffer, dst, this->stall_before_read, this->manager, read_event, src_page_index, pages_to_read);
            this->enqueue_command(command, false);
        } else {
            auto command = EnqueueReadInterleavedBufferCommand(
                this->id, this->device, buffer, dst, this->stall_before_read, this->manager, read_event, src_page_index, pages_to_read);
            this->enqueue_command(command, false);
        }
        uint32_t completion_num_bytes = align(EVENT_PADDED_SIZE + pages_to_read * padded_page_size, 32);
        this->manager.next_completion_queue_push_back(completion_num_bytes, this->id);

        total_pages_to_read -= pages_to_read;
        src_page_index += pages_to_read;
        unpadded_dst_offset += pages_to_read * buffer.page_size();
    }

    if (blocking) {
        this->finish();
    }
}

void HWCommandQueue::enqueue_write_buffer(std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<const Buffer>> buffer, HostDataType src, bool blocking) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    std::visit ([this, &buffer, &blocking](auto&& data) {
        using T = std::decay_t<decltype(data)>;
        std::visit ([this, &buffer, &blocking, &data](auto&& b) {
            using type_buf = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, const void*>) {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data, blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data, blocking);
                }
            } else {
                if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                    this->enqueue_write_buffer(*b, data.get() -> data(), blocking);
                } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                    this->enqueue_write_buffer(b.get(), data.get() -> data(), blocking);
                }
            }
        }, buffer);
    }, src);
}

void HWCommandQueue::enqueue_write_buffer(const Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer");
    // TODO(agrebenisan): Fix these asserts after implementing multi-core CQ
    // TODO (abhullar): Use eth mem l1 size when issue queue interface kernel is on ethernet core
    TT_ASSERT(
        buffer.page_size() < get_cq_data_buffer_size(false),
        "Buffer pages must fit within the command queue data section");

    if (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
        buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        convert_interleaved_to_sharded_on_host(src, buffer);
    }

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_write = buffer.num_pages();
    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    uint32_t dst_page_index = 0;
    while (total_pages_to_write > 0) {
        int32_t num_pages_available =
            (int32_t(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id)) -
             int32_t(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND)) /
            int32_t(padded_page_size);
        // If not even a single device command fits, we hit this edgecase
        num_pages_available = std::max(num_pages_available, 0);

        uint32_t pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);
        if (pages_to_write == 0) {
            // No space for command and data
            this->issue_wrap();
            num_pages_available = (int32_t(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id)) -
                                   int32_t(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND)) /
                                  int32_t(padded_page_size);
            pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);
        }

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);
        uint32_t event = this->manager.get_next_event(this->id);
        if (is_sharded(buffer.buffer_layout())) {
            auto command = EnqueueWriteShardedBufferCommand(
                this->id, this->device, buffer, src, this->manager, event, dst_page_index, pages_to_write);
            this->enqueue_command(command, false);
        } else {
            auto command = EnqueueWriteInterleavedBufferCommand(
                this->id, this->device, buffer, src, this->manager, event, dst_page_index, pages_to_write);
            this->enqueue_command(command, false);
        }
        this->manager.next_completion_queue_push_back(align(EVENT_PADDED_SIZE, 32), this->id);

        total_pages_to_write -= pages_to_write;
        dst_page_index += pages_to_write;
    }

    if (blocking) {
        this->finish();
    }
}

void HWCommandQueue::enqueue_program(
    Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");

    // Whether or not we should stall the producer from prefetching binary data. If the
    // data is cached, then we don't need to stall, otherwise we need to wait for the
    // data to land in DRAM first
    bool stall;
    if (not program.loaded_onto_device) {
        this->enqueue_write_buffer(*program.buffer, program.program_device_map.program_pages.data(), false);
        stall = true;
        program.loaded_onto_device = true;
    } else {
        stall = false;
    }

    tt::log_debug(tt::LogDispatch, "EnqueueProgram for channel {}", this->id);
    ProgramDeviceMap& program_device_map = program.program_device_map;
    uint32_t host_data_num_pages = program_device_map.num_transfers_in_runtime_arg_pages.at(PageTransferType::MULTICAST).size() + program_device_map.num_transfers_in_cb_config_pages.at(PageTransferType::MULTICAST).size();
    uint32_t host_data_and_device_command_size =
        DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + (host_data_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE);

    if ((this->manager.get_issue_queue_write_ptr(this->id)) + host_data_and_device_command_size >=
        this->manager.get_issue_queue_limit(this->id)) {
        TT_FATAL(
            host_data_and_device_command_size <= this->manager.get_issue_queue_size(this->id) - CQ_START,
            "EnqueueProgram command size too large");
        this->issue_wrap();
    }

    EnqueueProgramCommand command(
        this->id,
        this->device,
        program,
        this->manager,
        this->manager.get_next_event(this->id),
        stall,
        trace);
    this->enqueue_command(command, blocking);
    this->manager.next_completion_queue_push_back(align(EVENT_PADDED_SIZE, 32), this->id);
}

void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");
    uint32_t command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true;

    if ((this->manager.get_issue_queue_write_ptr(this->id)) + command_size >= this->manager.get_issue_queue_limit(this->id)) {
        this->issue_wrap();
    }

    auto command = EnqueueRecordEventCommand(this->id, this->device, this->manager, event->event_id);
    this->enqueue_command(command, false);
    this->manager.next_completion_queue_push_back(align(EVENT_PADDED_SIZE, 32), this->id);
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    uint32_t command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    if ((this->manager.get_issue_queue_write_ptr(this->id)) + command_size >= this->manager.get_issue_queue_limit(this->id)) {
        this->issue_wrap();
    }

    auto command = EnqueueWaitForEventCommand(this->id, this->device, this->manager, this->manager.get_next_event(this->id), *event);
    this->enqueue_command(command, false);
    this->manager.next_completion_queue_push_back(align(EVENT_PADDED_SIZE, 32), this->id);
}


void HWCommandQueue::enqueue_trace() {
    ZoneScopedN("HWCommandQueue_enqueue_trace");
    TT_THROW("Not implemented");
}

void HWCommandQueue::copy_into_user_space(uint32_t event, uint32_t read_ptr, chip_id_t mmio_device_id, uint16_t channel) {
    const auto& [buffer_layout, page_size, padded_page_size, dev_page_to_host_page_mapping, dst, dst_offset, num_pages_read, cur_host_page_id] =
        this->issued_reads.at(event);

    uint32_t padded_num_bytes = num_pages_read * padded_page_size;
    if (buffer_layout == TensorMemoryLayout::INTERLEAVED or
        buffer_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        void* contiguous_dst = (void*)(uint64_t(dst) + dst_offset);
        if ((page_size % 32) == 0) {
            tt::Cluster::instance().read_sysmem(
                contiguous_dst, padded_num_bytes, read_ptr + align(EVENT_PADDED_SIZE, 32), mmio_device_id, channel);
        } else {
            uint32_t dst_offset = 0;
            uint32_t read_src = read_ptr + align(EVENT_PADDED_SIZE, 32);
            for (uint32_t offset = 0; offset < padded_page_size * num_pages_read; offset += padded_page_size) {
                tt::Cluster::instance().read_sysmem(
                    (char*)(uint64_t(contiguous_dst) + dst_offset),
                    page_size,
                    read_src + offset,
                    mmio_device_id,
                    channel);
                dst_offset += page_size;
            }
        }
    } else if (
        buffer_layout == TensorMemoryLayout::WIDTH_SHARDED or
        buffer_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t host_page_id = cur_host_page_id;
        uint32_t read_src = read_ptr + align(EVENT_PADDED_SIZE, 32);
        for (uint32_t offset = 0; offset < padded_page_size * num_pages_read; offset += padded_page_size) {
            uint32_t device_page_id = dev_page_to_host_page_mapping[host_page_id];
            void* page_dst = (void*)(uint64_t(dst) + device_page_id * page_size);
            tt::Cluster::instance().read_sysmem(
                page_dst, page_size, read_src + offset, mmio_device_id, channel);
            host_page_id++;
        }
    }
    this->manager.completion_queue_pop_front(padded_num_bytes, this->id);
    this->issued_reads.erase(event);
}

void HWCommandQueue::read_completion_queue() {
    tracy::SetThreadName("COMPLETION QUEUE");
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        if (this->num_issued_commands > this->num_completed_commands) {
            uint32_t num_events_to_read = this->num_issued_commands - this->num_completed_commands;
            for (uint32_t i = 0; i < num_events_to_read; i++) {
                this->manager.completion_queue_wait_front(this->id, this->exit_condition);

                if (this->exit_condition) {  // Early exit
                    return;
                }
                uint32_t event;
                uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
                uint32_t read_toggle = this->manager.get_completion_queue_read_toggle(this->id);
                tt::Cluster::instance().read_sysmem(&event, 4, read_ptr, mmio_device_id, channel);

                if (this->issued_completion_wraps.count(event)) {
                    this->manager.wrap_completion_queue_rd_ptr(this->id);
                    this->manager.send_completion_queue_read_ptr(this->id);
                    this->issued_completion_wraps.erase(event);
                } else {
                    if (this->issued_reads.count(event)) {
                        this->copy_into_user_space(event, read_ptr, mmio_device_id, channel);
                    }
                    this->manager.completion_queue_pop_front(align(EVENT_PADDED_SIZE, 32), this->id);
                }
                this->manager.set_last_completed_event(this->id, event);
            }
            this->num_completed_commands += num_events_to_read;
        } else if (this->exit_condition) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void HWCommandQueue::finish() {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);

    if (tt::llrt::OptionsG.get_test_mode_enabled()) {
        while (this->num_issued_commands > this->num_completed_commands) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang. Mark state and early exit. Assert in main thread.
                this->exit_condition = true;
                this->dprint_server_hang = true;
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->exit_condition = true;
                this->illegal_noc_txn_hang = true;
                return;
            }
        }
    } else {
        while (this->num_issued_commands > this->num_completed_commands);
    }
}

void HWCommandQueue::issue_wrap() {
    ZoneScopedN("CommandQueue_wrap");
    tt::log_debug(tt::LogDispatch, "EnqueueWrap for channel {}", this->id);
    EnqueueIssueWrapCommand command(this->id, this->device, this->manager);
    command.process(); // Not increasing num issued commands
}

void HWCommandQueue::completion_wrap(uint32_t event) {
    ZoneScopedN("HWCommandQueue_wrap");
    tt::log_debug(tt::LogDispatch, "EnqueueWrap for channel {}", this->id);
    EnqueueCompletionWrapCommand command(this->id, this->device, this->manager, event);
    this->enqueue_command(command, false);
}

volatile bool HWCommandQueue::is_dprint_server_hung() {
    return dprint_server_hang;
}

volatile bool HWCommandQueue::is_noc_hung() {
    return illegal_noc_txn_hang;
}

void EnqueueAddBufferToProgram(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program, bool blocking) {
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ADD_BUFFER_TO_PROGRAM,
        .blocking = blocking,
        .buffer = buffer,
        .program = program,
    });
}

void EnqueueAddBufferToProgramImpl(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program) {
    std::visit([program] (auto&& b) {
        using buffer_type = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
            std::visit([&b] (auto&& p) {
                using program_type = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<program_type, std::reference_wrapper<Program>>) {
                    p.get().add_buffer(b);
                }
                else {
                    p->add_buffer(b);
                }
            }, program);
        }
    }, buffer);
}

void EnqueueUpdateRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::vector<uint32_t> &update_idx, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
            .update_idx = update_idx,
    };
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::UPDATE_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueUpdateRuntimeArgsImpl (const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    auto& kernel_runtime_args = runtime_args_md.kernel->runtime_args(runtime_args_md.core_coord);
    for (const auto& idx : runtime_args_md.update_idx) {
        kernel_runtime_args[idx] = resolved_runtime_args[idx];
    }
}

void EnqueueSetRuntimeArgs(CommandQueue& cq, const std::shared_ptr<Kernel> kernel, const CoreCoord &core_coord, std::shared_ptr<RuntimeArgs> runtime_args_ptr, bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata {
            .core_coord = core_coord,
            .runtime_args_ptr = runtime_args_ptr,
            .kernel = kernel,
    };
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::SET_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md,
    });
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata& runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto& arg : *(runtime_args_md.runtime_args_ptr)) {
        std::visit([&resolved_runtime_args] (auto&& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Buffer*>) {
                resolved_runtime_args.push_back(a -> address());
            } else {
                resolved_runtime_args.push_back(a);
            }
        }, arg);
    }
    runtime_args_md.kernel -> set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking) {
    cq.run_command( CommandInterface {
        .type = EnqueueCommandType::GET_BUF_ADDR,
        .blocking = blocking,
        .shadow_buffer = buffer,
        .dst = dst_buf_addr
    });
}

void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer) {
    *(static_cast<uint32_t*>(dst_buf_addr)) = buffer -> address();
}
void EnqueueAllocateBuffer(CommandQueue& cq, Buffer* buffer, bool bottom_up, bool blocking) {
    auto alloc_md = AllocBufferMetadata {
        .buffer = buffer,
        .allocator = *(buffer->device()->allocator_),
        .bottom_up = bottom_up,
    };
    cq.run_command(CommandInterface {
        .type = EnqueueCommandType::ALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md) {
    Buffer* buffer = alloc_md.buffer;
    uint32_t allocated_addr;
    if(is_sharded(buffer->buffer_layout())) {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, buffer->num_cores());
    }
    else {
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, std::nullopt);
    }
    buffer->set_address(static_cast<uint64_t>(allocated_addr));
}

void EnqueueDeallocateBuffer(CommandQueue& cq, Allocator& allocator, uint32_t device_address, BufferType buffer_type, bool blocking) {
    // Need to explictly pass in relevant buffer attributes here, since the Buffer* ptr can be deallocated a this point
    auto alloc_md = AllocBufferMetadata {
        .allocator = allocator,
        .buffer_type = buffer_type,
        .device_address = device_address,
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::DEALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md,
    });
}

void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md) {
    allocator::deallocate_buffer(alloc_md.allocator, alloc_md.device_address, alloc_md.buffer_type);
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& dst, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    Buffer & b = std::holds_alternative<std::shared_ptr<Buffer>>(buffer) ? *(std::get< std::shared_ptr<Buffer> > ( buffer )) :
                                                                            std::get<std::reference_wrapper<Buffer>>(buffer).get();
    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    std::visit ( [&dst](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>>) {
            dst.resize(b.get().page_size() * b.get().num_pages() / sizeof(uint32_t));
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
            dst.resize(b->page_size() * b->num_pages() / sizeof(uint32_t));
        }
    }, buffer);

    // TODO(agrebenisan): Move to deprecated
    EnqueueReadBuffer(cq, buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, vector<uint32_t>& src, bool blocking){
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, buffer, src.data(), blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .dst = dst
    });
}

void EnqueueReadBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, bool blocking) {
    std::visit ( [&cq, dst, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer> > ) {
            cq.hw_command_queue().enqueue_read_buffer(b, dst, blocking);
        }
    }, buffer);
}

void EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER,
        .blocking = blocking,
        .buffer = buffer,
        .src = src
    });
}

void EnqueueWriteBufferImpl(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer,
                                          HostDataType src, bool blocking) {
    std::visit ( [&cq, src, blocking](auto&& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>> ) {
            cq.hw_command_queue().enqueue_write_buffer(b, src, blocking);
        }
    }, buffer);
}

void EnqueueProgram(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    detail::DispatchStateCheck(true);
    if (cq.get_mode() != CommandQueue::CommandQueueMode::TRACE) {
        TT_FATAL(cq.id() == 0, "EnqueueProgram only supported on first command queue on device for time being.");
    }
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_PROGRAM,
        .blocking = blocking,
        .program = program
    });
}

void EnqueueProgramImpl(CommandQueue& cq, std::variant < std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking) {
    ZoneScoped;
    std::visit ( [&cq, blocking](auto&& program) {
        ZoneScoped;
        using T = std::decay_t<decltype(program)>;
        Device * device = cq.device();
        std::optional<std::reference_wrapper<Trace>> trace;
        if (cq.trace()) {
            trace = std::optional<std::reference_wrapper<Trace>>(*cq.trace());
        }
        if constexpr (std::is_same_v<T, std::reference_wrapper<Program>>) {
            detail::CompileProgram(device, program);
            program.get().allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(program, device);
            cq.hw_command_queue().enqueue_program(program, trace, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program.get().release_buffers();
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Program>>) {
            detail::CompileProgram(device, *program);
            program->allocate_circular_buffers();
            detail::ValidateCircularBufferRegion(*program, device);
            cq.hw_command_queue().enqueue_program(*program, trace, blocking);
            // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem leaks on device.
            program->release_buffers();
        }
    }, program);
}

void EnqueueRecordEvent(CommandQueue& cq, std::shared_ptr<Event> event) {
    TT_ASSERT(event->device == nullptr, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->event_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");
    TT_ASSERT(event->cq_id == -1, "EnqueueRecordEvent expected to be given an uninitialized event");

    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_RECORD_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueRecordEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    cq.hw_command_queue().enqueue_record_event(event);
}


void EnqueueWaitForEvent(CommandQueue& cq, std::shared_ptr<Event> event) {

    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT,
        .blocking = false,
        .event = event,
    });
}

void EnqueueWaitForEventImpl(CommandQueue& cq, std::shared_ptr<Event> event) {
    event->wait_until_ready(); // Block until event populated. Worker thread.
    log_trace(tt::LogMetal, "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(), event->cq_id, event->event_id, cq.device()->id(), cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}


void EventSynchronize(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready(); // Block until event populated. Parent thread.
    log_trace(tt::LogMetal, "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})", event->device->id(), event->cq_id, event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::OptionsG.get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
            TT_ASSERT(false, "Command Queue could not complete EventSynchronize. See {} for details.", tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

void Finish(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::FINISH,
        .blocking = true
    });
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_dprint_server_hung()),
              "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(!(cq.device() -> hw_command_queue(cq.id()).is_noc_hung()),
              "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
               tt::watcher_get_log_file_name());
}

void FinishImpl(CommandQueue& cq) {
    cq.hw_command_queue().finish();
}

CommandQueue& BeginTrace(Trace& trace) {
    TT_ASSERT(not trace.trace_complete, "Already completed this trace");
    TT_ASSERT(trace.queue().empty(), "Cannot begin trace on one that already captured commands");
    return trace.queue();
}

void EndTrace(Trace& trace) {
    TT_ASSERT(not trace.trace_complete, "Already completed this trace");
    trace.trace_complete = true;
    trace.validate();
}

uint32_t InstantiateTrace(Trace& trace, CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    TT_ASSERT(cq.trace() == nullptr, "Multiple traces on a CQ is not supported yet");
    uint32_t trace_id = trace.instantiate(cq);
    return trace_id;
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_ASSERT(cq.trace(), "A trace has not been instantiated on this command queue yet!");
    if (cq.trace()->trace_instances.count(trace_id) == 0) {
        TT_THROW("Trace instance " + std::to_string(trace_id) + " does not exist");
    }
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE,
        .blocking = blocking
    });
}

void EnqueueTraceImpl(CommandQueue& cq) {
    // STUB: Run the trace in eager mode for now
    auto& tq = cq.trace()->queue();
    for (const auto& cmd : tq.worker_queue) {
        cq.run_command_impl(cmd);
    }
}

CommandQueue::CommandQueue(Device* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device),
    trace_ptr(nullptr),
    cq_id(id),
    mode(mode),
    worker_state(CommandQueueState::IDLE) {
    if (this->async_mode()) {
        num_async_cqs++;
        // The main program thread launches the Command Queue
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
    }
}

CommandQueue::CommandQueue(Trace* trace) :
    device_ptr(nullptr),
    parent_thread_id(0),
    trace_ptr(trace),
    cq_id(-1),
    mode(CommandQueueMode::TRACE),
    worker_state(CommandQueueState::IDLE) {
    TT_ASSERT(this->trace_ptr, "A valid trace must be provided for a trace mode queue");
}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (this->trace_mode()) {
        TT_ASSERT(this->trace()->trace_complete, "Trace capture must be complete before desctruction");
    } else {
        TT_ASSERT(this->worker_queue.empty(), "CQ{} worker queue must be empty on destruction", this->cq_id);
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() {
    return this->device()->hw_command_queue(this->cq_id);
}

void CommandQueue::wait_until_empty() {
    log_trace(LogDispatch, "CQ{} WFI start", this->cq_id);
    if (this->async_mode()) {
        // Insert a flush token to push all prior commands to completion
        // Necessary to avoid implementing a peek and pop on the lock-free queue
        this->worker_queue.push(CommandInterface{.type = EnqueueCommandType::FLUSH});
    }
    while (true) {
        if (this->worker_queue.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    log_trace(LogDispatch, "CQ{} WFI complete", this->cq_id);
}

void CommandQueue::set_mode(const CommandQueueMode& mode) {
    TT_ASSERT(not this->trace_mode(), "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
    if (this->mode == mode) {
        // Do nothing if requested mode matches current CQ mode.
        return;
    }
    this->mode = mode;
    if (this->async_mode()) {
        num_async_cqs++;
        num_passthrough_cqs--;
        // Record parent thread-id and start worker.
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
        num_async_cqs--;
        // Wait for all cmds sent in async mode to complete and stop worker.
        this->wait_until_empty();
        this->stop_worker();
    }
}

void CommandQueue::start_worker() {
    if (this->worker_state == CommandQueueState::RUNNING) {
        return;  // worker already running, exit
    }
    this->worker_state = CommandQueueState::RUNNING;
    this->worker_thread = std::make_unique<std::thread>(std::thread(&CommandQueue::run_worker, this));
    tt::log_debug(tt::LogDispatch, "CQ{} started worker thread", this->cq_id);
}

void CommandQueue::stop_worker() {
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "CQ{} stopped worker thread", this->cq_id);
}

void CommandQueue::run_worker() {
    // forever loop checking for commands in the worker queue
    // Track the worker thread id, for cases where a command calls a sub command.
    // This is to detect cases where commands may be nested.
    worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if (this->worker_queue.empty()) {
            if (this->worker_state == CommandQueueState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::shared_ptr<CommandInterface> command(this->worker_queue.pop());
            run_command_impl(*command);
        }
    }
}

void CommandQueue::run_command(const CommandInterface& command) {
    log_trace(LogDispatch, "CQ{} received {} in {} mode", this->cq_id, command.type, this->mode);
    if (not this->passthrough_mode()) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == parent_thread_id or this->trace_mode()) {
            // Push to worker queue for trace or async mode. In trace mode, store the execution in the queue.
            // In async mode when parent pushes cmd, feed worker through queue.
            this->worker_queue.push(command);
            if (command.blocking.has_value() and *command.blocking == true) {
                TT_ASSERT(not this->trace_mode(), "Blocking commands cannot be traced!");
                this->wait_until_empty();
            }
        }
        else {
            // Handle case where worker pushes command to itself (passthrough)
            TT_ASSERT(std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id, "Only main thread or worker thread can run commands through the SW command queue");
            run_command_impl(command);
        }
    } else {
        this->run_command_impl(command);
    }
}

void CommandQueue::run_command_impl(const CommandInterface& command) {
    log_trace(LogDispatch, "CQ{} running {}", this->cq_id, command.type);
    switch (command.type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueReadBufferImpl(*this, command.buffer.value(), command.dst.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER:
            TT_ASSERT(command.src.has_value(), "Must provide a src!");
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueWriteBufferImpl(*this, command.buffer.value(), command.src.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueAllocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::DEALLOCATE_BUFFER:
            TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
            EnqueueDeallocateBufferImpl(command.alloc_md.value());
            break;
        case EnqueueCommandType::GET_BUF_ADDR:
            TT_ASSERT(command.dst.has_value(), "Must provide a dst address!");
            TT_ASSERT(command.shadow_buffer.has_value(), "Must provide a shadow buffer!");
            EnqueueGetBufferAddrImpl(command.dst.value(), command.shadow_buffer.value());
            break;
        case EnqueueCommandType::SET_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueSetRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::UPDATE_RUNTIME_ARGS:
            TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
            EnqueueUpdateRuntimeArgsImpl(command.runtime_args_md.value());
            break;
        case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
            TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            EnqueueAddBufferToProgramImpl(command.buffer.value(), command.program.value());
            break;
        case EnqueueCommandType::ENQUEUE_PROGRAM:
            TT_ASSERT(command.program.has_value(), "Must provide a program!");
            TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
            EnqueueProgramImpl(*this, command.program.value(), command.blocking.value());
            break;
        case EnqueueCommandType::ENQUEUE_TRACE:
            EnqueueTraceImpl(*this);
            break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueRecordEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT:
            TT_ASSERT(command.event.has_value(), "Must provide an event!");
            EnqueueWaitForEventImpl(*this, command.event.value());
            break;
        case EnqueueCommandType::FINISH:
            FinishImpl(*this);
            break;
        case EnqueueCommandType::FLUSH:
            // Used by CQ to push prior commands
            break;
        default:
            TT_THROW("Invalid command type");
    }
    log_trace(LogDispatch, "CQ{} running {} complete", this->cq_id, command.type);
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type) {
    switch (type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: os << "ENQUEUE_READ_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: os << "ENQUEUE_WRITE_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_PROGRAM: os << "ENQUEUE_PROGRAM"; break;
        case EnqueueCommandType::ENQUEUE_TRACE: os << "ENQUEUE_TRACE"; break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT: os << "ENQUEUE_RECORD_EVENT"; break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: os << "ENQUEUE_WAIT_FOR_EVENT"; break;
        case EnqueueCommandType::FINISH: os << "FINISH"; break;
        case EnqueueCommandType::FLUSH: os << "FLUSH"; break;
        default: TT_THROW("Invalid command type!");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type) {
    switch (type) {
        case CommandQueue::CommandQueueMode::PASSTHROUGH: os << "PASSTHROUGH"; break;
        case CommandQueue::CommandQueueMode::ASYNC: os << "ASYNC"; break;
        case CommandQueue::CommandQueueMode::TRACE: os << "TRACE"; break;
        default: TT_THROW("Invalid CommandQueueMode type!");
    }
    return os;
}
