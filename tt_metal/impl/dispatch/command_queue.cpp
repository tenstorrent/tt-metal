// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include "debug_tools.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/debug/dprint_server.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"
#include "dev_msgs.h"
#include <algorithm> // for copy() and assign()
#include <iterator> // for back_inserter

namespace tt::tt_metal {

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

uint32_t get_noc_multicast_encoding(const CoreCoord& top_left, const CoreCoord& bottom_right) {
    return NOC_MULTICAST_ENCODING(top_left.x, top_left.y, bottom_right.x, bottom_right.y);
}

uint32_t get_noc_unicast_encoding(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

ProgramMap ConstructProgramMap(const Device* device, Program& program) {
    /*
        TODO(agrebenisan): Move this logic to compile program
    */
    vector<transfer_info> runtime_arg_page_transfers;
    vector<transfer_info> cb_config_page_transfers;
    vector<transfer_info> program_page_transfers;
    vector<transfer_info> go_signal_page_transfers;
    vector<uint32_t> num_transfers_in_runtime_arg_pages; // Corresponds to the number of transfers within host data pages across all host data pages
    vector<uint32_t> num_transfers_in_cb_config_pages;
    vector<uint32_t> num_transfers_in_program_pages;
    vector<uint32_t> num_transfers_in_go_signal_pages;
    uint32_t num_transfers_within_page = 0;

    uint32_t src = 0;
    constexpr static uint32_t noc_transfer_alignment_in_bytes = 16;
    auto update_program_page_transfers = [&num_transfers_within_page](
                                             uint32_t src,
                                             uint32_t num_bytes,
                                             uint32_t dst,
                                             vector<transfer_info>& transfers,
                                             vector<uint32_t>& num_transfers_per_page,
                                             const vector<pair<uint32_t, uint32_t>>& dst_noc_transfer_info,
                                             bool linked = false) -> uint32_t {
        while (num_bytes) {
            uint32_t num_bytes_left_in_page = DeviceCommand::PROGRAM_PAGE_SIZE - (src % DeviceCommand::PROGRAM_PAGE_SIZE);
            uint32_t num_bytes_in_transfer = std::min(num_bytes_left_in_page, num_bytes);
            src = align(src + num_bytes_in_transfer, noc_transfer_alignment_in_bytes);

            uint32_t transfer_instruction_idx = 1;
            for (const auto& [dst_noc_encoding, num_receivers] : dst_noc_transfer_info) {
                bool last = transfer_instruction_idx == dst_noc_transfer_info.size();
                transfer_info transfer_instruction = {.size_in_bytes = num_bytes_in_transfer, .dst = dst, .dst_noc_encoding = dst_noc_encoding, .num_receivers = num_receivers, .last_transfer_in_group = last, .linked = linked};
                transfers.push_back(transfer_instruction);
                num_transfers_within_page++;
                transfer_instruction_idx++;
            }

            dst += num_bytes_in_transfer;
            num_bytes -= num_bytes_in_transfer;

            if ((src % DeviceCommand::PROGRAM_PAGE_SIZE) == 0) {
                num_transfers_per_page.push_back(num_transfers_within_page);
                num_transfers_within_page = 0;
            }
        }

        return src;
    };

    auto extract_dst_noc_multicast_info = [&device](const set<CoreRange>& ranges) -> vector<pair<uint32_t, uint32_t>> {
        // This API extracts all the pairs of noc multicast encodings given a set of core ranges
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info;
        for (const CoreRange& core_range : ranges) {
            CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
            CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

            uint32_t dst_noc_multicast_encoding = get_noc_multicast_encoding(physical_start, physical_end);

            uint32_t num_receivers = core_range.size();
            dst_noc_multicast_info.push_back(std::make_pair(dst_noc_multicast_encoding, num_receivers));
        }
        return dst_noc_multicast_info;
    };

    static const map<RISCV, uint32_t> processor_to_l1_arg_base_addr = {
        {RISCV::BRISC, BRISC_L1_ARG_BASE},
        {RISCV::NCRISC, NCRISC_L1_ARG_BASE},
        {RISCV::COMPUTE, TRISC_L1_ARG_BASE},
    };

    // Step 1: Get transfer info for runtime args (soon to just be host data). We
    // want to send host data first because of the higher latency to pull
    // in host data.
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        Kernel* kernel = detail::GetKernel(program, kernel_id);
        uint32_t dst = processor_to_l1_arg_base_addr.at(kernel->processor());
        for (const auto &core_coord : kernel->cores_with_runtime_args()) {
            CoreCoord physical_core = device->worker_core_from_logical_core(core_coord);
            const auto & runtime_args = kernel->runtime_args(core_coord);
            uint32_t num_bytes = runtime_args.size() * sizeof(uint32_t);
            uint32_t dst_noc = get_noc_unicast_encoding(physical_core);

            // Only one receiver per set of runtime arguments
            src = update_program_page_transfers(
                src, num_bytes, dst, runtime_arg_page_transfers, num_transfers_in_runtime_arg_pages, {{dst_noc, 1}});
        }
    }

    // Cleanup step of separating runtime arg pages from program pages
    if (num_transfers_within_page) {
        num_transfers_in_runtime_arg_pages.push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    src = 0; // Resetting since in a new page
    // Step 2: Continue constructing pages for circular buffer configs
    for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers()) {
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info = extract_dst_noc_multicast_info(cb->core_ranges().ranges());
        constexpr static uint32_t num_bytes = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
        for (const auto buffer_index : cb->buffer_indices()) {
            src = update_program_page_transfers(
                src,
                num_bytes,
                CIRCULAR_BUFFER_CONFIG_BASE + buffer_index * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t),
                cb_config_page_transfers,
                num_transfers_in_cb_config_pages,
                dst_noc_multicast_info);
        }
    }

    // Cleanup step of separating runtime arg pages from program pages
    if (num_transfers_within_page) {
        num_transfers_in_cb_config_pages.push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    static const map<RISCV, uint32_t> processor_to_local_mem_addr = {
        {RISCV::BRISC, MEM_BRISC_INIT_LOCAL_L1_BASE},
        {RISCV::NCRISC, MEM_NCRISC_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC0, MEM_TRISC0_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC1, MEM_TRISC1_INIT_LOCAL_L1_BASE},
        {RISCV::TRISC2, MEM_TRISC2_INIT_LOCAL_L1_BASE}};

    // Step 3: Determine the transfer information for each program binary
    src = 0; // Restart src since it begins in a new page
    for (const KernelGroup &kg: program.get_kernel_groups()) {

        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(kg.core_ranges.ranges());

        // So far, we don't support linking optimizations for kernel groups
        // which use multiple core ranges
        bool linked = dst_noc_multicast_info.size() == 1;

        vector<KernelHandle> kernel_ids;
        if (kg.riscv0_id) kernel_ids.push_back(kg.riscv0_id.value());
        if (kg.riscv1_id) kernel_ids.push_back(kg.riscv1_id.value());
        if (kg.compute_id) kernel_ids.push_back(kg.compute_id.value());

        uint32_t src_copy = src;
        for (size_t i = 0; i < kernel_ids.size(); i++) {
            KernelHandle kernel_id = kernel_ids[i];
            vector<RISCV> sub_kernels;
            const Kernel* kernel = detail::GetKernel(program, kernel_id);
            if (kernel->processor() == RISCV::COMPUTE) {
                sub_kernels = {RISCV::TRISC0, RISCV::TRISC1, RISCV::TRISC2};
            } else {
                sub_kernels = {kernel->processor()};
            }

            uint32_t sub_kernel_index = 0;
            const auto& binaries = kernel->binaries(device->id());
            for (size_t j = 0; j < binaries.size(); j++) {
                const ll_api::memory& kernel_bin = binaries[j];

                uint32_t k = 0;
                uint32_t num_spans = kernel_bin.num_spans();
                kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                    linked &= (i != kernel_ids.size() - 1) or (j != binaries.size() - 1) or (k != num_spans - 1);

                    uint32_t num_bytes = len * sizeof(uint32_t);
                    if ((dst & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
                        dst = (dst & ~MEM_LOCAL_BASE) + processor_to_local_mem_addr.at(sub_kernels[sub_kernel_index]);
                    } else if ((dst & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
                        dst = (dst & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE;
                    }

                    src = update_program_page_transfers(
                        src, num_bytes, dst, program_page_transfers, num_transfers_in_program_pages, dst_noc_multicast_info, linked);
                    k++;
                });
                sub_kernel_index++;
            }
        }
    }

    // Step 4: Continue constructing pages for semaphore configs
    for (const Semaphore& semaphore : program.semaphores()) {
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(semaphore.core_range_set().ranges());

        src = update_program_page_transfers(
            src,
            L1_ALIGNMENT,
            semaphore.address(),
            program_page_transfers,
            num_transfers_in_program_pages,
            dst_noc_multicast_info);
    }

    if (num_transfers_within_page) {
        num_transfers_in_program_pages.push_back(num_transfers_within_page);
        num_transfers_within_page = 0;
    }

    vector<uint32_t> program_pages(align(src, DeviceCommand::PROGRAM_PAGE_SIZE) / sizeof(uint32_t), 0);

    // Step 5: Continue constructing pages for GO signals
    src = 0;
    for (KernelGroup& kg : program.get_kernel_groups()) {
        kg.launch_msg.mode = DISPATCH_MODE_DEV;
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(kg.core_ranges.ranges());

        src = update_program_page_transfers(
            src,
            sizeof(launch_msg_t),
            GET_MAILBOX_ADDRESS_HOST(launch),
            go_signal_page_transfers,
            num_transfers_in_go_signal_pages,
            dst_noc_multicast_info
        );
    }

    if (num_transfers_within_page) {
        num_transfers_in_go_signal_pages.push_back(num_transfers_within_page);
    }

    // Allocate some more space for GO signal
    program_pages.resize(program_pages.size() + align(src, DeviceCommand::PROGRAM_PAGE_SIZE) / sizeof(uint32_t));

    // Create a vector of all program binaries/cbs/semaphores
    uint32_t program_page_idx = 0;
    for (const KernelGroup &kg: program.get_kernel_groups()) {
        vector<KernelHandle> kernel_ids;
        if (kg.riscv0_id) kernel_ids.push_back(kg.riscv0_id.value());
        if (kg.riscv1_id) kernel_ids.push_back(kg.riscv1_id.value());
        if (kg.compute_id) kernel_ids.push_back(kg.compute_id.value());
        for (KernelHandle kernel_id: kernel_ids) {
            const Kernel* kernel = detail::GetKernel(program, kernel_id);

            for (const ll_api::memory& kernel_bin : kernel->binaries(device->id())) {
                kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                    std::copy(mem_ptr, mem_ptr + len, program_pages.begin() + program_page_idx);
                    program_page_idx = align(program_page_idx + len, noc_transfer_alignment_in_bytes / sizeof(uint32_t));
                });
            }
        }
    }

    for (const Semaphore& semaphore : program.semaphores()) {
        program_pages[program_page_idx] = semaphore.initial_value();
        program_page_idx += 4;
    }

    // Since GO signal begin in a new page, I need to advance my idx
    program_page_idx = align(program_page_idx, DeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t));

    // uint32_t dispatch_core_word = ((uint32_t)dispatch_core.y << 16) | dispatch_core.x;
    for (KernelGroup& kg: program.get_kernel_groups()) {
        // TODO(agrebenisan): Hanging when we extend the launch msg. Needs to be investigated. For now,
        // only supporting enqueue program for cq 0 on a device.
        // kg.launch_msg.dispatch_core_x = dispatch_core.x;
        // kg.launch_msg.dispatch_core_y = dispatch_core.y;
        static_assert(sizeof(launch_msg_t) % sizeof(uint32_t) == 0);
        uint32_t *launch_message_data = (uint32_t *)&kg.launch_msg;
        for (int i = 0; i < sizeof(launch_msg_t) / sizeof(uint32_t); i++) {
            program_pages[program_page_idx + i] = launch_message_data[i];
        }
        program_page_idx += sizeof(launch_msg_t) / sizeof(uint32_t);
    }

    uint32_t num_workers = 0;
    if (program.logical_cores().find(CoreType::WORKER) != program.logical_cores().end()) {
        num_workers = program.logical_cores().at(CoreType::WORKER).size();
    }

    return {
        .num_workers = num_workers,
        .program_pages = std::move(program_pages),
        .program_page_transfers = std::move(program_page_transfers),
        .runtime_arg_page_transfers = std::move(runtime_arg_page_transfers),
        .cb_config_page_transfers = std::move(cb_config_page_transfers),
        .go_signal_page_transfers = std::move(go_signal_page_transfers),
        .num_transfers_in_program_pages = std::move(num_transfers_in_program_pages),
        .num_transfers_in_runtime_arg_pages = std::move(num_transfers_in_runtime_arg_pages),
        .num_transfers_in_cb_config_pages = std::move(num_transfers_in_cb_config_pages),
        .num_transfers_in_go_signal_pages = std::move(num_transfers_in_go_signal_pages),
    };
}

// EnqueueReadBufferCommandSection
EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    void* dst,
    SystemMemoryManager& manager,
    uint32_t src_page_index,
    std::optional<uint32_t> pages_to_read) :
    command_queue_id(command_queue_id), dst(dst), manager(manager), buffer(buffer), src_page_index(src_page_index), pages_to_read(pages_to_read.has_value() ? pages_to_read.value() : buffer.num_pages()) {
    this->device = device;
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(uint32_t dst_address) {
    DeviceCommand command;

    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t num_pages = this->pages_to_read;
    uint32_t buffer_address = this->buffer.address();
    uint32_t dst_page_index = 0;

    if (is_sharded(this->buffer.buffer_layout())) {
        uint32_t num_cores = this->buffer.num_cores();
        uint32_t shard_size = this->buffer.shard_size();
        //TODO: for now all shards are same size of pages
        vector<uint32_t> num_pages_in_shards(num_cores, shard_size);
        vector<uint32_t> core_id_x;
        core_id_x.reserve(num_cores);
        vector<uint32_t> core_id_y;
        core_id_y.reserve(num_cores);
        auto all_cores = this->buffer.all_cores();
        for (const auto & core: all_cores) {
            CoreCoord physical_core = device->worker_core_from_logical_core(core);
            core_id_x.push_back(physical_core.x);
            core_id_y.push_back(physical_core.y);
        }
        command.add_buffer_transfer_instruction_sharded(
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
            core_id_y
        );
    }
    else {
        command.add_buffer_transfer_instruction(
            buffer_address,
            dst_address,
            num_pages,
            padded_page_size,
            (uint32_t)this->buffer.buffer_type(),
            uint32_t(BufferType::SYSTEM_MEMORY),
            this->src_page_index,
            dst_page_index);
    }

    // Targeting fast dispatch on remote device means commands have to be tunneled through ethernet
    bool cmd_consumer_on_ethernet = not device->is_mmio_capable();
    uint32_t consumer_cb_num_pages = (get_consumer_data_buffer_size(cmd_consumer_on_ethernet) / padded_page_size);

    if (consumer_cb_num_pages >= 4) {
        consumer_cb_num_pages = (consumer_cb_num_pages / 4) * 4;
        command.set_producer_consumer_transfer_num_pages(consumer_cb_num_pages / 4);
    } else {
        command.set_producer_consumer_transfer_num_pages(1);
    }

    uint32_t consumer_cb_size = consumer_cb_num_pages * padded_page_size;
    uint32_t producer_cb_num_pages = consumer_cb_num_pages * 2;
    uint32_t producer_cb_size = producer_cb_num_pages * padded_page_size;

    command.set_stall();
    command.set_page_size(padded_page_size);
    command.set_producer_cb_size(producer_cb_size);
    command.set_consumer_cb_size(consumer_cb_size);
    command.set_producer_cb_num_pages(producer_cb_num_pages);
    command.set_consumer_cb_num_pages(consumer_cb_num_pages);
    command.set_num_pages(num_pages);

    if (is_sharded(this->buffer.buffer_layout())) {
        uint32_t num_cores = this->buffer.num_cores();
        auto core_bank_indices = this->buffer.core_bank_indices();
        command.set_sharded_buffer_num_cores(num_cores);
    }

    TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");
    return command;
}

void EnqueueReadBufferCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    this->read_buffer_addr = this->manager.get_completion_queue_read_ptr(this->command_queue_id);

    const DeviceCommand cmd = this->assemble_device_command(this->read_buffer_addr);

    this->manager.issue_queue_reserve_back(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
}

EnqueueCommandType EnqueueReadBufferCommand::type() { return this->type_; }

// EnqueueWriteBufferCommand section
EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    const void* src,
    SystemMemoryManager& manager,
    uint32_t dst_page_index,
    std::optional<uint32_t> pages_to_write) :
    command_queue_id(command_queue_id), manager(manager), src(src), buffer(buffer), dst_page_index(dst_page_index), pages_to_write(pages_to_write.has_value() ? pages_to_write.value() : buffer.num_pages()) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");
    this->device = device;
}

const DeviceCommand EnqueueWriteBufferCommand::assemble_device_command(uint32_t src_address) {
    DeviceCommand command;
    uint32_t num_pages = this->pages_to_write;
    uint32_t buffer_address = this->buffer.address();
    uint32_t padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) { // should buffer.size() be num_pages * page_size
        padded_page_size = align(this->buffer.page_size(), 32);
    }
    uint32_t src_page_index = 0;
    if (is_sharded(this->buffer.buffer_layout())) {
        uint32_t num_cores = this->buffer.num_cores();
        uint32_t shard_size = this->buffer.shard_size();
        //TODO: for now all shards are same size of pages
        vector<uint32_t> num_pages_in_shards(num_cores, shard_size);
        vector<uint32_t> core_id_x;
        core_id_x.reserve(num_cores);
        vector<uint32_t> core_id_y;
        core_id_y.reserve(num_cores);
        auto all_cores = this->buffer.all_cores();
        for (const auto & core: all_cores) {
            CoreCoord physical_core = device->worker_core_from_logical_core(core);
            core_id_x.push_back(physical_core.x);
            core_id_y.push_back(physical_core.y);
        }
        command.add_buffer_transfer_instruction_sharded(
            src_address,
            buffer_address,
            num_pages,
            padded_page_size,
            (uint32_t) BufferType::SYSTEM_MEMORY,
            (uint32_t) this->buffer.buffer_type(),
            src_page_index,
            this->dst_page_index,
            num_pages_in_shards,
            core_id_x,
            core_id_y
        );
    }
    else {
        command.add_buffer_transfer_instruction(
        src_address,
        buffer_address,
        num_pages,
        padded_page_size,
        (uint32_t) BufferType::SYSTEM_MEMORY,
        (uint32_t) this->buffer.buffer_type(),
        src_page_index,
        this->dst_page_index);
    }

    // Targeting fast dispatch on remote device means commands have to be tunneled through ethernet
    bool cmd_consumer_on_ethernet = not device->is_mmio_capable();
    uint32_t consumer_cb_num_pages = (get_consumer_data_buffer_size(cmd_consumer_on_ethernet) / padded_page_size);

    if (consumer_cb_num_pages >= 4) {
        consumer_cb_num_pages = (consumer_cb_num_pages / 4) * 4;
        command.set_producer_consumer_transfer_num_pages(consumer_cb_num_pages / 4);
    } else {
        command.set_producer_consumer_transfer_num_pages(1);
    }

    uint32_t consumer_cb_size = consumer_cb_num_pages * padded_page_size;
    uint32_t producer_cb_num_pages = consumer_cb_num_pages * 2;
    uint32_t producer_cb_size = producer_cb_num_pages * padded_page_size;

    command.set_page_size(padded_page_size);
    command.set_producer_cb_size(producer_cb_size);
    command.set_consumer_cb_size(consumer_cb_size);
    command.set_producer_cb_num_pages(producer_cb_num_pages);
    command.set_consumer_cb_num_pages(consumer_cb_num_pages);
    command.set_num_pages(num_pages);

    if (is_sharded(this->buffer.buffer_layout())) {
        uint32_t num_cores = this->buffer.num_cores();
        auto core_bank_indices = this->buffer.core_bank_indices();
        command.set_sharded_buffer_num_cores(num_cores);
    }
    command.set_data_size(padded_page_size * num_pages);

    TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");
    return command;
}

void EnqueueWriteBufferCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);
    uint32_t data_size_in_bytes = cmd.get_data_size();

    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);

    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();

    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        // If page size is not 32B-aligned, we cannot do a contiguous write
        uint32_t src_address_offset = unpadded_src_offset;
        uint32_t padded_page_size = align(this->buffer.page_size(), 32);
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_in_bytes; sysmem_address_offset += padded_page_size) {
            this->manager.cq_write((char*)this->src + src_address_offset, this->buffer.page_size(), system_memory_temporary_storage_address + sysmem_address_offset);
            src_address_offset += this->buffer.page_size();
        }
    } else {
        this->manager.cq_write((char*)this->src + unpadded_src_offset, data_size_in_bytes, system_memory_temporary_storage_address);
    }

    this->manager.issue_queue_push_back(cmd_size, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);

    auto cmd_desc = cmd.get_desc();
}

EnqueueCommandType EnqueueWriteBufferCommand::type() { return this->type_; }

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    Device* device,
    Buffer& buffer,
    ProgramMap& program_to_dev_map,
    SystemMemoryManager& manager,
    const Program& program,
    bool stall,
    std::optional<std::reference_wrapper<Trace>> trace
    ) :
    command_queue_id(command_queue_id), buffer(buffer), program_to_dev_map(program_to_dev_map), manager(manager), program(program), stall(stall) {
    this->device = device;
    this->trace = trace;
}

const DeviceCommand EnqueueProgramCommand::assemble_device_command(uint32_t host_data_src) {
    DeviceCommand command;
    command.set_num_workers(this->program_to_dev_map.num_workers);

    auto populate_program_data_transfer_instructions =
        [&command](const vector<uint32_t>& num_transfers_per_page, const vector<transfer_info>& transfers_in_pages) {
            uint32_t i = 0;
            for (uint32_t j = 0; j < num_transfers_per_page.size(); j++) {
                uint32_t num_transfers_in_page = num_transfers_per_page[j];
                command.write_program_entry(num_transfers_in_page);
                for (uint32_t k = 0; k < num_transfers_in_page; k++) {
                    const auto [num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group, linked] = transfers_in_pages[i];
                    command.add_write_page_partial_instruction(num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group, linked);
                    i++;
                }
            }
        };

    command.set_is_program();

    // Not used, since we specified that this is a program command, and the consumer just looks at the write program
    // info
    constexpr static uint32_t dummy_dst_addr = 0;
    constexpr static uint32_t dummy_buffer_type = 0;
    uint32_t num_runtime_arg_pages = this->program_to_dev_map.num_transfers_in_runtime_arg_pages.size();
    uint32_t num_cb_config_pages = this->program_to_dev_map.num_transfers_in_cb_config_pages.size();
    uint32_t num_program_binary_pages = this->program_to_dev_map.num_transfers_in_program_pages.size();
    uint32_t num_go_signal_pages = this->program_to_dev_map.num_transfers_in_go_signal_pages.size();
    uint32_t num_host_data_pages = num_runtime_arg_pages + num_cb_config_pages;
    uint32_t num_cached_pages = num_program_binary_pages + num_go_signal_pages;
    uint32_t total_num_pages = num_host_data_pages + num_cached_pages;

    command.set_page_size(DeviceCommand::PROGRAM_PAGE_SIZE);
    command.set_num_pages(DeviceCommand::TransferType::RUNTIME_ARGS, num_runtime_arg_pages);
    command.set_num_pages(DeviceCommand::TransferType::CB_CONFIGS, num_cb_config_pages);
    command.set_num_pages(DeviceCommand::TransferType::PROGRAM_PAGES, num_program_binary_pages);
    command.set_num_pages(DeviceCommand::TransferType::GO_SIGNALS, num_go_signal_pages);
    command.set_num_pages(total_num_pages);

    command.set_data_size(
        DeviceCommand::PROGRAM_PAGE_SIZE *
        num_host_data_pages);

    const uint32_t page_index_offset = 0;
    if (num_host_data_pages) {
        command.add_buffer_transfer_instruction(
            host_data_src,
            dummy_dst_addr,
            num_host_data_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            uint32_t(BufferType::SYSTEM_MEMORY),
            dummy_buffer_type, page_index_offset, page_index_offset);

        if (num_runtime_arg_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_runtime_arg_pages, this->program_to_dev_map.runtime_arg_page_transfers);
        }

        if (num_cb_config_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_cb_config_pages, this->program_to_dev_map.cb_config_page_transfers);
        }
    }

    if (num_cached_pages) {
        command.add_buffer_transfer_instruction(
            this->buffer.address(),
            dummy_dst_addr,
            num_cached_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            uint32_t(this->buffer.buffer_type()),
            dummy_buffer_type, page_index_offset, page_index_offset);

        if (num_program_binary_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_program_pages, this->program_to_dev_map.program_page_transfers);
        }

        if (num_go_signal_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_go_signal_pages, this->program_to_dev_map.go_signal_page_transfers);
        }
    }

    // TODO (abhullar): deduce whether the producer is on ethernet core rather than hardcoding assuming tensix worker
    const uint32_t producer_cb_num_pages = (get_producer_data_buffer_size(/*use_eth_l1=*/false) / DeviceCommand::PROGRAM_PAGE_SIZE);
    const uint32_t producer_cb_size = producer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

    // Targeting fast dispatch on remote device means commands have to be tunneled through ethernet
    bool cmd_consumer_on_ethernet = not device->is_mmio_capable();
    const uint32_t consumer_cb_num_pages = (get_consumer_data_buffer_size(cmd_consumer_on_ethernet) / DeviceCommand::PROGRAM_PAGE_SIZE);
    const uint32_t consumer_cb_size = consumer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

    command.set_producer_cb_size(producer_cb_size);
    command.set_consumer_cb_size(consumer_cb_size);
    command.set_producer_cb_num_pages(producer_cb_num_pages);
    command.set_consumer_cb_num_pages(consumer_cb_num_pages);

    // Should only ever be set if we are
    // enqueueing a program immediately
    // after writing it to a buffer
    if (this->stall) {
        command.set_stall();
    }

    // This needs to be quite small, since programs are small
    command.set_producer_consumer_transfer_num_pages(4);

    return command;
}

void EnqueueProgramCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);

    uint32_t data_size_in_bytes = cmd.get_data_size();
    const uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);

    bool tracing = this->trace.has_value();
    vector<uint32_t> trace_host_data;
    uint32_t start_addr = system_memory_temporary_storage_address;
    constexpr static uint32_t padding_alignment = 16;
    for (size_t kernel_id = 0; kernel_id < this->program.num_kernels(); kernel_id++) {
        Kernel* kernel = detail::GetKernel(program, kernel_id);
        for (const auto& c: kernel->cores_with_runtime_args()) {
            const auto & core_runtime_args = kernel->runtime_args(c);
            this->manager.cq_write(core_runtime_args.data(), core_runtime_args.size() * sizeof(uint32_t), system_memory_temporary_storage_address);
            system_memory_temporary_storage_address = align(system_memory_temporary_storage_address + core_runtime_args.size() * sizeof(uint32_t), padding_alignment);

            if (tracing) {
                trace_host_data.insert(trace_host_data.end(), core_runtime_args.begin(), core_runtime_args.end());
                trace_host_data.resize(align(trace_host_data.size(), padding_alignment / sizeof(uint32_t)));
            }
        }
    }

    system_memory_temporary_storage_address = start_addr + align(system_memory_temporary_storage_address - start_addr, DeviceCommand::PROGRAM_PAGE_SIZE);

    array<uint32_t, 4> cb_data;
    for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers()) {
        for (const auto buffer_index : cb->buffer_indices()) {
            cb_data = {cb->address() >> 4, cb->size() >> 4, cb->num_pages(buffer_index), cb->size() / cb->num_pages(buffer_index) >> 4};
            this->manager.cq_write(cb_data.data(), padding_alignment, system_memory_temporary_storage_address);
            system_memory_temporary_storage_address += padding_alignment;
            if (tracing) {
                // No need to resize since cb_data size is guaranteed to be 16 bytes
                trace_host_data.insert(trace_host_data.end(), cb_data.begin(), cb_data.end());
            }
        }
    }

    this->manager.issue_queue_push_back(cmd_size, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
    if (tracing) {
        uint32_t trace_node_num_bytes = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + sizeof(uint32_t) * trace_host_data.size();
        Trace::TraceNode trace_node = {.command = cmd, .data = trace_host_data, .command_type = this->type(), .num_bytes = trace_node_num_bytes};
        Trace& trace_ = trace.value();
        trace_.record(trace_node);
    }
}

EnqueueCommandType EnqueueProgramCommand::type() { return this->type_; }

FinishCommand::FinishCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager) : command_queue_id(command_queue_id), manager(manager) { this->device = device; }

const DeviceCommand FinishCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.finish();
    return command;
}

void FinishCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    const DeviceCommand cmd = this->assemble_device_command(0);
    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    this->manager.issue_queue_reserve_back(cmd_size, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->manager.issue_queue_push_back(cmd_size, false, this->command_queue_id);
}

EnqueueCommandType FinishCommand::type() { return this->type_; }

// EnqueueWrapCommand section
EnqueueWrapCommand::EnqueueWrapCommand(uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, DeviceCommand::WrapRegion wrap_region) : command_queue_id(command_queue_id), manager(manager), wrap_region(wrap_region) {
    this->device = device;
}

const DeviceCommand EnqueueWrapCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.wrap(this->wrap_region);
    return command;
}

void EnqueueWrapCommand::process() {
    uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
    uint32_t space_left_in_bytes = this->manager.get_issue_queue_limit(this->command_queue_id) - write_ptr;
    // There may not be enough space in the issue queue to submit another command
    // In that case we write as big of a vector as we can with the wrap index (0) set to wrap type
    // To ensure that the issue queue write pointer does wrap, we need the wrap packet to be the full size of the issue queue
    uint32_t wrap_packet_size_bytes = std::min(space_left_in_bytes, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);

    const DeviceCommand cmd = this->assemble_device_command(0);
    this->manager.issue_queue_reserve_back(wrap_packet_size_bytes, this->command_queue_id);
    this->manager.cq_write(cmd.get_desc().data(), wrap_packet_size_bytes, write_ptr);
    if (this->wrap_region == DeviceCommand::WrapRegion::COMPLETION) {
        // Wrap the read pointers for completion queue because device will start writing data at head of completion queue and there are no more reads to be done at current completion queue write pointer
        // If we don't wrap the read then the subsequent read buffer command may attempt to read past the total command queue size
        // because the read buffer command will see updated write pointer to compute num pages to read but the local read pointer is pointing to tail of completion queue
        this->manager.wrap_completion_queue_rd_ptr(this->command_queue_id);
        this->manager.issue_queue_push_back(wrap_packet_size_bytes, LAZY_COMMAND_QUEUE_MODE, this->command_queue_id);
    } else {
        this->manager.wrap_issue_queue_wr_ptr(this->command_queue_id);
    }
}

EnqueueCommandType EnqueueWrapCommand::type() { return this->type_; }

// CommandQueue section
CommandQueue::CommandQueue(Device* device, uint32_t id): manager(*device->manager) {
    this->device = device;
    this->id = id;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();

    tt_cxy_pair issue_q_reader_location = dispatch_core_manager::get(device->num_hw_cqs()).issue_queue_reader_core(device->id(), channel, this->id);
    tt_cxy_pair completion_q_writer_location = dispatch_core_manager::get(device->num_hw_cqs()).completion_queue_writer_core(device->id(), channel, this->id);

    this->issue_queue_reader_core = CoreCoord(issue_q_reader_location.x, issue_q_reader_location.y);
    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);
}

CommandQueue::~CommandQueue() {}

void CommandQueue::enqueue_command(Command& command, bool blocking) {
    // For the time-being, doing the actual work of enqueing in
    // the main thread.
    // TODO(agrebenisan): Perform the following in a worker thread
    command.process();

    if (blocking) {
        this->finish();
    }
}


//TODO: Currently converting page ordering from interleaved to sharded and then doing contiguous read/write
// Look into modifying command to do read/write of a page at a time to avoid doing copy
void convert_interleaved_to_sharded_on_host(const void * host, const int num_pages,
                                        const int page_size,
                                        const std::vector<uint32_t>& page_map,
                                        bool read=false) {

    const uint32_t size_in_bytes = num_pages * page_size;

    void * temp = malloc(size_in_bytes);
    memcpy(temp, host, size_in_bytes);

    const void * dst = host;
    std::set<uint32_t> pages_seen;
    for (uint32_t host_page_id = 0; host_page_id < num_pages; host_page_id++) {
        auto dev_page_id = page_map[host_page_id];

        TT_ASSERT(dev_page_id < num_pages and dev_page_id >= 0);
        if (read) {
            memcpy((char* )dst + dev_page_id*page_size,
                (char *)temp + host_page_id*page_size,
                page_size
                );
        }
        else {
            memcpy((char* )dst + host_page_id*page_size,
                (char *)temp + dev_page_id*page_size,
                page_size
                );
        }
    }
    free(temp);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion region
void CommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("CommandQueue_read_buffer");
    TT_FATAL(blocking, "EnqueueReadBuffer only has support for blocking mode currently");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    uint32_t read_buffer_command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;
    while (total_pages_to_read > 0) {
        if ((this->manager.get_issue_queue_write_ptr(this->id)) + read_buffer_command_size >= this->manager.get_issue_queue_limit(this->id)) {
            this->wrap(DeviceCommand::WrapRegion::ISSUE, blocking);
        }

        const uint32_t command_completion_limit = this->manager.get_completion_queue_limit(this->id);
        uint32_t num_pages_available = (command_completion_limit - get_cq_completion_wr_ptr<false>(this->device->id(), this->id, this->size_B)) / padded_page_size;
        uint32_t pages_to_read = std::min(total_pages_to_read, num_pages_available);
        if (pages_to_read == 0) {
            // Wrap the completion region because a single page won't fit in available space
            // Wrap needs to be blocking because host needs updated write pointer to compute how many pages can be read
            this->wrap(DeviceCommand::WrapRegion::COMPLETION, true);
            num_pages_available = (command_completion_limit - get_cq_completion_wr_ptr<false>(this->device->id(), this->id, this->size_B)) / padded_page_size;
            pages_to_read = std::min(total_pages_to_read, num_pages_available);
        }

        tt::log_debug(tt::LogDispatch, "EnqueueReadBuffer for channel {}", this->id);
        EnqueueReadBufferCommand command(this->id, this->device, buffer, dst, this->manager, src_page_index, pages_to_read);
        this->enqueue_command(command, blocking);
        this->manager.completion_queue_wait_front(this->id); // wait for device to write data

        uint32_t bytes_read = pages_to_read * padded_page_size;
        if ((buffer.page_size() % 32) != 0) {
            // If page size is not 32B-aligned, we cannot do a contiguous copy
            uint32_t dst_address_offset = unpadded_dst_offset;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < bytes_read; sysmem_address_offset += padded_page_size) {
                tt::Cluster::instance().read_sysmem((char*)dst + dst_address_offset, buffer.page_size(), command.read_buffer_addr + sysmem_address_offset, mmio_device_id, channel);
                dst_address_offset += buffer.page_size();
            }
        } else {
            tt::Cluster::instance().read_sysmem((char*)dst + unpadded_dst_offset, bytes_read, command.read_buffer_addr, mmio_device_id, channel);
        }

        this->manager.completion_queue_pop_front(bytes_read, this->id);
        total_pages_to_read -= pages_to_read;
        src_page_index += pages_to_read;
        unpadded_dst_offset += pages_to_read * buffer.page_size();
    }

    if (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
        buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        convert_interleaved_to_sharded_on_host(dst, buffer.num_pages(),
                                        buffer.page_size(),
                                        buffer.dev_page_to_host_page_mapping(),
                                        true);
    }
}

void CommandQueue::enqueue_write_buffer(Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("CommandQueue_write_buffer");

    // TODO(agrebenisan): Fix these asserts after implementing multi-core CQ
    // TODO (abhullar): Use eth mem l1 size when issue queue interface kernel is on ethernet core
    TT_ASSERT(
        buffer.page_size() < MEM_L1_SIZE - get_data_section_l1_address(false),
        "Buffer pages must fit within the command queue data section");

    if (buffer.buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED or
        buffer.buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        convert_interleaved_to_sharded_on_host(src, buffer.num_pages(),
                                    buffer.page_size(),
                                    buffer.dev_page_to_host_page_mapping());
    }

    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t total_pages_to_write = buffer.num_pages();
    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    uint32_t dst_page_index = 0;
    while (total_pages_to_write > 0) {
        int32_t num_pages_available = (int32_t(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id)) - int32_t(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND)) / int32_t(padded_page_size);
        // If not even a single device command fits, we hit this edgecase
        num_pages_available = std::max(num_pages_available, 0);

        uint32_t pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);
        if (pages_to_write == 0) {
            // No space for command and data
            this->wrap(DeviceCommand::WrapRegion::ISSUE, blocking);
            num_pages_available = (int32_t(command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id)) - int32_t(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND)) / int32_t(padded_page_size);
            pages_to_write = std::min(total_pages_to_write, (uint32_t)num_pages_available);
        }

        tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);
        EnqueueWriteBufferCommand command(this->id, this->device, buffer, src, this->manager, dst_page_index, pages_to_write);
        this->enqueue_command(command, blocking);

        total_pages_to_write -= pages_to_write;
        dst_page_index += pages_to_write;
    }
}

void CommandQueue::enqueue_program(Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking) {
    ZoneScopedN("CommandQueue_enqueue_program");
    TT_FATAL(not blocking, "EnqueueProgram only has support for non-blocking mode currently");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it
    const uint64_t program_id = program.get_id();

    // Whether or not we should stall the producer from prefetching binary data. If the
    // data is cached, then we don't need to stall, otherwise we need to wait for the
    // data to land in DRAM first
    bool stall = false;
    // No shared cache so far, can come at a later time
    map<uint64_t, unique_ptr<Buffer>>& program_to_buffer = this->program_to_buffer(this->device->id());
    if (not program_to_buffer.count(program_id)) {
        stall = true;
        ProgramMap program_to_device_map = ConstructProgramMap(this->device, program);

        vector<uint32_t>& program_pages = program_to_device_map.program_pages;
        uint32_t program_data_size_in_bytes = program_pages.size() * sizeof(uint32_t);

        uint32_t write_buffer_command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + program_data_size_in_bytes;

        program_to_buffer.emplace(
            program_id,
            std::make_unique<Buffer>(
                this->device, program_data_size_in_bytes, DeviceCommand::PROGRAM_PAGE_SIZE, BufferType::DRAM));

        this->enqueue_write_buffer(*program_to_buffer.at(program_id), program_pages.data(), blocking);

        map<uint64_t, ProgramMap>& program_to_dev_map = this->program_to_dev_map(device->id());
        program_to_dev_map.emplace(program_id, std::move(program_to_device_map));
    }

    tt::log_debug(tt::LogDispatch, "EnqueueProgram for channel {}", this->id);

    uint32_t host_data_num_pages = this->program_to_dev_map(this->device->id()).at(program_id).runtime_arg_page_transfers.size() + this->program_to_dev_map(this->device->id()).at(program_id).cb_config_page_transfers.size();

    uint32_t host_data_and_device_command_size =
        DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + (host_data_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE);

    if ((this->manager.get_issue_queue_write_ptr(this->id)) + host_data_and_device_command_size >=
        this->manager.get_issue_queue_size(this->id)) {
        TT_FATAL(
            host_data_and_device_command_size <= this->manager.get_issue_queue_size(this->id) - CQ_START, "EnqueueProgram command size too large");
        this->wrap(DeviceCommand::WrapRegion::ISSUE, blocking);
    }

    EnqueueProgramCommand command(
        this->id,
        this->device,
        *this->program_to_buffer(this->device->id()).at(program_id),
        this->program_to_dev_map(this->device->id()).at(program_id),
        this->manager,
        program,
        stall,
        trace);

    this->enqueue_command(command, blocking);
}

void CommandQueue::wait_finish() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());

    // Poll to check that we're done.
    uint32_t finish_addr_offset = this->id * this->size_B;
    uint32_t finish;
    do {
        tt::Cluster::instance().read_sysmem(&finish, 4, HOST_CQ_FINISH_PTR + finish_addr_offset, mmio_device_id, channel);

        // There's also a case where the device can be hung due to an unanswered DPRINT WAIT and
        // a full print buffer. Poll the print server for this case and throw if it happens.
        if (DPrintServerHangDetected()) {
            TT_THROW("Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
        }
    } while (finish != 1);
    // Reset this value to 0 before moving on
    finish = 0;
    tt::Cluster::instance().write_sysmem(&finish, 4, HOST_CQ_FINISH_PTR + finish_addr_offset, mmio_device_id, channel);
}

void CommandQueue::finish() {
    ZoneScopedN("CommandQueue_finish");
    if ((this->manager.get_issue_queue_write_ptr(this->id)) + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND >=
        this->manager.get_issue_queue_limit(this->id)) {
        this->wrap(DeviceCommand::WrapRegion::ISSUE, false);
    }
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);

    FinishCommand command(this->id, this->device, this->manager);
    this->enqueue_command(command, false);
    this->wait_finish();
}

void CommandQueue::wrap(DeviceCommand::WrapRegion wrap_region, bool blocking) {
    ZoneScopedN("CommandQueue_wrap");
    tt::log_debug(tt::LogDispatch, "EnqueueWrap for channel {}", this->id);
    EnqueueWrapCommand command(this->id, this->device, this->manager, wrap_region);
    this->enqueue_command(command, blocking);
}

void CommandQueue::reset() {
    // Place the cores into reset since need to update a lot of core info
    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->device->id(), this->device->worker_core_from_logical_core(this->issue_queue_reader_core)));
    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->device->id(), this->device->worker_core_from_logical_core(this->completion_queue_writer_core)));
    tt::Cluster::instance().l1_barrier(this->device->id());

    detail::CommandQueueInit(this->device);
}

void CommandQueue::launch(launch_msg_t& msg) {
    tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->device->id(), this->device->worker_core_from_logical_core(this->issue_queue_reader_core)));
    tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->device->id(), this->device->worker_core_from_logical_core(this->completion_queue_writer_core)));

    std::unordered_set<CoreCoord> not_done_cores = {
        this->device->worker_core_from_logical_core(this->issue_queue_reader_core),
        this->device->worker_core_from_logical_core(this->completion_queue_writer_core)};

    llrt::internal_::wait_until_cores_done(this->device->id(), RUN_MSG_INIT, not_done_cores);

    tt::llrt::write_launch_msg_to_core(this->device->id(), this->device->worker_core_from_logical_core(this->issue_queue_reader_core), &msg);
    tt::llrt::write_launch_msg_to_core(this->device->id(), this->device->worker_core_from_logical_core(this->completion_queue_writer_core), &msg);
    llrt::internal_::wait_until_cores_done(this->device->id(), RUN_MSG_GO, not_done_cores);
}

Trace::Trace(CommandQueue& command_queue): command_queue(command_queue) {
    this->trace_complete = false;
    this->num_bytes = 0;
}

void Trace::record(const TraceNode& trace_node) {
    TT_ASSERT(not this->trace_complete, "Cannot record any more for a completed trace");
    this->num_bytes += trace_node.num_bytes;
    this->history.push_back(trace_node);
}

void Trace::create_replay() {
    this->command_queue.reset();

    // Reconstruct the hugepage from the command cache
    SystemMemoryManager& manager = this->command_queue.manager;
    const uint32_t command_queue_id = this->command_queue.id;
    const bool lazy_push = true;
    for (auto& [device_command, data, command_type, num_bytes]: this->history) {
        uint32_t issue_write_ptr = manager.get_issue_queue_write_ptr(command_queue_id);
        device_command.update_buffer_transfer_src(0, issue_write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
        manager.cq_write(device_command.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_write_ptr);
        manager.issue_queue_push_back(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, lazy_push, command_queue_id);

        uint32_t host_data_size = align(data.size() * sizeof(uint32_t), 16);
        manager.cq_write(data.data(), host_data_size, manager.get_issue_queue_write_ptr(command_queue_id));
        vector<uint32_t> read_back(host_data_size / sizeof(uint32_t), 0);
        tt::Cluster::instance().read_sysmem(read_back.data(), host_data_size, issue_write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, 0, 0);
        manager.issue_queue_push_back(host_data_size, lazy_push, command_queue_id);
    }
}

void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& dst, bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    TT_FATAL(blocking, "Non-blocking EnqueueReadBuffer not yet supported");

    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    dst.resize(buffer.page_size() * buffer.num_pages() / sizeof(uint32_t));
    cq.enqueue_read_buffer(buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& src, bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.enqueue_write_buffer(buffer, src.data(), blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, void* dst, bool blocking) {
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.enqueue_read_buffer(buffer, dst, blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, const void* src, bool blocking) {
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.enqueue_write_buffer(buffer, src, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking, std::optional<std::reference_wrapper<Trace>> trace) {
    ZoneScoped;
    TT_ASSERT(cq.id == 0, "EnqueueProgram only supported on first command queue on device for time being.");
    detail::DispatchStateCheck(true);

    detail::CompileProgram(cq.device, program);

    program.allocate_circular_buffers();
    detail::ValidateCircularBufferRegion(program, cq.device);

    cq.enqueue_program(program, trace, blocking);
}

void Finish(CommandQueue& cq) {
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    cq.finish();
}

void ClearProgramCache(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.program_to_buffer(cq.device->id()).clear();
    cq.program_to_dev_map(cq.device->id()).clear();
}

Trace BeginTrace(CommandQueue& command_queue) {
    Finish(command_queue); // Ensures that nothing being executed in command queue prior to beginning our trace
    // Resets the command queue state
    command_queue.reset();
    Program& command_queue_program = *command_queue.device->command_queue_programs[command_queue.id];
    launch_msg_t msg = command_queue_program.kernels_on_core(command_queue.issue_queue_reader_core)->launch_msg;
    command_queue.launch(msg);

    return Trace(command_queue);
}

void EndTrace(Trace& trace) {
    TT_ASSERT(not trace.trace_complete, "Already completed this trace");
    Finish(trace.command_queue);
    SystemMemoryManager& manager = trace.command_queue.manager;
    const uint32_t command_queue_id = trace.command_queue.id;
    TT_FATAL(trace.num_bytes <= manager.get_issue_queue_limit(command_queue_id), "Trace does not fit in system memory");
    trace.trace_complete = true;
    trace.create_replay();

    // Resend dispatch kernels to device with updated limits
    manager.set_custom_issue_limit_for_trace(command_queue_id, trace.num_bytes);
    trace.command_queue.reset();
    CommandQueue& command_queue = trace.command_queue;
    Program& command_queue_program = *command_queue.device->command_queue_programs[command_queue_id];
    launch_msg_t msg = command_queue_program.kernels_on_core(command_queue.issue_queue_reader_core)->launch_msg;
    command_queue.launch(msg);
}

void EnqueueTrace(Trace& trace, bool blocking) {
    // Run the trace
    trace.command_queue.manager.issue_queue_push_back(trace.num_bytes, false, trace.command_queue.id);

    // This will block because the wr toggles will be different between the host and the device
    if (blocking) {
        trace.command_queue.manager.issue_queue_reserve_back(0, trace.command_queue.id);
    }
}

}  // namespace tt::tt_metal
