// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include "debug_tools.hpp"
#include "device_data.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
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

uint32_t align(uint32_t addr, uint32_t alignment) { return ((addr - 1) | (alignment - 1)) + 1; }


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
                                             const vector<pair<uint32_t, uint32_t>>& dst_noc_transfer_info) -> uint32_t {
        while (num_bytes) {
            uint32_t num_bytes_left_in_page = DeviceCommand::PROGRAM_PAGE_SIZE - (src % DeviceCommand::PROGRAM_PAGE_SIZE);
            uint32_t num_bytes_in_transfer = std::min(num_bytes_left_in_page, num_bytes);
            src = align(src + num_bytes_in_transfer, noc_transfer_alignment_in_bytes);

            uint32_t transfer_instruction_idx = 1;
            for (const auto& [dst_noc_encoding, num_receivers] : dst_noc_transfer_info) {
                bool last = transfer_instruction_idx == dst_noc_transfer_info.size();
                transfer_info transfer_instruction = {.size_in_bytes = num_bytes_in_transfer, .dst = dst, .dst_noc_encoding = dst_noc_encoding, .num_receivers = num_receivers, .last_transfer_in_group = last};
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
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        const Kernel* kernel = detail::GetKernel(program, kernel_id);
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(kernel->core_range_set().ranges());

        vector<RISCV> sub_kernels;
        if (kernel->processor() == RISCV::COMPUTE) {
            sub_kernels = {RISCV::TRISC0, RISCV::TRISC1, RISCV::TRISC2};
        } else {
            sub_kernels = {kernel->processor()};
        }

        uint32_t sub_kernel_index = 0;
        for (const ll_api::memory& kernel_bin : kernel->binaries(device->id())) {
            kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                uint32_t num_bytes = len * sizeof(uint32_t);
                if ((dst & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
                    dst = (dst & ~MEM_LOCAL_BASE) + processor_to_local_mem_addr.at(sub_kernels[sub_kernel_index]);
                } else if ((dst & MEM_NCRISC_IRAM_BASE) == MEM_NCRISC_IRAM_BASE) {
                    dst = (dst & ~MEM_NCRISC_IRAM_BASE) + MEM_NCRISC_INIT_IRAM_L1_BASE;
                }

                src = update_program_page_transfers(
                    src, num_bytes, dst, program_page_transfers, num_transfers_in_program_pages, dst_noc_multicast_info);
            });
            sub_kernel_index++;
        }
    }

    // Step 4: Continue constructing pages for semaphore configs
    for (const Semaphore& semaphore : program.semaphores()) {
        vector<pair<uint32_t, uint32_t>> dst_noc_multicast_info =
            extract_dst_noc_multicast_info(semaphore.core_range_set().ranges());

        src = update_program_page_transfers(
            src,
            SEMAPHORE_ALIGNMENT,
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
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        const Kernel* kernel = detail::GetKernel(program, kernel_id);

        for (const ll_api::memory& kernel_bin : kernel->binaries(device->id())) {
            kernel_bin.process_spans([&](vector<uint32_t>::const_iterator mem_ptr, uint64_t dst, uint32_t len) {
                std::copy(mem_ptr, mem_ptr + len, program_pages.begin() + program_page_idx);
                program_page_idx = align(program_page_idx + len, noc_transfer_alignment_in_bytes / sizeof(uint32_t));
            });
        }
    }

    for (const Semaphore& semaphore : program.semaphores()) {
        program_pages[program_page_idx] = semaphore.initial_value();
        program_page_idx += 4;
    }

    // Since GO signal begin in a new page, I need to advance my idx
    program_page_idx = align(program_page_idx, DeviceCommand::PROGRAM_PAGE_SIZE / sizeof(uint32_t));
    for (const KernelGroup& kg: program.get_kernel_groups()) {
        uint32_t *launch_message_data = (uint32_t *)&kg.launch_msg;
        program_pages[program_page_idx] = launch_message_data[0];
        program_pages[program_page_idx + 1] = launch_message_data[1];
        program_pages[program_page_idx + 2] = launch_message_data[2];
        program_pages[program_page_idx + 3] = launch_message_data[3];
        program_page_idx += 4;
    }

    return {
        .num_workers = uint32_t(program.logical_cores().size()),
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
    Device* device, Buffer& buffer, void* dst, SystemMemoryWriter& writer) :
    dst(dst), writer(writer), buffer(buffer) {
    this->device = device;
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(uint32_t dst_address) {
    DeviceCommand command;

    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t data_size_in_bytes = padded_page_size * this->buffer.num_pages();

    command.add_buffer_transfer_instruction(
        this->buffer.address(),
        dst_address,
        this->buffer.num_pages(),
        padded_page_size,
        (uint32_t)this->buffer.buffer_type(),
        uint32_t(BufferType::SYSTEM_MEMORY));

    uint32_t consumer_cb_num_pages = (DeviceCommand::CONSUMER_DATA_BUFFER_SIZE / padded_page_size);

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
    command.set_num_pages(this->buffer.num_pages());
    command.set_data_size(padded_page_size * this->buffer.num_pages());

    TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");
    return command;
}

void EnqueueReadBufferCommand::process() {
    uint32_t write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    this->read_buffer_addr = system_memory_temporary_storage_address;
    const auto cmd = this->assemble_device_command(system_memory_temporary_storage_address);

    uint32_t num_pages = this->buffer.size() / this->buffer.page_size();
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t data_size_in_bytes = cmd.get_data_size();
    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;

    this->writer.cq_reserve_back(cmd_size);
    this->writer.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->writer.cq_push_back(cmd_size);
}

EnqueueCommandType EnqueueReadBufferCommand::type() { return this->type_; }

// EnqueueWriteBufferCommand section
EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    Device* device, Buffer& buffer, const void* src, SystemMemoryWriter& writer) :
    writer(writer), src(src), buffer(buffer) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");

    this->device = device;
}

const DeviceCommand EnqueueWriteBufferCommand::assemble_device_command(uint32_t src_address) {
    DeviceCommand command;

    uint32_t padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) {
        padded_page_size = align(this->buffer.page_size(), 32);
    }
    uint32_t data_size_in_bytes = padded_page_size * this->buffer.num_pages();
    command.add_buffer_transfer_instruction(
        src_address,
        this->buffer.address(),
        this->buffer.num_pages(),
        padded_page_size,
        (uint32_t) BufferType::SYSTEM_MEMORY,
        (uint32_t) this->buffer.buffer_type());

    uint32_t consumer_cb_num_pages = (DeviceCommand::CONSUMER_DATA_BUFFER_SIZE / padded_page_size);

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
    command.set_num_pages(this->buffer.num_pages());
    command.set_data_size(padded_page_size * this->buffer.num_pages());

    TT_ASSERT(padded_page_size <= consumer_cb_size, "Page is too large to fit in consumer buffer");

    return command;
}

void EnqueueWriteBufferCommand::process() {
    uint32_t write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const auto cmd = this->assemble_device_command(system_memory_temporary_storage_address);
    uint32_t data_size_in_bytes = cmd.get_data_size();

    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->writer.cq_reserve_back(cmd_size);
    this->writer.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);

    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        // If page size is not 32B-aligned, we cannot do a contiguous write
        uint32_t src_address_offset = 0;
        uint32_t padded_page_size = align(this->buffer.page_size(), 32);
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_in_bytes; sysmem_address_offset += padded_page_size) {
            this->writer.cq_write((char*)this->src + src_address_offset, this->buffer.page_size(), system_memory_temporary_storage_address + sysmem_address_offset);
            src_address_offset += this->buffer.page_size();
        }
    } else {
        this->writer.cq_write(this->src, data_size_in_bytes, system_memory_temporary_storage_address);
    }

    this->writer.cq_push_back(cmd_size);
}

EnqueueCommandType EnqueueWriteBufferCommand::type() { return this->type_; }

EnqueueProgramCommand::EnqueueProgramCommand(
    Device* device,
    Buffer& buffer,
    ProgramMap& program_to_dev_map,
    SystemMemoryWriter& writer,
    const Program& program,
    bool stall
    ) :
    buffer(buffer), program_to_dev_map(program_to_dev_map), writer(writer), program(program), stall(stall) {
    this->device = device;
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
                    const auto [num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group] = transfers_in_pages[i];
                    command.add_write_page_partial_instruction(num_bytes, dst, dst_noc, num_receivers, last_multicast_in_group);
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

    if (num_host_data_pages) {
        command.add_buffer_transfer_instruction(
            host_data_src,
            dummy_dst_addr,
            num_host_data_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            uint32_t(BufferType::SYSTEM_MEMORY),
            dummy_buffer_type);

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
            dummy_buffer_type);

        if (num_program_binary_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_program_pages, this->program_to_dev_map.program_page_transfers);
        }

        if (num_go_signal_pages) {
            populate_program_data_transfer_instructions(
                this->program_to_dev_map.num_transfers_in_go_signal_pages, this->program_to_dev_map.go_signal_page_transfers);
        }
    }

    constexpr static uint32_t producer_cb_num_pages = (DeviceCommand::PRODUCER_DATA_BUFFER_SIZE / DeviceCommand::PROGRAM_PAGE_SIZE);
    constexpr static uint32_t producer_cb_size = producer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

    constexpr static uint32_t consumer_cb_num_pages = (DeviceCommand::CONSUMER_DATA_BUFFER_SIZE / DeviceCommand::PROGRAM_PAGE_SIZE);
    constexpr static uint32_t consumer_cb_size = consumer_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

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
    uint32_t write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    uint32_t system_memory_temporary_storage_address = write_ptr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);

    uint32_t data_size_in_bytes = cmd.get_data_size();
    const uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size_in_bytes;
    this->writer.cq_reserve_back(cmd_size);
    this->writer.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);

    uint32_t start_addr = system_memory_temporary_storage_address;
    constexpr static uint32_t padding_alignment = 16;
    for (size_t kernel_id = 0; kernel_id < this->program.num_kernels(); kernel_id++) {
        Kernel* kernel = detail::GetKernel(program, kernel_id);
        for (const auto& c: kernel->cores_with_runtime_args()) {
            const auto & core_runtime_args = kernel->runtime_args(c);
            this->writer.cq_write(core_runtime_args.data(), core_runtime_args.size() * sizeof(uint32_t), system_memory_temporary_storage_address);
            system_memory_temporary_storage_address = align(system_memory_temporary_storage_address + core_runtime_args.size() * sizeof(uint32_t), padding_alignment);
        }
    }

    system_memory_temporary_storage_address = start_addr + align(system_memory_temporary_storage_address - start_addr, DeviceCommand::PROGRAM_PAGE_SIZE);

    array<uint32_t, 4> cb_data;
    for (const shared_ptr<CircularBuffer>& cb : program.circular_buffers()) {
        for (const auto buffer_index : cb->buffer_indices()) {
            cb_data = {cb->address() >> 4, cb->size() >> 4, cb->num_pages(buffer_index), cb->size() / cb->num_pages(buffer_index) >> 4};
            this->writer.cq_write(cb_data.data(), padding_alignment, system_memory_temporary_storage_address);
            system_memory_temporary_storage_address += padding_alignment;
        }
    }

    this->writer.cq_push_back(cmd_size);
}

EnqueueCommandType EnqueueProgramCommand::type() { return this->type_; }

// FinishCommand section
FinishCommand::FinishCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) { this->device = device; }

const DeviceCommand FinishCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    command.finish();
    return command;
}

void FinishCommand::process() {
    uint32_t write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    const auto cmd = this->assemble_device_command(0);
    uint32_t cmd_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;

    this->writer.cq_reserve_back(cmd_size);
    this->writer.cq_write(cmd.get_desc().data(), DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, write_ptr);
    this->writer.cq_push_back(cmd_size);
}

EnqueueCommandType FinishCommand::type() { return this->type_; }

// EnqueueWrapCommand section
EnqueueWrapCommand::EnqueueWrapCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) {
    this->device = device;
}

const DeviceCommand EnqueueWrapCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    return command;
}

void EnqueueWrapCommand::process() {
    uint32_t write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    uint32_t space_left = DeviceCommand::HUGE_PAGE_SIZE - write_ptr;

    // Since all of the values will be 0, this will be equivalent to
    // a bunch of NOPs
    vector<uint32_t> command_vector(space_left / sizeof(uint32_t), 0);
    command_vector[0] = 1;  // wrap

    this->writer.cq_reserve_back(space_left);
    this->writer.cq_write(command_vector.data(), command_vector.size() * sizeof(uint32_t), write_ptr);
    this->writer.cq_push_back(space_left);
}

EnqueueCommandType EnqueueWrapCommand::type() { return this->type_; }

// Sending dispatch kernel. TODO(agrebenisan): Needs a refactor
void send_dispatch_kernel_to_device(Device* device) {
    ZoneScoped;
    // Ideally, this should be some separate API easily accessible in
    // TT-metal, don't like the fact that I'm writing this from scratch

    Program dispatch_program = CreateProgram();
    auto dispatch_cores = device->dispatch_cores().begin();
    CoreCoord producer_logical_core = *dispatch_cores++;
    CoreCoord consumer_logical_core = *dispatch_cores;

    CoreCoord producer_physical_core = device->worker_core_from_logical_core(producer_logical_core);
    CoreCoord consumer_physical_core = device->worker_core_from_logical_core(consumer_logical_core);

    std::map<string, string> producer_defines = {
        {"IS_DISPATCH_KERNEL", ""},
        {"CONSUMER_NOC_X", std::to_string(consumer_physical_core.x)},
        {"CONSUMER_NOC_Y", std::to_string(consumer_physical_core.y)},
    };
    std::map<string, string> consumer_defines = {
        {"PRODUCER_NOC_X", std::to_string(producer_physical_core.x)},
        {"PRODUCER_NOC_Y", std::to_string(producer_physical_core.y)},
    };
    std::vector<uint32_t> dispatch_compile_args = {DEVICE_DATA.TENSIX_SOFT_RESET_ADDR};
    tt::tt_metal::CreateKernel(
        dispatch_program,
        "tt_metal/impl/dispatch/kernels/command_queue_producer.cpp",
        producer_logical_core,
        tt::tt_metal::DataMovementConfig {
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = dispatch_compile_args,
            .defines = producer_defines});

    tt::tt_metal::CreateKernel(
        dispatch_program,
        "tt_metal/impl/dispatch/kernels/command_queue_consumer.cpp",
        consumer_logical_core,
        tt::tt_metal::DataMovementConfig {
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = dispatch_compile_args,
            .defines = consumer_defines});

    tt::tt_metal::CreateSemaphore(dispatch_program, producer_logical_core, 2);
    tt::tt_metal::CreateSemaphore(dispatch_program, consumer_logical_core, 0);

    detail::CompileProgram(device, dispatch_program);
    tt::tt_metal::detail::ConfigureDeviceWithProgram(device, dispatch_program);

    uint32_t fifo_addr = (HOST_CQ_FINISH_PTR + 32) >> 4;
    vector<uint32_t> fifo_addr_vector = {fifo_addr};
    tt::tt_metal::detail::WriteToDeviceL1(device, producer_logical_core, CQ_READ_PTR, fifo_addr_vector);
    tt::tt_metal::detail::WriteToDeviceL1(device, producer_logical_core, CQ_WRITE_PTR, fifo_addr_vector);

    tt::Cluster::instance().l1_barrier(device->id());

    const std::tuple<uint32_t, uint32_t> tlb_data = tt::Cluster::instance().get_tlb_data(tt_cxy_pair(device->id(), device->worker_core_from_logical_core(*device->dispatch_cores().begin()))).value();
    auto [tlb_offset, tlb_size] = tlb_data;
    // std::cout << "CORE: " << device->worker_core_from_logical_core(*device->dispatch_cores().begin()).str() << std::endl;
    // std::cout << "after sending pointers to device. my tlb_offset: " << tlb_offset << ", my tlb_size: " << tlb_size << std::endl;

    launch_msg_t msg = dispatch_program.kernels_on_core(producer_logical_core)->launch_msg;

    // TODO(pkeller): Should use detail::LaunchProgram once we have a mechanism to avoid running all RISCs
    tt::llrt::write_launch_msg_to_core(device->id(), producer_physical_core, &msg);
    tt::llrt::write_launch_msg_to_core(device->id(), consumer_physical_core, &msg);
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device): sysmem_writer(device) {
    vector<uint32_t> pointers(CQ_START / sizeof(uint32_t), 0);
    pointers[0] = CQ_START >> 4;

    tt::Cluster::instance().write_sysmem(pointers.data(), pointers.size() * sizeof(uint32_t), 0, 0);

    send_dispatch_kernel_to_device(device);
    this->device = device;
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

void CommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("CommandQueue_read_buffer");
    TT_FATAL(blocking, "EnqueueReadBuffer only has support for blocking mode currently");
    uint32_t read_buffer_command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + buffer.size();
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + read_buffer_command_size >= DeviceCommand::HUGE_PAGE_SIZE) {
        TT_ASSERT(read_buffer_command_size <= DeviceCommand::HUGE_PAGE_SIZE - CQ_START, "EnqueueReadBuffer command is too large");
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "EnqueueReadBuffer");

    EnqueueReadBufferCommand command(this->device, buffer, dst, this->sysmem_writer);

    // TODO(agrebenisan): Provide support so that we can achieve non-blocking
    // For now, make read buffer blocking since after the
    // device moves data into the buffer we want to read out
    // of, we then need to consume it into a vector. This
    // is easiest way to bring this up
    this->enqueue_command(command, blocking);

    uint32_t num_pages = buffer.size() / buffer.page_size();
    uint32_t padded_page_size = align(buffer.page_size(), 32);
    uint32_t data_size_in_bytes = padded_page_size * num_pages;

    if ((buffer.page_size() % 32) != 0) {
        // If page size is not 32B-aligned, we cannot do a contiguous copy
        uint32_t dst_address_offset = 0;
        for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_in_bytes; sysmem_address_offset += padded_page_size) {
            tt::Cluster::instance().read_sysmem((char*)dst + dst_address_offset, buffer.page_size(), command.read_buffer_addr + sysmem_address_offset, 0);
            dst_address_offset += buffer.page_size();
        }
    } else {
        tt::Cluster::instance().read_sysmem(dst, data_size_in_bytes, command.read_buffer_addr, 0);
    }
}

void CommandQueue::enqueue_write_buffer(Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("CommandQueue_write_buffer");
    TT_FATAL(not blocking, "EnqueueWriteBuffer only has support for non-blocking mode currently");

    // TODO(agrebenisan): Fix these asserts after implementing multi-core CQ
    TT_ASSERT(
        buffer.page_size() < MEM_L1_SIZE - DeviceCommand::DATA_SECTION_ADDRESS,
        "Buffer pages must fit within the command queue data section");

    uint32_t write_buffer_command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + buffer.size();
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + write_buffer_command_size >= DeviceCommand::HUGE_PAGE_SIZE) {
        TT_ASSERT(
            write_buffer_command_size <= DeviceCommand::HUGE_PAGE_SIZE - CQ_START,
            "EnqueueWriteBuffer command is too large: {}",
            write_buffer_command_size);
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer");

    // TODO(agrebenisan): This could just be a stack variable since we
    // are just running in one thread
    EnqueueWriteBufferCommand command(this->device, buffer, src, this->sysmem_writer);
    this->enqueue_command(command, blocking);
}

void CommandQueue::enqueue_program(Program& program, bool blocking) {
    ZoneScopedN("CommandQueue_enqueue_program");
    TT_FATAL(not blocking, "EnqueueProgram only has support for non-blocking mode currently");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it
    const uint64_t program_id = program.get_id();

    // Whether or not we should stall the producer from prefetching binary data. If the
    // data is cached, then we don't need to stall, otherwise we need to wait for the
    // data to land in DRAM first
    bool stall = false;
    if (not this->program_to_buffer.count(program_id)) {
        stall = true;
        ProgramMap program_to_device_map = ConstructProgramMap(this->device, program);

        vector<uint32_t>& program_pages = program_to_device_map.program_pages;
        uint32_t program_data_size_in_bytes = program_pages.size() * sizeof(uint32_t);

        uint32_t write_buffer_command_size = DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + program_data_size_in_bytes;

        this->program_to_buffer.emplace(
            program_id,
            std::make_unique<Buffer>(
                this->device, program_data_size_in_bytes, DeviceCommand::PROGRAM_PAGE_SIZE, BufferType::DRAM));

        this->enqueue_write_buffer(*this->program_to_buffer.at(program_id), program_pages.data(), blocking);
        this->program_to_dev_map.emplace(program_id, std::move(program_to_device_map));
    }

    tt::log_debug(tt::LogDispatch, "EnqueueProgram");

    uint32_t host_data_num_pages = this->program_to_dev_map.at(program_id).runtime_arg_page_transfers.size() + this->program_to_dev_map.at(program_id).cb_config_page_transfers.size();

    uint32_t host_data_and_device_command_size =
        DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + (host_data_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE);

    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + host_data_and_device_command_size >=
        DeviceCommand::HUGE_PAGE_SIZE) {
        TT_ASSERT(
            host_data_and_device_command_size <= DeviceCommand::HUGE_PAGE_SIZE - CQ_START, "EnqueueProgram command size too large");
        this->wrap();
    }

    EnqueueProgramCommand command(this->device,
        *this->program_to_buffer.at(program_id),
        this->program_to_dev_map.at(program_id),
        this->sysmem_writer,
        program,
        stall);

    this->enqueue_command(command, blocking);
}

void CommandQueue::finish() {
    ZoneScopedN("CommandQueue_finish");
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND >=
        DeviceCommand::HUGE_PAGE_SIZE) {
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "Finish");

    FinishCommand command(this->device, this->sysmem_writer);
    this->enqueue_command(command, false);

    // We then poll to check that we're done.
    uint32_t finish;
    do {
        tt::Cluster::instance().read_sysmem(&finish, 4, HOST_CQ_FINISH_PTR, 0);
    } while (finish != 1);

    // Reset this value to 0 before moving on
    finish = 0;
    tt::Cluster::instance().write_sysmem(&finish, 4, HOST_CQ_FINISH_PTR, 0);
}

void CommandQueue::wrap() {
    ZoneScopedN("CommandQueue_wrap");
    tt::log_debug(tt::LogDispatch, "EnqueueWrap");
    EnqueueWrapCommand command(this->device, this->sysmem_writer);
    this->enqueue_command(command, false);
}

// OpenCL-like APIs
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& dst, bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    detail::DispatchStateCheck(true);
    TT_FATAL(blocking, "Non-blocking EnqueueReadBuffer not yet supported");

    // Only resizing here to keep with the original implementation. Notice how in the void*
    // version of this API, I assume the user mallocs themselves
    dst.resize(buffer.page_size() * buffer.num_pages() / sizeof(uint32_t));
    cq.enqueue_read_buffer(buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& src, bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    detail::DispatchStateCheck(true);
    cq.enqueue_write_buffer(buffer, src.data(), blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, void* dst, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.enqueue_read_buffer(buffer, dst, blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, const void* src, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.enqueue_write_buffer(buffer, src, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    ZoneScoped;
    detail::DispatchStateCheck(true);

    detail::CompileProgram(cq.device, program);

    program.allocate_circular_buffers();
    detail::ValidateCircularBufferRegion(program, cq.device);

    cq.enqueue_program(program, blocking);
}

void Finish(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.finish();
}

void ClearProgramCache(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.program_to_buffer.clear();
    cq.program_to_dev_map.clear();
}

}  // namespace tt::tt_metal
