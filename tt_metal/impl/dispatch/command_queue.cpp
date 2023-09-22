// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools.hpp"
#include "device_data.hpp"
#include "noc/noc_parameters.h"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

namespace tt::tt_metal {

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

u32 get_noc_multicast_encoding(const CoreCoord& top_left, const CoreCoord& bottom_right) {
    return NOC_MULTICAST_ENCODING(top_left.x, top_left.y, bottom_right.x, bottom_right.y);
}

u32 align(u32 addr, u32 alignment) { return ((addr - 1) | (alignment - 1)) + 1; }

ProgramSrcToDstAddrMap ConstructProgramSrcToDstAddrMap(const Device* device, Program& program) {
    // This function retrieves all the required information to group program binaries into sections,
    // such that each section is the largest amount of data that can be read into the dispatch
    // core's L1 at a time. For each section, it also specifies the relay program information,
    // as described in device_command.hpp.
    ProgramSrcToDstAddrMap program_to_device_map;
    vector<u32>& program_vector = program_to_device_map.program_vector;
    vector<ProgramSection>& sections = program_to_device_map.program_sections;

    // Initialize the worker notify section
    for (const CoreRange& core_range : program.get_worker_core_range_set().ranges()) {
        CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
        CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);
        program_to_device_map.multicast_message_noc_coords.push_back(
            std::make_pair(get_noc_multicast_encoding(physical_start, physical_end), core_range.size()));
    }
    program_to_device_map.num_workers = program.logical_cores().size();

    // 'section' here refers to a piece of the program buffer
    // that we can read in one shot into dispatch core L1
    u32 current_section_idx = 0;
    auto initialize_section = [&sections]() {
        // The purpose of this function is to create a new 'section'
        // as described in the above comment.
        vector<transfer_info> init_vec;
        map<TransferType, vector<transfer_info>> init_map;
        init_map.emplace(TransferType::B, init_vec);
        init_map.emplace(TransferType::N, init_vec);
        init_map.emplace(TransferType::T0, init_vec);
        init_map.emplace(TransferType::T1, init_vec);
        init_map.emplace(TransferType::T2, init_vec);
        init_map.emplace(TransferType::CB, init_vec);
        init_map.emplace(TransferType::SEM, init_vec);
        ProgramSection section = {.section = init_map, .size_in_bytes = 0};
        sections.push_back(section);
    };

    u32 start_in_bytes = DEVICE_COMMAND_DATA_ADDR;

    const char* DISPATCH_MAP_DUMP = std::getenv("TT_METAL_DISPATCH_MAP_DUMP");
    std::ofstream dispatch_dump_file;

    if (DISPATCH_MAP_DUMP != nullptr) {
        dispatch_dump_file.open(DISPATCH_MAP_DUMP, std::ofstream::out | std::ofstream::trunc);
    }

    auto write_program_kernel_transfer = [&](const Kernel* kernel, vector<TransferType> transfer_types) {
        size_t i = 0;
        const vector<ll_api::memory>& kernel_bins = kernel->binaries();
        CoreRangeSet cr_set = kernel->core_range_set();

        for (TransferType transfer_type : transfer_types) {
            u32 dst_code_location;

            switch (transfer_type) {
                case TransferType::B: dst_code_location = MEM_BRISC_INIT_LOCAL_L1_BASE; break;
                case TransferType::N: dst_code_location = MEM_NCRISC_INIT_LOCAL_L1_BASE; break;
                case TransferType::T0: dst_code_location = MEM_TRISC0_INIT_LOCAL_L1_BASE; break;
                case TransferType::T1: dst_code_location = MEM_TRISC1_INIT_LOCAL_L1_BASE; break;
                case TransferType::T2: dst_code_location = MEM_TRISC2_INIT_LOCAL_L1_BASE; break;
                default: TT_THROW("Invalid riscv type");
            }

            const ll_api::memory& kernel_bin = kernel_bins.at(i);
            i++;

            u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
            u32 num_new_bytes = kernel_bin.size() * sizeof(u32);

            if (num_bytes_so_far + num_new_bytes > MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR) {
                current_section_idx++;
                initialize_section();
            }

            // Appends the binary to a vector
            vector<pair<u32, u32>> binary_destination_and_size;
            kernel_bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len) {
                program_vector.insert(program_vector.end(), mem_ptr, mem_ptr + len);

                // Each section needs to be aligned to 16B
                u32 padding = (align(len * sizeof(u32), 16) / sizeof(u32)) - len;
                for (u32 i = 0; i < padding; i++) {
                    program_vector.push_back(0);
                }

                u32 destination = tt::llrt::relocate_dev_addr(addr, dst_code_location);
                u32 transfer_size_in_bytes = len * sizeof(u32);

                binary_destination_and_size.push_back(std::make_pair(destination, transfer_size_in_bytes));

                sections.at(current_section_idx).size_in_bytes += (transfer_size_in_bytes + sizeof(u32) * padding);

                if (DISPATCH_MAP_DUMP != nullptr) {
                    string name = "BINARY SPAN " + transfer_type_to_string(transfer_type);
                    update_dispatch_map_dump(name, std::move(vector<u32>(mem_ptr, mem_ptr + len)), dispatch_dump_file);
                }
            });

            // Need to initialize to 0 to avoid 'maybe-not-initialized' error
            u32 start_in_bytes_copy = 0;
            for (const CoreRange& core_range : cr_set.ranges()) {
                CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
                CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

                u32 noc_multicast_encoding = get_noc_multicast_encoding(physical_start, physical_end);

                start_in_bytes_copy = start_in_bytes;
                for (const auto& [destination, transfer_size_in_bytes] : binary_destination_and_size) {
                    sections.at(current_section_idx)
                        .at(transfer_type)
                        .push_back(std::make_tuple(
                            destination,
                            start_in_bytes_copy,
                            transfer_size_in_bytes,
                            noc_multicast_encoding,
                            core_range.size()));
                    start_in_bytes_copy = align(start_in_bytes_copy + transfer_size_in_bytes, 16);
                }
            }
            start_in_bytes = start_in_bytes_copy;
        }
    };

    auto write_cb_config_transfer = [&](const CircularBuffer& cb) {
        u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
        u32 num_new_bytes = 16;

        if (num_bytes_so_far + num_new_bytes > MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR) {
            current_section_idx++;
            initialize_section();
        }

        program_vector.push_back(cb.address() >> 4);
        program_vector.push_back(cb.size() >> 4);
        program_vector.push_back(cb.num_tiles());

        u32 page_size = cb.size() / cb.num_tiles();
        program_vector.push_back(page_size >> 4);  // Padding

        if (DISPATCH_MAP_DUMP != nullptr) {
            vector<u32> cb_config = {cb.address() >> 4, cb.size() >> 4, cb.num_tiles()};
            for (auto buffer_index : cb.buffer_indices()) {
                string name = "CB: " + std::to_string(buffer_index);
                update_dispatch_map_dump(name, cb_config, dispatch_dump_file);
            }
        }

        CoreRangeSet cr_set = cb.core_range_set();

        for (const CoreRange& core_range : cr_set.ranges()) {
            CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
            CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

            u32 noc_multicast_encoding = get_noc_multicast_encoding(physical_start, physical_end);

            for (auto buffer_index : cb.buffer_indices()) {
                sections.at(current_section_idx)
                    .at(TransferType::CB)
                    .push_back(std::make_tuple(
                        CIRCULAR_BUFFER_CONFIG_BASE +
                            buffer_index * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32),
                        start_in_bytes,
                        16,
                        noc_multicast_encoding,
                        core_range.size()));
            }
        }
        start_in_bytes += 16;
        sections.at(current_section_idx).size_in_bytes += 16;
    };

    // If only we could template lambdas in C++17, then wouldn't need this code duplication!
    // Maybe we can move to C++20 soon :)
    auto write_sem_config_transfer = [&](const Semaphore& sem) {
        u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
        u32 num_new_bytes = 16;

        if (num_bytes_so_far + num_new_bytes > MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR) {
            current_section_idx++;
            initialize_section();
        }

        program_vector.push_back(sem.initial_value());
        for (u32 i = 0; i < (SEMAPHORE_ALIGNMENT / sizeof(u32)) - 1; i++) {
            program_vector.push_back(0);
        }

        if (DISPATCH_MAP_DUMP != nullptr) {
            vector<u32> sem_config = {sem.initial_value()};
            update_dispatch_map_dump("SEM", sem_config, dispatch_dump_file);
        }

        CoreRangeSet cr_set = sem.core_range_set();

        for (const CoreRange& core_range : cr_set.ranges()) {
            CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
            CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

            u32 noc_multicast_encoding = get_noc_multicast_encoding(physical_start, physical_end);

            sections.at(current_section_idx)
                .at(TransferType::SEM)
                .push_back(
                    std::make_tuple(sem.address(), start_in_bytes, 4, noc_multicast_encoding, core_range.size()));
        }
        start_in_bytes += 16;
        sections.at(current_section_idx).size_in_bytes += 16;
    };

    initialize_section();
    const static std::map<tt::RISCV, vector<TransferType>> processor_to_transfer_type = {
        {tt::RISCV::BRISC, {TransferType::B}},
        {tt::RISCV::NCRISC, {TransferType::N}},
        {tt::RISCV::TRISC0, {TransferType::T0}},
        {tt::RISCV::TRISC1, {TransferType::T1}},
        {tt::RISCV::TRISC2, {TransferType::T2}},
        {tt::RISCV::COMPUTE, {TransferType::T0, TransferType::T1, TransferType::T2}},
    };

    for (KernelID kernel_id : program.kernel_ids()) {
        auto kernel = detail::GetKernel(program, kernel_id);
        vector<TransferType> riscv_type = processor_to_transfer_type.at(kernel->processor());
        write_program_kernel_transfer(kernel, riscv_type);
    }

    for (const CircularBuffer& cb : program.circular_buffers()) {
        write_cb_config_transfer(cb);
    }

    for (auto sem : program.semaphores()) {
        write_sem_config_transfer(sem);
    }

    TT_ASSERT(current_section_idx == 0, "Testing for just one section so far");

    return program_to_device_map;
}

string EnqueueCommandTypeToString(EnqueueCommandType ctype) {
    switch (ctype) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: return "EnqueueReadBuffer";
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: return "EnqueueWriteBuffer";
        default: TT_THROW("Invalid command type");
    }
}

u32 noc_coord_to_u32(CoreCoord coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }

// EnqueueReadBufferCommandSection
EnqueueReadBufferCommand::EnqueueReadBufferCommand(
    Device* device, Buffer& buffer, vector<u32>& dst, SystemMemoryWriter& writer) :
    dst(dst), writer(writer), buffer(buffer) {
    this->device = device;
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(u32 dst_address) {
    DeviceCommand command;

    u32 num_pages = this->buffer.size() / this->buffer.page_size();
    u32 padded_page_size = align(this->buffer.page_size(), 32);
    u32 data_size_in_bytes = padded_page_size * num_pages;
    command.set_data_size_in_bytes(data_size_in_bytes);

    u32 available_l1 = MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR;
    u32 burst_size = (available_l1 / padded_page_size) * padded_page_size;

    vector<CoreCoord> pcie_cores = this->device->cluster()->get_soc_desc(this->device->id()).pcie_cores;
    TT_ASSERT(pcie_cores.size() == 1, "Should only have one pcie core");
    const CoreCoord& pcie_core = pcie_cores.at(0);

    command.add_read_buffer_instruction(
        dst_address,
        NOC_XY_ENCODING(pcie_core.x, pcie_core.y),
        this->buffer.address(),

        data_size_in_bytes,
        burst_size,
        this->buffer.page_size(),
        padded_page_size,
        (u32) this->buffer.buffer_type());

    return command;
}

void EnqueueReadBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();
    this->read_buffer_addr = system_memory_temporary_storage_address;
    const auto command_desc = this->assemble_device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    u32 num_pages = this->buffer.size() / this->buffer.page_size();
    u32 padded_page_size = align(this->buffer.page_size(), 32);
    u32 data_size_in_bytes = padded_page_size * num_pages;
    u32 cmd_size = DeviceCommand::size_in_bytes() + data_size_in_bytes;

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueReadBufferCommand::type() { return this->type_; }

// EnqueueWriteBufferCommand section
EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
    Device* device, Buffer& buffer, vector<u32>& src, SystemMemoryWriter& writer) :
    writer(writer), src(src), buffer(buffer) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");

    this->device = device;
}

const DeviceCommand EnqueueWriteBufferCommand::assemble_device_command(u32 src_address) {
    DeviceCommand command;

    u32 num_pages = this->buffer.size() / this->buffer.page_size();
    u32 padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) {
        padded_page_size = align(this->buffer.page_size(), 32);
    }
    u32 data_size_in_bytes = padded_page_size * num_pages;
    command.set_data_size_in_bytes(data_size_in_bytes);

    u32 available_l1 = MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR;
    u32 burst_size = (available_l1 / padded_page_size) * padded_page_size;

    vector<CoreCoord> pcie_cores = this->device->cluster()->get_soc_desc(this->device->id()).pcie_cores;
    TT_ASSERT(pcie_cores.size() == 1, "Should only have one pcie core");
    const CoreCoord& pcie_core = pcie_cores.at(0);

    command.add_write_buffer_instruction(
        src_address,
        NOC_XY_ENCODING(pcie_core.x, pcie_core.y),
        this->buffer.address(),

        data_size_in_bytes,
        burst_size,
        this->buffer.page_size(),
        padded_page_size,
        (u32)(this->buffer.buffer_type()));

    return command;
}

void EnqueueWriteBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();

    const auto command_desc = this->assemble_device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    u32 num_pages = this->buffer.size() / this->buffer.page_size();

    u32 padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) {
        padded_page_size = align(this->buffer.page_size(), 32);
    }

    u32 data_size_in_bytes = padded_page_size * num_pages;

    u32 cmd_size = DeviceCommand::size_in_bytes() + data_size_in_bytes;
    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);

    // Need to deal with the edge case where our page
    // size is not 32B aligned
    if (this->buffer.page_size() % 32 != 0 and this->buffer.page_size() != this->buffer.size()) {
        vector<u32>::const_iterator src_iterator = this->src.begin();
        u32 num_u32s_in_page = this->buffer.page_size() / sizeof(u32);
        u32 num_pages = this->buffer.size() / this->buffer.page_size();
        u32 dst = system_memory_temporary_storage_address;
        for (u32 i = 0; i < num_pages; i++) {
            vector<u32> src_page(src_iterator, src_iterator + num_u32s_in_page);
            this->writer.cq_write(this->device, src_page, dst);
            src_iterator += num_u32s_in_page;
            dst = align(dst + this->buffer.page_size(), 32);
        }
    } else {
        this->writer.cq_write(this->device, this->src, system_memory_temporary_storage_address);
    }

    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueWriteBufferCommand::type() { return this->type_; }

EnqueueProgramCommand::EnqueueProgramCommand(
    Device* device,
    Buffer& buffer,
    ProgramSrcToDstAddrMap& program_to_dev_map,
    SystemMemoryWriter& writer,
    const RuntimeArgs& runtime_args) :
    buffer(buffer), program_to_dev_map(program_to_dev_map), writer(writer), runtime_args(runtime_args) {
    this->device = device;
}

const DeviceCommand EnqueueProgramCommand::assemble_device_command(u32 runtime_args_src) {
    DeviceCommand command;
    command.set_num_workers(this->program_to_dev_map.num_workers);
    command.set_num_multicast_messages(this->program_to_dev_map.multicast_message_noc_coords.size());

    // Set the noc coords for all the worker cores
    for (const auto& [multicast_message_noc_coord, num_messages] :
         this->program_to_dev_map.multicast_message_noc_coords) {
        command.set_multicast_message_noc_coord(multicast_message_noc_coord, num_messages);
    }

    u32 program_src = this->buffer.address();
    u32 program_src_noc = noc_coord_to_u32(this->buffer.noc_coordinates());

    for (const ProgramSection& section : this->program_to_dev_map.program_sections) {
        vector<TrailingWriteCommand> trailing_write_commands;

        // Kernel section
        for (const auto& [transfer_type, transfer_info_vector] : section.section) {
            for (const auto& [dst_addr, src, size_in_bytes, noc_multicast_encoding, num_receivers] :
                 transfer_info_vector) {
                TrailingWriteCommand trailing_write = {
                    .src = src,
                    .dst = dst_addr,
                    .dst_noc = noc_multicast_encoding,
                    .transfer_size = size_in_bytes,
                    .num_receivers = num_receivers};

                trailing_write_commands.push_back(trailing_write);
            }
        }

        // This is not fully correct since if there are multiple sections, they are not starting at the correct
        // part of the program buffer... a simpler method would be for there to be multiple buffers, where each
        // buffer owns a section... that is definitely a TODO(agrebenisan)
        command.add_read_multi_write_instruction(
            program_src, program_src_noc, section.size_in_bytes, trailing_write_commands);
    }

    // Deal with runtime args
    u32 data_size_in_bytes = 0;
    vector<TrailingWriteCommand> trailing_write_commands;
    u32 rt_args_src = DEVICE_COMMAND_DATA_ADDR;
    for (const auto& [core_coord, rt_arg_map] : this->runtime_args) {
        // u32 dst_noc = noc_coord_to_u32(core_coord);
        CoreCoord worker_dst_noc = this->device->worker_core_from_logical_core(core_coord);
        u32 dst_noc_multicast_encoding = get_noc_multicast_encoding(worker_dst_noc, worker_dst_noc);

        for (const auto& [riscv, rt_args_for_core] : rt_arg_map) {
            u32 dst;
            switch (riscv) {
                case tt::RISCV::BRISC: dst = BRISC_L1_ARG_BASE; break;
                case tt::RISCV::NCRISC: dst = NCRISC_L1_ARG_BASE; break;
                case tt::RISCV::COMPUTE: dst = TRISC_L1_ARG_BASE; break;
                default: TT_THROW("Invalid RISCV for runtime args");
            }

            u32 transfer_size = u32(rt_args_for_core.size() * sizeof(u32));

            TrailingWriteCommand trailing_write = {
                .src = rt_args_src,
                .dst = dst,
                .dst_noc = dst_noc_multicast_encoding,
                .transfer_size = transfer_size,
                .num_receivers = 1 /* Due to unicasting */};

            trailing_write_commands.push_back(trailing_write);
            data_size_in_bytes = align(data_size_in_bytes + transfer_size, 32);
            rt_args_src = align(rt_args_src + transfer_size, 32);
        }
    }

    u32 device_id = 0;
    vector<CoreCoord> pcie_cores = this->device->cluster()->get_soc_desc(device_id).pcie_cores;
    TT_ASSERT(pcie_cores.size() == 1, "Should only have one pcie core");
    const CoreCoord& pcie_core = pcie_cores.at(0);

    u32 host_noc_addr = noc_coord_to_u32({pcie_core.x, pcie_core.y});

    if (data_size_in_bytes) {
        command.add_read_multi_write_instruction(
            runtime_args_src, host_noc_addr, data_size_in_bytes, trailing_write_commands);
    }

    command.set_data_size_in_bytes(data_size_in_bytes);
    return command;
}

void EnqueueProgramCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();

    const DeviceCommand cmd = this->assemble_device_command(system_memory_temporary_storage_address);
    const auto command_desc = cmd.get_desc();

    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    vector<u32> rt_args_vector;
    for (const auto& [core_coord, rt_arg_map] : this->runtime_args) {
        for (const auto& [riscv, rt_args_for_core] : rt_arg_map) {
            rt_args_vector.insert(rt_args_vector.end(), rt_args_for_core.begin(), rt_args_for_core.end());
        }
    }

    const u32 cmd_size = DeviceCommand::size_in_bytes() + cmd.get_data_size_in_bytes();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, rt_args_vector, system_memory_temporary_storage_address);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueProgramCommand::type() { return this->type_; }

// FinishCommand section
FinishCommand::FinishCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) { this->device = device; }

const DeviceCommand FinishCommand::assemble_device_command(u32) {
    DeviceCommand command;
    command.finish();
    return command;
}

void FinishCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    const auto command_desc = this->assemble_device_command(0).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());

    u32 cmd_size = DeviceCommand::size_in_bytes();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType FinishCommand::type() { return this->type_; }

// EnqueueWrapCommand section
EnqueueWrapCommand::EnqueueWrapCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) {
    this->device = device;
}

const DeviceCommand EnqueueWrapCommand::assemble_device_command(u32) {
    DeviceCommand command;
    return command;
}

void EnqueueWrapCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;

    u32 space_left = HUGE_PAGE_SIZE - write_ptr;

    // Since all of the values will be 0, this will be equivalent to
    // a bunch of NOPs
    vector<u32> command_vector(space_left / sizeof(u32), 0);
    command_vector.at(0) = 1;  // wrap

    this->writer.cq_reserve_back(this->device, space_left);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_push_back(this->device, space_left);
}

EnqueueCommandType EnqueueWrapCommand::type() { return this->type_; }

// Sending dispatch kernel. TODO(agrebenisan): Needs a refactor
void send_dispatch_kernel_to_device(Device* device) {
    ZoneScoped;
    // Ideally, this should be some separate API easily accessible in
    // TT-metal, don't like the fact that I'm writing this from scratch

    Program dispatch_program = Program();
    CoreCoord dispatch_logical_core = *device->dispatch_cores().begin();
    vector<CoreCoord> pcie_cores = device->cluster()->get_soc_desc(device->id()).pcie_cores;
    TT_ASSERT(pcie_cores.size() == 1, "Should only have one pcie core");

    std::map<string, string> dispatch_defines = {
        {"IS_DISPATCH_KERNEL", ""},
        {"PCIE_NOC_X", std::to_string(pcie_cores[0].x)},
        {"PCIE_NOC_Y", std::to_string(pcie_cores[0].y)},
    };
    const char* DISPATCH_MAP_DUMP = std::getenv("TT_METAL_DISPATCH_MAP_DUMP");
    if (DISPATCH_MAP_DUMP) {
        dispatch_defines["TT_METAL_DISPATCH_MAP_DUMP"] = "";
    }
    std::vector<uint32_t> dispatch_compile_args = {DEVICE_DATA.TENSIX_SOFT_RESET_ADDR};
    auto dispatch_kernel = tt::tt_metal::CreateDataMovementKernel(
        dispatch_program,
        "tt_metal/impl/dispatch/kernels/command_queue.cpp",
        dispatch_logical_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = dispatch_compile_args,
            .defines = dispatch_defines
        }
    );

    detail::CompileProgram(device, dispatch_program);
    tt::tt_metal::detail::ConfigureDeviceWithProgram(device, dispatch_program);

    u32 fifo_addr = (HOST_CQ_FINISH_PTR + 32) >> 4;
    vector<u32> fifo_addr_vector = {fifo_addr};
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_READ_PTR, fifo_addr_vector);
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_WRITE_PTR, fifo_addr_vector);

    // Initialize wr toggle
    vector<u32> toggle_start_vector = {0};
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_READ_TOGGLE, toggle_start_vector);
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_logical_core, CQ_WRITE_TOGGLE, toggle_start_vector);

    // XXXX move this to llrt
    CoreCoord dpc = device->worker_core_from_logical_core(dispatch_logical_core);
    tt::llrt::disable_ncrisc(device->cluster(), device->id(), dpc);
    tt::llrt::disable_triscs(device->cluster(), device->id(), dpc);
    std::vector<uint32_t> run_mailbox_go_val = {RUN_MESSAGE_GO};
    tt::llrt::write_hex_vec_to_core(device->cluster(), device->id(), dpc, run_mailbox_go_val, MEM_RUN_MAILBOX_ADDRESS);
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device) {
    vector<u32> pointers(CQ_START / sizeof(u32), 0);
    pointers[0] = CQ_START >> 4;  // rd ptr (96 >> 4 = 6)

    device->cluster()->write_sysmem_vec(pointers, 0, 0);

    send_dispatch_kernel_to_device(device);
    this->device = device;
}

CommandQueue::~CommandQueue() {
    // if (this->device->cluster_is_initialized()) {
    //     this->finish();
    // }
}

void CommandQueue::enqueue_command(shared_ptr<Command> command, bool blocking) {
    // For the time-being, doing the actual work of enqueing in
    // the main thread.
    // TODO(agrebenisan): Perform the following in a worker thread
    command->process();

    if (blocking) {
        this->finish();
    }
}

void CommandQueue::enqueue_read_buffer(Buffer& buffer, vector<u32>& dst, bool blocking) {
    ZoneScopedN("CommandQueue_read_buffer");
    u32 read_buffer_command_size = DeviceCommand::size_in_bytes() + buffer.size();
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + read_buffer_command_size >= HUGE_PAGE_SIZE) {
        tt::log_assert(read_buffer_command_size <= HUGE_PAGE_SIZE - 96, "EnqueueReadBuffer command is too large");
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "EnqueueReadBuffer");

    shared_ptr<EnqueueReadBufferCommand> command =
        std::make_shared<EnqueueReadBufferCommand>(this->device, buffer, dst, this->sysmem_writer);

    // TODO(agrebenisan): Provide support so that we can achieve non-blocking
    // For now, make read buffer blocking since after the
    // device moves data into the buffer we want to read out
    // of, we then need to consume it into a vector. This
    // is easiest way to bring this up
    TT_ASSERT(blocking, "EnqueueReadBuffer only has support for blocking mode currently");
    this->enqueue_command(command, blocking);

    u32 num_pages = buffer.size() / buffer.page_size();
    u32 padded_page_size = align(buffer.page_size(), 32);
    u32 data_size_in_bytes = padded_page_size * num_pages;

    this->device->cluster()->read_sysmem_vec(dst, command->read_buffer_addr, data_size_in_bytes, 0);

    // This vector is potentially padded due to alignment constraints, so need to now remove the padding
    if ((buffer.page_size() % 32) != 0) {
        vector<u32> new_dst(buffer.size() / sizeof(u32), 0);
        u32 padded_page_size_in_u32s = align(buffer.page_size(), 32) / sizeof(u32);
        u32 new_dst_counter = 0;
        for (u32 i = 0; i < dst.size(); i += padded_page_size_in_u32s) {
            for (u32 j = 0; j < buffer.page_size() / sizeof(u32); j++) {
                new_dst[new_dst_counter] = dst[i + j];
                new_dst_counter++;
            }
        }
        dst = new_dst;
    }
}

void CommandQueue::enqueue_write_buffer(Buffer& buffer, vector<u32>& src, bool blocking) {
    ZoneScopedN("CommandQueue_write_buffer");
    TT_ASSERT(not blocking, "EnqueueWriteBuffer only has support for non-blocking mode currently");
    uint32_t src_size_bytes = src.size() * sizeof(uint32_t);
    TT_ASSERT(
        src_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer", src_size_bytes, buffer.size());
    TT_ASSERT(
        buffer.page_size() < MEM_L1_SIZE - DEVICE_COMMAND_DATA_ADDR,
        "Buffer pages must fit within the command queue data section");

    u32 write_buffer_command_size = DeviceCommand::size_in_bytes() + buffer.size();
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + write_buffer_command_size >= HUGE_PAGE_SIZE) {
        tt::log_assert(
            write_buffer_command_size <= HUGE_PAGE_SIZE - 96,
            "EnqueueWriteBuffer command is too large: {}",
            write_buffer_command_size);
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer");

    shared_ptr<EnqueueWriteBufferCommand> command =
        std::make_shared<EnqueueWriteBufferCommand>(this->device, buffer, src, this->sysmem_writer);
    this->enqueue_command(command, blocking);
}

void CommandQueue::enqueue_program(Program& program, bool blocking) {
    ZoneScopedN("CommandQueue_enqueue_program");
    TT_ASSERT(not blocking, "EnqueueProgram only has support for non-blocking mode currently");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it

    const u64 program_id = program.get_id();
    if (not this->program_to_buffer.count(program_id)) {
        ProgramSrcToDstAddrMap program_to_device_map = ConstructProgramSrcToDstAddrMap(this->device, program);

        vector<u32>& program_vector = program_to_device_map.program_vector;
        u32 program_data_size_in_bytes = program_vector.size() * sizeof(u32);

        u32 write_buffer_command_size = DeviceCommand::size_in_bytes() + program_data_size_in_bytes;

        this->program_to_buffer.emplace(
            program_id,
            std::make_unique<Buffer>(
                this->device, program_data_size_in_bytes, program_data_size_in_bytes, BufferType::DRAM));

        this->enqueue_write_buffer(*this->program_to_buffer.at(program_id), program_vector, blocking);

        this->program_to_dev_map.emplace(program_id, std::move(program_to_device_map));
    }
    tt::log_debug(tt::LogDispatch, "EnqueueProgram");

    auto get_current_runtime_args = [&program]() {
        RuntimeArgs runtime_args;
        for (const auto kernel_id : program.kernel_ids()) {
            const auto kernel = detail::GetKernel(program, kernel_id);
            tt::RISCV processor = kernel->processor();
            for (const auto& [logical_core, rt_args] : kernel->runtime_args()) {
                u32 padding = (align(rt_args.size() * sizeof(u32), 32) / sizeof(u32)) - rt_args.size();

                runtime_args[logical_core][processor] = rt_args;
                for (u32 i = 0; i < padding; i++) {
                    runtime_args[logical_core][processor].push_back(0);
                }
            }
        }
        return runtime_args;
    };

    auto rt_args = get_current_runtime_args();

    u32 runtime_args_and_device_command_size = DeviceCommand::size_in_bytes() + (rt_args.size() * sizeof(u32));
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + runtime_args_and_device_command_size >=
        HUGE_PAGE_SIZE) {
        tt::log_assert(
            runtime_args_and_device_command_size <= HUGE_PAGE_SIZE - 96, "EnqueueProgram command size too large");
        this->wrap();
    }

    shared_ptr<EnqueueProgramCommand> command = std::make_shared<EnqueueProgramCommand>(
        this->device,
        *this->program_to_buffer.at(program_id),
        this->program_to_dev_map.at(program_id),
        this->sysmem_writer,
        rt_args);

    this->enqueue_command(command, blocking);
}

void CommandQueue::finish() {
    ZoneScopedN("CommandQueue_finish");
    if ((this->sysmem_writer.cq_write_interface.fifo_wr_ptr << 4) + DeviceCommand::size_in_bytes() >= HUGE_PAGE_SIZE) {
        this->wrap();
    }
    tt::log_debug(tt::LogDispatch, "Finish");

    FinishCommand command(this->device, this->sysmem_writer);
    shared_ptr<FinishCommand> p = std::make_shared<FinishCommand>(std::move(command));
    this->enqueue_command(p, false);

    // We then poll to check that we're done.
    vector<u32> finish_vec;
    do {
        this->device->cluster()->read_sysmem_vec(finish_vec, HOST_CQ_FINISH_PTR, 4, 0);
    } while (finish_vec.at(0) != 1);

    // Reset this value to 0 before moving on
    finish_vec.at(0) = 0;
    this->device->cluster()->write_sysmem_vec(finish_vec, HOST_CQ_FINISH_PTR, 0);
}

void CommandQueue::wrap() {
    ZoneScopedN("CommandQueue_wrap");
    tt::log_debug(tt::LogDispatch, "EnqueueWrap");
    EnqueueWrapCommand command(this->device, this->sysmem_writer);
    shared_ptr<EnqueueWrapCommand> p = std::make_shared<EnqueueWrapCommand>(std::move(command));
    this->enqueue_command(p, false);
}

// OpenCL-like APIs
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_ASSERT(blocking, "Non-blocking EnqueueReadBuffer not yet supported");
    cq.enqueue_read_buffer(buffer, dst, blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking) {
    detail::DispatchStateCheck(true);
    cq.enqueue_write_buffer(buffer, src, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    detail::DispatchStateCheck(true);
    const char* COMPARE_DISPATCH_DEVICE_TO_HOST = std::getenv("TT_METAL_COMPARE_DISPATCH_DEVICE_TO_HOST");
    const char* DISPATCH_MAP_DUMP = std::getenv("TT_METAL_DISPATCH_MAP_DUMP");

    detail::CompileProgram(cq.device, program);

    if (COMPARE_DISPATCH_DEVICE_TO_HOST != nullptr) {
        TT_ASSERT(
            DISPATCH_MAP_DUMP != nullptr,
            "Cannot compare dispatch device output to host when dispatch map dump not enabled");
    }

    detail::ValidateCircularBufferRegion(program, cq.device);
    cq.enqueue_program(program, blocking);

    if (COMPARE_DISPATCH_DEVICE_TO_HOST != nullptr) {
        internal::wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(DISPATCH_MAP_DUMP, cq.device);
    }
}

void Finish(CommandQueue& cq) {
    detail::DispatchStateCheck(true);
    cq.finish();
}

} // namespace tt::tt_metal
