#include "debug_tools.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/llrt/tt_debug_print_server.hpp"

u64 get_noc_multicast_encoding(const CoreCoord& top_left, const CoreCoord& bottom_right) {
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
    for (const CoreRange& core_range : program.logical_core_range_set().ranges()) {
        CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
        CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);
        program_to_device_map.multicast_message_noc_coords.push_back(std::make_pair(
            get_noc_multicast_encoding(physical_start, physical_end), core_range.size()));
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

            if (num_bytes_so_far + num_new_bytes > 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR) {
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

    auto write_cb_config_transfer = [&](const CircularBuffer* cb) {
        u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
        u32 num_new_bytes = 16;

        if (num_bytes_so_far + num_new_bytes > 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR) {
            current_section_idx++;
            initialize_section();
        }

        program_vector.push_back(cb->address() >> 4);
        program_vector.push_back(cb->size() >> 4);
        program_vector.push_back(cb->num_tiles());
        program_vector.push_back(0);  // Padding

        if (DISPATCH_MAP_DUMP != nullptr) {
            vector<u32> cb_config = {cb->address() >> 4, cb->size() >> 4, cb->num_tiles()};
            for (auto buffer_index : cb->buffer_indices()) {
                string name = "CB: " + std::to_string(buffer_index);
                update_dispatch_map_dump(name, cb_config, dispatch_dump_file);
            }
        }

        CoreRangeSet cr_set = cb->core_range_set();

        for (const CoreRange& core_range : cr_set.ranges()) {
            CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
            CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

            u32 noc_multicast_encoding = get_noc_multicast_encoding(physical_start, physical_end);

            for (auto buffer_index : cb->buffer_indices()) {
                sections.at(current_section_idx)
                    .at(TransferType::CB)
                    .push_back(std::make_tuple(
                        CIRCULAR_BUFFER_CONFIG_BASE +
                            buffer_index * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32),
                        start_in_bytes,
                        12,  // Only 3 of the UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG actually represent CB config data,
                             // the last one just used for 16B alignment... need some constant for this somewhere
                        noc_multicast_encoding,
                        core_range.size()));
            }
        }
        start_in_bytes += 16;
        sections.at(current_section_idx).size_in_bytes += 16;
    };

    // If only we could template lambdas in C++17, then wouldn't need this code duplication!
    // Maybe we can move to C++20 soon :)
    auto write_sem_config_transfer = [&](const Semaphore* sem) {
        u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
        u32 num_new_bytes = 16;

        if (num_bytes_so_far + num_new_bytes > 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR) {
            current_section_idx++;
            initialize_section();
        }

        program_vector.push_back(sem->initial_value());
        for (u32 i = 0; i < (SEMAPHORE_ALIGNMENT / sizeof(u32)) - 1; i++) {
            program_vector.push_back(0);
        }

        if (DISPATCH_MAP_DUMP != nullptr) {
            vector<u32> sem_config = {sem->initial_value()};
            update_dispatch_map_dump("SEM", sem_config, dispatch_dump_file);
        }

        CoreRangeSet cr_set = sem->core_range_set();

        for (const CoreRange& core_range : cr_set.ranges()) {
            CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
            CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

            u32 noc_multicast_encoding = get_noc_multicast_encoding(physical_start, physical_end);

            sections.at(current_section_idx)
                .at(TransferType::SEM)
                .push_back(
                    std::make_tuple(sem->address(), start_in_bytes, 4, noc_multicast_encoding, core_range.size()));
        }
        start_in_bytes += 16;
        sections.at(current_section_idx).size_in_bytes += 16;
    };

    // TODO(agrebenisan): Once Almeet gets rid of kernel polymorphism,
    // need to come back and clean this up. Ideally this should be as
    // simple as just getting the type from the kernel.
    initialize_section();
    for (Kernel* kernel : program.kernels()) {
        vector<TransferType> riscv_type;
        switch (kernel->kernel_type()) {
            case (KernelType::DataMovement): {
                auto dm_kernel = dynamic_cast<DataMovementKernel*>(kernel);
                switch (dm_kernel->data_movement_processor()) {
                    case (DataMovementProcessor::RISCV_0): riscv_type = {TransferType::B}; break;
                    case (DataMovementProcessor::RISCV_1): riscv_type = {TransferType::N}; break;
                    default: TT_THROW("Invalid kernel type");
                }
            } break;
            case (KernelType::Compute): riscv_type = {TransferType::T0, TransferType::T1, TransferType::T2}; break;
            default: TT_THROW("Invalid kernel type");
        }

        write_program_kernel_transfer(kernel, riscv_type);
    }

    for (const CircularBuffer* cb : program.circular_buffers()) {
        write_cb_config_transfer(cb);
    }

    for (const Semaphore* sem : program.semaphores()) {
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

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(u32 dst) {
    DeviceCommand command;
    command.set_data_size_in_bytes(this->buffer.size());

    u32 available_l1 = 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR;
    u32 potential_burst_size = available_l1;
    u32 num_bursts = this->buffer.size() / (available_l1);
    u32 num_pages_per_burst = potential_burst_size / this->buffer.page_size();
    u32 burst_size = num_pages_per_burst * this->buffer.page_size();
    u32 remainder_burst_size = this->buffer.size() - (num_bursts * burst_size);
    u32 num_pages_per_remainder_burst = remainder_burst_size / this->buffer.page_size();

    // Need to make a PCIE coordinate variable
    command.add_read_buffer_instruction(
        dst,
        NOC_XY_ENCODING(NOC_X(0), NOC_Y(4)),
        this->buffer.address(),
        noc_coord_to_u32(this->buffer.noc_coordinates()),
        num_bursts,
        burst_size,
        num_pages_per_burst,
        this->buffer.page_size(),
        remainder_burst_size,
        num_pages_per_remainder_burst,
        (u32)(this->buffer.buffer_type()));

    return command;
}

void EnqueueReadBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();
    this->read_buffer_addr = system_memory_temporary_storage_address;
    const auto command_desc = this->assemble_device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    u32 cmd_size = DeviceCommand::size_in_bytes() + this->buffer.size();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, this->dst, system_memory_temporary_storage_address);
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

    command.set_data_size_in_bytes(this->buffer.size());

    u32 available_l1 = 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR;
    u32 potential_burst_size = available_l1;
    u32 num_bursts = this->buffer.size() / (available_l1);
    u32 num_pages_per_burst = potential_burst_size / this->buffer.page_size();
    u32 burst_size = num_pages_per_burst * this->buffer.page_size();
    u32 remainder_burst_size = this->buffer.size() - (num_bursts * burst_size);
    u32 num_pages_per_remainder_burst = remainder_burst_size / this->buffer.page_size();

    // Need to make a PCIE coordinate variable
    command.add_write_buffer_instruction(
        src_address,
        NOC_XY_ENCODING(NOC_X(0), NOC_Y(4)),
        this->buffer.address(),
        noc_coord_to_u32(this->buffer.noc_coordinates()),
        num_bursts,
        burst_size,
        num_pages_per_burst,
        this->buffer.page_size(),
        remainder_burst_size,
        num_pages_per_remainder_burst,
        (u32)(this->buffer.buffer_type()));

    return command;
}

void EnqueueWriteBufferCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    u32 system_memory_temporary_storage_address = write_ptr + DeviceCommand::size_in_bytes();

    const auto command_desc = this->assemble_device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    u32 cmd_size = DeviceCommand::size_in_bytes() + this->buffer.size();

    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, this->src, system_memory_temporary_storage_address);
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
    for (const auto& [multicast_message_noc_coord, num_messages] : this->program_to_dev_map.multicast_message_noc_coords) {
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
    u32 host_noc_addr = noc_coord_to_u32({0, 4});
    command.add_read_multi_write_instruction(
        runtime_args_src, host_noc_addr, data_size_in_bytes, trailing_write_commands);

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

            // Need 32B alignment
            u32 padding = (align(rt_args_for_core.size() * sizeof(u32), 32) / sizeof(u32)) - rt_args_for_core.size();
            for (u32 i = 0; i < padding; i++) {
                rt_args_vector.push_back(0);
            }
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

// Sending dispatch kernel. TODO(agrebenisan): Needs a refactor
void send_dispatch_kernel_to_device(Device* device) {
    // Ideally, this should be some separate API easily accessible in
    // TT-metal, don't like the fact that I'm writing this from scratch
    std::string arch_name = tt::get_string_lowercase(device->arch());
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary", "command_queue");

    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    // Hard-coding as BRISC for now, could potentially be NCRISC
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/dispatch/command_queue.cpp";
    std::map<string, string> brisc_defines = {{"IS_DISPATCH_KERNEL", ""}};

    const char* DISPATCH_MAP_DUMP = std::getenv("TT_METAL_DISPATCH_MAP_DUMP");
    if (DISPATCH_MAP_DUMP) {
        brisc_defines.emplace("TT_METAL_DISPATCH_MAP_DUMP", "");
    }

    build_kernel_for_riscv_options.brisc_defines = brisc_defines;
    bool profile = false;

    GenerateBankToNocCoordHeaders(device, &build_kernel_for_riscv_options, "command_queue");
    generate_binary_for_risc(
        RISCID::BR, &build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name, 0, {}, profile);

    // Currently hard-coded. TODO(agrebenisan): Once we add support for multiple dispatch cores, this can be refactored,
    // but don't yet have a plan for where this variable should exist.
    CoreCoord dispatch_core = {1, 11};
    tt::llrt::test_load_write_read_risc_binary(device->cluster(), "command_queue/brisc/brisc.hex", 0, dispatch_core, 0);

    // Initialize cq pointers
    u32 fifo_addr = (HOST_CQ_FINISH_PTR + 32) >> 4;
    vector<u32> fifo_addr_vector = {fifo_addr};
    tt::llrt::write_hex_vec_to_core(device->cluster(), 0, {1, 11}, fifo_addr_vector, CQ_READ_PTR);
    tt::llrt::write_hex_vec_to_core(device->cluster(), 0, {1, 11}, fifo_addr_vector, CQ_WRITE_PTR);

    // Deassert reset of dispatch core BRISC. TODO(agrebenisan): Refactor once Paul's changes in
    tt::llrt::internal_::setup_riscs_on_specified_core(
        device->cluster(), 0, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {dispatch_core});
    device->cluster()->set_remote_tensix_risc_reset(tt_cxy_pair(0, dispatch_core), TENSIX_DEASSERT_SOFT_RESET);

    u32 chip_id = 0;  // TODO(agrebenisan): Remove hardcoding
    const auto& sdesc = device->cluster()->get_soc_desc(chip_id);

    for (const CoreCoord& worker : sdesc.workers) {
        tt::llrt::program_brisc_startup_addr(device->cluster(), 0, {worker.x, worker.y});
    }
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device) {
    // Zeroing out the read/write pointers
    vector<u32> zeros(96 / sizeof(u32), 0);
    device->cluster()->write_sysmem_vec(zeros, 0, 0);

    // BUG: Potential race since I don't initialize the device's write pointer copy, it does it
    // in its own deassert. Easy fix, just need to do it.
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
    shared_ptr<EnqueueReadBufferCommand> command =
        std::make_shared<EnqueueReadBufferCommand>(this->device, buffer, dst, this->sysmem_writer);

    // TODO(agrebenisan): Provide support so that we can achieve non-blocking
    // For now, make read buffer blocking since after the
    // device moves data into the buffer we want to read out
    // of, we then need to consume it into a vector. This
    // is easiest way to bring this up
    TT_ASSERT(blocking, "EnqueueReadBuffer only has support for blocking mode currently");
    this->enqueue_command(command, blocking);

    this->device->cluster()->read_sysmem_vec(dst, command->read_buffer_addr, command->buffer.size(), 0);
}

void CommandQueue::enqueue_write_buffer(Buffer& buffer, vector<u32>& src, bool blocking) {
    TT_ASSERT(not blocking, "EnqueueWriteBuffer only has support for non-blocking mode currently");
    TT_ASSERT(
        buffer.page_size() < 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR,
        "Buffer pages must fit within the command queue data section");

    shared_ptr<EnqueueWriteBufferCommand> command =
        std::make_shared<EnqueueWriteBufferCommand>(this->device, buffer, src, this->sysmem_writer);
    this->enqueue_command(command, blocking);
}

void CommandQueue::enqueue_program(Program& program, bool blocking) {
    TT_ASSERT(not blocking, "EnqueueProgram only has support for non-blocking mode currently");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it
    static int channel_id = 0;  // Are there issues with this being static?

    const u64 program_id = program.get_id();
    if (not this->program_to_buffer.count(program_id)) {
        ProgramSrcToDstAddrMap program_to_device_map = ConstructProgramSrcToDstAddrMap(this->device, program);

        vector<u32>& program_vector = program_to_device_map.program_vector;
        u32 program_data_size_in_bytes = program_vector.size() * sizeof(u32);

        this->program_to_buffer.emplace(
            program_id,
            std::make_unique<Buffer>(
                this->device, program_data_size_in_bytes, channel_id, program_data_size_in_bytes, BufferType::DRAM));

        this->enqueue_write_buffer(*this->program_to_buffer.at(program_id), program_vector, blocking);

        // TODO(agrebenisan): Right now, write/read buffer device APIs assume we start at bank 0,
        // need to add support to start at a non-0-starting buffer
        // channel_id =
        //     (channel_id + 1) % this->device->cluster()
        //                            ->get_soc_desc(0)
        //                            .dram_cores.size();  // TODO(agrebenisan): Pull in num DRAM banks from SOC
        //                            descriptor

        this->program_to_dev_map.emplace(program_id, std::move(program_to_device_map));
    }

    auto get_current_runtime_args = [&program]() {
        RuntimeArgs runtime_args;
        for (const auto kernel : program.kernels()) {
            tt::RISCV processor = kernel->processor();
            for (const auto& [logical_core, rt_args] : kernel->runtime_args()) {
                runtime_args[logical_core][processor] = rt_args;
            }
        }
        return runtime_args;
    };

    shared_ptr<EnqueueProgramCommand> command = std::make_shared<EnqueueProgramCommand>(
        this->device,
        *this->program_to_buffer.at(program_id),
        this->program_to_dev_map.at(program_id),
        this->sysmem_writer,
        get_current_runtime_args());

    this->enqueue_command(command, blocking);
}

void CommandQueue::finish() {
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

// OpenCL-like APIs
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking) {
    tt::log_debug(tt::LogDispatch, "EnqueueReadBuffer");

    TT_ASSERT(blocking, "Non-blocking EnqueueReadBuffer not yet supported");
    cq.enqueue_read_buffer(buffer, dst, blocking);
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking) {
    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer");

    cq.enqueue_write_buffer(buffer, src, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    tt::log_debug(tt::LogDispatch, "EnqueueProgram");
    const char* COMPARE_DISPATCH_DEVICE_TO_HOST = std::getenv("TT_METAL_COMPARE_DISPATCH_DEVICE_TO_HOST");
    const char* DISPATCH_MAP_DUMP = std::getenv("TT_METAL_DISPATCH_MAP_DUMP");

    if (COMPARE_DISPATCH_DEVICE_TO_HOST != nullptr) {
        tt::log_assert(
            DISPATCH_MAP_DUMP != nullptr,
            "Cannot compare dispatch device output to host when dispatch map dump not enabled");

        auto hart_mask = DPRINT_HART_BR;

        string device_dispatch_dump_file = "device_" + string(DISPATCH_MAP_DUMP);
        tt_start_debug_print_server(cq.device->cluster(), {0}, {{1, 11}}, hart_mask, device_dispatch_dump_file.c_str());
    }

    cq.enqueue_program(program, blocking);

    if (COMPARE_DISPATCH_DEVICE_TO_HOST != nullptr) {
        internal::wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(DISPATCH_MAP_DUMP, cq.device);
    }
}

void Finish(CommandQueue& cq) {
    tt::log_debug(tt::LogDispatch, "Finish");

    cq.finish();
}
