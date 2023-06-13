#include "command_queue.hpp"

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
        ProgramSection section = {.section = init_map, .size_in_bytes = 0};
        sections.push_back(section);
    };

    u32 start_in_bytes = DEVICE_COMMAND_DATA_ADDR;
    auto write_program_kernel_transfer = [&](const Kernel* kernel, vector<TransferType> transfer_types) {
        size_t i = 0;
        const vector<ll_api::memory>& kernel_bins = kernel->binaries();
        CoreRangeSet cr_set = kernel->core_range_set();

        for (TransferType transfer_type : transfer_types) {
            const ll_api::memory& kernel_bin = kernel_bins.at(i);
            i++;

            u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
            u32 num_new_bytes = kernel_bin.size() * sizeof(u32);

            if (num_bytes_so_far + num_new_bytes > 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR) {
                current_section_idx++;
                initialize_section();
            }

            kernel_bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len) {
                program_vector.insert(program_vector.end(), mem_ptr, mem_ptr + len);

                // Each section needs to be aligned to 16B
                u32 padding = (align(len * sizeof(u32), 16) / sizeof(u32)) - len;
                for (u32 i = 0; i < padding; i++) {
                    program_vector.push_back(0);
                }
            });

            for (const CoreRange& core_range : cr_set.ranges()) {
                CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
                CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

                u32 start_x = physical_start.x;
                u32 start_y = physical_start.y;
                u32 end_x = physical_end.x;
                u32 end_y = physical_end.y;

                u32 noc_multicast_encoding = NOC_MULTICAST_ENCODING(start_x, start_y, end_x, end_y);

                kernel_bin.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr,
                                             uint64_t addr,
                                             uint32_t len) {
                    u32 transfer_size_in_bytes = len * sizeof(u32);


                    sections.at(current_section_idx)
                        .at(transfer_type)
                        .push_back(std::make_tuple(
                            addr, start_in_bytes, transfer_size_in_bytes, noc_multicast_encoding, core_range.size()));
                    start_in_bytes = align(start_in_bytes + transfer_size_in_bytes, 16);
                });
            }
            sections.at(current_section_idx).size_in_bytes += align(kernel_bin.size() * sizeof(u32), 16);
        }
    };

    auto write_cb_config_transfer = [&](const CircularBuffer* cb) {
        u32 num_bytes_so_far = program_vector.size() * sizeof(u32);
        u32 num_new_bytes = 16;

        if (num_bytes_so_far + num_new_bytes > 1024 * 1024 - DEVICE_COMMAND_DATA_ADDR) {
            current_section_idx++;
            initialize_section();
        }

        program_vector.push_back(cb->address());
        program_vector.push_back(cb->size());
        program_vector.push_back(cb->num_tiles());
        program_vector.push_back(0);  // Padding

        CoreRangeSet cr_set = cb->core_range_set();

        for (const CoreRange& core_range : cr_set.ranges()) {
            CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
            CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

            u32 start_x = physical_start.x;
            u32 start_y = physical_start.y;
            u32 end_x = physical_end.x;
            u32 end_y = physical_end.y;

            u32 noc_multicast_encoding = NOC_MULTICAST_ENCODING(start_x, start_y, end_x, end_y);

            sections.at(current_section_idx)
                .at(TransferType::CB)
                .push_back(std::make_tuple(
                    CIRCULAR_BUFFER_CONFIG_BASE + cb->buffer_index() * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32),
                    start_in_bytes,
                    12,  // Only 3 of the UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG actually represent CB config data, the
                         // last one just used for 16B alignment... need some constant for this somewhere
                    noc_multicast_encoding,
                    core_range.size()));

            start_in_bytes += 16;
            sections.at(current_section_idx).size_in_bytes += 16;
        }
    };

    auto write_program_transfer = [&](TransferType transfer_type, const Kernel* kernel) {
        vector<TransferType> transfer_types;
        switch (transfer_type) {
            case TransferType::C: {
                transfer_types = {TransferType::T0, TransferType::T1, TransferType::T2};
                write_program_kernel_transfer(kernel, transfer_types);
            } break;
            case TransferType::N:
            case TransferType::B: {
                transfer_types = {transfer_type};
                write_program_kernel_transfer(kernel, transfer_types);
            } break;
            case TransferType::CB: break;
            default: TT_THROW("Invalid riscv_type");
        }
    };

    // TODO(agrebenisan): Once Almeet gets rid of kernel polymorphism,
    // need to come back and clean this up. Ideally this should be as
    // simple as just getting the type from the kernel.
    initialize_section();
    for (Kernel* kernel : program.kernels()) {
        TransferType riscv_type;
        switch (kernel->kernel_type()) {
            case (KernelType::DataMovement): {
                auto dm_kernel = dynamic_cast<DataMovementKernel*>(kernel);
                switch (dm_kernel->data_movement_processor()) {
                    case (DataMovementProcessor::RISCV_0): riscv_type = TransferType::B; break;
                    case (DataMovementProcessor::RISCV_1): riscv_type = TransferType::N; break;
                    default: TT_THROW("Invalid kernel type");
                }
            } break;
            case (KernelType::Compute): riscv_type = TransferType::C; break;
            default: TT_THROW("Invalid kernel type");
        }

        write_program_transfer(riscv_type, kernel);
    }

    for (const CircularBuffer* cb : program.circular_buffers()) {
        write_cb_config_transfer(cb);
    }

    // TODO(agrebenisan): Need to add support for sem configs
    for (const Semaphore* sem : program.semaphores()) {
        TT_THROW("Semaphores not yet supported in command queue");
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
    RuntimeArgs runtime_args) :
    buffer(buffer), program_to_dev_map(program_to_dev_map), writer(writer) {
    this->device = device;
    this->rt_args = runtime_args;
}

const DeviceCommand EnqueueProgramCommand::assemble_device_command(u32 runtime_args_src) {
    DeviceCommand command;
    command.launch();

    u32 program_src = this->buffer.address();
    u32 program_src_noc = noc_coord_to_u32(this->buffer.noc_coordinates());

    for (const ProgramSection& section : this->program_to_dev_map.program_sections) {
        u32 transfer_size = section.size_in_bytes;
        vector<TrailingWriteCommand> trailing_write_commands;
        i32 dst_code_location = 0;

        // Kernel section
        for (const auto& [transfer_type, transfer_info_vector] : section.section) {
            bool is_kernel = true;
            switch (transfer_type) {
                case TransferType::CB: is_kernel = false; break;
                case TransferType::B: dst_code_location = MEM_BRISC_INIT_LOCAL_L1_BASE; break;
                case TransferType::N: dst_code_location = MEM_NCRISC_INIT_LOCAL_L1_BASE; break;
                case TransferType::T0: dst_code_location = MEM_TRISC0_INIT_LOCAL_L1_BASE; break;
                case TransferType::T1: dst_code_location = MEM_TRISC1_INIT_LOCAL_L1_BASE; break;
                case TransferType::T2: dst_code_location = MEM_TRISC2_INIT_LOCAL_L1_BASE; break;
                default: TT_THROW("Invalid riscv type");
            }

            for (const auto& [dst_addr, src, size_in_bytes, noc_multicast_encoding, num_receivers] :
                 transfer_info_vector) {
                // If kernel, we need to put it into the address that will be relocated upon deassert
                u32 relocate_dst_addr;
                if (is_kernel) {
                    relocate_dst_addr = tt::llrt::relocate_dev_addr(dst_addr, dst_code_location);
                } else {
                    relocate_dst_addr = dst_addr;
                }

                TrailingWriteCommand trailing_write = {
                    .src = src,
                    .dst = relocate_dst_addr,
                    .dst_noc = noc_multicast_encoding,
                    .transfer_size = size_in_bytes,
                    .num_receivers = num_receivers};

                trailing_write_commands.push_back(trailing_write);
            }
        }

        // This is not fully correct since if there are multiple sections, they are not starting at the correct
        // part of the program buffer... a simpler method would be for there to be multiple buffers, where each
        // buffer owns a section... that is definitely a TODO(agrebenisan)
        command.add_read_multi_write_instruction(program_src, program_src_noc, transfer_size, trailing_write_commands);
    }

    // Deal with runtime args
    u32 data_size_in_bytes = 0;
    vector<TrailingWriteCommand> trailing_write_commands;
    u32 rt_args_src = DEVICE_COMMAND_DATA_ADDR;
    for (const auto& [core_coord, rt_arg_map] : this->rt_args) {
        // u32 dst_noc = noc_coord_to_u32(core_coord);
        CoreCoord worker_dst_noc = this->device->worker_core_from_logical_core(core_coord);
        u32 dst_noc_multicast_encoding =
            NOC_MULTICAST_ENCODING(worker_dst_noc.x, worker_dst_noc.y, worker_dst_noc.x, worker_dst_noc.y);

        for (const auto& [riscv, rt_args_for_core] : rt_arg_map) {
            u32 dst;
            switch (riscv) {
                case Riscv::B: dst = BRISC_L1_ARG_BASE; break;
                case Riscv::N: dst = NCRISC_L1_ARG_BASE; break;
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
    command.add_read_multi_write_instruction(runtime_args_src, host_noc_addr, data_size_in_bytes, trailing_write_commands);

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
    for (const auto& [core_coord, rt_arg_map] : this->rt_args) {
        for (const auto& [riscv, rt_args_for_core] : rt_arg_map) {
            rt_args_vector.insert(rt_args_vector.end(), rt_args_for_core.begin(), rt_args_for_core.end());

            // Need 16B alignment
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
    build_kernel_for_riscv_options.brisc_defines = brisc_defines;
    bool profile = false;

    GenerateBankToNocCoordHeaders(device, &build_kernel_for_riscv_options, "command_queue");
    generate_binary_for_risc(
        RISCID::BR, &build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name, 0, {}, profile);

    // Currently hard-coded. TODO(agrebenisan): Once we add support for multiple dispatch cores, this can be refactored,
    // but don't yet have a plan for where this variable should exist.
    CoreCoord dispatch_core = {1, 11};
    tt::llrt::test_load_write_read_risc_binary(device->cluster(), "command_queue/brisc/brisc.hex", 0, dispatch_core, 0);

    // Deassert reset of dispatch core BRISC. TODO(agrebenisan): Refactor once Paul's changes in
    tt::llrt::internal_::setup_riscs_on_specified_core(
        device->cluster(), 0, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {dispatch_core});
    device->cluster()->set_remote_tensix_risc_reset(tt_cxy_pair(0, dispatch_core), TENSIX_DEASSERT_SOFT_RESET);

    // TODO(agrebenisan): REMOVE!!! For time being, hardcoding in the core I am setting up the mem[0] jump for
    // the only worker core
    tt::llrt::program_brisc_startup_addr(device->cluster(), 0, {1, 1});
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device) {

    // Zeroing out the read/write pointers
    vector<u32> zeros(96 / sizeof(u32), 0);
    device->cluster()->write_sysmem_vec(zeros, 0, 0);

    send_dispatch_kernel_to_device(device);
    this->device = device;
}

CommandQueue::~CommandQueue() {
    if (this->device->cluster_is_initialized()) {
        this->finish();
    }
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
    shared_ptr<EnqueueWriteBufferCommand> command =
        std::make_shared<EnqueueWriteBufferCommand>(this->device, buffer, src, this->sysmem_writer);
    this->enqueue_command(command, blocking);
}

void CommandQueue::enqueue_program(Program& program, const RuntimeArgs& runtime_args, bool blocking) {
    TT_ASSERT(not blocking, "EnqueueProgram only has support for non-blocking mode currently");

    // Need to relay the program into DRAM if this is the first time
    // we are seeing it
    static int channel_id = 0;  // Are there issues with this being static?
    if (not this->program_to_buffer.count(&program)) {
        ProgramSrcToDstAddrMap program_to_device_map = ConstructProgramSrcToDstAddrMap(this->device, program);

        vector<u32>& program_vector = program_to_device_map.program_vector;
        u32 program_data_size_in_bytes = program_vector.size() * sizeof(u32);

        this->program_to_buffer.emplace(&program, std::make_unique<Buffer>(
            this->device, program_data_size_in_bytes, channel_id, program_data_size_in_bytes, BufferType::DRAM));

        this->enqueue_write_buffer(*this->program_to_buffer.at(&program), program_vector, blocking);

        // TODO(agrebenisan): Right now, write/read buffer device APIs assume we start at bank 0,
        // need to add support to start at a non-0-starting buffer
        // channel_id =
        //     (channel_id + 1) % this->device->cluster()
        //                            ->get_soc_desc(0)
        //                            .dram_cores.size();  // TODO(agrebenisan): Pull in num DRAM banks from SOC descriptor

        this->program_to_dev_map.emplace(&program, std::move(program_to_device_map));
    }

    shared_ptr<EnqueueProgramCommand> command = std::make_shared<EnqueueProgramCommand>(
        this->device,
        *this->program_to_buffer.at(&program),
        this->program_to_dev_map.at(&program),
        this->sysmem_writer,
        runtime_args);

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

void EnqueueProgram(CommandQueue& cq, Program& program, const RuntimeArgs& runtime_args, bool blocking) {
    tt::log_debug(tt::LogDispatch, "EnqueueProgram");

    cq.enqueue_program(program, runtime_args, blocking);
}

void Finish(CommandQueue& cq) {
    tt::log_debug(tt::LogDispatch, "Finish");

    cq.finish();
}
