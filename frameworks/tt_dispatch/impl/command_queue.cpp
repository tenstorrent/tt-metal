#include "command_queue.hpp"

#include "tt_metal/llrt/tt_debug_print_server.hpp"

string EnqueueCommandTypeToString(EnqueueCommandType ctype) {
    switch (ctype) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: return "EnqueueReadBuffer";
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: return "EnqueueWriteBuffer";
        default: TT_THROW("Invalid command type");
    }
}

u32 noc_coord_to_u32(tt_xy_pair coord) { return NOC_XY_ENCODING(NOC_X(coord.x), NOC_Y(coord.y)); }


// EnqueueReadBufferCommandSection
EnqueueReadBufferCommand::EnqueueReadBufferCommand(Device* device, Buffer& buffer, vector<u32>& dst, SystemMemoryWriter& writer) :
    dst(dst), writer(writer), buffer(buffer) {
    this->device = device;
}

const DeviceCommand EnqueueReadBufferCommand::device_command(u32 dst) {
    DeviceCommand command;
    command.set_data_size_in_bytes(this->buffer.size());

    u32 available_l1 = 1024 * 1024 - UNRESERVED_BASE;
    u32 potential_burst_size = available_l1;
    u32 num_bursts = this->buffer.size() / (available_l1);
    u32 num_pages_per_burst = potential_burst_size / this->buffer.page_size();
    u32 burst_size = num_pages_per_burst * this->buffer.page_size();
    u32 remainder_burst_size = this->buffer.size() - (num_bursts * burst_size);
    u32 num_pages_per_remainder_burst = remainder_burst_size / this->buffer.page_size();

    // Need to make a PCIE coordinate variable
    command.add_read_relay(
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
    const auto command_desc = this->device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    u32 cmd_size = DeviceCommand::size_in_bytes() + this->buffer.size();

    // Change noc write name
    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, this->dst, system_memory_temporary_storage_address);

    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueReadBufferCommand::type() { return this->type_; }

// EnqueueWriteBufferCommand section
EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(Device* device, Buffer& buffer, vector<u32>& src, SystemMemoryWriter& writer) :
    writer(writer), src(src), buffer(buffer) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM or buffer.buffer_type() == BufferType::L1,
        "Trying to write to an invalid buffer");
    this->device = device;
}

const DeviceCommand EnqueueWriteBufferCommand::device_command(u32 src_address) {
    DeviceCommand command;


    command.set_data_size_in_bytes(this->buffer.size());

    u32 available_l1 = 1024 * 1024 - UNRESERVED_BASE;
    u32 potential_burst_size = available_l1;
    u32 num_bursts = this->buffer.size() / (available_l1);
    u32 num_pages_per_burst = potential_burst_size / this->buffer.page_size();
    u32 burst_size = num_pages_per_burst * this->buffer.page_size();
    u32 remainder_burst_size = this->buffer.size() - (num_bursts * burst_size);
    u32 num_pages_per_remainder_burst = remainder_burst_size / this->buffer.page_size();

    // Need to make a PCIE coordinate variable
    command.add_write_relay(
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
    const auto command_desc = this->device_command(system_memory_temporary_storage_address).get_desc();
    vector<u32> command_vector(command_desc.begin(), command_desc.end());
    u32 cmd_size = DeviceCommand::size_in_bytes() + this->buffer.size();

    // Change noc write name
    this->writer.cq_reserve_back(this->device, cmd_size);
    this->writer.cq_write(this->device, command_vector, write_ptr);
    this->writer.cq_write(this->device, this->src, system_memory_temporary_storage_address);
    this->writer.cq_push_back(this->device, cmd_size);
}

EnqueueCommandType EnqueueWriteBufferCommand::type() { return this->type_; }

// FinishCommand section
FinishCommand::FinishCommand(Device* device, SystemMemoryWriter& writer) : writer(writer) { this->device = device; }

const DeviceCommand FinishCommand::device_command(u32) {
    DeviceCommand command;
    command.finish();

    return command;
}

void FinishCommand::process() {
    u32 write_ptr = this->writer.cq_write_interface.fifo_wr_ptr << 4;
    const auto command_desc = this->device_command(0).get_desc();
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
    std::string root_dir = tt::utils::get_root_dir();
    std::string arch_name = tt::utils::get_env_arch_name();
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary", "command_queue");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    // Hard-coding as BRISC for now, could potentially be NCRISC
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/dispatch/command_queue.cpp";
    std::map<string, string> brisc_defines = {{"IS_DISPATCH_KERNEL", ""}, {"DEVICE_DISPATCH_MODE", ""}};
    build_kernel_for_riscv_options.brisc_defines = brisc_defines;
    bool profile = false;
    generate_binary_for_risc(RISCID::BR, &build_kernel_for_riscv_options, out_dir_path, arch_name, 0, {}, profile);

    // Currently hard-coded. TODO(agrebenisan): Once we add support for multiple dispatch cores, this can be refactored,
    // but don't yet have a plan for where this variable should exist.
    tt_xy_pair dispatch_core = {1, 11};
    tt::llrt::test_load_write_read_risc_binary(
        device->cluster(), "built_kernels/command_queue/brisc/brisc.hex", 0, dispatch_core, 0);

    // Deassert reset of dispatch core BRISC. TODO(agrebenisan): Refactor once Paul's changes in
    tt::llrt::internal_::setup_riscs_on_specified_core(
        device->cluster(), 0, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {dispatch_core});
    device->cluster()->set_remote_tensix_risc_reset(tt_cxy_pair(0, dispatch_core), TENSIX_DEASSERT_SOFT_RESET);
}

// CommandQueue section
CommandQueue::CommandQueue(Device* device) {
    send_dispatch_kernel_to_device(device);

    this->device = device;
}

CommandQueue::~CommandQueue() {
    this->finish();

    // For time being, asserting reset of the whole board. Will need
    // to rethink once we get to multiple command queues
    tt::llrt::assert_reset_for_all_chips(this->device->cluster());
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

void Finish(CommandQueue& cq) {
    tt::log_debug(tt::LogDispatch, "Finish");

    cq.finish();
}
