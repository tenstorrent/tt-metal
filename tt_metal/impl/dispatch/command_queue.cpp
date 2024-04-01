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
#include "tt_metal/common/logger.hpp"
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
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

const DeviceCommand EnqueueReadShardedBufferCommand::create_buffer_transfer_instruction(
    uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;
    return command;
}

const DeviceCommand EnqueueReadInterleavedBufferCommand::create_buffer_transfer_instruction(
    uint32_t dst_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;
    return command;
}

const DeviceCommand EnqueueReadBufferCommand::assemble_device_command(uint32_t dst_address) {
    uint32_t padded_page_size = align(this->buffer.page_size(), 32);
    uint32_t num_pages = this->pages_to_read;
    DeviceCommand command = this->create_buffer_transfer_instruction(dst_address, padded_page_size, num_pages);
    return command;
}

void EnqueueReadBufferCommand::process() {
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
    TT_ASSERT(buffer.is_dram() or buffer.is_l1(), "Trying to write to an invalid buffer");
    this->device = device;
    this->event = event;
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

const DeviceCommand EnqueueWriteInterleavedBufferCommand::create_buffer_transfer_instruction(
    uint32_t src_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;
    return command;
}

const DeviceCommand EnqueueWriteShardedBufferCommand::create_buffer_transfer_instruction(
    uint32_t src_address, uint32_t padded_page_size, uint32_t num_pages) {
    DeviceCommand command;
    return command;
}

const DeviceCommand EnqueueWriteBufferCommand::assemble_device_command(uint32_t src_address) {
    uint32_t num_pages = this->pages_to_write;
    uint32_t padded_page_size = this->buffer.page_size();
    if (this->buffer.page_size() != this->buffer.size()) {  // should buffer.size() be num_pages * page_size
        padded_page_size = align(this->buffer.page_size(), 32);
    }

    DeviceCommand command = this->create_buffer_transfer_instruction(src_address, padded_page_size, num_pages);

    return command;
}

void EnqueueWriteBufferCommand::process() {
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
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

const DeviceCommand EnqueueProgramCommand::assemble_device_command(uint32_t host_data_src) {
    DeviceCommand command;
    return command;
}

void EnqueueProgramCommand::process() {
}

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event):
    command_queue_id(command_queue_id), device(device), manager(manager), event(event) {
}


const DeviceCommand EnqueueRecordEventCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    return command;
}

void EnqueueRecordEventCommand::process() {
}

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
    uint32_t command_queue_id, Device* device, SystemMemoryManager& manager, uint32_t event, const Event& sync_event):
    command_queue_id(command_queue_id), device(device), manager(manager), event(event), sync_event(sync_event) {
        this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
        // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
        TT_ASSERT(command_queue_id != sync_event.cq_id || event != sync_event.event_id,
            "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. Event ID: {} CQ ID: {}",
            event, command_queue_id);
}

const DeviceCommand EnqueueWaitForEventCommand::assemble_device_command(uint32_t) {
    DeviceCommand command;
    return command;
}

void EnqueueWaitForEventCommand::process() {
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
        dispatch_core_manager::get(device->num_hw_cqs()).prefetcher_core(device->id(), channel, this->id);
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
void convert_interleaved_to_sharded_on_host(void * swapped, const void* host, const Buffer& buffer) {
    const uint32_t num_pages = buffer.num_pages();
    const uint32_t page_size = buffer.page_size();

    std::set<uint32_t> pages_seen;
    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
    uint32_t shard_width_in_pages = buffer.shard_spec().tensor_shard_spec.shape[1] / buffer.shard_spec().page_shape[1];
    for (uint32_t page_id = 0; page_id < num_pages; page_id++) {
        uint32_t local_num_pages;
        auto host_page_id = page_id;
        auto dev_page_id = buffer_page_mapping.host_page_to_dev_page_mapping_[host_page_id];
        TT_ASSERT(host_page_id < num_pages and host_page_id >= 0);
        memcpy((char*)swapped + dev_page_id * page_size, (char*)host + host_page_id * page_size, page_size);
    }
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void* dst, bool blocking) {
    this->enqueue_read_buffer(*buffer, dst, blocking);
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion region
void HWCommandQueue::enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking) {
    ZoneScopedN("HWCommandQueue_read_buffer");
    TT_THROW("Unsupported in FD 2.0");
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

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void HWCommandQueue::enqueue_write_buffer(const Buffer& buffer, const void* src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer");
    TT_THROW("Unsupported in FD 2.0");
}

void HWCommandQueue::enqueue_program(
    Program& program, std::optional<std::reference_wrapper<Trace>> trace, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");
    TT_THROW("Unsupported in FD 2.0");
}

void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");
    TT_THROW("Unsupported in FD 2.0");
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> event) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");
    TT_THROW("Unsupported in FD 2.0");
}


void HWCommandQueue::enqueue_trace() {
    ZoneScopedN("HWCommandQueue_enqueue_trace");
    TT_THROW("Not implemented");
}

void HWCommandQueue::copy_into_user_space(uint32_t event, uint32_t read_ptr, chip_id_t mmio_device_id, uint16_t channel) {
    const auto& [buffer_layout, page_size, padded_page_size, dev_page_to_host_page_mapping, dst, dst_offset, num_pages_read, cur_dev_page_id] =
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
        uint32_t dev_page_id = cur_dev_page_id;
        uint32_t read_src = read_ptr + align(EVENT_PADDED_SIZE, 32);
        for (uint32_t offset = 0; offset < padded_page_size * num_pages_read; offset += padded_page_size) {
            auto host_page_id = dev_page_to_host_page_mapping[dev_page_id];
            if(host_page_id.has_value()) {
                void* page_dst = (void*)(uint64_t(dst) + host_page_id.value() * page_size);
                tt::Cluster::instance().read_sysmem(
                    page_dst, page_size, read_src + offset, mmio_device_id, channel);
            }
            dev_page_id++;
        }
    }
    this->manager.completion_queue_pop_front(padded_num_bytes, this->id);
    this->issued_reads.erase(event);
}

void HWCommandQueue::read_completion_queue() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        if (this->exit_condition) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void HWCommandQueue::finish() {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);
    TT_THROW("Unsupported in FD 2.0");
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
        allocated_addr = allocator::allocate_buffer(*(buffer->device()->allocator_), buffer->shard_spec().size() * buffer->num_cores() * buffer->page_size(), buffer->page_size(), buffer->buffer_type(), alloc_md.bottom_up, buffer->num_cores());
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
    TT_THROW("EnqueueProgram currently unsupported in FD2.0");
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
        std::optional<std::reference_wrapper<Trace>> trace;  // TODO TMZ: remove trace from enqueue_program interface
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

bool EventQuery(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready(); // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(tt::LogMetal, "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed, event->device->id(), event->cq_id, event->event_id);
    return event_completed;
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
    log_debug(LogMetalTrace, "Begin trace capture");
    trace.begin_capture();
    return trace.queue();
}

void EndTrace(Trace& trace) {
    trace.end_capture();
    log_debug(LogMetalTrace, "End trace capture");
}

uint32_t InstantiateTrace(Trace& trace, CommandQueue& cq) {
    uint32_t trace_id = trace.instantiate(cq);
    return trace_id;
}

void ReleaseTrace(uint32_t trace_id) {
    if (trace_id == -1) {
        Trace::release_all();
    } else if (Trace::has_instance(trace_id)) {
        Trace::remove_instance(trace_id);
    }
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(Trace::has_instance(trace_id), "Trace instance " + std::to_string(trace_id) + " must exist on device");
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE,
        .blocking = blocking
    });
}

void EnqueueTraceImpl(CommandQueue& cq) {
    // STUB: Run the trace in eager mode for now
    // auto& tq = cq.trace()->queue();
    // for (const auto& cmd : tq.worker_queue) {
    //     cq.run_command_impl(cmd);
    // }
    TT_THROW("EnqueueTrace is not yet implemented!");
}

CommandQueue::CommandQueue(Device* device, uint32_t id, CommandQueueMode mode) :
    device_ptr(device),
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

CommandQueue::CommandQueue(Trace& trace) :
    device_ptr(nullptr),
    parent_thread_id(0),
    cq_id(-1),
    mode(CommandQueueMode::TRACE),
    worker_state(CommandQueueState::IDLE) {
}

CommandQueue::~CommandQueue() {
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (not this->trace_mode()) {
        TT_FATAL(this->worker_queue.empty(), "{} worker queue must be empty on destruction", this->name());
    }
}

HWCommandQueue& CommandQueue::hw_command_queue() {
    return this->device()->hw_command_queue(this->cq_id);
}

void CommandQueue::dump() {
    int cid = 0;
    log_info(LogMetalTrace, "Dumping {}, mode={}", this->name());
    for (const auto& cmd : this->worker_queue) {
        log_info(LogMetalTrace, "[{}]: {}", cid, cmd.type);
        cid++;
    }
}

std::string CommandQueue::name() {
    if (this->mode == CommandQueueMode::TRACE) {
        return "TraceQueue";
    }
    return "CQ" + std::to_string(this->cq_id);
}

void CommandQueue::wait_until_empty() {
    log_trace(LogDispatch, "{} WFI start", this->name());
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
    log_trace(LogDispatch, "{} WFI complete", this->name());
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
    tt::log_debug(tt::LogDispatch, "{} started worker thread", this->name());
}

void CommandQueue::stop_worker() {
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "{} stopped worker thread", this->name());
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
    log_trace(LogDispatch, "{} received {} in {} mode", this->name(), command.type, this->mode);
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
    log_trace(LogDispatch, "{} running {}", this->name(), command.type);
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
    log_trace(LogDispatch, "{} running {} complete", this->name(), command.type);
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
