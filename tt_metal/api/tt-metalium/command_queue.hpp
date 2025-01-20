// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "env_lib.hpp"
#include "command_queue_interface.hpp"
#include "device_command.hpp"
#include "lock_free_queue.hpp"
#include "program_command_sequence.hpp"
#include "worker_config_buffer.hpp"
#include "program_impl.hpp"
#include "trace_buffer.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class CommandQueue;
class BufferRegion;
class Event;
class Trace;
using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;

}  // namespace v0

class HWCommandQueue;

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType {
    ENQUEUE_READ_BUFFER,
    ENQUEUE_WRITE_BUFFER,
    GET_BUF_ADDR,
    ADD_BUFFER_TO_PROGRAM,
    SET_RUNTIME_ARGS,
    ENQUEUE_PROGRAM,
    ENQUEUE_TRACE,
    ENQUEUE_RECORD_EVENT,
    ENQUEUE_WAIT_FOR_EVENT,
    FINISH,
    FLUSH,
    TERMINATE,
    INVALID
};

string EnqueueCommandTypeToString(EnqueueCommandType ctype);

class CommandInterface;

using WorkerQueue = LockFreeQueue<CommandInterface>;

class Command {
   public:
    Command() {}
    virtual void process() {};
    virtual EnqueueCommandType type() = 0;
};

class EnqueueProgramCommand : public Command {
   private:
    uint32_t command_queue_id;
    IDevice* device;
    NOC noc_index;
    Program& program;
    SystemMemoryManager& manager;
    WorkerConfigBufferMgr& config_buffer_mgr;
    CoreCoord dispatch_core;
    CoreType dispatch_core_type;
    uint32_t expected_num_workers_completed;
    uint32_t packed_write_max_unicast_sub_cmds;
    uint32_t dispatch_message_addr;
    uint32_t multicast_cores_launch_message_wptr = 0;
    uint32_t unicast_cores_launch_message_wptr = 0;
    // TODO: There will be multiple ids once programs support spanning multiple sub_devices
    SubDeviceId sub_device_id = SubDeviceId{0};

   public:
    EnqueueProgramCommand(
        uint32_t command_queue_id,
        IDevice* device,
        NOC noc_index,
        Program& program,
        CoreCoord& dispatch_core,
        SystemMemoryManager& manager,
        WorkerConfigBufferMgr& config_buffer_mgr,
        uint32_t expected_num_workers_completed,
        uint32_t multicast_cores_launch_message_wptr,
        uint32_t unicast_cores_launch_message_wptr,
        SubDeviceId sub_device_id);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_PROGRAM; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueRecordEventCommand : public Command {
   private:
    uint32_t command_queue_id;
    IDevice* device;
    NOC noc_index;
    SystemMemoryManager& manager;
    uint32_t event_id;
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    tt::stl::Span<const SubDeviceId> sub_device_ids;
    bool clear_count;
    bool write_barrier;

   public:
    EnqueueRecordEventCommand(
        uint32_t command_queue_id,
        IDevice* device,
        NOC noc_index,
        SystemMemoryManager& manager,
        uint32_t event_id,
        tt::stl::Span<const uint32_t> expected_num_workers_completed,
        tt::stl::Span<const SubDeviceId> sub_device_ids,
        bool clear_count = false,
        bool write_barrier = true);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_RECORD_EVENT; }

    constexpr bool has_side_effects() { return false; }
};

class EnqueueWaitForEventCommand : public Command {
   private:
    uint32_t command_queue_id;
    IDevice* device;
    SystemMemoryManager& manager;
    const Event& sync_event;
    CoreType dispatch_core_type;
    bool clear_count;

   public:
    EnqueueWaitForEventCommand(
        uint32_t command_queue_id,
        IDevice* device,
        SystemMemoryManager& manager,
        const Event& sync_event,
        bool clear_count = false);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT; }

    constexpr bool has_side_effects() { return false; }
};

class EnqueueTraceCommand : public Command {
   private:
    uint32_t command_queue_id;
    Buffer& buffer;
    IDevice* device;
    SystemMemoryManager& manager;
    std::shared_ptr<detail::TraceDescriptor>& descriptor;
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed;
    bool clear_count;
    NOC noc_index;
    CoreCoord dispatch_core;
   public:
    EnqueueTraceCommand(
        uint32_t command_queue_id,
        IDevice* device,
        SystemMemoryManager& manager,
        std::shared_ptr<detail::TraceDescriptor>& descriptor,
        Buffer& buffer,
        std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
        NOC noc_index,
        CoreCoord dispatch_core);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_TRACE; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueTerminateCommand : public Command {
   private:
    uint32_t command_queue_id;
    IDevice* device;
    SystemMemoryManager& manager;

   public:
    EnqueueTerminateCommand(uint32_t command_queue_id, IDevice* device, SystemMemoryManager& manager);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::TERMINATE; }

    constexpr bool has_side_effects() { return false; }
};

struct RuntimeArgsMetadata {
    CoreCoord core_coord;
    std::shared_ptr<RuntimeArgs> runtime_args_ptr;
    std::shared_ptr<Kernel> kernel;
    std::vector<uint32_t> update_idx;
};

// Common interface for all command queue types
struct CommandInterface {
    EnqueueCommandType type;
    std::optional<bool> blocking;
    std::optional<std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>> buffer;
    Program* program;
    std::optional<RuntimeArgsMetadata> runtime_args_md;
    std::optional<const Buffer*> shadow_buffer;
    std::optional<HostDataType> src;
    std::optional<void*> dst;
    std::optional<std::shared_ptr<Event>> event;
    std::optional<uint32_t> trace_id;
    std::optional<BufferRegion> region;
    tt::stl::Span<const SubDeviceId> sub_device_ids;
};

inline namespace v0 {

class CommandQueue {
    friend class Trace;

public:
    enum class CommandQueueMode {
        PASSTHROUGH = 0,
        ASYNC = 1,
        TRACE = 2,
    };
    enum class CommandQueueState {
        IDLE = 0,
        RUNNING = 1,
        TERMINATE = 2,
    };

    CommandQueue() = delete;

    CommandQueue(IDevice* device, uint32_t id, CommandQueueMode mode = CommandQueue::default_mode());
    ~CommandQueue();

    // Trace queue constructor
    CommandQueue(Trace& trace);

    // Getters for private members
    IDevice* device() const { return this->device_ptr; }
    uint32_t id() const { return this->cq_id; }

    // Blocking method to wait for all commands to drain from the queue
    // Optional if used in passthrough mode (async_mode = false)
    void wait_until_empty();

    // Schedule a command to be run on the device
    // Blocking if in passthrough mode. Non-blocking if in async mode
    void run_command(CommandInterface&& command);

    // API for setting/getting the mode of the command queue
    // TODO: disallow changing the mode of the queue. This is error prone, because changing mode requires
    // accordingly updating the higher-level abstractions.
    void set_mode(const CommandQueueMode& mode);
    CommandQueueMode get_mode() const { return this->mode; }

    // Reference to the underlying hardware command queue, non-const because side-effects are allowed
    HWCommandQueue& hw_command_queue();

    // The empty state of the worker queue
    bool empty() const { return this->worker_queue.empty(); }

    // Dump methods for name and pending commands in the queue
    void dump();
    std::string name();

    static CommandQueueMode default_mode() {
        // Envvar is used for bringup and debug only. Will be removed in the future and should not be relied on in
        // production.
        static int value =
            parse_env<int>("TT_METAL_CQ_ASYNC_MODE", /*default_value=*/static_cast<int>(CommandQueueMode::PASSTHROUGH));
        return static_cast<CommandQueue::CommandQueueMode>(value);
    }
    // Determine if any CQ is using Async mode
    static bool async_mode_set() { return num_async_cqs > 0; }

   private:
    // Initialize Command Queue Mode based on the env-var. This will be default, unless the user excplictly sets the
    // mode using set_mode.
    CommandQueueMode mode;
    CommandQueueState worker_state;
    std::unique_ptr<std::thread> worker_thread;
    WorkerQueue worker_queue;
    uint32_t cq_id = 0;
    IDevice* device_ptr = nullptr;
    Trace* trace_ptr = nullptr;

    void start_worker();
    void stop_worker();
    void run_worker();
    void run_command_impl(const CommandInterface& command);

    bool async_mode() { return this->mode == CommandQueueMode::ASYNC; }
    bool trace_mode() { return this->mode == CommandQueueMode::TRACE; }
    bool passthrough_mode() { return this->mode == CommandQueueMode::PASSTHROUGH; }

    std::atomic<std::size_t> worker_thread_id = -1;
    std::atomic<std::size_t> parent_thread_id = -1;
    // Track the number of CQs using async vs pt mode
    inline static uint32_t num_async_cqs = 0;
    inline static uint32_t num_passthrough_cqs = 0;
};

}  // namespace v0

// Primitives used to place host only operations on the SW Command Queue.
// These are used in functions exposed through tt_metal.hpp or host_api.hpp
void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking);
void EnqueueSetRuntimeArgs(
    CommandQueue& cq,
    const std::shared_ptr<Kernel>& kernel,
    const CoreCoord& core_coord,
    std::shared_ptr<RuntimeArgs> runtime_args_ptr,
    bool blocking);
void EnqueueAddBufferToProgram(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    Program& program,
    bool blocking);

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, tt::tt_metal::EnqueueCommandType const& type);
std::ostream& operator<<(std::ostream& os, tt::tt_metal::CommandQueue::CommandQueueMode const& type);
