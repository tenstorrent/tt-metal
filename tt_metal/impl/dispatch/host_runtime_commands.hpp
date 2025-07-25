// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "command_queue_interface.hpp"
#include "core_coord.hpp"
#include "device_command.hpp"
#include "env_lib.hpp"
#include "multi_producer_single_consumer_queue.hpp"
#include "dispatch_settings.hpp"
#include "tt-metalium/program.hpp"
#include "sub_device_types.hpp"
#include "trace/trace_buffer.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"
#include "worker_config_buffer.hpp"
#include "program/dispatch.hpp"

enum class CoreType;
namespace tt {
namespace tt_metal {
class IDevice;
class Program;
class SystemMemoryManager;
class WorkerConfigBufferMgr;
enum NOC : uint8_t;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

class BufferRegion;
class Event;
class Trace;

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

class Command {
public:
    Command() = default;
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
    uint32_t multicast_cores_launch_message_wptr = 0;
    uint32_t unicast_cores_launch_message_wptr = 0;
    // TODO: There will be multiple ids once programs support spanning multiple sub_devices
    SubDeviceId sub_device_id = SubDeviceId{0};
    program_dispatch::ProgramDispatchMetadata& dispatch_metadata;

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
        SubDeviceId sub_device_id,
        program_dispatch::ProgramDispatchMetadata& dispatch_md);

    void process() override;

    EnqueueCommandType type() override { return EnqueueCommandType::ENQUEUE_PROGRAM; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueTerminateCommand : public Command {
private:
    uint32_t command_queue_id;
    IDevice* device;
    SystemMemoryManager& manager;

public:
    EnqueueTerminateCommand(uint32_t command_queue_id, IDevice* device, SystemMemoryManager& manager);

    void process() override;

    EnqueueCommandType type() override { return EnqueueCommandType::TERMINATE; }

    constexpr bool has_side_effects() { return false; }
};

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::EnqueueCommandType& type);
