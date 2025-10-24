// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
// pruned unused includes

// pruned unused includes
#include "tt-metalium/program.hpp"
// pruned unused includes
#include "program/dispatch.hpp"

#include <umd/device/types/core_coordinates.hpp>

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

struct BufferRegion;
struct Event;
class Trace;

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType {
    ENQUEUE_READ_BUFFER,
    ENQUEUE_WRITE_BUFFER,
    GET_BUF_ADDR,
    ADD_BUFFER_TO_PROGRAM,
    SET_RUNTIME_ARGS,
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
    virtual ~Command() = default;
    virtual void process() {};
    virtual EnqueueCommandType type() = 0;
};

class EnqueueTerminateCommand : public Command {
private:
    uint32_t command_queue_id;
    SystemMemoryManager& manager;

public:
    EnqueueTerminateCommand(uint32_t command_queue_id, IDevice* device, SystemMemoryManager& manager);

    void process() override;

    EnqueueCommandType type() override { return EnqueueCommandType::TERMINATE; }

    constexpr bool has_side_effects() { return false; }
};

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::EnqueueCommandType& type);
