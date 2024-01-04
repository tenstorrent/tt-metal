// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <fstream>


#include "jit_build/build.hpp"
#include "tt_metal/impl/dispatch/thread_safe_queue.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "noc/noc_parameters.h"

namespace tt::tt_metal {

using std::pair;
using std::set;
using std::shared_ptr;
using std::tuple;
using std::unique_ptr;

struct transfer_info {
    uint32_t size_in_bytes;
    uint32_t dst;
    uint32_t dst_noc_encoding;
    uint32_t num_receivers;
    bool last_transfer_in_group;
    bool linked;
};

struct ProgramMap {
    uint32_t num_workers;
    vector<uint32_t> program_pages;
    vector<transfer_info> program_page_transfers;
    vector<transfer_info> runtime_arg_page_transfers;
    vector<transfer_info> cb_config_page_transfers;
    vector<transfer_info> go_signal_page_transfers;
    vector<uint32_t> num_transfers_in_program_pages;
    vector<uint32_t> num_transfers_in_runtime_arg_pages;
    vector<uint32_t> num_transfers_in_cb_config_pages;
    vector<uint32_t> num_transfers_in_go_signal_pages;
};

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, ENQUEUE_PROGRAM, FINISH, WRAP, INVALID };

string EnqueueCommandTypeToString(EnqueueCommandType ctype);

// TEMPORARY! TODO(agrebenisan): need to use proper macro based on loading noc
#define NOC_X(x) x
#define NOC_Y(y) y

uint32_t get_noc_unicast_encoding(CoreCoord coord);

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void process(){};
    virtual EnqueueCommandType type() = 0;
    virtual const DeviceCommand assemble_device_command(uint32_t buffer_size) = 0;
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    SystemMemoryManager& manager;
    void* dst;
    uint32_t src_page_index;
    uint32_t pages_to_read;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_READ_BUFFER;
    uint32_t command_queue_channel;

   public:
    Buffer& buffer;
    uint32_t read_buffer_addr;
    EnqueueReadBufferCommand(
        uint32_t command_queue_channel,
        Device* device,
        Buffer& buffer,
        void* dst,
        SystemMemoryManager& manager,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt);

    const DeviceCommand assemble_device_command(uint32_t dst);

    void process();

    EnqueueCommandType type();
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;

    SystemMemoryManager& manager;
    const void* src;
    uint32_t dst_page_index;
    uint32_t pages_to_write;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_WRITE_BUFFER;
    uint32_t command_queue_channel;
   public:
    EnqueueWriteBufferCommand(
        uint32_t command_queue_channel,
        Device* device,
        Buffer& buffer,
        const void* src,
        SystemMemoryManager& manager,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt);

    const DeviceCommand assemble_device_command(uint32_t src_address);

    void process();

    EnqueueCommandType type();
};

class EnqueueProgramCommand : public Command {
   private:
    uint32_t command_queue_channel;
    Device* device;
    Buffer& buffer;
    ProgramMap& program_to_dev_map;
    const Program& program;
    SystemMemoryManager& manager;
    bool stall;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_PROGRAM;

   public:
    EnqueueProgramCommand(uint32_t command_queue_channel, Device*, Buffer&, ProgramMap&, SystemMemoryManager&, const Program& program, bool stall);

    const DeviceCommand assemble_device_command(uint32_t src_address);

    void process();

    EnqueueCommandType type();
};
// write to address chosen by us for finish... that way we don't need
// to mess with checking recv and acked
class FinishCommand : public Command {
   private:
    Device* device;
    SystemMemoryManager& manager;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::FINISH;
    uint32_t command_queue_channel;

   public:
    FinishCommand(uint32_t command_queue_channel, Device* device, SystemMemoryManager& manager);

    const DeviceCommand assemble_device_command(uint32_t);

    void process();

    EnqueueCommandType type();
};

class EnqueueWrapCommand : public Command {
   private:
    Device* device;
    SystemMemoryManager& manager;
    DeviceCommand::WrapRegion wrap_region;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::WRAP;
    uint32_t command_queue_channel;

   public:
    EnqueueWrapCommand(uint32_t command_queue_channel, Device* device, SystemMemoryManager& manager, DeviceCommand::WrapRegion wrap_region);

    const DeviceCommand assemble_device_command(uint32_t);

    void process();

    EnqueueCommandType type();
};

// Fwd declares
namespace detail{
    CommandQueue &GetCommandQueue(Device *device);
}

class CommandQueue {
   public:
    CommandQueue(Device* device, uint32_t command_queue_channel);

    ~CommandQueue();

   private:
    CoreCoord dispatch_core;
    uint32_t command_queue_channel;
    uint32_t command_queue_channel_size;
    SystemMemoryManager& manager;

    Device* device;

    map<uint64_t, unique_ptr<Buffer>>& program_to_buffer(const chip_id_t chip_id) {
        static map<chip_id_t, map<uint64_t, unique_ptr<Buffer>>> chip_to_program_to_buffer;
        if (chip_to_program_to_buffer.count(chip_id)) {
            return chip_to_program_to_buffer[chip_id];
        }
        map<uint64_t, unique_ptr<Buffer>> dummy;
        chip_to_program_to_buffer.emplace(chip_id, std::move(dummy));
        return chip_to_program_to_buffer[chip_id];
    }

    map<uint64_t, ProgramMap>& program_to_dev_map(const chip_id_t chip_id) {
        static map<chip_id_t, map<uint64_t, ProgramMap>> chip_to_program_to_dev_map;
        if (chip_to_program_to_dev_map.count(chip_id)) {
            return chip_to_program_to_dev_map[chip_id];
        }
        map<uint64_t, ProgramMap> dummy;
        chip_to_program_to_dev_map.emplace(chip_id, std::move(dummy));
        return chip_to_program_to_dev_map[chip_id];
    };

    void enqueue_command(Command& command, bool blocking);

    void enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking);

    void enqueue_write_buffer(Buffer& buffer, const void* src, bool blocking);

    void enqueue_program(Program& program, bool blocking);

    void finish();

    void wrap(DeviceCommand::WrapRegion wrap_region = DeviceCommand::WrapRegion::ISSUE, bool blocking = false);

    friend void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& dst, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<uint32_t>& src, bool blocking);
    friend void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, void* dst, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, const void* src, bool blocking);
    friend void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking);
    friend void Finish(CommandQueue& cq);
    friend void ClearProgramCache(CommandQueue& cq);
    friend CommandQueue &detail::GetCommandQueue(Device *device);
};

inline bool LAZY_COMMAND_QUEUE_MODE = false;

} // namespace tt::tt_metal
