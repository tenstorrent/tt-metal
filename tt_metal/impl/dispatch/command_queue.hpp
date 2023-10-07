/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <fstream>


#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/dispatch/thread_safe_queue.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/impl/program.hpp"
#include "noc/noc_parameters.h"


namespace tt::tt_metal {

using std::pair;
using std::set;
using std::shared_ptr;
using std::tuple;
using std::unique_ptr;

struct transfer_info {
    u32 size_in_bytes;
    u32 dst;
    u32 dst_noc_multicast_encoding;
    u32 num_receivers;
    bool last_multicast_in_group;
};

struct ProgramMap {
    u32 num_workers;
    vector<pair<u32, u32>> multicast_message_noc_coords;
    vector<u32> program_pages;
    vector<transfer_info> program_page_transfers;
    vector<transfer_info> host_page_transfers;
    vector<u32> num_transfers_in_program_pages;
    vector<u32> num_transfers_in_host_data_pages;
};

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, ENQUEUE_PROGRAM, FINISH, WRAP, INVALID };

string EnqueueCommandTypeToString(EnqueueCommandType ctype);

// TEMPORARY! TODO(agrebenisan): need to use proper macro based on loading noc
#define NOC_X(x) x
#define NOC_Y(y) y

u32 noc_coord_to_u32(CoreCoord coord);

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void process(){};
    virtual EnqueueCommandType type() = 0;
    virtual const DeviceCommand assemble_device_command(u32 buffer_size) = 0;
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    SystemMemoryWriter& writer;
    vector<u32>& dst;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_READ_BUFFER;

   public:
    Buffer& buffer;
    u32 read_buffer_addr;
    EnqueueReadBufferCommand(Device* device, Buffer& buffer, vector<u32>& dst, SystemMemoryWriter& writer);

    const DeviceCommand assemble_device_command(u32 dst);

    void process();

    EnqueueCommandType type();
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;

    SystemMemoryWriter& writer;
    vector<u32>& src;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_WRITE_BUFFER;

   public:
    EnqueueWriteBufferCommand(Device* device, Buffer& buffer, vector<u32>& src, SystemMemoryWriter& writer);

    const DeviceCommand assemble_device_command(u32 src_address);

    void process();

    EnqueueCommandType type();
};

class EnqueueProgramCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;
    ProgramMap& program_to_dev_map;
    vector<u32>& host_data;
    SystemMemoryWriter& writer;
    bool stall;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_PROGRAM;

   public:
    EnqueueProgramCommand(Device*, Buffer&, ProgramMap&, SystemMemoryWriter&, vector<u32>& host_data, bool stall);

    const DeviceCommand assemble_device_command(u32);

    void process();

    EnqueueCommandType type();
};

// Easiest way for us to process finish is to explicitly have the device
// write to address chosen by us for finish... that way we don't need
// to mess with checking recv and acked
class FinishCommand : public Command {
   private:
    Device* device;
    SystemMemoryWriter& writer;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::FINISH;

   public:
    FinishCommand(Device* device, SystemMemoryWriter& writer);

    const DeviceCommand assemble_device_command(u32);

    void process();

    EnqueueCommandType type();
};

class EnqueueWrapCommand : public Command {
   private:
    Device* device;
    SystemMemoryWriter& writer;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::WRAP;

   public:
    EnqueueWrapCommand(Device* device, SystemMemoryWriter& writer);

    const DeviceCommand assemble_device_command(u32);

    void process();

    EnqueueCommandType type();
};

void send_dispatch_kernel_to_device(Device* device);

class CommandQueue {
   public:
    CommandQueue(Device* device);

    ~CommandQueue();

   private:
    Device* device;
    SystemMemoryWriter sysmem_writer;
    TSQueue<shared_ptr<Command>>
        processing_thread_queue;  // These are commands that have not been placed in system memory
    // thread processing_thread;
    map<u64, unique_ptr<Buffer>>
        program_to_buffer;

    map<u64, ProgramMap> program_to_dev_map;

    void enqueue_command(shared_ptr<Command> command, bool blocking);

    void enqueue_read_buffer(Buffer& buffer, vector<u32>& dst, bool blocking);

    void enqueue_write_buffer(Buffer& buffer, vector<u32>& src, bool blocking);

    void enqueue_program(Program& program, bool blocking);

    void finish();

    void wrap();

    friend void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking);
    friend void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking);
    friend void Finish(CommandQueue& cq);
};

} // namespace tt::tt_metal
