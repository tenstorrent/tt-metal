
#include <memory>
#include <thread>
#include <algorithm>
#include <chrono>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "frameworks/tt_dispatch/impl/command_queue_interface.hpp"
#include "frameworks/tt_dispatch/impl/thread_safe_queue.hpp"
#include "llrt/tt_debug_print_server.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/src/firmware/riscv/grayskull/noc/noc_parameters.h"

using namespace tt::tt_metal;
using std::shared_ptr;
using std::thread;
using std::unique_ptr;

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, FINISH, INVALID };

string EnqueueCommandTypeToString(EnqueueCommandType ctype);

// TEMPORARY! TODO(agrebenisan): need to use proper macro based on loading noc
#define NOC_X(x) x
#define NOC_Y(y) y

u32 noc_coord_to_u32(tt_xy_pair coord);

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void process(){};
    virtual EnqueueCommandType type() = 0;
    virtual const DeviceCommand device_command(u32 buffer_size) = 0;
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

    const DeviceCommand device_command(u32 dst);

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

    const DeviceCommand device_command(u32 src_address);

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

    const DeviceCommand device_command(u32);

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
    thread processing_thread;

    void enqueue_command(shared_ptr<Command> command, bool blocking);

    void enqueue_read_buffer(Buffer& buffer, vector<u32>& dst, bool blocking);

    void enqueue_write_buffer(Buffer& buffer, vector<u32>& src, bool blocking);

    void finish();

    friend void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking);
    friend void Finish(CommandQueue& cq);
};

void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking);

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking);

void Finish(CommandQueue& cq);
